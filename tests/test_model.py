"""LayaNative on tiny random encoders (CPU): the Laya forward is reproduced, images do not disturb the text half."""
import os

import pytest
import torch

from conftest import fake_batch, tiny_model
from lightnav.model import forward_embeds, load, save
from lightnav.questions import ANSWERS, KEYS, QUESTION

INSTRUCTIONS = ["Walk past the sofa and turn left at the door.",
                "Go up the stairs, then wait by the second window on your right. Stop there."]


def test_forward_embeds_equals_laya_forward(tok, model):
    """Gate G2: with no image tokens, embeddings in == ids in."""
    from laya.common import collate_items
    from laya.train import encode_item, encode_state, make_item, target_from_expected, to_internal

    questions = {
        "action": (QUESTION, "left 30"),
        "arrived": ({"type": "choice", "instructions": "Has the robot arrived?",
                     "criteria": {"A": "yes", "B": "no"}}, "B"),
        "amount": ({"type": "score", "instructions": "How large should the move be?",
                    "criteria": ["small", "medium", "large"]}, 1),
    }
    state = encode_state(tok, {"instruction": INSTRUCTIONS[1]}, 448)
    items = []
    for qid, (question, expected) in questions.items():
        q = to_internal(qid, question)
        item, reason = make_item(tok, q, target_from_expected(q, expected), state, 192, 448)
        assert reason is None
        items.append(encode_item(tok, item, 448, 192))
    batch = collate_items([items], tok.pad_token_id)
    args = [batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    with torch.no_grad():
        want, _ = model.dm(*args)
        embeds = model.dm.encoder.get_input_embeddings()(args[0])
        got = forward_embeds(model.dm, embeds, *args[1:])
    assert got.shape == want.shape == (3, 10)
    assert torch.allclose(got, want, atol=1e-5)
    assert (got[1, 2:] == -1e4).all()          # a 2-option question masks the unused slots


def test_question_fits_the_head_budget(tok):
    from laya.common import build_head, render_options
    from laya.train import to_internal

    q = to_internal("action", QUESTION)
    ids, markers, stats = build_head(tok, q, 192)
    roomy, _, _ = build_head(tok, q, 4096)
    assert ids == roomy and len(ids) <= 192                 # nothing was cut to make it fit
    assert stats == {"options": 10, "options_distinct": 10, "tokens_per_option": None}
    assert all(ids[m] == tok.mask_token_id for m in markers) and len(markers) == 10
    assert [o.split(":")[0] for o in render_options(q)] == KEYS and len(KEYS) == len(ANSWERS)


def test_images_go_before_the_last_sep_and_markers_stay(tok, model):
    batch = fake_batch(tok, model, INSTRUCTIONS, n_frames=[3, 40])
    emb = model.dm.encoder.get_input_embeddings()
    with torch.no_grad():
        tokens = model.frame_tokens(model.encode_frames(batch["pixels"]), batch["frame_slot"])
        embeds, mask = model.assemble(batch["input_ids"], batch["text_len"], tokens, batch["frame_owner"])
        sep, marker = emb.weight[tok.sep_token_id], emb.weight[tok.mask_token_id]
    assert [t.shape[0] for t in tokens] == [49, 49, 196] + [49] * 7 + [196]
    n_image = [2 * 49 + 196, 7 * 49 + 196]
    for b in range(2):
        length = int(mask[b].sum())
        assert length == int(batch["text_len"][b]) + n_image[b]
        assert torch.equal(embeds[b, length - 1], sep)                     # still closes the sequence
        assert not torch.equal(embeds[b, length - 2], sep)
        for m in batch["marker_pos"][b].tolist():
            assert torch.equal(embeds[b, m], marker)
        text_before = int(batch["text_len"][b]) - 1
        assert torch.equal(embeds[b, :text_before], emb(batch["input_ids"][b, :text_before]))
    assert int(mask[0].sum()) < mask.shape[1] == int(mask[1].sum())        # the shorter row is padded


def test_forward_shape_and_padding_invariance(tok, model):
    batch = fake_batch(tok, model, INSTRUCTIONS, n_frames=[3, 40])
    with torch.no_grad():
        logits = model(batch)
        alone = model(fake_batch(tok, model, INSTRUCTIONS[:1], n_frames=[3]))
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    assert torch.allclose(logits[0], alone[0], atol=1e-4)                  # padding does not change a row


def test_images_change_the_answer(tok, model):
    with torch.no_grad():
        a = model(fake_batch(tok, model, INSTRUCTIONS[:1], n_frames=[9], seed=0))
        b = model(fake_batch(tok, model, INSTRUCTIONS[:1], n_frames=[9], seed=1))
    assert not torch.allclose(a, b, atol=1e-4)


def test_cached_pooled_history_gives_the_same_logits(tok, model):
    batch = fake_batch(tok, model, INSTRUCTIONS[:1], n_frames=[20])
    with torch.no_grad():
        feats = model.encode_frames(batch["pixels"])
        cached = [model.pool_history(f[None])[0] for f in feats[:-1]] + [feats[-1]]
        assert torch.allclose(model(batch, feats=cached), model(batch), atol=1e-5)


def test_align_stage_trains_only_the_new_parts(tok):
    model = tiny_model(tok)
    model.set_stage("align")
    new = {id(p) for p in model.new_parameters()}
    assert all(p.requires_grad == (id(p) in new) for p in model.parameters())
    model(fake_batch(tok, model, INSTRUCTIONS, n_frames=[3, 40])).sum().backward()
    assert all(p.grad is None for p in model.dm.parameters())
    assert all(p.grad is None for p in model.vision.parameters())
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.new_parameters())
    assert len(model.param_groups(1e-5, 1e-4, 1e-5)) == 1

    model.set_stage("nav")
    assert all(p.requires_grad for p in model.parameters())
    groups = model.param_groups(1e-5, 1e-4, 1e-6)
    assert [g["lr"] for g in groups] == [1e-5, 1e-6, 1e-4]
    assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in model.parameters())


def test_rlcd_loss_step_lowers_the_loss(tok):
    from laya.train import rlcd_loss, soft_ce_loss

    model = tiny_model(tok)
    model.set_stage("nav")
    batch = fake_batch(tok, model, INSTRUCTIONS, n_frames=[3, 40])
    target = torch.zeros(2, 10)
    target[0, 4] = target[1, 0] = 1.0
    mask, qtype = torch.ones(2, 10, dtype=torch.bool), torch.zeros(2, dtype=torch.long)
    opt = torch.optim.AdamW(model.param_groups(1e-3, 1e-3, 1e-3))
    model.eval()                     # no dropout, so the before/after comparison is exact
    before = soft_ce_loss(model(batch), target, mask).item()
    for _ in range(8):
        opt.zero_grad()
        rlcd_loss(model(batch), target, mask, qtype, sigma=0.4).backward()
        opt.step()
    assert soft_ce_loss(model(batch), target, mask).item() < before


def test_save_and_load(tok, model, tmp_path):
    batch = fake_batch(tok, model, INSTRUCTIONS, n_frames=[3, 40])
    save(model, tok, str(tmp_path / "ckpt"))
    again, tok2 = load(str(tmp_path / "ckpt"))
    with torch.no_grad():
        assert torch.allclose(again.eval()(batch), model(batch), atol=1e-5)
    assert tok2.mask_token_id == tok.mask_token_id and again.cfg == model.cfg


@pytest.mark.skipif(not os.environ.get("LIGHTNAV_REAL_CKPT"), reason="set LIGHTNAV_REAL_CKPT=english to run")
def test_forward_embeds_equals_laya_forward_on_the_real_checkpoint():
    from laya.common import collate_items
    from laya.train import encode_item, encode_state, load_checkpoint, make_item, target_from_expected, to_internal

    dm, tok, cfg = load_checkpoint(os.environ["LIGHTNAV_REAL_CKPT"])
    dm.eval()
    q = to_internal("action", QUESTION)
    state = encode_state(tok, {"instruction": INSTRUCTIONS[1]}, cfg["max_len"])
    item, _ = make_item(tok, q, target_from_expected(q, "stop"), state, cfg["head_max_len"], cfg["max_len"])
    batch = collate_items([[encode_item(tok, item, cfg["max_len"], cfg["head_max_len"])]], tok.pad_token_id)
    args = [batch[k] for k in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")]
    with torch.no_grad():
        want, _ = dm(*args)
        got = forward_embeds(dm, dm.encoder.get_input_embeddings()(args[0]), *args[1:])
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_pooled_current_frame(tok):
    from lightnav.model import LayaNative

    base = tiny_model(tok)
    cfg = base.cfg
    cfg.cur_grid, cfg.hist_grid = 2, 1
    model = LayaNative(cfg, base.dm, base.vision).eval()
    batch = fake_batch(tok, model, INSTRUCTIONS[:1], n_frames=[12])
    with torch.no_grad():
        tokens = model.frame_tokens(model.encode_frames(batch["pixels"]), batch["frame_slot"])
        logits = model(batch)
    sizes = {s: t.shape[0] for s, t in zip(batch["frame_slot"].tolist(), tokens)}
    assert sizes[7] == 4 and all(n == 1 for s, n in sizes.items() if s != 7)
    assert logits.shape == (1, 10) and torch.isfinite(logits).all()
