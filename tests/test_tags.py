"""Landmark tags (lightnav.model, design B 7.2.3) on a tiny random model: which words, where, save / load."""
import random

import torch
import torch.nn.functional as F

from conftest import fake_batch, tiny_model
from lightnav import tasks as T
from lightnav.model import LayaNative, load, save, tag_prompt, tag_vocabulary

VOCAB = ["bathroom", "sofa", "living room", "chairs", "kitchen", "bed"]
INSTRUCTION = ["Walk past the sofa into the bathroom and stop."]


def tagged(tok, current=2, past=1):
    base = tiny_model(tok)
    cfg = base.cfg
    cfg.tags_current, cfg.tags_history, cfg.tag_vocab = current, past, list(VOCAB)
    model = LayaNative(cfg, base.dm, base.vision)
    torch.manual_seed(1)
    model.tag_text.copy_(F.normalize(torch.randn(len(VOCAB), model.vision.config.hidden_size), dim=-1))
    model.set_tag_tokens(tok)
    return model.eval()


def test_vocabulary_and_prompts():
    v = tag_vocabulary()
    assert v == T.tag_phrases() and "living room" in v and "chairs" in v and len(v) == len(set(v))
    assert tag_prompt("chairs") == "a photo of chairs." and tag_prompt("sofa") == "a photo of a sofa."


def test_tag_ids_read_like_the_instruction(tok):
    model = tagged(tok)
    for i, p in enumerate(VOCAB):
        assert tok.decode([t for t in model.tag_ids[i].tolist() if t >= 0]).strip() == p
    sofa = [t for t in model.tag_ids[1].tolist() if t >= 0]
    inst = tok(INSTRUCTION[0], add_special_tokens=False)["input_ids"]
    assert any(inst[j:j + len(sofa)] == sofa for j in range(len(inst)))      # the very same token(s)


def test_row_tags_current_first_then_past_without_repeats(tok):
    model = tagged(tok, current=2, past=1)
    batch = fake_batch(tok, model, INSTRUCTION * 2, n_frames=[12, 3])
    with torch.no_grad():
        feats = model.encode_frames(batch["pixels"])
        scores = model.tag_scores(feats)
        words = model.row_tags(feats, batch["frame_slot"], batch["frame_owner"], 2)
    for b in range(2):
        idx = [i for i, o in enumerate(batch["frame_owner"].tolist()) if o == b]
        slots = batch["frame_slot"][idx].tolist()
        cur = idx[slots.index(7)]
        want, seen = [], set()
        for v in scores[cur].topk(2).indices.tolist():
            want.append((v, True)) if v not in seen else None
            seen.add(v)
        for i in sorted((i for i in idx if i != cur), key=lambda i: int(batch["frame_slot"][i])):
            v = scores[i].topk(1).indices.item()
            if v not in seen:
                want.append((v, False))
                seen.add(v)
        assert words[b] == want


def test_segment_sits_between_text_and_images(tok):
    model = tagged(tok)
    plain = tiny_model(tok).eval()
    batch = fake_batch(tok, model, INSTRUCTION, n_frames=[12])
    with torch.no_grad():
        feats = model.encode_frames(batch["pixels"])
        words = model.row_tags(feats, batch["frame_slot"], batch["frame_owner"], 1)[0]
        seg = model.tag_segment(words)
        e_tag, _ = model.assemble(batch["input_ids"], batch["text_len"], model.frame_tokens(feats, batch["frame_slot"]),
                                  batch["frame_owner"], [seg])
        e_plain, _ = model.assemble(batch["input_ids"], batch["text_len"],
                                    model.frame_tokens(feats, batch["frame_slot"]), batch["frame_owner"])
    n = int(batch["text_len"][0])
    assert e_tag.shape[1] == e_plain.shape[1] + seg.shape[0]
    assert torch.allclose(e_tag[0, n - 1:n - 1 + seg.shape[0]], seg, atol=1e-6)
    assert torch.allclose(e_tag[0, n - 1 + seg.shape[0]:], e_plain[0, n - 1:], atol=1e-6)
    ids = model.tag_prefix.tolist()
    text = tok.decode(ids + [t for j, (v, _) in enumerate(words)
                             for t in [x for x in model.tag_ids[v].tolist() if x >= 0]
                             + (model.tag_sep.tolist() if j + 1 < len(words) else [])])
    assert text.strip().startswith("seen:") and VOCAB[words[0][0]] in text
    with torch.no_grad():
        assert not torch.allclose(plain(batch), model(batch), atol=1e-4)
    assert not any("tag" in k for k in plain.state_dict())


def test_override_replaces_the_words(tok):
    model = tagged(tok)
    batch = fake_batch(tok, model, INSTRUCTION * 2, n_frames=[12, 12])
    batch["tag_override"] = torch.tensor([[4, 5, -1], [-1, -1, -1]])
    with torch.no_grad():
        feats = model.encode_frames(batch["pixels"])
        words = model.row_tags(feats, batch["frame_slot"], batch["frame_owner"], 2, batch["tag_override"])
        own = model.row_tags(feats, batch["frame_slot"], batch["frame_owner"], 2)
    assert words[0] == [(4, False), (5, False)] and words[1] == own[1]


def test_statistics_normalise_and_survive_save_load(tok, tmp_path):
    model = tagged(tok)
    batch = fake_batch(tok, model, INSTRUCTION, n_frames=[12])
    with torch.no_grad():
        feats = model.encode_frames(batch["pixels"])
        raw = model.tag_scores(feats, normalise=False)
        model.set_tag_stats(raw)
        z = model.tag_scores(feats)
    assert torch.allclose(z.mean(0), torch.zeros(len(VOCAB)), atol=1e-4)
    save(model, tok, str(tmp_path / "ckpt"))
    again, _ = load(str(tmp_path / "ckpt"))
    again.eval()
    assert again.cfg.tag_vocab == VOCAB and torch.equal(again.tag_mu, model.tag_mu)
    assert torch.equal(again.tag_text, model.tag_text) and torch.equal(again.tag_ids, model.tag_ids)
    with torch.no_grad():
        assert torch.allclose(model(batch), again(batch), atol=1e-5)


def test_tag_type_trains_and_the_table_does_not(tok):
    model = tagged(tok).train()
    assert any(p is model.tag_type for p in model.new_parameters())
    assert not any(p is model.tag_text for p in model.parameters())
    model(fake_batch(tok, model, INSTRUCTION, n_frames=[12])).sum().backward()
    assert model.tag_type.grad is not None and model.tag_type.grad.abs().sum() > 0


def test_synthetic_matching_tags_are_separable(tok, tmp_path):
    from test_tasks import EPISODES, write_episodes

    rows, roots = write_episodes(tmp_path, EPISODES)
    model = tagged(tok)
    ds = T.MatchDataset(T.routes_from_rows(rows), roots, tok, model.cfg, synthetic=1.0, synthetic_words=4)
    random.seed(0)
    for _ in range(40):
        it = ds[random.randrange(len(ds))]
        if "tag_override" not in it:
            continue
        shown = tok.decode(it["ids"])
        own = {VOCAB.index(T.phrase(g, w)) for g, w in T.landmarks(shown) if T.phrase(g, w) in VOCAB}
        hit = bool(own & set(it["tag_override"]))
        assert hit == (it["kind"] == "pos") and len(it["tag_override"]) == 4
