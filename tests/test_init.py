"""Starting a run from another checkpoint (lightnav.model.init_from, design_C_review.md 5.5): the vision encoder may
change (224 -> 256 px) and the current or past frames' grid may grow (2 x 2 -> 4 x 4, 1 x 1 -> 2 x 2); everything else
is copied."""
import pytest
import torch
import torch.nn.functional as F

from conftest import fake_batch, tiny_model
from lightnav.model import LayaNative, _resize_positions, init_from, load, save

VOCAB = ["bathroom", "sofa", "living room", "chairs", "kitchen", "bed"]
INSTRUCTIONS = ["Walk past the sofa into the bathroom and stop.", "Turn right at the kitchen and wait by the bed."]


def student(tok, image_size, cur_grid, encoder, hist_grid=1, seed=0, vocab=VOCAB):
    """A tagged tiny student on a vision tower of 4-pixel patches (image_size / 4 patches a side)."""
    from transformers import SiglipVisionConfig, SiglipVisionModel

    base = tiny_model(tok, seed=seed)
    torch.manual_seed(seed + 10)
    vision = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                                  intermediate_size=64, image_size=image_size, patch_size=4))
    cfg = base.cfg
    cfg.vision_encoder, cfg.image_size = encoder, image_size
    cfg.cur_grid, cfg.hist_grid, cfg.tags_current, cfg.tags_history, cfg.tag_vocab = cur_grid, hist_grid, 3, 1, list(vocab)
    model = LayaNative(cfg, base.dm, vision)
    model.tag_text.copy_(F.normalize(torch.randn(len(vocab), 32), dim=-1))
    model.set_tag_stats(torch.randn(50, len(vocab)))
    model.set_tag_tokens(tok)
    return model.eval()


def old_student(tok, tmp_path):
    """Run D's layout at 32 px (8 x 8 patches, a 2 x 2 current frame), saved."""
    model = student(tok, 32, 2, "tiny-224", seed=0)
    for p in (model.vis_proj[0].weight, model.pos_cur, model.pos_hist, model.frame_emb.weight, model.tag_type):
        p.data.normal_()                                       # as if trained: not the fresh initialisation
    save(model, tok, str(tmp_path / "old"))
    return model, str(tmp_path / "old")


def test_resize_positions_is_bilinear_and_row_major():
    pos = torch.arange(4.0)[:, None].repeat(1, 3)              # 2 x 2: top-left 0, top-right 1, bottom 2 and 3
    out = _resize_positions(pos, 16).view(4, 4, 3)[..., 0]
    assert out[0, 0] == 0 and out[0, 3] == 1 and out[3, 0] == 2 and out[3, 3] == 3     # corners keep their places
    assert torch.allclose(out[0], torch.tensor([0, 0.25, 0.75, 1]))
    assert torch.allclose(_resize_positions(torch.ones(4, 5), 16), torch.ones(16, 5))
    assert torch.equal(_resize_positions(pos, 4), pos)
    with pytest.raises(ValueError):
        _resize_positions(torch.zeros(6, 2), 16)


def test_same_model_is_a_plain_copy(tok, tmp_path):
    old, path = old_student(tok, tmp_path)
    new = student(tok, 32, 2, "tiny-224", seed=3)
    assert init_from(new, path) == []
    batch = fake_batch(tok, old, INSTRUCTIONS, n_frames=[3, 40])
    with torch.no_grad():
        assert torch.allclose(new(batch), old(batch), atol=1e-5)
    assert torch.equal(new.tag_mu, old.tag_mu)


def test_new_encoder_and_grid(tok, tmp_path):
    old, path = old_student(tok, tmp_path)
    new = student(tok, 64, 4, "tiny-256", seed=3)               # 16 x 16 patches, a 4 x 4 current frame
    own = {k: v.clone() for k, v in new.state_dict().items()}
    notes = init_from(new, path)
    assert len(notes) == 2 and "tiny-224 -> tiny-256" in notes[0] and "2 x 2 -> 4 x 4" in notes[1]
    sd, before = new.state_dict(), old.state_dict()
    for k, v in sd.items():
        if k.startswith("vision.") or k in ("tag_text", "tag_mu", "tag_sd"):
            assert torch.equal(v, own[k]), k                    # the new encoder's, untouched
        elif k == "pos_cur":
            assert v.shape == (16, before[k].shape[1])
            assert torch.allclose(v.view(4, 4, -1)[0, 0], before[k][0]) and torch.allclose(v.view(4, 4, -1)[3, 3], before[k][3])
        else:
            assert torch.equal(v, before[k]), k                 # language side, head, projection, embeddings
    batch = fake_batch(tok, new, INSTRUCTIONS, n_frames=[3, 40])
    with torch.no_grad():
        tokens = new.frame_tokens(new.encode_frames(batch["pixels"]), batch["frame_slot"])
        logits = new(batch)
    sizes = [t.shape[0] for t in tokens]
    assert sizes.count(16) == 2 and set(sizes) == {1, 16}      # 4 x 4 for each current frame, 1 per past frame
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()


def test_past_frames_grow_to_2x2(tok, tmp_path):
    """Run H1 (design_C_review.md 5.10): D's encoder and current frame, each past frame as 2 x 2 tokens."""
    old, path = old_student(tok, tmp_path)
    new = student(tok, 32, 2, "tiny-224", hist_grid=2, seed=3)
    notes = init_from(new, path)
    assert notes == ["past frames 1 x 1 -> 2 x 2 tokens: position embeddings interpolated"]
    sd, before = new.state_dict(), old.state_dict()
    for k, v in sd.items():
        if k == "pos_hist":
            assert v.shape == (4, before[k].shape[1]) and torch.allclose(v, before[k].expand(4, -1))
        else:
            assert torch.equal(v, before[k]), k
    batch = fake_batch(tok, new, INSTRUCTIONS, n_frames=[3, 40])
    with torch.no_grad():
        tokens = new.frame_tokens(new.encode_frames(batch["pixels"]), batch["frame_slot"])
        logits = new(batch)
    sizes = [t.shape[0] for t in tokens]
    assert sizes == [4] * len(sizes) and len(sizes) == 3 + 8    # 3 frames, then 8: 4 tokens each
    assert logits.shape == (2, 10) and torch.isfinite(logits).all()
    save(new, tok, str(tmp_path / "h1"))
    again, _ = load(str(tmp_path / "h1"))
    assert again.cfg.hist_grid == 2
    with torch.no_grad():
        assert torch.allclose(again.eval()(batch), logits, atol=1e-5)


def test_the_new_student_saves_and_loads_at_its_own_size(tok, tmp_path):
    """What scripts/local_drive.sh relies on: the agent reads the image size and encoder from the checkpoint."""
    _, path = old_student(tok, tmp_path)
    new = student(tok, 64, 4, "tiny-256", seed=3)
    init_from(new, path)
    save(new, tok, str(tmp_path / "new"))
    again, _ = load(str(tmp_path / "new"))
    assert again.cfg.image_size == 64 and again.cfg.cur_grid == 4 and again.grid == 16
    assert again.cfg.vision_encoder == "tiny-256"
    batch = fake_batch(tok, new, INSTRUCTIONS, n_frames=[3, 40])
    assert batch["pixels"].shape[-1] == 64
    with torch.no_grad():
        assert torch.allclose(again.eval()(batch), new(batch), atol=1e-5)


def test_what_cannot_be_carried_over(tok, tmp_path):
    _, path = old_student(tok, tmp_path)
    with pytest.raises(ValueError, match="tag phrases"):
        init_from(student(tok, 32, 2, "tiny-224", vocab=VOCAB[:5]), path)
