"""CPU smoke test for Laya-S2 with tiny random encoders (no downloads)."""

import numpy as np
import pytest
import torch
from PIL import Image

transformers = pytest.importorskip("transformers")
from transformers import ModernBertConfig, ModernBertModel, SiglipVisionConfig, SiglipVisionModel  # noqa: E402

from internnav.dataset.laya_s2_dataset import (  # noqa: E402
    GOAL,
    KEY_FIELDS,
    LayaS2Dataset,
    collate_laya_s2,
    enumerate_samples,
    frame_path,
    sample_key,
)
from internnav.model.basemodel.laya_s2 import (  # noqa: E402
    LayaS2,
    LayaS2Config,
    LossWeights,
    compute_loss,
)


def tiny_model():
    cfg = LayaS2Config(image_size=32, num_history=3, hist_pool=2, cur_pool=2, n_query=2, latent_dim=16, max_text_len=12)
    text = ModernBertModel(
        ModernBertConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=96,
            global_attn_every_n_layers=1,
            pad_token_id=0,
            attn_implementation="sdpa",
        )
    )
    vision = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=8
        )
    )
    return LayaS2(cfg, text, vision)


def fake_batch(B=4, H=3, S=32, Lt=7):
    torch.manual_seed(0)
    hist_mask = torch.tensor([[1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=torch.bool)[:B]
    lens = torch.tensor([7, 5, 3, 6])[:B]
    return dict(
        input_ids=torch.randint(1, 100, (B, Lt)),
        text_mask=torch.arange(Lt)[None] < lens[:, None],
        hist_pixels=torch.randn(B, H, 3, S, S),
        hist_mask=hist_mask,
        cur_pixels=torch.randn(B, 3, S, S),
        down_pixels=torch.randn(B, 3, S, S),
        is_goal=torch.tensor([True, False, True, False])[:B],
        action_idx=torch.tensor([-1, 2, -1, 0])[:B],
        goal_xy=torch.tensor([[0.1, 0.9], [0.0, 0.0], [0.55, 0.3], [0.0, 0.0]])[:B],
        latent=torch.randn(B, 2, 16),
        latent_mask=torch.tensor([True, False, False, False])[:B],
    )


def test_forward_backward_and_decide():
    model = tiny_model()
    b = fake_batch()
    out = model(**{k: b[k] for k in ("input_ids", "text_mask", "hist_pixels", "hist_mask", "cur_pixels", "down_pixels")},
                goal_xy=b["goal_xy"])
    A, P = model.n_action, model.n_patch
    assert out["logits"].shape == (4, A + P) and P == 16
    assert out["offsets"].shape == (4, P, 2)
    assert out["latent"].shape == (4, 2, 16)
    loss, stats = compute_loss(model, out, b, LossWeights())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.scorer[-1].weight.grad is not None and model.latent_queries.grad is not None

    model.eval()
    with torch.no_grad():
        out = model(**{k: b[k] for k in ("input_ids", "text_mask", "hist_pixels", "hist_mask", "cur_pixels", "down_pixels")})
    dec = model.decide(out)
    assert len(dec) == 4 and all(d["kind"] in ("action", "goal") for d in dec)


def test_padding_invariance():
    """A sample's output must not depend on what else is in the batch."""
    model = tiny_model().eval()
    b = fake_batch()
    keys = ("input_ids", "text_mask", "hist_pixels", "hist_mask", "cur_pixels", "down_pixels")
    with torch.no_grad():
        full = model(**{k: b[k] for k in keys}, goal_xy=b["goal_xy"])
        one = model(**{k: b[k][1:2] for k in keys}, goal_xy=b["goal_xy"][1:2])
    torch.testing.assert_close(full["logits"][1:2], one["logits"], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(full["latent"][1:2], one["latent"], atol=1e-4, rtol=1e-4)


def test_goal_patch_roundtrip():
    model = tiny_model()
    xy = torch.tensor([[0.0, 0.0], [0.99, 0.5], [0.3, 0.7]])
    idx, off = model.goal_to_patch(xy)
    torch.testing.assert_close(model.patch_to_goal(idx, off), xy, atol=1e-5, rtol=0)


def test_save_load(tmp_path):
    model = tiny_model().eval()
    model.save_pretrained(str(tmp_path))
    loaded = LayaS2.from_pretrained(str(tmp_path)).eval()
    b = fake_batch()
    keys = ("input_ids", "text_mask", "hist_pixels", "hist_mask", "cur_pixels", "down_pixels")
    with torch.no_grad():
        torch.testing.assert_close(model(**{k: b[k] for k in keys})["logits"], loaded(**{k: b[k] for k in keys})["logits"])


class _Tok:
    pad_token_id = 0

    def __call__(self, text, truncation=True, max_length=None):
        ids = [1] + [2 + (ord(c) % 90) for c in text][: max_length - 2] + [3]
        return {"input_ids": ids}


def test_dataset_pipeline(tmp_path):
    video = str(tmp_path / "scene" / "videos" / "chunk-000")
    ann = {
        "episodes": [
            {
                "id": 7,
                "instructions": "walk past the sofa and stop at the door",
                "video": video,
                # actions[1:] + [0] -> [2, 1, 1, 1, 1, 1, 1, 1, 1, 0]
                "actions": [-1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
                "pixel_goals": [[-1, [-1, -1]]] + [[4, [64, 96]]] * 9,
            }
        ]
    }
    goals, turns, stops = enumerate_samples(ann, 125, 0, 30, sample_step=4, num_future_steps=4)
    assert [s["start"] for s in goals] == [4, 8] and [s["kind"] for s in turns] == [1] and stops[0]["start"] == 9
    for s in goals + turns + stops:
        for f in range(10):
            for down in (False, True):
                p = frame_path(s, f, look_down=down)
                (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((96, 128, 3), f * 20, np.uint8)).save(p)

    cfg = LayaS2Config(image_size=32, num_history=3, n_query=2, latent_dim=16)
    store = tmp_path / "latents"
    store.mkdir()
    np.save(store / "latents.npy", np.ones((1, 2, 16), np.float16))
    (store / "keys.txt").write_text(sample_key(*(goals[0][k] for k in KEY_FIELDS)) + "\n")

    ds = LayaS2Dataset(goals + turns + stops, _Tok(), cfg, teacher_latents=str(store))
    items = [ds[i] for i in range(len(ds))]
    g = next(it for it in items if it["is_goal"])
    torch.testing.assert_close(g["goal_xy"], torch.tensor([0.5, 1.0]))  # (64 / 128, 96 / 96)
    assert g["latent_mask"] and g["hist_mask"].sum() == 3
    turn = next(it for it in items if not it["is_goal"] and it["action_idx"] == cfg.actions.index(2))
    assert turn["hist_mask"].sum() == 0  # turn at frame 0 has no history
    batch = collate_laya_s2(items, pad_id=0)
    assert batch["hist_pixels"].shape == (4, 3, 3, 32, 32)
    assert batch["is_goal"].tolist() == [True, True, False, False]
    assert GOAL == 0


def test_agent_decide():
    from internnav.model.basemodel.laya_s2.agent import LayaS2Agent

    model = tiny_model().eval()
    agent = LayaS2Agent(model, s1=None, tokenizer=_Tok(), device=torch.device("cpu"))
    img = Image.fromarray(np.random.randint(0, 255, (96, 128, 3), np.uint8))
    for history in ([], [img] * 5):  # first step (no history) and more frames than slots
        d = agent.decide("go to the kitchen", history, img, img)
        assert d["kind"] in ("action", "goal") and 0.0 <= d["escalate_prob"] <= 1.0
        if d["kind"] == "goal":
            assert d["latent"].shape == (2, 16) and all(0 <= v <= 1 for v in d["goal_xy"])
