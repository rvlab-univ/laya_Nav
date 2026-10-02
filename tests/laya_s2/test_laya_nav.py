"""LayaNav: single model (Laya-S2 decision + trajectory head) — CPU tests with tiny random encoders."""

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")
pytest.importorskip("scipy")
from test_laya_s2_smoke import _Tok, fake_batch, tiny_model  # noqa: E402

from internnav.dataset.laya_s2_dataset import (  # noqa: E402
    LayaS2Dataset,
    collate_laya_s2,
    enumerate_samples,
    frame_path,
    trajectory_frame_ids,
    trajectory_target,
)
from internnav.model.basemodel.laya_s2 import (  # noqa: E402
    LayaNav,
    LayaNavConfig,
    TrajLossWeights,
    traj_loss,
)

KEYS = ("input_ids", "text_mask", "hist_pixels", "hist_mask", "cur_pixels", "down_pixels")


def tiny_nav():
    s2 = tiny_model()
    cfg = LayaNavConfig(**{**vars(s2.cfg), "traj_dim": 32, "traj_layers": 2})
    return LayaNav(cfg, s2.text, s2.vision)


def camera_poses_along_x(n, step=0.25, pitch=30):
    """Camera extrinsics of a robot driving straight along its x axis (inverse of the dataset transform)."""
    t_r2c = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    r = np.radians(pitch)
    t_deg = np.array([[1, 0, 0, 0], [0, np.cos(-r), -np.sin(-r), 0], [0, np.sin(-r), np.cos(-r), 0], [0, 0, 0, 1]])
    poses = []
    for i in range(n):
        t = np.eye(4)
        t[0, 3] = i * step
        poses.append((t @ t_r2c @ t_deg).tolist())
    return poses


def test_forward_plan_and_c1_gradients():
    model = tiny_nav()
    b = fake_batch()
    traj_mask = b["is_goal"].clone()
    traj_pixels = torch.randn(4, 3, 32, 32)
    # c1: only the trajectory head is trainable
    model.requires_grad_(False)
    for p in model.traj_parameters():
        p.requires_grad_(True)
    out = model(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"], traj_pixels=traj_pixels, traj_mask=traj_mask)
    assert out["traj"].shape == (2, 32, 3) and out["traj_idx"].tolist() == [0, 2]
    loss, stats = traj_loss(out["traj"], torch.randn(2, 32, 3), torch.ones(2, dtype=torch.bool), TrajLossWeights())
    loss.backward()
    trained = {n for n, p in model.named_parameters() if p.grad is not None}
    assert trained and all(n.startswith("traj_") for n in trained)
    assert stats["n_traj"] == 2 and stats["fde"] >= 0

    # inference path: decide, then plan from the cached memory with a new frame (System 1 rate)
    model.eval()
    with torch.no_grad():
        out = model(**{k: b[k][:1] for k in KEYS})
        memory = model.plan_memory(out)
        t1 = model.plan(memory, out["down_feat"], model.encode_frame(torch.randn(1, 3, 32, 32)))
        # same frames -> same trajectory as the training-time forward
        cur = torch.randn(1, 3, 32, 32)
        t2 = model.plan(memory, out["down_feat"], model.encode_frame(cur))
        t3 = model(**{k: b[k][:1] for k in KEYS}, traj_pixels=cur)["traj"]
    assert t1.shape == (1, 32, 3)
    torch.testing.assert_close(t2, t3)


def test_init_from_laya_s2_and_save_load(tmp_path):
    s2 = tiny_model().eval()
    s2.save_pretrained(str(tmp_path / "s2"))
    nav = LayaNav.load_any(str(tmp_path / "s2"), traj_dim=32, traj_layers=2).eval()
    b = fake_batch()
    with torch.no_grad():  # decision part is exactly the Laya-S2 checkpoint
        torch.testing.assert_close(nav(**{k: b[k] for k in KEYS})["logits"], s2(**{k: b[k] for k in KEYS})["logits"])
    nav.save_pretrained(str(tmp_path / "nav"))
    again = LayaNav.load_any(str(tmp_path / "nav")).eval()
    assert again.cfg.model_type == "laya_nav" and again.cfg.traj_dim == 32
    with torch.no_grad():
        x = dict(traj_pixels=torch.randn(4, 3, 32, 32))
        torch.testing.assert_close(nav(**{k: b[k] for k in KEYS}, **x)["traj"], again(**{k: b[k] for k in KEYS}, **x)["traj"])


def test_trajectory_target_straight_line():
    goal_len = 8
    s = dict(start=2, goal_len=goal_len, pitch_2=30, poses=camera_poses_along_x(20))
    ids = trajectory_frame_ids(goal_len)
    assert ids.tolist() == [0, 2, 4, 6]
    t = trajectory_target(s, 0, 32)
    assert t.shape == (32, 3)
    xy = np.cumsum(t[:, :2] / 4, 0)
    assert abs(xy[-1, 0] - goal_len * 0.25) < 0.05 and np.abs(xy[:, 1]).max() < 1e-3  # 2 m straight ahead
    assert abs(t[0, 0] - 0.4) < 0.02  # 0.1 m steps, scaled by 4
    later = trajectory_target(s, 4, 32)  # starting halfway: 1 m left
    assert abs(np.cumsum(later[:, 0] / 4)[-1] - 1.0) < 0.05


def test_dataset_with_traj_and_agent(tmp_path):
    video = str(tmp_path / "scene" / "videos" / "chunk-000")
    n = 12
    ann = {
        "episodes": [
            {
                "id": 0,
                "instructions": "walk straight to the door",
                "video": video,
                "actions": [-1] + [1] * (n - 1),
                "pixel_goals": [[8, [64, 90]]] * n,
                "poses_125cm_30deg": camera_poses_along_x(n),
            }
        ]
    }
    goals, _, _ = enumerate_samples(ann, 125, 0, 30, sample_step=4, num_future_steps=4)
    assert goals and goals[0]["poses"] is ann["episodes"][0]["poses_125cm_30deg"]  # shared, not copied
    for f in range(n + 1):
        for down in (False, True):
            p = frame_path(goals[0], f, look_down=down)
            (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((96, 128, 3), f * 10, np.uint8)).save(p)
    cfg = tiny_nav().cfg
    ds = LayaS2Dataset(goals, _Tok(), cfg, with_traj=True, traj_steps=cfg.traj_steps)
    batch = collate_laya_s2([ds[i] for i in range(len(ds))], pad_id=0)
    assert batch["traj_mask"].all() and batch["traj"].shape == (len(ds), 32, 3)
    assert (batch["traj"][:, 0, 0] > 0.3).all()  # moving forward

    from internnav.model.basemodel.laya_s2.agent import LayaNavAgent

    agent = LayaNavAgent(tiny_nav().eval(), _Tok(), torch.device("cpu"))
    img = Image.fromarray(np.random.randint(0, 255, (96, 128, 3), np.uint8))
    agent.decide("go to the door", [img] * 3, img, img)
    agent.start_goal({}, img, torch.zeros(224, 224))
    traj = agent.plan(img, torch.zeros(224, 224))
    assert traj.shape == (1, 32, 3) and torch.isfinite(traj).all()
