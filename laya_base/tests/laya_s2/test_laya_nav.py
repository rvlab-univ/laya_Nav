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


def tiny_nav(**head):
    s2 = tiny_model()
    cfg = LayaNavConfig(**{**vars(s2.cfg), "traj_dim": 32, "traj_layers": 2, **head})
    return LayaNav(cfg, s2.text, s2.vision)


NEW_HEAD = dict(traj_fuse_layers=1, traj_goal_mark=True)


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
    assert batch["traj_mask"].all() and batch["traj"].shape == (len(ds), 1, 32, 3)
    assert (batch["traj"][:, 0, 0, 0] > 0.3).all()  # moving forward

    # several starting frames per goal sample (goal_len 8 -> starts 0, 2, 4, 6)
    multi = LayaS2Dataset(goals, _Tok(), cfg, with_traj=True, traj_steps=cfg.traj_steps, traj_starts=6)
    item = multi[0]
    assert item["traj_pixels"].shape == (6, 3, 32, 32) and item["traj_mask"].tolist() == [True] * 4 + [False] * 2
    remaining = (item["traj"][:4, :, 0] / 4).sum(1)  # metres left from each start: 2.0, 1.5, 1.0, 0.5
    assert torch.allclose(remaining, torch.tensor([2.0, 1.5, 1.0, 0.5]), atol=0.06)
    assert LayaS2Dataset._pick_starts(np.array([0, 2, 4, 6]), 1, False) == [4]  # validation, one start: the middle
    picked = LayaS2Dataset._pick_starts(np.array([0, 2, 4, 6]), 3, True)
    assert len(set(picked)) == 3 and set(picked) <= {0, 2, 4, 6}

    from internnav.model.basemodel.laya_s2.agent import LayaNavAgent

    img = Image.fromarray(np.random.randint(0, 255, (96, 128, 3), np.uint8))
    for head in ({}, NEW_HEAD):
        agent = LayaNavAgent(tiny_nav(**head).eval(), _Tok(), torch.device("cpu"))
        agent.decide("go to the door", [img] * 3, img, img)
        agent.start_goal({}, img, torch.zeros(224, 224))
        traj = agent.plan(img, torch.zeros(224, 224))
        assert traj.shape == (1, 32, 3) and torch.isfinite(traj).all()


def test_new_head_options():
    """Fusion layers + goal mark: c1 gradients stay in the head, inference path = training forward, goal matters."""
    model = tiny_nav(**NEW_HEAD)
    assert model.traj_fuse is not None and all(
        n.startswith("traj_") for n in ("traj_fuse", "traj_goal_xy", "traj_goal_mark")
    )
    b = fake_batch()
    model.requires_grad_(False)
    for p in model.traj_parameters():
        p.requires_grad_(True)
    x = dict(traj_pixels=torch.randn(4, 3, 32, 32), traj_mask=b["is_goal"])
    out = model(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"], **x)
    traj_loss(out["traj"], torch.randn(2, 32, 3), torch.ones(2, dtype=torch.bool), TrajLossWeights())[0].backward()
    trained = {n for n, p in model.named_parameters() if p.grad is not None}
    assert {"traj_goal_mark", "traj_goal_xy.proj.weight"} <= trained and all(n.startswith("traj_") for n in trained)

    model.eval()
    cur = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model(**{k: b[k][:1] for k in KEYS})
        t_inf = model.plan(model.plan_memory(out), out["down_feat"], model.encode_frame(cur), goal_xy=out["goal_xy"])
        t_fwd = model(**{k: b[k][:1] for k in KEYS}, traj_pixels=cur)["traj"]
        moved = model(**{k: b[k][:1] for k in KEYS}, goal_xy=torch.tensor([[0.9, 0.1]]), traj_pixels=cur)["traj"]
        again = model(**{k: b[k][:1] for k in KEYS}, goal_xy=torch.tensor([[0.1, 0.9]]), traj_pixels=cur)["traj"]
    torch.testing.assert_close(t_inf, t_fwd)
    assert not torch.allclose(moved, again)  # the trajectory depends on where the goal is


def test_multi_start_forward_matches_single_start():
    model = tiny_nav(**NEW_HEAD).eval()
    b = fake_batch()
    starts = torch.randn(4, 3, 3, 32, 32)
    mask = torch.tensor([[1, 1, 0], [0, 0, 0], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        out = model(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"], traj_pixels=starts, traj_mask=mask)
        assert out["traj_idx"].tolist() == [0, 0, 2, 2] and out["traj_slot"].tolist() == [0, 1, 0, 2]
        for n, (i, k) in enumerate(zip(out["traj_idx"].tolist(), out["traj_slot"].tolist())):
            one = model(**{key: b[key] for key in KEYS}, goal_xy=b["goal_xy"], traj_pixels=starts[:, k])["traj"]
            torch.testing.assert_close(out["traj"][n], one[i], atol=1e-5, rtol=1e-4)


def test_head_config_compatibility(tmp_path):
    """Older checkpoints (no new config fields) load with the old head; new heads start from a Laya-S2 checkpoint."""
    import json

    old = tiny_nav().eval()
    old.save_pretrained(str(tmp_path / "old"))
    cfg_path = tmp_path / "old" / "laya_s2_config.json"
    cfg = json.loads(cfg_path.read_text())
    for k in ("traj_fuse_layers", "traj_goal_mark"):
        cfg.pop(k)  # as written by the first LayaNav version
    cfg_path.write_text(json.dumps(cfg))
    loaded = LayaNav.load_any(str(tmp_path / "old")).eval()
    assert loaded.traj_fuse is None and not loaded.cfg.traj_goal_mark
    b = fake_batch()
    x = dict(traj_pixels=torch.randn(4, 3, 32, 32))
    with torch.no_grad():
        t_old, t_loaded = (m(**{k: b[k] for k in KEYS}, **x)["traj"] for m in (old, loaded))
    torch.testing.assert_close(t_old, t_loaded)

    tiny_model().save_pretrained(str(tmp_path / "s2"))
    nav = LayaNav.load_any(str(tmp_path / "s2"), traj_dim=32, traj_layers=2, **NEW_HEAD)
    nav.save_pretrained(str(tmp_path / "nav"))
    again = LayaNav.load_any(str(tmp_path / "nav"))
    assert again.cfg.traj_fuse_layers == 1 and again.cfg.traj_goal_mark


def test_eval_traj_metrics():
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train", "laya_s2", "eval_traj.py")
    spec = importlib.util.spec_from_file_location("eval_traj", path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    straight = torch.zeros(2, 32, 3)
    straight[:, :, 0] = 0.4  # 0.1 m per step
    ade, fde, ang = ev.pair_errors(torch.zeros_like(straight), straight)  # standing still
    assert abs(float(fde[0]) - 3.2) < 1e-4 and abs(float(ade[0]) - 0.1 * 33 / 2) < 1e-4
    left = straight.clone()
    left[:, :, :2] = torch.tensor([0.0, 0.4])
    assert ang.isnan().all()  # no direction without motion
    _, _, ang = ev.pair_errors(left, straight)
    assert torch.allclose(ang, torch.tensor([90.0, 90.0]))
    rows = dict(pred=straight, target=straight, frac=torch.tensor([0.0, 0.5]))
    summary = ev.summarize(rows, mean_traj=torch.zeros(32, 3))
    assert summary["all"]["model (GT goal)"]["ade"] == 0 and summary["start: decision frame"]["n"] == 1
    assert summary["path: >= 3.2 m (clipped)"]["n"] == 2
