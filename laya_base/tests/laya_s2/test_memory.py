"""v4 memory: history frame choice, pose / motion features, model inputs, upgrading a v3 checkpoint, agent cache -
CPU tests with tiny random encoders."""

import argparse
import importlib.util
import os

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")
pytest.importorskip("scipy")
from test_laya_nav import KEYS, NEW_HEAD, camera_poses_along_x, tiny_nav  # noqa: E402
from test_laya_s2_smoke import _Tok, fake_batch, tiny_model  # noqa: E402

from internnav.dataset.laya_s2_dataset import (  # noqa: E402
    LayaS2Dataset,
    collate_laya_s2,
    enumerate_samples,
    frame_path,
    history_ids,
    memory_features,
    robot_poses,
)
from internnav.model.basemodel.laya_s2 import LayaNav, LossWeights, TrajLossWeights  # noqa: E402

MEM = dict(num_history=6, hist_recent=3, hist_pose=True, motion_tokens=4, hist_grad=False)
INS = ["walk to the door", "turn left at the sofa", "go to the kitchen", "stop at the stairs"]


def v4(**extra):
    nav = tiny_nav(**NEW_HEAD)
    return LayaNav(nav.cfg.__class__(**{**vars(nav.cfg), **MEM, **extra}), nav.text, nav.vision)


def mem_batch(B=4, H=6, S=32, M=4):
    hist_mask = torch.zeros(B, H, dtype=torch.bool)
    hist_mask[0, :6], hist_mask[1, :2], hist_mask[3, :4] = True, True, True
    return dict(fake_batch(B), hist_pixels=torch.randn(B, H, 3, S, S), hist_mask=hist_mask,
                hist_pose=torch.randn(B, H, 5), motion_act=torch.randint(0, 5, (B, M)), motion_disp=torch.randn(B, M, 3))


def load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train", "laya_s2", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_history_ids():
    assert history_ids(0, 8) == [] and history_ids(5, 8) == [0, 1, 2, 3, 4]
    assert history_ids(100, 8) == np.unique(np.linspace(0, 99, 8, dtype=np.int32)).tolist()  # as before v4
    ids = history_ids(100, 32, 16)
    assert len(ids) == 32 and ids == sorted(set(ids)) and ids[0] == 0 and ids[-16:] == list(range(84, 100))
    assert history_ids(10, 32, 16) == list(range(10)) and history_ids(20, 32, 16) == list(range(20))


def test_memory_features_straight_then_left():
    # 5 steps forward (0.25 m), then two 15 degree left turns in place
    xyyaw = np.array([[0.25 * i, 0, 0] for i in range(6)] + [[1.25, 0, np.radians(15)], [1.25, 0, np.radians(30)]])
    acts = [1] * 5 + [2, 2]
    hp, ma, md = memory_features(xyyaw, [0, 4], 7, acts, n_hist=3, n_motion=4)
    assert hp[0, 0] < 0 and hp[0, 1] > 0  # the start lies behind, and to the left now that we face 30 degrees left
    assert np.hypot(*hp[0, :2]) * 4 == pytest.approx(1.25)
    assert np.allclose(hp[0, 2:4], [np.sin(np.radians(-30)), np.cos(np.radians(-30))])
    assert hp[0, 4] == pytest.approx(7 / 64) and not hp[2].any()  # unused slot
    assert ma.tolist() == [2, 2, 3, 3]  # forward, forward, left, left
    assert np.allclose(md[0], [1, 0, 0], atol=1e-6) and np.allclose(md[3], [0, 0, 1], atol=1e-6)
    hp0, ma0, md0 = memory_features(None, [0, 4], 7, acts, 3, 4)  # no poses: zeros, actions still known
    assert not hp0.any() and not md0.any() and ma0.tolist() == [2, 2, 3, 3]
    p = robot_poses(np.asarray(camera_poses_along_x(5), np.float32), 30)
    assert np.allclose(p[:, 0], 0.25 * np.arange(5), atol=1e-5) and np.allclose(p[:, 1:], 0, atol=1e-5)


def test_dataset_gives_memory_inputs(tmp_path):
    video = str(tmp_path / "scene" / "videos" / "chunk-000")
    n = 12
    ann = {"episodes": [{"id": 0, "instructions": "walk straight to the door", "video": video,
                         "actions": [-1] + [1] * (n - 1), "pixel_goals": [[8, [64, 90]]] * n,
                         "poses_125cm_30deg": camera_poses_along_x(n)}]}
    goals, turns, stops = enumerate_samples(ann, 125, 0, 30, sample_step=4, num_future_steps=4)
    for f in range(n + 1):
        for down in (False, True):
            p = frame_path(goals[0], f, look_down=down)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(np.full((96, 128, 3), f * 10, np.uint8)).save(p)
    cfg = v4().cfg
    ds = LayaS2Dataset(goals + turns + stops, _Tok(), cfg, with_traj=True, traj_steps=cfg.traj_steps)
    item = next(ds[i] for i in range(len(ds)) if ds.samples[i]["start"] == 8)
    assert item["hist_pose"].shape == (6, 5) and item["hist_mask"].sum() == 6
    assert item["motion_act"].tolist() == [2, 2, 2, 2] and torch.allclose(item["motion_disp"][:, 0], torch.ones(4), atol=1e-4)
    assert (item["hist_pose"][:6, 0] < 0).all()  # every history frame was seen from behind
    batch = collate_laya_s2([ds[i] for i in range(len(ds))], pad_id=0)
    assert batch["hist_pose"].shape == (len(ds), 6, 5) and batch["motion_disp"].shape == (len(ds), 4, 3)


def test_v4_forward_history_without_gradients_and_cached_features():
    model = v4().train()
    b = mem_batch()
    b["hist_pixels"].requires_grad_(True)
    b["cur_pixels"].requires_grad_(True)
    mem = {k: b[k] for k in ("hist_pose", "motion_act", "motion_disp")}
    out = model(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"], **mem)
    out["logits"].logsumexp(-1).sum().backward()
    assert b["hist_pixels"].grad is None and b["cur_pixels"].grad is not None  # history encoded without gradients
    assert model.hist_pose_proj[-1].weight.grad.abs().sum() > 0 and model.motion_disp[-1].weight.grad.abs().sum() > 0

    model.eval()
    x = {k: b[k].detach() for k in KEYS}
    with torch.no_grad():
        ref = model(**x, **mem)
        feats = model.encode_frame(x["hist_pixels"][x["hist_mask"]])
        cached = model(**{**x, "hist_pixels": None}, hist_feats=feats, **mem)
    torch.testing.assert_close(cached["logits"], ref["logits"])


def test_upgrade_v3_checkpoint_to_v4(tmp_path):
    old = tiny_nav(**NEW_HEAD).eval()
    old.save_pretrained(str(tmp_path / "v3"))
    new = LayaNav.load_any(str(tmp_path / "v3"), addons=MEM).eval()
    assert new.cfg.num_history == 6 and new.hist_slot_emb.weight.shape[0] == 6
    torch.testing.assert_close(new.hist_slot_emb.weight[0], old.hist_slot_emb.weight[0])  # oldest / newest slots kept
    torch.testing.assert_close(new.hist_slot_emb.weight[-1], old.hist_slot_emb.weight[-1])
    b = mem_batch()
    x = {k: b[k] for k in KEYS}
    with torch.no_grad():  # the pose embedding starts at zero output: poses do not matter yet
        a = new(**x, hist_pose=torch.randn(4, 6, 5), motion_act=b["motion_act"], motion_disp=b["motion_disp"])
        c = new(**x, hist_pose=torch.randn(4, 6, 5), motion_act=b["motion_act"], motion_disp=b["motion_disp"])
    torch.testing.assert_close(a["logits"], c["logits"])

    s2 = tiny_model()
    s2.save_pretrained(str(tmp_path / "s2"))  # a Laya-S2 checkpoint, new trajectory head
    nav = LayaNav.load_any(str(tmp_path / "s2"), addons=MEM, traj_dim=32, traj_layers=2)
    assert nav.cfg.motion_tokens == 4 and nav.hist_slot_emb.weight.shape[0] == 6


def test_agent_memory_and_feature_cache():
    from internnav.model.basemodel.laya_s2.agent import LayaNavAgent

    model = v4().eval()
    agent = LayaNavAgent(model, _Tok(), torch.device("cpu"))
    calls, orig = [], model.encode_frame
    model.encode_frame = lambda px: calls.append(px.shape[0]) or orig(px)
    frames = [Image.fromarray(np.full((96, 128, 3), 10 * i, np.uint8)) for i in range(12)]
    poses = np.array([(0.25 * i, 0.0, 0.0) for i in range(12)])
    d = agent.decide("go to the door", frames[:10], frames[10], frames[10], poses=poses[:11], actions=[1] * 10)
    assert d["kind"] in ("action", "goal") and calls == [6, 2]  # 6 history frames once, then current + look-down
    agent.decide("go to the door", frames[:11], frames[11], frames[11], poses=poses, actions=[1] * 11)
    assert calls == [6, 2, 1, 2]  # frames 0, 3, 7, 8, 9, 10: only frame 10 was not seen before
    agent.decide("go to the door", [], frames[0], frames[0])  # first step, no poses: still works
    agent.reset()
    assert not agent._feat_cache

    old = tiny_nav().eval()  # a checkpoint without v4 memory, history already cut by the caller as before
    agent = LayaNavAgent(old, _Tok(), torch.device("cpu"))
    agent.decide("go to the door", frames[:3], frames[3], frames[3])
    tf = agent.transform
    with torch.no_grad():
        ref = old(torch.tensor([_Tok()("go to the door", max_length=old.cfg.max_text_len)["input_ids"]]),
                  torch.ones(1, len(_Tok()("go to the door", max_length=old.cfg.max_text_len)["input_ids"]), dtype=torch.bool),
                  torch.stack([tf(f) for f in frames[:3]])[None], torch.ones(1, 3, dtype=torch.bool),
                  tf(frames[3])[None], tf(frames[3])[None])
    torch.testing.assert_close(agent._out["logits"], ref["logits"])


def test_trainer_with_memory_and_matching():
    tr = load_script("train_laya_nav")
    model = v4(match_head=True)
    b = mem_batch()
    batch = dict(b, instructions=INS, traj_pixels=torch.randn(4, 2, 3, 32, 32), traj=torch.randn(4, 2, 32, 3),
                 traj_mask=torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0]]).bool())
    loss, st = tr.run(model, batch, LossWeights(match=0.5), TrajLossWeights(), "c2")
    assert torch.isfinite(loss) and "l_match" in st
    loss.backward()
    args = argparse.Namespace(num_history=None, hist_recent=16, hist_pose=1, motion_tokens=None, hist_grad=0)
    assert tr.base.memory_overrides(args) == dict(hist_recent=16, hist_pose=True, hist_grad=False)
