"""v5, against copying the previous action: change points, left-right mirroring, memory dropout, the change-point loss
weight and the shortcut checks of the diagnosis - CPU tests with tiny random encoders."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")
pytest.importorskip("scipy")
from test_laya_nav import KEYS, NEW_HEAD, tiny_nav  # noqa: E402
from test_laya_s2_smoke import _Tok, fake_batch  # noqa: E402
from test_memory import MEM, load_script, v4  # noqa: E402

from internnav.dataset.laya_s2_dataset import (  # noqa: E402
    CHANGE,
    GOAL,
    KEEP,
    NO_PREV,
    TURN,
    LayaS2Dataset,
    change_flag,
    collate_laya_s2,
    enumerate_samples,
    frame_path,
    mirror_instruction,
    mirror_poses,
    robot_poses,
)
from internnav.model.basemodel.laya_s2 import LossWeights, TrajLossWeights, compute_loss  # noqa: E402


def camera_poses(xyyaw, pitch=30):
    """Camera extrinsics of a robot at these (x, y, counter-clockwise yaw) poses (inverse of the dataset transform)."""
    t_r2c = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    r = np.radians(pitch)
    t_deg = np.array([[1, 0, 0, 0], [0, np.cos(-r), -np.sin(-r), 0], [0, np.sin(-r), np.cos(-r), 0], [0, 0, 0, 1]])
    out = []
    for x, y, yaw in xyyaw:
        t = np.eye(4)
        t[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        t[:2, 3] = x, y
        out.append((t @ t_r2c @ t_deg).tolist())
    return out


def path_with_left_turn():
    """4 steps forward, two 15 degree left turns, then forward along the new heading, drifting a little to the
    left (12 frames)."""
    p = [(0.25 * i, 0.0, 0.0) for i in range(5)] + [(1.0, 0.0, np.radians(15)), (1.0, 0.0, np.radians(30))]
    for _ in range(5):
        x, y, yaw = p[-1]
        p.append((x + 0.25 * np.cos(yaw) - 0.05 * np.sin(yaw), y + 0.25 * np.sin(yaw) + 0.05 * np.cos(yaw), yaw))
    raw_actions = [-1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1]  # action i + 1 is taken at frame i; STOP is appended
    return np.array(p), raw_actions


def write_turn_episode(root, instruction="Turn left, then walk to the door on the right."):
    xyyaw, raw = path_with_left_turn()
    n = len(raw)
    acts = raw[1:] + [0]
    video = str(root / "scene" / "videos" / "chunk-000")
    ann = {"episodes": [{"id": 0, "instructions": instruction, "video": video, "actions": raw,
                         "pixel_goals": [[-1, [0, 0]] if a in (2, 3) else [5, [30, 40]] for a in acts],
                         "poses_125cm_30deg": camera_poses(xyyaw)}]}
    goals, turns, stops = enumerate_samples(ann, 125, 0, 30, sample_step=4, num_future_steps=4)
    rng = np.random.default_rng(0)
    for f in range(n + 1):
        for down in (False, True):
            p = frame_path((goals + turns + stops)[0], f, look_down=down)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)).save(p)  # not left-right symmetric
    return goals + turns + stops, xyyaw


def test_mirror_instruction():
    assert mirror_instruction("Turn left, then walk to the right of the sofa.") == \
        "Turn right, then walk to the left of the sofa."
    assert mirror_instruction("Turn LEFT at the Right door") == "Turn RIGHT at the Left door"
    assert mirror_instruction("Take the leftmost door, then the right-hand stairs") == \
        "Take the rightmost door, then the left-hand stairs"
    assert mirror_instruction("Rotate clockwise, then counter-clockwise") == "Rotate counterclockwise, then clockwise"
    assert mirror_instruction("Go to the kitchen.") == "Go to the kitchen."  # mirrored scene, nothing to swap
    assert mirror_instruction("Pass the brightest lamp") == "Pass the brightest lamp"  # no word inside words
    t = "Turn left and pass the right sofa"
    assert mirror_instruction(mirror_instruction(t)) == t
    # no direction, or not English: not mirrored at all
    assert mirror_instruction("Stop right in front of the door") is None
    assert mirror_instruction("Go back to the room you just left") is None
    assert mirror_instruction("दाईं ओर मुड़ें और दरवाज़े तक जाएँ") is None


def test_change_flag(tmp_path):
    samples, _ = write_turn_episode(tmp_path)
    by_start = {s["start"]: s for s in samples}
    assert by_start[0]["kind"] == GOAL and change_flag(by_start[0]) == NO_PREV  # first step
    assert by_start[4]["kind"] == TURN and by_start[4]["action"] == 2
    assert change_flag(by_start[4]) == CHANGE  # walking, now turning
    assert by_start[8]["kind"] == GOAL and change_flag(by_start[8]) == KEEP  # walking on
    stop = next(s for s in samples if s["kind"] == 2)
    assert change_flag(stop) == NO_PREV
    turning_goal = dict(by_start[8], start=6)  # a goal right after the turns
    assert change_flag(turning_goal) == CHANGE
    still_turning = dict(by_start[4], start=5)
    assert change_flag(still_turning) == KEEP


def test_dataset_mirror_and_memory_dropout(tmp_path):
    samples, xyyaw = write_turn_episode(tmp_path)
    p = robot_poses(np.asarray(camera_poses(xyyaw), np.float32), 30)
    assert np.allclose(p, xyyaw, atol=1e-4)  # the test helper inverts the dataset transform
    assert np.allclose(mirror_poses(p)[:, 1:], -p[:, 1:]) and np.allclose(mirror_poses(p)[:, 0], p[:, 0])

    cfg = v4().cfg
    kw = dict(with_traj=True, traj_steps=cfg.traj_steps, traj_starts=2)
    plain = LayaS2Dataset(samples, _Tok(), cfg, **kw)
    mirrored = LayaS2Dataset(samples, _Tok(), cfg, mirror=1.0, **kw)
    i_goal = next(i for i, s in enumerate(plain.samples) if s["start"] == 8)
    i_turn = next(i for i, s in enumerate(plain.samples) if s["start"] == 4)
    a, b = plain[i_goal], mirrored[i_goal]
    assert b["instruction"] == "Turn right, then walk to the door on the left."
    for k in ("hist_pixels", "cur_pixels", "down_pixels", "traj_pixels"):
        assert torch.equal(b[k], a[k].flip(-1)) and not torch.equal(a[k], a[k].flip(-1))
    assert b["goal_xy"][0] == pytest.approx(1 - a["goal_xy"][0]) and b["goal_xy"][1] == a["goal_xy"][1]
    assert b["traj_mask"].all() and torch.equal(b["traj"][..., 0], a["traj"][..., 0])
    assert torch.equal(b["traj"][..., 1:], -a["traj"][..., 1:]) and a["traj"][..., 1].abs().sum() > 0
    # memory: frames seen from the mirrored side, and the left turns became right turns
    assert a["motion_act"].tolist() == [3, 3, 2, 2] and b["motion_act"].tolist() == [4, 4, 2, 2]
    assert torch.allclose(b["hist_pose"][:, [0, 3, 4]], a["hist_pose"][:, [0, 3, 4]])
    assert torch.allclose(b["hist_pose"][:, 1:3], -a["hist_pose"][:, 1:3]) and a["hist_pose"][:, 2].abs().sum() > 0
    assert torch.allclose(b["motion_disp"][:, 1:], -a["motion_disp"][:, 1:]) and a["motion_disp"][:, 2].abs().sum() > 0
    assert a["change"] == b["change"] == KEEP
    t_a, t_b = plain[i_turn], mirrored[i_turn]
    assert cfg.actions[t_a["action_idx"]] == 2 and cfg.actions[t_b["action_idx"]] == 3 and t_a["change"] == CHANGE

    off = LayaS2Dataset(samples, _Tok(), cfg, memory_dropout=1.0, **kw)[i_goal]
    assert not off["hist_pose"].any() and not off["motion_act"].any() and not off["motion_disp"].any()
    assert torch.equal(off["cur_pixels"], a["cur_pixels"]) and off["instruction"] == a["instruction"]
    swap = LayaS2Dataset(samples, _Tok(), cfg, swap_directions=1.0, **kw)[i_goal]
    assert swap["instruction"] == b["instruction"] and torch.equal(swap["cur_pixels"], a["cur_pixels"])

    batch = collate_laya_s2([plain[i] for i in range(len(plain))], pad_id=0)
    assert batch["change"].tolist() == [change_flag(s) for s in plain.samples]


def test_change_weight_in_loss_and_trainer():
    model = tiny_nav(**NEW_HEAD)
    b = fake_batch()
    out = model(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"])
    l_plain, st = compute_loss(model, out, b, LossWeights(change=3.0))  # no change flags: plain mean
    assert "acc_change" not in st
    flags = dict(b, change=torch.tensor([CHANGE, KEEP, KEEP, NO_PREV]))
    l1, st1 = compute_loss(model, out, flags, LossWeights())
    l3, st3 = compute_loss(model, out, flags, LossWeights(change=3.0))
    torch.testing.assert_close(l1, l_plain)
    assert not torch.isclose(l1, l3) and float(st3["n_change"]) == 1 and float(st3["n_keep"]) == 2
    same = dict(b, change=torch.full((4,), CHANGE))  # every sample a change point: same mean
    torch.testing.assert_close(compute_loss(model, out, same, LossWeights(change=3.0))[0], l1)

    tr = load_script("train_laya_nav")
    batch = dict(flags, instructions=["turn left", "go", "walk to the door", "stop"],
                 traj_pixels=torch.randn(4, 2, 3, 32, 32), traj=torch.randn(4, 2, 32, 3),
                 traj_mask=torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0]]).bool())
    loss, st = tr.run(model, batch, LossWeights(change=3.0), TrajLossWeights(), "c2")
    assert torch.isfinite(loss) and "acc_change" in st and "acc_keep" in st
    assert tr.METRIC_WEIGHT["acc_change"] == "n_change"
    loss.backward()


def test_diagnosis_shortcut_checks(tmp_path):
    dg = load_script("diagnose_decision")
    samples, _ = write_turn_episode(tmp_path / "a")
    samples += write_turn_episode(tmp_path / "b", "Stop right in front of the door.")[0]  # not mirrored
    samples += write_turn_episode(tmp_path / "c", "Go to the kitchen.")[0]  # mirrored, no direction word

    torch.manual_seed(0)
    model = v4().eval()
    kept, full = dg.profile(model, samples, _Tok(), batch_size=3, num_workers=0)
    checks = dg.run_checks(model, kept, _Tok(), dg.CHECKS, batch_size=3, num_workers=0)
    n_a = sum(s["instruction"].startswith("Turn left") for s in kept)
    assert len(checks["memory_off"]["idx"]) == len(kept)
    assert len(checks["swap_directions"]["idx"]) == n_a
    assert len(checks["mirror"]["idx"]) == len(kept) - sum(s["instruction"].startswith("Stop right") for s in kept)
    assert "memory_off" not in dg.run_checks(tiny_nav().eval(), kept, _Tok(), dg.CHECKS, batch_size=3, num_workers=0)

    acts, g = list(model.cfg.actions), model.grid
    A = len(acts)
    assert dg.mirrored_top(acts.index(2), acts, g) == acts.index(3) and dg.mirrored_top(acts.index(0), acts, g) == 0
    assert dg.mirrored_top(A + 0, acts, g) == A + g - 1
    assert all(dg.mirrored_top(dg.mirrored_top(t, acts, g), acts, g) == t for t in range(A + g * g))

    sc = dg.summarize_checks(kept, {"m/step_1": full}, {"m/step_1": checks})
    r = sc["m/step_1"]
    assert r["n_change"] == sum(change_flag(s) == CHANGE for s in kept) and r["n_keep"] > 0
    assert 0 <= r["mirror"]["agreement"] <= 1 and r["swap_directions"]["n"] == n_a
    assert set(r["memory_off"]) == {"acc", "change", "keep"}

    labels = dg.label_stats(kept)
    assert labels["all"]["change"] == r["n_change"]
    summary = dg.summarize(kept, {"m/step_1": full})
    out = tmp_path / "diag"
    dg.write_report(str(out), labels, None, summary, kept, {"m/step_1": full}, cfg_len=12, shortcuts=sc)
    text = (out / "report.md").read_text()
    assert "## Shortcut checks" in text and "mirror: agreement" in text
