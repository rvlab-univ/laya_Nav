"""Each frame's pose by dead reckoning (lightnav.pose, design_C_review.md 5.11): the same numbers in training, in the
request a driving client sends, and in the model, which starts from a checkpoint without poses unchanged."""
import math

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import fake_batch
from lightnav.data import NavDataset, collate, episode_poses, load_rows
from lightnav.model import LayaNativeConfig, init_from, load, save
from lightnav.pose import dead_reckoning, mirrored, relative
from lightnav.vlnce import decode_request, encode_request
from test_data import annotation, write_annotations
from test_init import INSTRUCTIONS, old_student, student


def test_dead_reckoning_and_relative_poses():
    walk = dead_reckoning([1, 1, 1, 1] + [2] * 6 + [1, 1, 1, 1, 0])        # 1 m ahead, left 90 degrees, 1 m
    assert np.allclose(walk[4], [1, 0, 0]) and np.allclose(walk[10], [1, 0, math.pi / 2])
    assert np.allclose(walk[14], [1, 1, math.pi / 2]) and np.allclose(walk[15], walk[14])      # stop does not move
    start, turn_point = relative(walk[[0, 4]], walk[14])
    assert np.allclose(start, [-0.2, 0.2, -1, 0], atol=1e-6)     # seen from the end: 1 m behind, 1 m to the left
    assert np.allclose(turn_point, [-0.2, 0, -1, 0], atol=1e-6)
    assert np.allclose(relative(walk[[14]], walk[14]), [[0, 0, 0, 1]])
    swapped = dead_reckoning([1, 1, 1, 1] + [3] * 6 + [1, 1, 1, 1, 0])
    assert np.allclose(mirrored(walk), swapped)


def _episode(tmp_path):
    """Left 45, forward 75, forward 75, stop: frames 1 -> 4 -> 7 -> 10, solid frames."""
    root = tmp_path / "frames" / "7"
    root.mkdir(parents=True)
    for i in range(10):
        Image.fromarray(np.full((32, 32, 3), 20 * i, dtype=np.uint8)).save(root / ("frame_%d.jpg" % i))
    path = str(tmp_path / "a.json")
    write_annotations(path, [annotation("7", 0, 1, 6), annotation("7", 1, 4, 3), annotation("7", 2, 7, 3),
                             annotation("7", 3, 10, 0)])
    return load_rows(path, "r2r"), {"r2r": str(tmp_path / "frames")}


def test_samples_carry_the_poses_of_their_frames(tok, tmp_path):
    rows, roots = _episode(tmp_path)
    poses = episode_poses(rows)
    walk = poses["r2r:7"]
    assert walk.shape == (11, 3) and np.allclose(walk[9], [1.5 * math.cos(math.pi / 4), 1.5 * math.sin(math.pi / 4),
                                                           math.pi / 4])
    cfg = LayaNativeConfig(image_size=32)
    item = NavDataset(rows, roots, tok, cfg, poses=poses)[3]                 # the stop: frames 0..9, slots 0..7
    frames = [0, 1, 2, 3, 5, 6, 7, 9]
    assert item["slots"] == list(range(8)) and np.allclose(item["poses"], relative(walk[frames], walk[9]))
    flipped = NavDataset(rows, roots, tok, cfg, poses=poses, mirror_prob=1.0)[3]
    assert np.allclose(flipped["poses"], item["poses"] * [1, -1, -1, 1])     # right of the walk, turned the other way
    assert "poses" not in NavDataset(rows, roots, tok, cfg)[3]
    batch = collate([item, {**NavDataset(rows, roots, tok, cfg)[1], "task": "nav"}], tok.pad_token_id)
    assert batch["frame_pose"].shape == (8 + 4, 4) and not batch["frame_pose"][8:].any()   # no poses: unknown


def test_requests_carry_poses_or_not():
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 2
    walk = dead_reckoning([1, 2, 1])
    req = decode_request(encode_request("go", frames, [6, 7], "e1", walk[[1, 3]]))
    assert np.allclose(req["poses"], walk[[1, 3]]) and req["slots"] == [6, 7]
    assert decode_request(encode_request("go", frames, [6, 7], "e1"))["poses"] is None


def test_a_pose_model_starts_where_its_checkpoint_was(tok, tmp_path):
    old, path = old_student(tok, tmp_path)
    new = student(tok, 32, 2, "tiny-224", seed=3, pose=True)
    assert init_from(new, path) == ["pose embedding: new, adds nothing at the start"]
    batch = fake_batch(tok, new, INSTRUCTIONS, n_frames=[3, 40])
    n = batch["frame_slot"].shape[0]
    with_poses = {**batch, "frame_pose": torch.randn(n, 4)}
    with torch.no_grad():
        assert torch.allclose(new(batch), old(batch), atol=1e-5)
        assert torch.allclose(new(with_poses), old(batch), atol=1e-5)
        new.pose_emb[2].weight.normal_()                                     # as if trained
        assert not torch.allclose(new(with_poses), new(batch), atol=1e-3)
    save(new, tok, str(tmp_path / "p1"))
    again, _ = load(str(tmp_path / "p1"))
    assert again.cfg.pose
    with torch.no_grad():
        assert torch.allclose(again.eval()(with_poses), new(with_poses), atol=1e-5)
    with pytest.raises(ValueError, match="pose_emb"):                        # poses cannot be dropped silently
        init_from(student(tok, 32, 2, "tiny-224", seed=4), str(tmp_path / "p1"))


def test_dagger_rollouts_get_the_poses_of_the_actions_that_ran(tmp_path):
    """A DAgger row's answer is the expert's; the walk follows "run" (here the student's right 45, not left 45)."""
    import json

    from lightnav.data import rollout_poses

    rows = [{"step": 0, "frames": 1, "label": 6, "run": 9}, {"step": 1, "frames": 4, "label": 3, "run": 3},
            {"step": 2, "frames": 7, "label": 0, "run": 0}]
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps({"episode_id": "6", "rows": rows}) + "\n")
    walk = rollout_poses(str(path), "dagger2")["dagger2:6"]
    assert np.allclose(walk, dead_reckoning([3, 3, 3, 1, 1, 1])) and walk.shape[0] == 7
    rows[1]["frames"] = 3                                                   # 2 frames for a 3-step answer
    path.write_text(json.dumps({"episode_id": "6", "rows": rows}) + "\n")
    with pytest.raises(ValueError, match="2 frames"):
        rollout_poses(str(path), "dagger2")
