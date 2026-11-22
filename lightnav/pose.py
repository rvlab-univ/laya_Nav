"""Where each frame was taken, from the commanded actions alone (design_C_review.md 5.11).

The analogue of the map RSMP keeps of the places it passed (CLASH, arXiv 2512.10360), without its odometry: dead
reckoning of the low-level actions. NaVILA's annotations, the student's drives and the robot all know their actions,
so training and driving compute the same numbers with the same function. A step blocked by a wall still counts as
0.25 m here; only the frames show it.

A pose is (x, y, heading) in the episode's start frame: x ahead at the start, y to the left, heading counter-clockwise
in radians. The model sees each frame's pose relative to the current frame (`relative`).
"""
import math
from typing import Sequence

import numpy as np

FORWARD_M = 0.25                # VLN-CE's forward step
TURN_DEG = 15.0                 # VLN-CE's turn
SCALE_M = 5.0                   # metres per unit of the model input (R2R paths are about 10 m)
FEATURES = 4                    # dx, dy in the current frame's heading (/ SCALE_M), sin and cos of the heading change


def dead_reckoning(actions: Sequence[int], turn_deg: float = TURN_DEG, forward_m: float = FORWARD_M) -> np.ndarray:
    """[len(actions) + 1, 3]: frame i's pose is where the first i low-level actions lead (0 stop, 1 forward,
    2 left, 3 right, as in `lightnav.questions.low_level`)."""
    out = np.zeros((len(actions) + 1, 3), dtype=np.float64)
    x = y = h = 0.0
    turn = math.radians(turn_deg)
    for i, a in enumerate(actions):
        if a == 1:
            x, y = x + forward_m * math.cos(h), y + forward_m * math.sin(h)
        elif a == 2:
            h += turn
        elif a == 3:
            h -= turn
        out[i + 1] = (x, y, h)
    return out


def mirrored(poses: np.ndarray) -> np.ndarray:
    """The poses of the left-right mirrored walk (left and right turns swapped)."""
    return poses * np.array([1.0, -1.0, -1.0])


def relative(poses: np.ndarray, current: np.ndarray) -> np.ndarray:
    """[k, FEATURES] model inputs for poses [k, 3] seen from the pose `current` [3]."""
    d = poses[:, :2] - current[:2]
    c, s = math.cos(current[2]), math.sin(current[2])
    dh = poses[:, 2] - current[2]
    return np.stack([(c * d[:, 0] + s * d[:, 1]) / SCALE_M, (c * d[:, 1] - s * d[:, 0]) / SCALE_M,
                     np.sin(dh), np.cos(dh)], -1).astype(np.float32)


def unknown(k: int) -> np.ndarray:
    """[k, FEATURES] for frames without a pose (the matching task, a client that sends none). No real pose gives
    these: sin and cos of a heading change are never both 0."""
    return np.zeros((k, FEATURES), dtype=np.float32)
