"""scripts/teacher_signal.py on made-up teacher labels: same-path pairs, the reference pairs, reading parts."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from lightnav.data import Row                               # noqa: E402
from teacher_signal import read_teacher, signal             # noqa: E402

PATHS = {"a": [1, 4, 0], "b": [2, 7, 0]}                    # answer labels of each path's decisions


def make_rows():
    rows, ep = [], 0
    for name, labels in PATHS.items():
        for _ in range(3):                                   # 3 instructions per path, as R2R
            ep += 1
            for step, label in enumerate(labels):
                rows.append(Row("r2r", "%d-%d" % (ep, step), str(ep), step, "go %s %d" % (name, ep), label,
                                step + 1, "%d/" % ep))
    return rows


def onehot_like(label, sharp=0.9):
    p = np.full(10, (1 - sharp) / 9)
    p[label] = sharp
    return p


def test_identical_distributions_give_zero_and_the_reference_does_not():
    rows = make_rows()
    keys = [r.key for r in rows]
    probs = np.stack([onehot_like(r.label, 0.9 if r.episode in "123" else 0.5) for r in rows])
    out = signal(rows, keys, probs)
    assert out["labeled"] == 18 and out["paths_with_2_or_more"] == 2 and out["decisions_with_2_or_more"] == 6
    assert out["same_path"]["pairs"] == 6 * 3
    assert out["same_path"]["top1_differs"] == 0 and out["same_path"]["kl"] < 1e-9
    assert out["teacher_top1_equals_label"] == 1.0
    # stop (label 0) is shared by both paths, so a reference pair there compares 0.9 with 0.5
    assert out["other_path_same_label"]["pairs"] > 0


def test_instruction_dependent_teacher_shows_up():
    rows = make_rows()
    keys = [r.key for r in rows]
    probs = np.stack([onehot_like(r.label if int(r.episode) % 3 else (r.label + 1) % 10) for r in rows])
    out = signal(rows, keys, probs)
    # per decision, one instruction of three answers differently: 2 of its 3 pairs differ
    assert abs(out["same_path"]["top1_differs"] - 2 / 3) < 1e-9
    assert out["same_path"]["kl"] > 1.0 and out["same_path"]["label_prob_gap"] > 0.5


def test_only_labeled_rows_count():
    rows = make_rows()
    keys = [r.key for r in rows if r.episode in ("1", "4")]    # one instruction per path: nothing to compare
    probs = np.stack([onehot_like(1) for _ in keys])
    out = signal(rows, keys, probs)
    assert out["labeled"] == 6 and out["same_path"] is None


def test_read_teacher_skips_unfinished_parts(tmp_path):
    shard = tmp_path / "shard_0"
    shard.mkdir()
    np.savez(shard / "part_00000.npz", keys=np.array(["r2r:1-0", "r2r:1-1"]), labels=np.array([1, 4]),
             logp=np.log(np.stack([onehot_like(1), onehot_like(4)])).astype(np.float32), gen=np.array([-2, -2]))
    np.savez(shard / "part_00001.npz.tmp.npz", keys=np.array(["r2r:9-0"]), labels=np.array([1]),
             logp=np.zeros((1, 10), np.float32), gen=np.array([-2]))
    (shard / "part_00002.npz").write_bytes(b"not a zip")
    keys, probs = read_teacher(str(tmp_path))
    assert keys == ["r2r:1-0", "r2r:1-1"]
    assert np.allclose(probs.sum(1), 1) and probs[1].argmax() == 4
