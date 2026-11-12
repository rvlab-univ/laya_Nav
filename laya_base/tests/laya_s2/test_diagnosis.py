"""STOP repetition after the split, own-goal mixing for the trajectory head, STOP / turn metrics and the decision
diagnosis script - CPU tests with tiny random encoders."""

import csv
import importlib.util
import os

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")
from test_laya_nav import KEYS, NEW_HEAD, tiny_nav  # noqa: E402
from test_laya_s2_smoke import _Tok, fake_batch  # noqa: E402

from internnav.dataset.laya_s2_dataset import GOAL, STOP, enumerate_samples, frame_path, repeat_stops  # noqa: E402
from internnav.model.basemodel.laya_s2 import LossWeights, TrajLossWeights, compute_loss  # noqa: E402

INS = ["walk to the door", "turn left at the sofa", "go to the kitchen", "stop at the stairs"]


def load_script(name):
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train", "laya_s2", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def key(s):
    return (s["video"], s["ep_id"], s["instruction"], s["start"], s["kind"])


def test_stop_repeat_after_split_reproduces_old_lists(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from internnav.dataset.laya_s2_dataset import load_vln_samples

    n = 12
    for sc in ("sceneA", "sceneB", "sceneC"):
        scene = tmp_path / "traj_data" / "r2r" / sc
        (scene / "meta").mkdir(parents=True)
        (scene / "meta" / "episodes.jsonl").write_text(
            "".join(f'{{"episode_index": {e}, "tasks": ["go to door {e}"], "length": {n}}}\n' for e in range(3))
        )
        chunk = scene / "data" / "chunk-000"
        chunk.mkdir(parents=True)
        for e in range(3):
            pd.DataFrame({
                "action": [-1, 2] + [1] * (n - 2),
                "pose.125cm_30deg": [np.eye(4).flatten() for _ in range(n)],
                "goal.125cm_30deg": [np.array([-1, -1])] + [np.array([64, 96])] * (n - 1),
                "relative_goal_frame_id.125cm_30deg": [-1] + [4] * (n - 1),
            }).to_parquet(chunk / f"episode_{e:06d}.parquet")
    monkeypatch.chdir(tmp_path)
    base = load_script("train_laya_s2")

    old = load_vln_samples("r2r_125cm_0_30")  # STOP repeated before the split, as the runs before this change
    once = load_vln_samples("r2r_125cm_0_30", stop_repeat=1)
    assert len(repeat_stops(once, 5)) == len(old)
    for max_val in (-1, 7):
        tr_old, va_old = base.split_samples(old, 0.5, "episode", max_val)
        tr_new, va_new = base.split_samples(once, 0.5, "episode", max_val, train_stop_repeat=5, val_stop_repeat=5)
        assert [key(s) for s in tr_new] == [key(s) for s in tr_old]
        assert [key(s) for s in va_new] == [key(s) for s in va_old]
    tr, va = base.split_samples(once, 0.5, "episode", train_stop_repeat=5, val_stop_repeat=1)
    assert va and tr and len({key(s) for s in va}) == len(va)  # each STOP once in validation
    assert sum(s["kind"] == STOP for s in tr) == 5 * len({key(s) for s in tr if s["kind"] == STOP})
    assert base.dataset_of(va[0]) == "r2r"


def test_own_goal_for_the_trajectory_head():
    model = tiny_nav(**NEW_HEAD).eval()
    b = fake_batch()
    x = {k: b[k] for k in KEYS}
    with torch.no_grad():
        own = model(**x)  # inference: the model's own goal
        ref = model(**x, goal_xy=b["goal_xy"])  # teacher-forced
        far = model(**x, goal_xy=b["goal_xy"], pred_goal_mix=torch.ones(4, dtype=torch.bool), pred_goal_radius=100)
        none = model(**x, goal_xy=b["goal_xy"], pred_goal_mix=torch.ones(4, dtype=torch.bool), pred_goal_radius=-1)
    assert not ref["goal_from_pred"].any() and far["goal_from_pred"].all() and not none["goal_from_pred"].any()
    torch.testing.assert_close(far["goal_xy"], own["goal_xy"])
    torch.testing.assert_close(far["goal_token"], own["goal_token"])
    torch.testing.assert_close(none["goal_xy"], ref["goal_xy"])
    torch.testing.assert_close(none["latent"], ref["latent"])


def test_stop_and_turn_metrics():
    model = tiny_nav()
    b = fake_batch()  # actions [STOP, LEFT, RIGHT]: row 1 is RIGHT (turn), row 3 is STOP
    out = model(**{k: b[k] for k in KEYS}, goal_xy=b["goal_xy"])
    _, st = compute_loss(model, out, b, LossWeights())
    assert float(st["n_stop"]) == 1 and float(st["n_turn"]) == 1 and float(st["n_action"]) == 2
    assert 0 <= float(st["acc_stop"]) <= 1 and 0 <= float(st["acc_turn"]) <= 1


def test_trainer_mixes_own_goals():
    tr = load_script("train_laya_nav")
    model = tiny_nav(**NEW_HEAD)
    batch = dict(fake_batch(), instructions=INS, traj_pixels=torch.randn(4, 2, 3, 32, 32),
                 traj=torch.randn(4, 2, 32, 3), traj_mask=torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0]]).bool())
    loss, st = tr.run(model, batch, LossWeights(), TrajLossWeights(), "c2", pred_goal=1.0, pred_goal_radius=100.0)
    assert float(st["pred_goal_used"]) == 1.0 and "acc_stop" in st
    loss.backward()
    _, st0 = tr.run(model, batch, LossWeights(), TrajLossWeights(), "c2")
    assert "pred_goal_used" not in st0


def write_episode(root, dataset, scene, instruction, ep_id=0, n=12):
    video = str(root / "traj_data" / dataset / scene / "videos" / "chunk-000")
    ann = {"episodes": [{
        "id": ep_id,
        "instructions": instruction,
        "video": video,
        "actions": [-1, 2] + [1] * (n - 2),  # turn right at frame 0, then forward, STOP at the end
        "pixel_goals": [[-1, [0, 0]]] + [[6, [64, 90]]] * (n - 1),
    }]}
    goals, turns, stops = enumerate_samples(ann, 125, 0, 30, sample_step=4, num_future_steps=4)
    for f in range(n):
        for down in (False, True):
            p = frame_path((goals + turns + stops)[0], f, look_down=down)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            Image.fromarray(np.full((96, 128, 3), 20 * (f % 10), np.uint8)).save(p)
    return goals + turns + stops


def test_diagnosis_end_to_end(tmp_path):
    dg = load_script("diagnose_decision")
    samples = write_episode(tmp_path, "r2r", "sceneA", "walk to the door and stop")
    samples += write_episode(tmp_path, "rxr", "sceneB", "turn right then walk along the long hallway " * 3)
    labels = dg.label_stats(samples, stop_repeat=5)
    u = labels["all"]["unique"]
    assert u["stop"] == 2 and labels["all"]["stop_in_training"] == pytest.approx(10 / (u["goal"] + u["turn"] + 10))
    trunc = dg.truncation_stats(samples, _Tok(), [20, 500])
    assert trunc["rxr"]["cut"][20]["truncated"] == 1.0 and trunc["r2r"]["cut"][500]["truncated"] == 0.0

    torch.manual_seed(0)
    runs, kept = {}, None
    for name in ("a/step_1", "a/step_2"):
        kept, runs[name] = dg.profile(tiny_nav().eval(), samples, _Tok(), batch_size=3, num_workers=0)
    assert len(kept) == len(samples) and set(runs["a/step_1"]["pred"].tolist()) <= {-1, 0, 2, 3}
    summary = dg.summarize(kept, runs)
    r = summary["runs"]["a/step_1"]
    assert set(r["accuracy"]) == {"r2r", "rxr", "all"} and r["accuracy"]["all"]["stop"]["n"] == 2
    assert 0 <= r["calibration"]["all"]["ece"] <= 1
    assert sum(r["stop"]["confusion"].values()) == len(kept)
    c = summary["consistency"]["all/all"]
    assert c["always_right"] + c["always_wrong"] + c["mixed"] == pytest.approx(1)

    out = tmp_path / "diag"
    dg.write_report(str(out), labels, trunc, summary, kept, runs, cfg_len=12)
    assert "STOP: precision" in (out / "report.md").read_text()
    with open(out / "per_sample.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(kept) and rows[0]["dataset"] in ("r2r", "rxr") and "a/step_2:correct" in rows[0]
    assert sum(row["kind"] == "goal" for row in rows) == sum(s["kind"] == GOAL for s in kept)


def test_auroc_and_ece():
    dg = load_script("diagnose_decision")
    assert dg.auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False])) == 1.0
    assert dg.auroc(np.array([0.5, 0.5]), np.array([True, False])) == 0.5
    conf = np.full(100, 0.8)
    assert dg.ece(conf, np.r_[np.ones(80), np.zeros(20)].astype(bool)) == pytest.approx(0.0)
    assert dg.ece(conf, np.zeros(100, bool)) == pytest.approx(0.8)
