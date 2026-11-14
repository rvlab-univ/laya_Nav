"""DAgger data (scripts/collect_dagger.py) as training samples: NaVILA's format, and training-only extra sources."""
import importlib.util
import json
import os
import random
import sys
from types import SimpleNamespace

import numpy as np

from lightnav import features as FS
from lightnav import tasks as T
from lightnav.data import NavDataset, collate, load_rows
from lightnav.questions import ANSWERS, KEYS, STOP
from test_features import EPISODES, student, write
from test_tasks import _train_module

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _collector():
    sys.path.insert(0, SCRIPTS)
    spec = importlib.util.spec_from_file_location("lightnav_collect_dagger_test", os.path.join(SCRIPTS, "collect_dagger.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decision(step, frames, label, student, by, strayed=False):
    return {"step": step, "frames": frames, "label": label, "student": student, "by": by,
            "run": label if by == "expert" else student, "strayed": strayed}


# another walk of episode 1: the student turned right 45 where the expert turns left 45, and the expert leads back
ROLLOUT = {"episode_id": "1", "scene": "s", "q": EPISODES["1"][0], "frames": 9, "end": "stop", "stops_not_run": 1,
           "final_distance": 0.4,
           "rows": [_decision(0, 1, KEYS.index("left 45"), KEYS.index("right 45"), "student"),
                    _decision(1, 4, KEYS.index("left 45"), KEYS.index("forward 75"), "expert", True),
                    _decision(2, 7, KEYS.index("left 30"), STOP, "student", True),
                    _decision(3, 9, STOP, STOP, "expert", True)]}


def test_rollouts_become_navila_annotations_with_navilas_copies(tmp_path):
    c = _collector()
    assert [c.copies(i) for i in range(len(KEYS))] == [3, 1, 1, 1, 1, 2, 2, 1, 2, 2]      # as in R2R's annotations
    rows = c.annotation_rows(ROLLOUT)
    assert len(rows) == 2 + 2 + 2 + 3
    assert rows[0] == {"video_id": "1-0", "q": ROLLOUT["q"], "a": ANSWERS[KEYS.index("left 45")],
                       "frames": ["1/frame_0.jpg"]}
    assert rows[-1]["a"] == ANSWERS[STOP] and rows[-1]["frames"][-1] == "1/frame_8.jpg" and len(rows[-1]["frames"]) == 9
    (tmp_path / "d.json").write_text(json.dumps(rows))
    loaded = load_rows(str(tmp_path / "d.json"), "dagger1", dedupe=True)
    assert [(r.step, r.n_frames, r.label) for r in loaded] == [(d["step"], d["frames"], d["label"]) for d in ROLLOUT["rows"]]
    assert len(c.annotation_rows(ROLLOUT, balance=False)) == 4

    s = c.summary([ROLLOUT])
    assert s["decisions"] == 4 and s["driven_by_student"] == 0.5 and s["decisions_off_path"] == 0.75
    assert s["student_agrees"] == 0.25 and s["student_agrees_kind"] == 0.25 and s["stops_not_run"] == 1
    assert s["ended_by_stop"] == 1.0 and s["ended_within_3m"] == 1.0 and abs(sum(s["labels"].values()) - 1) < 1e-6


def test_extra_sources_train_only_and_borrow_r2rs_other_instructions(tok, tmp_path):
    c, train = _collector(), _train_module()
    model = student(tok)
    rows, _, stored = write(tmp_path, model)
    store = stored["r2r"].root
    (tmp_path / "d.json").write_text(json.dumps(c.annotation_rows(ROLLOUT)))
    rng = np.random.default_rng(3)
    FS.save_episode(store, "dagger1", "1/", rng.normal(size=(2, 9, 5, 32)).astype(np.float16))
    (tmp_path / "split.json").write_text(json.dumps({"mode": "episode", "val_episodes": {"r2r": ["3"]}}))
    args = SimpleNamespace(r2r_ann=str(tmp_path / "a.json"), r2r_frames="-", rxr_ann=None, rxr_frames="-",
                           r2r_scenes=None, rxr_scenes=None, features=store, only_available=True,
                           split=str(tmp_path / "split.json"), out=str(tmp_path), val_frac=0.05, seed=0,
                           extra=[["dagger1", str(tmp_path / "d.json")]])
    train_rows, val_rows, roots = train.load_data(args)
    assert {r.source for r in val_rows} == {"r2r"} and {r.episode for r in val_rows} == {"3"}
    extra = [r for r in train_rows if r.source == "dagger1"]
    assert len(extra) == 9 and train_rows[-9:] == extra
    assert {r.episode for r in train_rows if r.source == "r2r"} == {"1", "2"}

    base = [r for r in train_rows if r.source == "r2r"]
    assert T.OtherInstruction(T.routes_from_rows(base))(extra[0], random.Random(0)) is None
    other = T.OtherInstruction(T.routes_from_rows(base), alias={"dagger1": "r2r"})
    assert {other(extra[0], random.Random(k)) for k in range(8)} == {EPISODES["2"][0]}     # never episode 1's own

    ds = NavDataset(train_rows, roots, tok, model.cfg, negative=other, negative_prob=1.0)
    batch = collate([ds[len(ds) - 1], ds[0]], tok.pad_token_id)
    assert batch["keys"][0] == "dagger1:1-3" and batch["label"][0] == STOP and "neg_input_ids" in batch
    import torch
    with torch.no_grad():
        assert torch.isfinite(model(batch)).all()

    args.extra = [["r2r", str(tmp_path / "d.json")]]                  # a name that is taken
    try:
        train.load_data(args)
        raise AssertionError("an extra source named like a base source must be refused")
    except SystemExit:
        pass
