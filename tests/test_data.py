"""Frame selection against NaVILA's own code, annotation rows, mirroring, the validation split."""
import ast
import copy
import gzip
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import DATA, NAVILA, ROOT
from lightnav.data import (NavDataset, choose_val_episodes, clean_instruction, collate, dropped_instruction_tokens,
                           frame_slots, load_rows, load_split, low_level_actions, make_val_split, mirror_instruction,
                           rxr_episode_scenes, split_rows, target_distribution)
from lightnav.model import LayaNativeConfig
from lightnav.questions import ANSWERS, MIRROR, QUESTION


def navila_function(path, name):
    """A function taken out of a NaVILA source file without importing the module around it."""
    path = os.path.join(NAVILA, path)
    if not os.path.exists(path):
        pytest.skip("no NaVILA source at %s" % path)
    with open(path) as f:
        tree = ast.parse(f.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"np": np, "Image": Image, "copy": copy}
    exec(compile(ast.Module([node], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def numbered_frames(n):
    return [Image.new("RGB", (4, 4), (i % 200 + 1, i // 200 + 1, 9)) for i in range(n)]


def picked(images):
    """The frame number behind each returned image, -1 for NaVILA's black padding."""
    out = []
    for im in images:
        r, g, b = im.convert("RGB").getpixel((0, 0))
        out.append(-1 if (r, g, b) == (0, 0, 0) else (g - 1) * 200 + r - 1)
    return out


@pytest.mark.parametrize("source,name", [("llava/mm_utils.py", "vlnce_frame_sampling"),
                                         ("evaluation/vlnce_baselines/navila_trainer.py", "sample_and_pad_images")])
def test_frame_slots_match_navila(source, name):
    navila = navila_function(source, name)
    for n in (1, 2, 5, 7, 8, 9, 15, 16, 29, 100, 501):
        assert frame_slots(n) == picked(navila(numbered_frames(n), 8)), n


def test_frame_slots_shape():
    assert frame_slots(1) == [-1] * 7 + [0]
    assert frame_slots(3) == [-1] * 5 + [0, 1, 2]
    for n in (1, 8, 50):
        slots = frame_slots(n)
        assert len(slots) == 8 and slots[-1] == n - 1
        real = [s for s in slots if s >= 0]
        assert real == sorted(set(real))       # no frame twice, oldest first


def test_clean_instruction():
    assert clean_instruction("walk past the Table .  then turn left.\nstop there .") == \
        "Walk past the table.  then turn left. Stop there."


def test_mirror_instruction():
    assert mirror_instruction("Turn left, then go Right at the sofa.") == "Turn right, then go Left at the sofa."
    assert mirror_instruction("Go right in front of the door.") is None      # "right" is no direction here
    assert mirror_instruction("बाएं मुड़ें") is None


def write_annotations(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f)


def annotation(episode, step, n_frames, answer, q="Turn left and stop."):
    return {"video_id": "%s-%d" % (episode, step), "q": q, "a": ANSWERS[answer],
            "frames": ["%s/frame_%d.jpg" % (episode, i) for i in range(n_frames)]}


def test_load_rows_dedupe_and_language(tmp_path):
    path = str(tmp_path / "a.json")
    write_annotations(path, [annotation("7", 0, 1, 6), annotation("7", 0, 1, 6), annotation("7", 1, 4, 0),
                             annotation("8", 0, 1, 0, q="बाएं मुड़ें")])
    rows = load_rows(path, "r2r")
    assert [(r.video_id, r.label, r.n_frames) for r in rows] == [("7-0", 6, 1), ("7-0", 6, 1), ("7-1", 0, 4)]
    assert rows[0].key == "r2r:7-0" and rows[2].frame(3) == "7/frame_3.jpg" and rows[2].step == 1
    assert len(load_rows(path, "r2r", dedupe=True)) == 2
    assert len(load_rows(path, "r2r", english_only=False)) == 4


def test_load_rows_rejects_frames_that_are_not_contiguous(tmp_path):
    path = str(tmp_path / "a.json")
    bad = annotation("7", 0, 3, 1)
    bad["frames"] = ["7/frame_0.jpg", "7/frame_2.jpg", "7/frame_5.jpg"]
    write_annotations(path, [bad])
    with pytest.raises(ValueError):
        load_rows(path, "r2r")


def test_low_level_actions(tmp_path):
    path = str(tmp_path / "a.json")
    # left 45, forward 50, stop: frames 1 -> 4 -> 6
    write_annotations(path, [annotation("7", 0, 1, 6), annotation("7", 1, 4, 2), annotation("7", 2, 6, 0)])
    assert low_level_actions(load_rows(path, "r2r")) == {"r2r:7": [2, 2, 2, 1, 1, 0]}
    write_annotations(path, [annotation("7", 0, 1, 6), annotation("7", 1, 3, 0)])
    with pytest.raises(ValueError):
        low_level_actions(load_rows(path, "r2r"))


def test_low_level_actions_on_r2r(r2r_annotations):
    rows = load_rows(r2r_annotations, "r2r", dedupe=True)
    actions = low_level_actions(rows)        # raises on any decision whose frame count does not fit
    assert len(actions) == 10819 and len(rows) == 288594
    assert all(a[-1] == 0 and 0 not in a[:-1] for a in actions.values())


def test_val_split_holds_out_whole_scenes():
    episodes = [str(i) for i in range(200)]
    scenes = {e: "scene%d" % (int(e) % 20) for e in episodes}
    val = choose_val_episodes(episodes, scenes, frac=0.1, seed=0)
    assert val == choose_val_episodes(episodes, scenes, frac=0.1, seed=0)
    assert len({scenes[e] for e in val}) == 2
    assert not {scenes[e] for e in val} & {scenes[e] for e in episodes if e not in val}
    with pytest.raises(KeyError):
        choose_val_episodes(episodes + ["999"], scenes)


def test_joint_split_holds_out_the_same_scenes_everywhere():
    r2r = [str(i) for i in range(100)]
    rxr = [str(1000 + i) for i in range(80)]
    scenes = {"r2r": {e: "scene%d" % (int(e) % 20) for e in r2r}, "rxr": {e: "scene%d" % (int(e) % 16) for e in rxr}}
    val, held = make_val_split({"r2r": r2r, "rxr": rxr}, scenes, frac=0.1, seed=0)
    assert len(held) == 2
    for name, eps in (("r2r", r2r), ("rxr", rxr)):
        assert {scenes[name][e] for e in val[name]} <= held
        assert not {scenes[name][e] for e in eps if e not in val[name]} & held
    assert make_val_split({"r2r": r2r, "rxr": rxr}, scenes, frac=0.1, seed=0) == (val, held)
    # without scenes for every source each one is split on its own
    val2, held2 = make_val_split({"r2r": r2r, "rxr": rxr}, {"r2r": scenes["r2r"], "rxr": None}, frac=0.1)
    assert held2 is None and val2["r2r"] == choose_val_episodes(r2r, scenes["r2r"], 0.1, 0)


def test_rxr_episode_scenes_by_instruction_text(tmp_path):
    guide = tmp_path / "guide.jsonl.gz"
    with gzip.open(guide, "wt") as f:
        for scan, text in (("sA", "Turn left at the sofa."), ("sB", "Walk past the bed, then stop."),
                           ("sC", "Go up."), ("sD", "Go up.")):
            f.write(json.dumps({"scan": scan, "instruction": text}) + "\n")
    path = str(tmp_path / "a.json")
    write_annotations(path, [annotation("41", 0, 1, 6, q="turn left at the sofa"),
                             annotation("41", 1, 4, 0, q="turn left at the sofa"),
                             annotation("42", 0, 1, 0, q="Walk past the bed then stop")])
    assert rxr_episode_scenes(str(guide), load_rows(path, "rxr")) == {"41": "sA", "42": "sB"}
    write_annotations(path, [annotation("43", 0, 1, 0, q="Go up.")])           # text in two scenes
    with pytest.raises(KeyError):
        rxr_episode_scenes(str(guide), load_rows(path, "rxr"))


def test_committed_split_is_scene_level():
    path = os.path.join(ROOT, "splits", "val_scenes.json")
    if not os.path.exists(path):
        pytest.skip("no splits/val_scenes.json")
    with open(path) as f:
        split = json.load(f)
    assert json.loads(split["mode"]) == {"r2r": "scene", "rxr": "scene"} and len(split["held_scenes"]) == 3
    assert load_split(path).keys() == {"r2r", "rxr"}


def test_split_keeps_copies_together(tmp_path):
    path = str(tmp_path / "a.json")
    write_annotations(path, [annotation("7", 0, 1, 0), annotation("7", 0, 1, 0), annotation("8", 0, 1, 0)])
    train, val = split_rows(load_rows(path, "r2r"), {"r2r": {"7"}})
    assert [r.episode for r in train] == ["8"] and [r.episode for r in val] == ["7", "7"]


def test_target_distribution():
    assert target_distribution(3).tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    teacher = np.zeros(10, dtype=np.float32)
    teacher[[3, 4]] = [0.6, 0.4]
    t = target_distribution(3, teacher, alpha=0.5)
    assert t.sum() == pytest.approx(1.0) and t[3] == pytest.approx(0.8) and t[4] == pytest.approx(0.2)


def sample_rows(tmp_path, n_frames):
    """One episode of solid-colour frames whose left half is brighter, so a mirror is visible."""
    root = tmp_path / "frames"
    (root / "7").mkdir(parents=True)
    for i in range(max(n_frames)):
        px = np.zeros((32, 32, 3), dtype=np.uint8)
        px[:, :16] = 200
        Image.fromarray(px).save(root / "7" / ("frame_%d.jpg" % i))
    path = str(tmp_path / "a.json")
    write_annotations(path, [annotation("7", s, n, 5, q="Turn left at the door.") for s, n in enumerate(n_frames)])
    return load_rows(path, "r2r"), {"r2r": str(root)}


def test_dataset_item_and_mirror(tok, tmp_path):
    rows, roots = sample_rows(tmp_path, [1, 3, 12])
    cfg = LayaNativeConfig(image_size=32)
    plain = NavDataset(rows, roots, tok, cfg)
    item = plain[1]
    assert item["slots"] == [5, 6, 7] and item["pixels"].shape == (3, 3, 32, 32)       # black padding left out
    assert plain[0]["slots"] == [7] and plain[2]["slots"] == list(range(8))
    assert item["label"] == 5 and item["target"].argmax() == 5 and not item["mirrored"]
    assert item["pixels"][0, 0, :, :16].mean() > item["pixels"][0, 0, :, 16:].mean()

    flipped = NavDataset(rows, roots, tok, cfg, mirror_prob=1.0)[1]
    assert flipped["mirrored"] and flipped["label"] == MIRROR[5] == 8
    assert flipped["pixels"][0, 0, :, :16].mean() < flipped["pixels"][0, 0, :, 16:].mean()
    assert "right" in tok.decode(flipped["ids"]) and "left at" not in tok.decode(flipped["ids"])

    batch = collate([plain[0], plain[1], plain[2]], tok.pad_token_id)
    assert batch["pixels"].shape[0] == 12 and batch["frame_owner"].tolist() == [0] + [1] * 3 + [2] * 8
    assert batch["marker_pos"].shape == (3, 10) and batch["target"].shape == (3, 10)
    assert torch.equal(batch["input_ids"][0, :batch["text_len"][0]], torch.tensor(plain[0]["ids"]))


def test_long_instruction_is_counted_as_truncated(tok):
    from laya.train import to_internal
    q = to_internal("action", QUESTION)
    assert dropped_instruction_tokens(tok, q, "Walk forward.", 448, 192) == 0
    assert dropped_instruction_tokens(tok, q, "Walk past the red chair and turn left. " * 60, 448, 192) > 0


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA, "R2R", "train_sample")), reason="no local frames")
def test_real_sample_frames_load(tok, r2r_annotations):
    root = os.path.join(DATA, "R2R", "train_sample")
    have = set(os.listdir(root))
    rows = [r for r in load_rows(r2r_annotations, "r2r", dedupe=True) if r.episode in have][:50]
    ds = NavDataset(rows, {"r2r": root}, tok, LayaNativeConfig())
    item = ds[len(rows) - 1]
    assert item["pixels"].shape[1:] == (3, 224, 224) and len(item["markers"]) == 10
