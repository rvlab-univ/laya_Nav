"""The instruction-route matching task (design B 4.1 (A)) and batches that mix it with navigation samples."""
import json
import random
import re

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import tiny_model
from lightnav import tasks as T
from lightnav.data import NavDataset, collate, load_rows
from lightnav.model import LayaNativeConfig
from lightnav.questions import ANSWERS, NO, NUM_ACTIONS, YES


def test_swap_replaces_one_landmark_everywhere_within_its_group():
    text = "Walk past the sofa into the bedroom. Stop next to the sofa by the door."
    seen = set()
    for seed in range(40):
        new, old, rep = T.swap_landmark(text, random.Random(seed))
        seen.add(old)
        words, new_words = re.findall(r"[a-z]+", text.lower()), re.findall(r"[a-z]+", new.lower())
        assert old in ("sofa", "bedroom") and rep not in words      # "bed" inside "bedroom" is no mention
        group = "object" if old == "sofa" else "room"
        assert rep in T.GROUPS[group]
        assert new_words.count(rep) == words.count(old) and old not in new_words
        assert "door" in new                                           # structural words stay
    assert seen == {"sofa", "bedroom"}


def test_swap_keeps_case_number_and_compounds():
    new, old, rep = T.swap_landmark("Chairs line the hall. Walk past the chairs.", random.Random(0))
    assert old == "chairs" and rep in T.OBJECTS_PLURAL and new.startswith(rep.capitalize())
    new, old, rep = T.swap_landmark("Go into the living room and wait.", random.Random(0))
    assert old == "living" and rep in T.ROOM_MODIFIERS and new.endswith("room and wait.")
    assert T.landmarks("Turn counter clockwise and walk down the hallway.") == []
    assert T.swap_landmark("Turn left and walk down the hallway to the stairs.", random.Random(0)) is None


def write_episodes(tmp_path, episodes):
    """episodes: {episode: (instruction, [answer index per decision])}; frames are solid grey."""
    root, rows = tmp_path / "frames", []
    for ep, (q, labels) in episodes.items():
        sizes = [1]
        for a in labels[:-1]:
            sizes.append(sizes[-1] + (a - 1) % 3 + 1)
        (root / ep).mkdir(parents=True)
        for i in range(sizes[-1]):
            Image.fromarray(np.full((32, 32, 3), 90, dtype=np.uint8)).save(root / ep / ("frame_%d.jpg" % i))
        for s, (n, a) in enumerate(zip(sizes, labels)):
            rows.append({"video_id": "%s-%d" % (ep, s), "q": q, "a": ANSWERS[a],
                         "frames": ["%s/frame_%d.jpg" % (ep, i) for i in range(n)]})
    path = tmp_path / "a.json"
    path.write_text(json.dumps(rows))
    return load_rows(str(path), "r2r"), {"r2r": str(root)}


EPISODES = {
    "1": ("Walk past the sofa and stop at the bed.", [3, 6, 2, 0]),
    "2": ("Go by the couch, then stop near the bed.", [3, 6, 2, 0]),       # same path as 1, other instruction
    "3": ("Turn right into the kitchen and stop at the sink.", [9, 1, 0]),
    "4": ("Walk into the bathroom and wait by the mirror.", [1, 2, 4, 0]),
}


def test_routes_and_other_negatives_never_share_the_path(tok, tmp_path):
    rows, roots = write_episodes(tmp_path, EPISODES)
    routes = T.routes_from_rows(rows)
    assert [r.episode for r in routes] == ["1", "2", "3", "4"]
    assert routes[0].path == routes[1].path != routes[2].path
    assert routes[0].n_frames == 1 + 3 + 3 + 2
    ds = T.MatchDataset(routes, roots, tok, LayaNativeConfig(image_size=32))
    rng = random.Random(0)
    others = {ds.other_instruction(routes[0], rng) for _ in range(50)}
    assert others == {EPISODES["3"][0], EPISODES["4"][0]}


def test_fixed_validation_samples(tok, tmp_path):
    rows, roots = write_episodes(tmp_path, EPISODES)
    routes = T.routes_from_rows(rows)
    a = T.MatchDataset(routes, roots, tok, LayaNativeConfig(image_size=32), fixed=True)
    b = T.MatchDataset(routes, roots, tok, LayaNativeConfig(image_size=32), fixed=True)
    assert len(a) == 8
    items = [a[i] for i in range(8)]
    assert [it["label"] for it in items] == [YES, NO] * 4
    assert [it["kind"] for it in items[::2]] == ["pos"] * 4 and all(it["kind"] != "pos" for it in items[1::2])
    assert all(a[i]["ids"] == b[i]["ids"] for i in range(8))
    it = items[1]
    assert len(it["markers"]) == 2 and it["target"].shape == (NUM_ACTIONS,) and it["target"][NO] == 1
    assert it["slots"] == list(range(8)) and it["pixels"].shape == (8, 3, 32, 32)
    assert "Does the instruction describe this route?" in tok.decode(it["ids"])


def test_mixed_batch_masks_the_unused_options(tok, tmp_path):
    rows, roots = write_episodes(tmp_path, EPISODES)
    model = tiny_model(tok).eval()
    cfg = model.cfg
    nav = NavDataset(rows, roots, tok, cfg)
    match = T.MatchDataset(T.routes_from_rows(rows), roots, tok, cfg, fixed=True)
    mixed = T.Mixed(nav, match, 0.25)
    assert len(mixed) == len(nav) + round(len(nav) / 3)
    batch = collate([nav[0], match[1], nav[3]], tok.pad_token_id)
    assert batch["marker_pos"].shape == (3, NUM_ACTIONS)
    assert batch["marker_mask"].sum(-1).tolist() == [NUM_ACTIONS, 2, NUM_ACTIONS]
    assert batch["task"] == ["nav", "match", "nav"]
    with torch.no_grad():
        p = torch.softmax(model(batch), -1)
    assert torch.allclose(p[1, :2].sum(), torch.tensor(1.0)) and p[1, 2:].max() < 1e-6
    assert p[0].min() > 0

    from laya.train import rlcd_loss
    model.train()
    loss = rlcd_loss(model(batch), batch["target"], batch["marker_mask"], torch.zeros(3, dtype=torch.long), 0.4)
    assert torch.isfinite(loss)
    loss.backward()


def test_match_metrics():
    probs = np.zeros((4, NUM_ACTIONS))
    probs[[0, 1, 2, 3], [YES, YES, NO, NO]] = 1
    m = T.match_metrics(["pos", "swap", "other", "pos"], probs)
    assert m["acc"] == 0.5 and m["acc_pos"] == 0.5 and m["acc_swap"] == 0.0 and m["acc_other"] == 1.0


def test_shuffled_text_moves_every_instruction():
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "train", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    rows = [T.Route("r2r", str(i), "instruction %d" % i, 1, "", ()) for i in range(6)]
    out = train.shuffled_text(rows)
    assert [r.instruction for r in out] == ["instruction %d" % ((i + 3) % 6) for i in range(6)]
    assert [r.episode for r in out] == [r.episode for r in rows]


@pytest.mark.parametrize("text", ["Exit the bedroom, walk past the dining table and wait by the fridge.",
                                  "Go up the stairs and stop in front of the TV."])
def test_real_looking_instructions_have_landmarks(text):
    assert T.landmarks(text)


def test_other_instruction_skips_episodes_of_the_same_path(tok, tmp_path):
    rows, _ = write_episodes(tmp_path, EPISODES)
    other = T.OtherInstruction(T.routes_from_rows(rows))
    rng = random.Random(0)
    row = next(r for r in rows if r.episode == "1")
    got = {other(row, rng) for _ in range(60)}
    assert got == {EPISODES["3"][0], EPISODES["4"][0]}         # never episode 2: same path, other words


def _train_module():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "train.py")
    spec = importlib.util.spec_from_file_location("lightnav_train_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_contrast_pass_reuses_the_right_frames(tok, tmp_path):
    """With the own instruction as the 'other' one, the second pass must give exactly the first pass's logits."""
    rows, roots = write_episodes(tmp_path, EPISODES)
    model = tiny_model(tok).eval()
    cfg = model.cfg
    same = NavDataset(rows, roots, tok, cfg, negative=lambda r, rng: r.instruction, negative_prob=1.0)
    match = T.MatchDataset(T.routes_from_rows(rows), roots, tok, cfg, fixed=True)
    batch = collate([same[0], match[1], same[5], same[3]], tok.pad_token_id)
    assert batch["neg_rows"].tolist() == [0, 2, 3]
    train = _train_module()
    with torch.no_grad():
        feats = model.encode_frames(batch["pixels"])
        logits = model(batch, feats=feats)
        loss, gap = train.contrast_loss(model, batch, feats, logits, margin=0.5)
    assert abs(gap) < 1e-4 and abs(float(loss) - 0.5) < 1e-4

    other = T.OtherInstruction(T.routes_from_rows(rows))
    ds = NavDataset(rows, roots, tok, cfg, negative=other, negative_prob=1.0)
    batch = collate([ds[0], ds[4]], tok.pad_token_id)
    model.train()
    feats = model.encode_frames(batch["pixels"])
    logits = model(batch, feats=feats)
    loss, gap = train.contrast_loss(model, batch, feats, logits, margin=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert batch["neg_input_ids"].shape[0] == 2
    assert "sofa" in tok.decode(batch["input_ids"][0]) and "sofa" not in tok.decode(batch["neg_input_ids"][0])
