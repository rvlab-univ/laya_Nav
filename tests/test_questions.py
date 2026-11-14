"""Answer sentences, option order and the left-right mirror."""
import json
import os
import re

import pytest

from lightnav.questions import (ANSWERS, KEYS, MIRROR, NUM_ACTIONS, QUESTION, STOP, answer_index, decide,
                                from_low_level, kind, low_level)


def navila_parse(sentence):
    """What NaVILA's evaluation loop does with an answer (navila_trainer.py): (action, repeat count)."""
    patterns = {0: r"\bstop\b", 1: r"\bis move forward\b", 2: r"\bis turn left\b", 3: r"\bis turn right\b"}
    action = next((a for a, p in patterns.items() if re.search(p, sentence, re.IGNORECASE)), None)
    if action == 1:
        return 1, int(re.search(r"move forward (\d+) cm", sentence).group(1)) // 25
    if action in (2, 3):
        side = "left" if action == 2 else "right"
        return action, int(re.search(r"turn %s (\d+) degree" % side, sentence).group(1)) // 15
    return action, 1


def test_navila_reads_every_answer_as_the_intended_action():
    for i, sentence in enumerate(ANSWERS):
        assert navila_parse(sentence) == low_level(i), sentence


def test_low_level_round_trip():
    assert low_level(STOP) == (0, 1)
    for i in range(1, NUM_ACTIONS):
        assert from_low_level(*low_level(i)) == i
    assert [kind(i) for i in range(NUM_ACTIONS)] == [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_option_order_matches_answers():
    assert len(KEYS) == NUM_ACTIONS == 10
    for key, sentence in zip(KEYS, ANSWERS):
        if key == "stop":
            assert "stop" in sentence
        else:
            direction, amount = key.split()
            assert direction in sentence and " %s " % amount in sentence
        assert QUESTION["criteria"][key]


def test_mirror_swaps_left_and_right_only():
    assert [MIRROR[MIRROR[i]] for i in range(NUM_ACTIONS)] == list(range(NUM_ACTIONS))
    for i, sentence in enumerate(ANSWERS):
        mirrored = ANSWERS[MIRROR[i]]
        if "left" in sentence:
            assert mirrored == sentence.replace("left", "right")
        elif "right" in sentence:
            assert mirrored == sentence.replace("right", "left")
        else:
            assert mirrored == sentence


def test_decide():
    p = [0.30, 0.05, 0.05, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10]
    assert decide(p) == 0
    assert decide(p, tau_stop=0.5) == 3      # stop is the top answer but under the threshold
    assert decide(p, tau_stop=0.2) == 0


def test_every_annotation_answer_is_known(r2r_annotations):
    answers = set()
    for name in ("R2R", "RxR"):
        path = os.path.join(os.path.dirname(os.path.dirname(r2r_annotations)), name, "annotations.json")
        if os.path.exists(path):
            with open(path) as f:
                answers |= {x["a"] for x in json.load(f)}
    assert answers == set(ANSWERS)
    with pytest.raises(KeyError):
        answer_index("The next action is move forward 100 cm.")


def test_summary_reports_what_a_constant_answer_scores():
    """6 forward 75, 2 left 15, 1 left 45, 1 stop: the best constant answer gets 0.6, the best constant kind 0.6."""
    import numpy as np

    from lightnav.metrics import summarize
    labels = np.array([3] * 6 + [4, 4, 6, 0])
    probs = np.full((10, 10), 0.01)
    probs[np.arange(10), [3] * 6 + [4, 5, 6, 3]] = 0.91          # one left 15 answered as left 30, the stop missed
    s = summarize(labels, probs)
    assert s["constant_acc"] == 0.6 and s["constant_kind_acc"] == 0.6
    assert abs(s["acc"] - 0.8) < 1e-9 and abs(s["kind_acc"] - 0.9) < 1e-9
