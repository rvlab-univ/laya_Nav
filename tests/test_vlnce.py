"""VLN-CE evaluation pieces: NaVILA's answer parsing, ground-truth sentences, the request format, the student
agent and its HTTP server."""
import gzip
import json
import os
import threading
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest
import torch

from conftest import DATA, tiny_model
from lightnav.questions import ANSWERS, QUESTION
from lightnav.vlnce import (decode_request, decode_response, encode_request, encode_response, gt_answers,
                            navila_actions)

GT = os.path.join(os.path.dirname(DATA), "vlnce", "r2r", "val_unseen_gt.json.gz")


def test_navila_actions_for_the_ten_answers():
    want = [[0], [1], [1, 1], [1, 1, 1], [2], [2, 2], [2, 2, 2], [3], [3, 3], [3, 3, 3]]
    assert [navila_actions(a) for a in ANSWERS] == want


def test_navila_actions_follows_navila_rules_for_other_text():
    assert navila_actions("The next action is move forward 60 cm.") == [1, 1]          # snapped to 50
    assert navila_actions("The next action is move forward 100 cm.") == [1] * 4        # a multiple of 25 runs as is
    assert navila_actions("The next action is turn left 20 degree.") == [2]           # snapped to 15
    assert navila_actions("The next action is turn right.") == [3]                     # no number: one step
    assert navila_actions("I will stop and turn left") == [0]                          # stop is looked for first
    assert navila_actions("Empty video.") is None


def test_gt_answers_small():
    assert gt_answers([2, 2, 2, 2, 1, 3, 3, 0]) == [ANSWERS[6], ANSWERS[4], ANSWERS[1], ANSWERS[8], ANSWERS[0]]
    assert gt_answers([1, 1]) == [ANSWERS[2], ANSWERS[0]]                                # no STOP: one is added


def test_gt_answers_replay_every_r2r_val_unseen_episode():
    if not os.path.exists(GT):
        pytest.skip("no %s" % GT)
    with gzip.open(GT, "rt") as f:
        gt = json.load(f)
    for ep, v in gt.items():
        replay = [a for s in gt_answers(v["actions"]) for a in navila_actions(s)]
        assert replay == v["actions"], ep


def test_request_round_trip():
    frames = [np.random.default_rng(i).integers(0, 255, (8, 6, 4), dtype=np.uint8) for i in range(3)]
    req = decode_request(encode_request("Walk to the sofa.", frames, [5, 6, 7], "123"))
    assert req["instruction"] == "Walk to the sofa." and req["slots"] == [5, 6, 7] and req["episode"] == "123"
    assert all(np.array_equal(a, b[..., :3]) for a, b in zip(req["frames"], frames))   # RGBA in, RGB out
    assert decode_response(encode_response(ANSWERS[3], [0.1] * 10, 12.5)) == (ANSWERS[3], [0.1] * 10, 12.5)


def tiny_agent(tok):
    from laya.train import to_internal

    from lightnav import data as D
    from lightnav.agent import StudentAgent

    agent = StudentAgent.__new__(StudentAgent)
    agent.model, agent.tok, agent.device = tiny_model(tok).eval(), tok, torch.device("cpu")
    agent.cfg = agent.model.cfg
    agent.question = to_internal("action", QUESTION)
    agent.transform = D.make_image_transform(agent.cfg.image_size, agent.cfg.image_mean, agent.cfg.image_std)
    agent.tau_stop, agent.params = None, sum(p.numel() for p in agent.model.parameters())
    return agent


def test_agent_decides_one_of_the_answers(tok):
    agent = tiny_agent(tok)
    frames = [np.full((64, 64, 3), 40 * i, dtype=np.uint8) for i in range(3)]
    answer, probs, ms = agent.decide("Turn left and stop at the sofa.", frames, [5, 6, 7])
    assert answer in ANSWERS and len(probs) == 10 and abs(sum(probs) - 1) < 1e-4 and ms > 0
    # the same input gives the same answer
    assert agent.decide("Turn left and stop at the sofa.", frames, [5, 6, 7])[1] == pytest.approx(probs)


def test_server_round_trip(tok):
    from lightnav.server import make_handler

    agent = tiny_agent(tok)
    info = {"ckpt": "tiny", "params": agent.params, "device": "cpu", "tau_stop": None, "decisions": 0}
    server = HTTPServer(("127.0.0.1", 0), make_handler(agent, info))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d" % server.server_address[1]
    try:
        with urllib.request.urlopen(url + "/health") as r:
            assert json.loads(r.read())["ckpt"] == "tiny"
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        req = urllib.request.Request(url + "/decide", data=encode_request("Stop here.", frames, [7], "1"))
        with urllib.request.urlopen(req) as r:
            answer, probs, ms = decode_response(r.read())
        assert answer in ANSWERS and len(probs) == 10 and info["decisions"] == 1
    finally:
        server.shutdown()
