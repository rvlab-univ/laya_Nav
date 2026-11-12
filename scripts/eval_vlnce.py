"""VLN-CE closed-loop evaluation with NaVILA's protocol (run in env_vlnce: Python 3.8, Habitat 0.1.7).

The loop is navila_trainer.py's: one frame per low-level step, NaVILA's 8-frame selection, NaVILA's answer parsing
and action queue, NaVILA's task configs (512 x 512 RGB, 90 degree HFOV, 0.25 m / 15 degrees, 500 steps, success
within 3 m) and VLN-CE's measures. Only the answer comes from elsewhere:

    --agent http   the student process (lightnav/server.py) at --url
    --agent gt     the ground-truth actions in NaVILA's sentences (check G3: the loop itself)
    --agent fixed  always "move forward 75 cm" (lower bound)

    env_vlnce/bin/python scripts/eval_vlnce.py --split r2r --agent gt --episodes 50 --out runs/eval/gt/r2r_50
    env_vlnce/bin/python scripts/eval_vlnce.py --split r2r --agent http --url http://127.0.0.1:8765 --out runs/eval/N1/r2r

Results: <out>/episodes.jsonl (one line per episode) and <out>/subset.json. A rerun skips finished episodes.
Summaries: scripts/eval_summary.py.
"""
import argparse
import collections
import gzip
import json
import os
import random
import sys
import time
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAVILA_EVAL = os.path.join(ROOT, "NaVILA", "evaluation")
sys.path.insert(0, ROOT)

from lightnav.data import frame_slots                                         # noqa: E402
from lightnav.questions import ANSWERS, KEYS                                 # noqa: E402
from lightnav.vlnce import (STOP_ACTION, decode_response, encode_request, gt_answers,  # noqa: E402
                            navila_actions)

SPLITS = {
    "r2r": {"exp": "vlnce_baselines/config/r2r_baselines/navila.yaml",
            "data": "data/vlnce/r2r/{split}.json.gz", "gt": "data/vlnce/r2r/{split}_gt.json.gz", "languages": None},
    "rxr": {"exp": "vlnce_baselines/config/rxr_baselines/navila.yaml",
            "data": "data/vlnce/rxr/{split}/{split}_{role}.json.gz",
            "gt": "data/vlnce/rxr/{split}/{split}_{role}_gt.json.gz", "languages": ["en-US"]},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=sorted(SPLITS), required=True)
    p.add_argument("--agent", choices=("http", "gt", "fixed"), required=True)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=None, help="evaluate a fixed random subset of this size")
    p.add_argument("--seed", type=int, default=0, help="seed of the subset")
    p.add_argument("--scenes_dir", default=os.path.join(ROOT, "data", "scene_datasets"))
    p.add_argument("--gpu", type=int, default=0, help="GPU that renders the scenes")
    p.add_argument("--turn_angle", type=int, default=None,
                   help="override the turn angle (diagnosis only: NaVILA evaluates with 15 degrees)")
    return p.parse_args()


def episode_ids(cfg, split="val_unseen"):
    """All evaluated episode ids of a set, as strings, in file order."""
    path = os.path.join(ROOT, cfg["data"].format(split=split, role="guide"))
    with gzip.open(path, "rt") as f:
        eps = json.load(f)["episodes"]
    if cfg["languages"]:
        eps = [e for e in eps if e["instruction"]["language"] in cfg["languages"]]
    return [str(e["episode_id"]) for e in eps]


def make_config(args, cfg, allowed, split="val_unseen"):
    """NaVILA's experiment config with the same changes navila_trainer.py makes, and our data paths."""
    import importlib.util

    # vlnce_baselines/config/default.py on its own: importing the vlnce_baselines package would pull in every
    # trainer, including NaVILA's (llava) and tensorflow
    spec = importlib.util.spec_from_file_location("vlnce_config_default",
                                                  os.path.join(NAVILA_EVAL, "vlnce_baselines", "config", "default.py"))
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)
    config = defaults.get_config(cfg["exp"], [])
    config.defrost()
    c = config.TASK_CONFIG
    c.DATASET.SPLIT = split
    c.DATASET.ROLES = ["guide"]
    if cfg["languages"]:
        c.DATASET.LANGUAGES = cfg["languages"]
    c.DATASET.DATA_PATH = os.path.join(ROOT, cfg["data"])
    c.DATASET.SCENES_DIR = args.scenes_dir
    c.DATASET.EPISODES_ALLOWED = list(allowed)
    c.DATASET.NUM_CHUNKS, c.DATASET.CHUNK_IDX = 1, 0      # NaVILA's dataset reads these (navila_trainer.py sets them)
    c.TASK.NDTW.SPLIT = split
    c.TASK.NDTW.GT_PATH = os.path.join(ROOT, cfg["gt"])
    c.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    c.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    c.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu
    if args.turn_angle:
        c.SIMULATOR.TURN_ANGLE = args.turn_angle
    config.freeze()
    return config


class GtAgent:
    def __init__(self, cfg, split="val_unseen"):
        path = os.path.join(ROOT, cfg["gt"].format(split=split, role="guide"))
        with gzip.open(path, "rt") as f:
            self.gt = json.load(f)
        self.todo = []

    def reset(self, episode_id):
        self.todo = gt_answers(self.gt[episode_id]["actions"])

    def __call__(self, instruction, frames, slots, episode_id):
        return (self.todo.pop(0) if self.todo else ANSWERS[0]), None, 0.0


class FixedAgent:
    def reset(self, episode_id):
        pass

    def __call__(self, instruction, frames, slots, episode_id):
        return ANSWERS[3], None, 0.0


class HttpAgent:
    def __init__(self, url):
        self.url = url.rstrip("/")
        with urllib.request.urlopen(self.url + "/health", timeout=600) as r:
            print("student:", r.read().decode(), flush=True)

    def reset(self, episode_id):
        pass

    def __call__(self, instruction, frames, slots, episode_id):
        req = urllib.request.Request(self.url + "/decide", data=encode_request(instruction, frames, slots, episode_id),
                                     headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return decode_response(r.read())


def run_episode(env, agent, obs, num_frames=8):
    """One episode with navila_trainer.py's loop. Returns the record of what the agent did."""
    ep = env.current_episode
    instruction = ep.instruction.instruction_text
    agent.reset(ep.episode_id)
    past, queue = [], []
    answers, ms, unparsed = collections.Counter(), [], 0
    while not env.episode_over:
        if queue:
            action = queue.pop(0)
        else:
            frames = past + [obs["rgb"]]
            slots = frame_slots(len(frames), num_frames)
            real = [(s, frames[f]) for s, f in enumerate(slots) if f >= 0]
            answer, _, t = agent(instruction, [f for _, f in real], [s for s, _ in real], ep.episode_id)
            ms.append(t)
            actions = navila_actions(answer)
            if actions is None:              # NaVILA would crash here; count it and stop
                unparsed += 1
                actions = [STOP_ACTION]
            hit = [k for k, a in zip(KEYS, ANSWERS) if a == answer]
            answers[hit[0] if hit else "other"] += 1
            action, queue = actions[0], actions[1:]
        past.append(obs["rgb"])             # navila_trainer.py keeps the frame seen before each low-level step
        obs = env.step(action)
    return {"decisions": sum(answers.values()), "answers": dict(answers), "ms": ms, "unparsed": unparsed}


def main():
    args = parse_args()
    cfg = SPLITS[args.split]
    os.makedirs(args.out, exist_ok=True)
    out = os.path.abspath(args.out)

    ids = episode_ids(cfg)
    subset_file = os.path.join(out, "subset.json")
    if os.path.exists(subset_file):
        with open(subset_file) as f:
            chosen = json.load(f)["episodes"]
    else:
        chosen = sorted(random.Random(args.seed).sample(ids, args.episodes), key=ids.index) if args.episodes else ids
        with open(subset_file, "w") as f:
            json.dump({"split": args.split, "seed": args.seed, "episodes": chosen}, f)
    results = os.path.join(out, "episodes.jsonl")
    done = set()
    if os.path.exists(results):
        with open(results) as f:
            done = {json.loads(line)["episode_id"] for line in f if line.strip()}
    todo = [e for e in chosen if e not in done]
    print("%s: %d episodes, %d done, %d to go" % (args.split, len(chosen), len(done), len(todo)), flush=True)
    if not todo:
        return

    os.chdir(NAVILA_EVAL)                   # NaVILA's configs use paths relative to its evaluation folder
    sys.path.insert(0, NAVILA_EVAL)
    import habitat
    import habitat_extensions  # noqa: F401  (registers VLN-CE's task, measures and datasets)

    agent = {"gt": lambda: GtAgent(cfg), "fixed": FixedAgent, "http": lambda: HttpAgent(args.url)}[args.agent]()
    env = habitat.Env(config=make_config(args, cfg, todo).TASK_CONFIG)
    started = time.time()
    with open(results, "a") as f:
        for i in range(len(env.episodes)):
            t = time.time()
            obs = env.reset()
            ep = env.current_episode
            rec = run_episode(env, agent, obs)
            metrics = {k: float(v) for k, v in env.get_metrics().items()
                       if isinstance(v, (int, float, np.integer, np.floating))}
            rec = dict(episode_id=ep.episode_id, scene=os.path.basename(ep.scene_id).split(".")[0],
                       metrics=metrics, wall_s=time.time() - t, **rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(env.episodes):
                rate = (time.time() - started) / (i + 1)
                print("%d/%d episodes, %.1f s each, %.1f h left | last: success %s, steps %s"
                      % (i + 1, len(env.episodes), rate, rate * (len(env.episodes) - i - 1) / 3600,
                         metrics.get("success"), metrics.get("steps_taken")), flush=True)
    env.close()


if __name__ == "__main__":
    main()
