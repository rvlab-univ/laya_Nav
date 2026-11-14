"""DAgger data on this PC (design_C_review.md, step 2; Python 3.8, env_vlnce): the student drives, the expert labels.

A rollout is an R2R training episode outside the validation split. At every decision the student
(lightnav/server.py at --url) answers from the frames so far, and NaVILA's expert (scripts/collect_vlnce.py) gives
the answer NaVILA's data would hold for that state: its next action and how often it would repeat it (at most 3).
With probability --beta the expert's answer is run, otherwise the student's (Ross et al. 2011; VLN-CE uses
beta = 0.75^n in round n). The sample is the state with the expert's answer, whoever drove.

    - Until the student's answer is run for the first time the rollout is NaVILA's own episode; --beta 1 needs no
      student and must give NaVILA's annotations back (it is checked and reported).
    - Off the expert's path the expert aims at the nearest target it has not passed, and at the goal again when the
      student has left it.
    - A stop the expert does not agree with is not run (the expert's answer is): the rollout would end there and the
      rest of the path would never be visited. The sample still says what the expert would do.
    - A rollout ends when stop is run, or after twice the frames of NaVILA's episode plus 30.

Output, in NaVILA's format so scripts/cache_features.py and scripts/train.py --extra read it:
    <out>/config.json            beta, seed, the student, the chosen episodes (a rerun continues the same set)
    <out>/episodes.jsonl         one line per finished rollout: its decisions and how it ended
    <out>/annotations.json       the samples, with NaVILA's copies (stop x3, 30 / 45 degree turns x2)
    <out>/frames/<episode>/frame_<i>.jpg

    HF_HOME=assets/hf_cache env/bin/python -m lightnav.server --ckpt runs/D/last --port 8766 &
    env_vlnce/bin/python scripts/collect_dagger.py --out data/dagger/round1 --url http://127.0.0.1:8766 \\
        --beta 0.75 --episodes 5000 --seed 1 --limit 500
    # exit code 3: nothing left to do. scripts/dagger_round.sh runs a whole round.
"""
import argparse
import collections
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_vlnce as E                                                          # noqa: E402
from collect_vlnce import JPEG_QUALITY, ReferenceExpert                         # noqa: E402
from lightnav.data import frame_slots                                           # noqa: E402
from lightnav.questions import ANSWERS, KEYS, STOP, answer_index, from_low_level, kind, low_level   # noqa: E402

MAX_FRAMES = 500         # VLN-CE ends an episode after 500 steps


class DaggerExpert(ReferenceExpert):
    """The reference expert for states the student led to. `strayed` is set once an answer other than the expert's
    has been run; from then on the next target in order may lie behind, so every answer first re-aims."""

    strayed = False

    def _aim(self):
        pos = self.sim.get_agent_state().position
        first = min(self.i, len(self.targets) - 1)          # the goal is a target again once the student has left it
        dist = [self.sim.geodesic_distance(pos, t) for t in self.targets[first:]]
        self.i = first + int(np.argmin(dist))

    def next_action(self):
        if self.strayed:
            self._aim()
        return super().next_action()

    def label(self):
        """The answer NaVILA's data would hold for the current state, as an index into ANSWERS. NaVILA merges up to
        3 equal expert actions in a row into one answer, so the expert is walked ahead and the agent put back."""
        a = self.next_action()
        if a == 0:
            return STOP
        state, i, n = self.sim.get_agent_state(), self.i, 1
        while n < 3:
            self.sim.step(a)
            if self.next_action() != a:
                break
            n += 1
        self.sim.set_agent_state(state.position, state.rotation, reset_sensors=False)
        self.i = i
        return from_low_level(a, n)


def navila_episodes(path):
    """episode id -> {"q": the instruction as NaVILA wrote it, "rows": [(frames, answer index), ...] by decision}."""
    with open(path) as f:
        rows = json.load(f)
    by = collections.defaultdict(dict)
    q = {}
    for r in rows:
        ep, step = r["video_id"].rsplit("-", 1)
        by[ep][int(step)] = (len(r["frames"]), answer_index(r["a"]))
        q[ep] = r["q"]
    return {ep: {"q": q[ep], "rows": [d[s] for s in sorted(d)]} for ep, d in by.items()}


def copies(label):
    """How often NaVILA's R2R annotations repeat a sample: stop x3, 30 / 45 degree turns x2."""
    action, repeat = low_level(label)
    return 3 if label == STOP else (2 if action != 1 and repeat >= 2 else 1)


def annotation_rows(record, balance=True):
    """One finished rollout (a line of episodes.jsonl) as NaVILA annotation rows."""
    ep, out = record["episode_id"], []
    for d in record["rows"]:
        row = {"video_id": "%s-%d" % (ep, d["step"]), "q": record["q"], "a": ANSWERS[d["label"]],
               "frames": ["%s/frame_%d.jpg" % (ep, i) for i in range(d["frames"])]}
        out += [row] * (copies(d["label"]) if balance else 1)
    return out


def read_records(out):
    path = os.path.join(out, "episodes.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_annotations(out):
    records = read_records(out)
    rows = [r for rec in records for r in annotation_rows(rec)]
    tmp = os.path.join(out, "annotations.json.tmp")
    with open(tmp, "w") as f:
        json.dump(rows, f)
    os.replace(tmp, os.path.join(out, "annotations.json"))
    return records, rows


def summary(records):
    """What the rollouts looked like: who drove, how often the student agreed with the expert, how they ended."""
    rows = [d for rec in records for d in rec["rows"]]
    asked = [d for d in rows if d["student"] is not None]
    n, m = max(1, len(rows)), max(1, len(asked))
    labels = collections.Counter(KEYS[d["label"]] for d in rows)
    return {
        "episodes": len(records), "decisions": len(rows), "frames": sum(rec["frames"] for rec in records),
        "driven_by_student": sum(d["by"] == "student" for d in rows) / n,
        "student_agrees": sum(d["student"] == d["label"] for d in asked) / m,
        "student_agrees_kind": sum(kind(d["student"]) == kind(d["label"]) for d in asked) / m,
        "student_agrees_off_path": (lambda off: sum(d["student"] == d["label"] for d in off) / max(1, len(off)))(
            [d for d in asked if d["strayed"]]),
        "decisions_off_path": sum(d["strayed"] for d in rows) / n,
        "stops_not_run": sum(rec["stops_not_run"] for rec in records),
        "ended_by_stop": sum(rec["end"] == "stop" for rec in records) / max(1, len(records)),
        "ended_within_3m": sum(rec["final_distance"] < 3.0 for rec in records) / max(1, len(records)),
        "labels": {k: round(labels[k] / n, 3) for k in KEYS},
    }


def rollout(env, obs, expert, agent, q, beta, rng, folder, cap):
    """One rollout. Frames go to `folder` as NaVILA's dataset has them; returns (decisions, how it ended)."""
    from PIL import Image

    past, rows, queue, stops_not_run, end = [], [], [], 0, "cap"
    while True:
        if not queue:
            strayed = expert.strayed
            label = expert.label()
            student = None
            if agent is not None:
                frames = past + [obs["rgb"]]
                real = [(s, frames[f]) for s, f in enumerate(frame_slots(len(frames))) if f >= 0]
                answer, _, _ = agent(q, [f for _, f in real], [s for s, _ in real], "")
                student = ANSWERS.index(answer)
            by = "expert" if agent is None or rng.random() < beta else "student"
            run = label if by == "expert" else student
            if run == STOP and label != STOP:
                run, stops_not_run = label, stops_not_run + 1
            if run != label:
                expert.strayed = True
            rows.append({"step": len(rows), "frames": len(past) + 1, "label": label, "student": student, "by": by,
                         "run": run, "strayed": strayed})
            action, repeat = low_level(run)
            queue = [action] * repeat
        Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(folder, "frame_%d.jpg" % len(past)),
                                                  quality=JPEG_QUALITY, subsampling=2)
        past.append(obs["rgb"])
        action = queue.pop(0)
        if action == 0:
            end = "stop"
            break
        if len(past) >= cap or env.episode_over:
            break
        obs = env.step(action)
    return rows, len(past), end, stops_not_run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8766", help="the student (not asked with --beta 1)")
    p.add_argument("--beta", type=float, required=True, help="probability that the expert's answer is run")
    p.add_argument("--episodes", type=int, default=5000, help="rollouts of the round (a random set of episodes)")
    p.add_argument("--seed", type=int, default=1, help="seed of the set and of who drives; use the round number")
    p.add_argument("--limit", type=int, default=None, help="rollouts in this run")
    p.add_argument("--only", nargs="+", default=None, help="just these episode ids (for checks)")
    p.add_argument("--ann", default=os.path.join(ROOT, "data", "NaVILA-Dataset", "R2R", "annotations.json"))
    p.add_argument("--split", default=os.path.join(ROOT, "splits", "val_scenes.json"),
                   help="validation split: its R2R episodes are never driven")
    p.add_argument("--scenes_dir", default=os.path.join(ROOT, "data", "scene_datasets"))
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    navila = navila_episodes(args.ann)
    with open(args.split) as f:
        held = set(json.load(f)["val_episodes"]["r2r"])
    ids = [e for e in E.episode_ids(E.SPLITS["r2r"], "train") if e in navila and e not in held]

    student = None
    if args.beta < 1:
        with urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=600) as r:
            student = json.loads(r.read().decode())
    config_file = os.path.join(out, "config.json")
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
        if config["beta"] != args.beta or (student or {}).get("ckpt") != (config["student"] or {}).get("ckpt"):
            sys.exit("%s was started with beta %s and student %s" % (out, config["beta"], config["student"]))
    else:
        chosen = [e for e in ids if e in args.only] if args.only else (
            sorted(random.Random(args.seed).sample(ids, args.episodes), key=ids.index)
            if args.episodes < len(ids) else ids)
        config = {"beta": args.beta, "seed": args.seed, "student": student, "split": os.path.basename(args.split),
                  "max_frames": "min(%d, 2 x NaVILA's frames + 30)" % MAX_FRAMES, "episodes": chosen}
        with open(config_file, "w") as f:
            json.dump(config, f)
    done = {rec["episode_id"] for rec in read_records(out)}
    todo = [e for e in config["episodes"] if e not in done]
    print("beta %.3f, student %s: %d rollouts, %d done, %d to go%s" % (
        args.beta, (student or {}).get("ckpt"), len(config["episodes"]), len(done), len(todo),
        ", %d in this run" % min(len(todo), args.limit) if args.limit else ""), flush=True)
    if not todo:
        write_annotations(out)
        sys.exit(3)
    todo = todo[:args.limit] if args.limit else todo

    os.chdir(E.NAVILA_EVAL)
    sys.path.insert(0, E.NAVILA_EVAL)
    import habitat
    import habitat_extensions  # noqa: F401

    agent = E.HttpAgent(args.url) if args.beta < 1 else None
    eargs = SimpleNamespace(scenes_dir=args.scenes_dir, gpu=args.gpu, turn_angle=None)
    env = habitat.Env(config=E.make_config(eargs, E.SPLITS["r2r"], todo, "train").TASK_CONFIG)
    started, new, same = time.time(), [], 0
    with open(os.path.join(out, "episodes.jsonl"), "a") as f:
        for k in range(len(env.episodes)):
            obs = env.reset()
            ep = env.current_episode
            eid = str(ep.episode_id)
            folder = os.path.join(out, "frames", eid)
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)
            ref = navila[eid]
            cap = min(MAX_FRAMES, 2 * ref["rows"][-1][0] + 30)
            rng = random.Random("%d:%s" % (args.seed, eid))
            rows, frames, end, stops_not_run = rollout(env, obs, DaggerExpert(env.sim, ep), agent, ref["q"],
                                                       args.beta, rng, folder, cap)
            goal = ep.goals[0].position
            rec = {"episode_id": eid, "scene": os.path.basename(ep.scene_id).split(".")[0], "q": ref["q"],
                   "rows": rows, "frames": frames, "end": end, "stops_not_run": stops_not_run,
                   "final_distance": float(env.sim.geodesic_distance(env.sim.get_agent_state().position, goal))}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            new.append(rec)
            same += [(d["frames"], d["label"]) for d in rows] == [tuple(x) for x in ref["rows"]]
            if (k + 1) % 25 == 0 or k + 1 == len(env.episodes):
                s = summary(new)
                rate = (time.time() - started) / (k + 1)
                print("%d / %d rollouts, %.1f s each, %.1f h left | student drove %.2f, agrees with the expert %.3f "
                      "(off the path %.3f), ended by stop %.2f, within 3 m %.2f" % (
                          k + 1, len(env.episodes), rate, rate * (len(env.episodes) - k - 1) / 3600,
                          s["driven_by_student"], s["student_agrees"], s["student_agrees_off_path"],
                          s["ended_by_stop"], s["ended_within_3m"]), flush=True)
    env.close()
    records, rows = write_annotations(out)
    if args.beta >= 1:
        print("beta 1: %d of %d rollouts are NaVILA's annotations decision for decision" % (same, len(new)), flush=True)
    s = summary(records)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    print("so far: %d rollouts, %d decisions (%d rows with copies), %d frames" % (
        s["episodes"], s["decisions"], len(rows), s["frames"]), flush=True)
    print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
