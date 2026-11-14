"""Can this PC reproduce NaVILA's expert? (design_C_review.md, step 1)

NaVILA's R2R labels are not VLN-CE's published ground-truth actions (57 of 10,819 episodes agree), and NaVILA does
not publish the code that made them. DAgger needs the same expert on states the student reaches, so this script runs
candidate experts in our Habitat setup on train episodes and compares, per episode:
  - the expert's low-level actions with the ones NaVILA's annotations imply (exact match, shared prefix)
  - the rendered first frame with NaVILA's frame_0.jpg (mean absolute pixel difference), where the frame is on disk

Result (2026-11-14, 40 episodes): `ShortestPathFollowerCompat` (Habitat v0.1.4's follower) through the reference
path point by point with radius 0.5 m, then the goal with radius 0.5 m, gives NaVILA's actions exactly (40 / 40).
The current follower on the same points gives VLN-CE's ground truth instead. First frames differ by 2.6 / 255 on
average (JPEG noise).

    env_vlnce/bin/python scripts/expert_check.py --episodes 40 --out runs/expert_check.json
"""
import argparse
import collections
import json
import os
import re
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_vlnce as E  # noqa: E402


def navila_low_level(path):
    """episode id -> the low-level actions (1 forward, 2 left, 3 right, 0 stop) its answers stand for."""
    with open(path) as f:
        rows = json.load(f)
    by = collections.defaultdict(dict)
    for r in rows:
        ep, step = r["video_id"].rsplit("-", 1)
        by[ep][int(step)] = r["a"]
    out = {}
    for ep, d in by.items():
        seq = []
        for s in sorted(d):
            a = d[s]
            if a.startswith("I think"):
                seq.append(0)
                continue
            m = re.search(r"(move forward|turn left|turn right) (\d+)", a)
            kind = {"move forward": 1, "turn left": 2, "turn right": 3}[m.group(1)]
            seq += [kind] * (int(m.group(2)) // (25 if kind == 1 else 15))
        out[ep] = seq
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--ann", default=os.path.join(ROOT, "data", "NaVILA-Dataset", "R2R", "annotations.json"))
    p.add_argument("--frames", default=os.path.join(ROOT, "data", "NaVILA-Dataset", "R2R", "train_sample"))
    p.add_argument("--goal_radius", type=float, nargs="+", default=[0.5])
    p.add_argument("--waypoint_radius", type=float, nargs="+", default=[0.5],
                   help="for the variants that follow the reference path point by point")
    p.add_argument("--out", required=True)
    p.add_argument("--scenes_dir", default=os.path.join(ROOT, "data", "scene_datasets"))
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    want = navila_low_level(args.ann)
    on_disk = sorted(d for d in os.listdir(args.frames) if d in want) if os.path.isdir(args.frames) else []
    chosen = on_disk[:args.episodes] or sorted(want)[:args.episodes]
    cfg = E.SPLITS["r2r"]
    os.chdir(E.NAVILA_EVAL)
    sys.path.insert(0, E.NAVILA_EVAL)
    import habitat
    import habitat_extensions  # noqa: F401
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
    from PIL import Image

    eargs = SimpleNamespace(scenes_dir=args.scenes_dir, gpu=args.gpu, turn_angle=None)
    env = habitat.Env(config=E.make_config(eargs, cfg, chosen, "train").TASK_CONFIG)
    variants = [("%s r=%.2f" % (name, r), cls, r, None) for r in args.goal_radius
                for name, cls in (("follower", ShortestPathFollower), ("compat", ShortestPathFollowerCompat))]
    # the reference path point by point (R2R's route on the nav-graph), then the goal
    variants += [("%s via path w=%.2f" % (name, w), cls, args.goal_radius[0], w) for w in args.waypoint_radius
                 for name, cls in (("follower", ShortestPathFollower), ("compat", ShortestPathFollowerCompat))]
    res = {v[0]: {"same": 0, "prefix": [], "len_diff": []} for v in variants}
    frame_mae, per_episode = [], []
    for _ in range(len(env.episodes)):
        obs = env.reset()
        ep = env.current_episode
        ref = want[str(ep.episode_id)]
        f0 = os.path.join(args.frames, str(ep.episode_id), "frame_0.jpg")
        if os.path.exists(f0):
            theirs = np.asarray(Image.open(f0).convert("RGB"), dtype=np.float32)
            ours = np.asarray(obs["rgb"], dtype=np.float32)[..., :3]
            mae = float(np.abs(theirs - ours).mean()) if theirs.shape == ours.shape else None
            frame_mae.append({"episode": ep.episode_id, "mae": mae, "shape_ours": list(ours.shape),
                              "shape_theirs": list(theirs.shape)})
        start = env.sim.get_agent_state()
        row = {"episode": ep.episode_id, "navila_len": len(ref), "navila": ref}
        for name, cls, radius, via in variants:
            env.sim.set_agent_state(start.position, start.rotation)
            targets = [(ep.goals[0].position, radius)]
            if via is not None:
                targets = [(q, via) for q in ep.reference_path[1:-1]] + targets
            seq = []
            for target, rad in targets:
                follower = cls(env.sim, rad, return_one_hot=False)
                while len(seq) < 500:
                    a = follower.get_next_action(target)
                    if a is None or a == 0:
                        break
                    seq.append(int(a))
                    env.sim.step(a)
            seq.append(0)
            k = 0
            while k < min(len(seq), len(ref)) and seq[k] == ref[k]:
                k += 1
            res[name]["same"] += seq == ref
            res[name]["prefix"].append(k / max(1, len(ref)))
            res[name]["len_diff"].append(len(seq) - len(ref))
            row[name] = {"len": len(seq), "prefix": k, "same": seq == ref, "seq": seq}
        per_episode.append(row)
    n = len(per_episode)
    summary = {name: {"episodes": n, "identical": r["same"], "mean_prefix_share": float(np.mean(r["prefix"])),
                      "mean_len_diff": float(np.mean(r["len_diff"]))} for name, r in res.items()}
    good = [f["mae"] for f in frame_mae if f["mae"] is not None]
    summary["first_frame"] = {"compared": len(good), "mean_abs_diff": float(np.mean(good)) if good else None,
                              "max_abs_diff": float(np.max(good)) if good else None,
                              "shapes": frame_mae[0] if frame_mae else None}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out, "w") as f:
        json.dump({"summary": summary, "episodes": per_episode, "frames": frame_mae}, f, indent=1)


if __name__ == "__main__":
    main()
