"""Render training frames in Habitat on this PC (design_C_review.md, step 1; Python 3.8, env_vlnce).

replay: NaVILA's own R2R training episodes. The low-level actions its annotations stand for are run again and every
frame is saved the way NaVILA's dataset has it (<out>/<episode>/frame_<i>.jpg, 512 x 512, JPEG quality 75), so
scripts/cache_features.py can read it with NaVILA's annotations.json and the 19.7 GB archive is not needed. Along
the way the reference expert (see scripts/expert_check.py) is asked at every state, and how often it agrees with
the replayed action is reported: that is the expert DAgger will use.

Episodes are skipped when their frames are complete (a `done` file) or their features are already stored
(--skip_features). --limit ends a run after that many episodes, so rendering and feature extraction can alternate
and the frames never take much disk:

    env_vlnce/bin/python scripts/collect_vlnce.py --mode replay --out data/collected/r2r_train --limit 500 \\
        --skip_features data/features/siglip2_g2/r2r
    # exit code 3: nothing left to do
"""
import argparse
import collections
import json
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import eval_vlnce as E                      # noqa: E402
from lightnav.vlnce import navila_actions   # noqa: E402

JPEG_QUALITY = 75        # NaVILA's frames: the quantisation tables of quality 75 with 4:2:0 chroma


def navila_low_level(path):
    """episode id -> (low-level actions its answers stand for, frames of its last decision)."""
    with open(path) as f:
        rows = json.load(f)
    by = collections.defaultdict(dict)
    for r in rows:
        ep, step = r["video_id"].rsplit("-", 1)
        by[ep][int(step)] = (r["a"], len(r["frames"]))
    out = {}
    for ep, d in by.items():
        seq = []
        for s in sorted(d):
            seq += navila_actions(d[s][0])
        out[ep] = (seq, max(n for _, n in d.values()))
    return out


class ReferenceExpert:
    """NaVILA's expert: Habitat v0.1.4's shortest-path follower through the reference path point by point (radius
    0.5 m), then to the goal (radius 0.5 m). `next_action()` is the low-level action from the simulator's current
    state: 1 forward, 2 left, 3 right, 0 stop."""

    def __init__(self, sim, episode, radius=0.5):
        from habitat_extensions.shortest_path_follower import ShortestPathFollowerCompat
        self._cls, self.sim, self.radius = ShortestPathFollowerCompat, sim, radius
        self.targets = list(episode.reference_path[1:-1]) + [episode.goals[0].position]
        self.i = 0
        self.follower = self._cls(sim, radius, return_one_hot=False)

    def next_action(self):
        while self.i < len(self.targets):
            a = self.follower.get_next_action(self.targets[self.i])
            if a is not None and a != 0:
                return int(a)
            self.i += 1
            self.follower = self._cls(self.sim, self.radius, return_one_hot=False)
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("replay",), default="replay")
    p.add_argument("--out", required=True, help="frames go to <out>/<episode>/frame_<i>.jpg")
    p.add_argument("--ann", default=os.path.join(ROOT, "data", "NaVILA-Dataset", "R2R", "annotations.json"))
    p.add_argument("--part", default="train")
    p.add_argument("--limit", type=int, default=None, help="episodes in this run")
    p.add_argument("--only", nargs="+", default=None, help="just these episode ids (for checks)")
    p.add_argument("--skip_features", default=None, help="folder of stored features: episodes in it are skipped")
    p.add_argument("--scenes_dir", default=os.path.join(ROOT, "data", "scene_datasets"))
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    skip = args.skip_features and (args.skip_features if os.path.isabs(args.skip_features)
                                   else os.path.join(ROOT, args.skip_features))
    want = navila_low_level(args.ann)
    ids = [e for e in E.episode_ids(E.SPLITS["r2r"], args.part) if e in want and (not args.only or e in args.only)]
    # `bad`: rendered before with another frame count than the annotations (never retried, so a loop over
    # chunks ends)
    todo = [e for e in ids if not os.path.exists(os.path.join(out, e, "done"))
            and not os.path.exists(os.path.join(out, e, "bad"))
            and not (skip and os.path.exists(os.path.join(skip, e + ".npy")))]
    print("%s: %d episodes with annotations, %d to render%s" % (
        args.part, len(ids), len(todo), ", %d in this run" % min(len(todo), args.limit) if args.limit else ""), flush=True)
    if not todo:
        sys.exit(3)
    todo = todo[:args.limit] if args.limit else todo

    os.chdir(E.NAVILA_EVAL)
    sys.path.insert(0, E.NAVILA_EVAL)
    import habitat
    import habitat_extensions  # noqa: F401
    from PIL import Image

    eargs = SimpleNamespace(scenes_dir=args.scenes_dir, gpu=args.gpu, turn_angle=None)
    env = habitat.Env(config=E.make_config(eargs, E.SPLITS["r2r"], todo, args.part).TASK_CONFIG)
    started, frames, agree, asked, bad = time.time(), 0, 0, 0, []
    for k in range(len(env.episodes)):
        obs = env.reset()
        ep = env.current_episode
        actions, n_frames = want[str(ep.episode_id)]
        folder = os.path.join(out, str(ep.episode_id))
        os.makedirs(folder, exist_ok=True)
        expert = ReferenceExpert(env.sim, ep)
        saved = 0
        for a in actions:
            Image.fromarray(obs["rgb"][..., :3]).save(os.path.join(folder, "frame_%d.jpg" % saved),
                                                      quality=JPEG_QUALITY, subsampling=2)
            saved += 1
            asked += 1
            agree += expert.next_action() == a
            if a == 0 or env.episode_over:
                break
            obs = env.step(a)
        frames += saved
        if saved == n_frames:
            open(os.path.join(folder, "done"), "w").close()
        else:
            bad.append((str(ep.episode_id), saved, n_frames))
            with open(os.path.join(folder, "bad"), "w") as f:
                f.write("%d frames saved, %d in the annotations\n" % (saved, n_frames))
        if (k + 1) % 100 == 0 or k + 1 == len(env.episodes):
            print("%d / %d episodes, %.0f frames/s, expert agrees with the replayed action %.4f, frame count wrong in %d"
                  % (k + 1, len(env.episodes), frames / (time.time() - started), agree / max(1, asked), len(bad)), flush=True)
    if bad:
        print("frame count differs from the annotations (episode, saved, expected):", bad[:10], flush=True)


if __name__ == "__main__":
    main()
