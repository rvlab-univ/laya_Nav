"""Write the validation split every run shares: whole MP3D scenes held out of R2R and RxR together.

The file goes into git (splits/), so the server needs neither the VLN-CE episodes nor the RxR guide file.

    env/bin/python scripts/make_split.py \
        --r2r_scenes data/vlnce/r2r/train.json.gz --rxr_guide data/vlnce/rxr/rxr_train_guide.jsonl.gz

Episode files (not in git):
- R2R: train/train.json.gz from R2R_VLNCE_v1-3_preprocessed.zip (Google Drive id 1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr,
  linked from NaVILA's README). NaVILA's R2R episode numbers are its episode_id (all 10,819 match, instructions too).
- RxR: https://storage.googleapis.com/rxr-data/rxr_train_guide.jsonl.gz. NaVILA numbers RxR episodes differently,
  so they are matched by instruction text (lightnav.data.rxr_episode_scenes).
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lightnav import data as D  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.path.join(ROOT, "data", "NaVILA-Dataset")
    p.add_argument("--r2r_ann", default=os.path.join(data, "R2R", "annotations.json"))
    p.add_argument("--rxr_ann", default=os.path.join(data, "RxR", "annotations.json"))
    p.add_argument("--r2r_scenes", required=True, help="VLN-CE R2R train.json.gz")
    p.add_argument("--rxr_guide", required=True, help="RxR rxr_train_guide.jsonl.gz")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "splits", "val_scenes.json"))
    args = p.parse_args()

    rows = {"r2r": D.load_rows(args.r2r_ann, "r2r"), "rxr": D.load_rows(args.rxr_ann, "rxr")}
    scenes = {"r2r": D.episode_scenes(args.r2r_scenes), "rxr": D.rxr_episode_scenes(args.rxr_guide, rows["rxr"])}
    episodes = {n: [r.episode for r in rs] for n, rs in rows.items()}
    val, held = D.make_val_split(episodes, scenes, args.val_frac, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    D.save_split(args.out, val, json.dumps({n: "scene" for n in rows}), held)
    all_scenes = {scenes[n][e] for n, eps in episodes.items() for e in eps}
    print("scenes: %d in R2R + RxR, %d held out: %s" % (len(all_scenes), len(held), ", ".join(sorted(held))))
    for n, rs in rows.items():
        unique = {r.key for r in rs}
        held_rows = [r for r in rs if r.episode in val[n]]
        per_scene = collections.Counter(scenes[n][r.episode] for r in held_rows)
        print("%s: %d / %d episodes, %d / %d rows (%d unique) held out; rows per held scene %s"
              % (n, len(val[n]), len(set(episodes[n])), len(held_rows), len(rs),
                 len({r.key for r in held_rows}), dict(per_scene)))
        assert len(unique) > 0
    print("wrote", args.out)


if __name__ == "__main__":
    main()
