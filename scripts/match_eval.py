"""Instruction-route matching accuracy of trained students on train routes and on validation routes.

Training logs matching accuracy on the validation routes only. Chance there (0.5) can mean two things: the model
learned the training routes but not something that carries to new scenes, or it learned nothing yet. This
script tells them apart. Both route sets are scored the same way as training's validation (scripts/train.py:
two fixed samples per route, positive and negative).

    env/bin/python scripts/match_eval.py --ckpt runs/A/last --out runs/diag/match_A.json   # server: match_eval.sh
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lightnav import data as D                    # noqa: E402
from lightnav import tasks as T                   # noqa: E402
from lightnav.model import load                   # noqa: E402


def _train_module():
    spec = importlib.util.spec_from_file_location("lightnav_train", os.path.join(ROOT, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.path.join(ROOT, "data", "NaVILA-Dataset")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--r2r_ann", default=os.path.join(data, "R2R", "annotations.json"))
    p.add_argument("--r2r_frames", default=os.path.join(data, "R2R", "train"))
    p.add_argument("--rxr_ann", default=None)
    p.add_argument("--rxr_frames", default=os.path.join(data, "RxR", "train"))
    p.add_argument("--only_available", action="store_true")
    p.add_argument("--split", default=os.path.join(ROOT, "splits", "val_scenes.json"))
    p.add_argument("--train_routes", type=int, default=1000, help="train routes to score (random, fixed)")
    p.add_argument("--match_swap", type=float, default=0.6)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if not os.path.exists(args.split):
        raise SystemExit("no split file %s" % args.split)
    train_mod = _train_module()
    targs = argparse.Namespace(r2r_ann=args.r2r_ann, r2r_frames=args.r2r_frames, rxr_ann=args.rxr_ann,
                               rxr_frames=args.rxr_frames, only_available=args.only_available, split=args.split,
                               out=None, r2r_scenes=None, rxr_scenes=None, val_frac=None, seed=0)
    train_rows, val_rows, roots = train_mod.load_data(targs)
    train_routes, val_routes = T.routes_from_rows(train_rows), T.routes_from_rows(val_rows)
    if len(train_routes) > args.train_routes:
        idx = np.random.default_rng(0).choice(len(train_routes), args.train_routes, replace=False)
        train_routes = [train_routes[i] for i in sorted(idx)]
    print("routes: %d train (scored), %d validation" % (len(train_routes), len(val_routes)), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"train_routes": len(train_routes), "val_routes": len(val_routes), "runs": []}
    for ckpt in args.ckpt:
        model, tok = load(ckpt)
        model.to(device).eval()
        collate = lambda b: D.collate(b, tok.pad_token_id)                       # noqa: E731
        rec = {"ckpt": ckpt}
        short = os.path.basename(ckpt.rstrip("/"))
        if short == "last":
            short = os.path.basename(os.path.dirname(ckpt.rstrip("/")))
        # "other" negatives come from the same route set, so each set gets its own dataset
        for name, routes in (("train", train_routes), ("val", val_routes)):
            ds = T.MatchDataset(routes, roots, tok, model.cfg, fixed=True, swap_frac=args.match_swap)
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate,
                                num_workers=args.workers)
            _, probs, kinds = train_mod.predict(model, loader, device)
            m = T.match_metrics(kinds, probs)
            yes = probs[:, T.YES] / np.clip(probs[:, :2].sum(1), 1e-9, None)
            m["p_yes_mean"] = float(yes.mean())
            m["p_yes_std"] = float(yes.std())
            rec[name] = m
            print("%s %s: acc %.3f (true %.3f, swapped landmark %.3f, other episode %.3f), p(yes) %.3f +- %.3f"
                  % (short, name, m["acc"], m["acc_pos"],
                     m["acc_swap"] or float("nan"), m["acc_other"] or float("nan"), m["p_yes_mean"], m["p_yes_std"]),
                  flush=True)
        out["runs"].append(rec)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
