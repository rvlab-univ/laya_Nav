"""Trajectory head diagnostics for a LayaNav checkpoint on the held-out scenes of the training split.

Every starting frame of every validation goal sample is scored (the training validation uses a few), next to
two baselines that ignore the images: no motion, and the mean training trajectory. A head that does not clearly
beat the mean trajectory has not learned to read the goal or the scene. Errors are broken down by where along
the path the robot starts (0 = the frame the goal was chosen in) and by the length of the remaining path.

    python scripts/train/laya_s2/eval_traj.py --ckpt checkpoints/laya_nav_c1/last \
        --vln_dataset_use r2r_125cm_0_30 [--own_goal] [--max_samples 3000] [--out logs/eval_traj_c1.json]

Use the same --val_ratio / --val_split as training so that the scenes are the unseen ones.
"""

import argparse
import importlib.util
import json
import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from internnav.dataset.laya_s2_dataset import (
    LayaS2Dataset,
    collate_laya_s2,
    load_vln_samples,
    trajectory_target,
)
from internnav.model.basemodel.laya_s2 import LayaNav

_spec = importlib.util.spec_from_file_location(
    "train_laya_s2", os.path.join(os.path.dirname(__file__), "train_laya_s2.py")
)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

MAX_STARTS = 12  # trajectory_frame_ids never returns more
HEADING_STEP = 5  # direction of the first 0.5 m
POS_BINS = [
    (0.0, 0.0, "decision frame"),
    (0.0, 1 / 3, "first third"),
    (1 / 3, 2 / 3, "middle third"),
    (2 / 3, 1.0, "last third"),
]
LEN_BINS = [
    (0.0, 1.0, "< 1 m"),
    (1.0, 2.0, "1-2 m"),
    (2.0, 3.15, "2-3.2 m"),
    (3.15, float("inf"), ">= 3.2 m (clipped)"),
]


class StartsDataset(LayaS2Dataset):
    """Adds where along the path each starting frame lies (cid / goal_len)."""

    def __getitem__(self, i):
        item = super().__getitem__(i)
        s = self.samples[i]
        frac = torch.zeros(self.traj_starts)
        cids = self._pick_starts(self._traj_start_ids(s), self.traj_starts, self.augment)
        for j, cid in enumerate(cids):
            frac[j] = cid / s["goal_len"]
        item["traj_frac"] = frac
        return item


def collate(batch, pad_id):
    out = collate_laya_s2(batch, pad_id)
    out["traj_frac"] = torch.stack([b["traj_frac"] for b in batch])
    return out


def path_xy(traj: torch.Tensor) -> torch.Tensor:
    """[N, T, 3] DualVLN deltas (dx, dy scaled by 4) -> [N, T, 2] positions in metres."""
    return torch.cumsum(traj[..., :2].float() / 4, 1)


def pair_errors(pred: torch.Tensor, target: torch.Tensor):
    """ade / fde in metres and heading error of the first 0.5 m in degrees (nan for shorter paths or no motion)."""
    p, t = path_xy(pred), path_xy(target)
    dist = (p - t).norm(dim=-1)
    hp, ht = p[:, HEADING_STEP - 1], t[:, HEADING_STEP - 1]
    ang = torch.rad2deg(torch.atan2(hp[:, 1], hp[:, 0]) - torch.atan2(ht[:, 1], ht[:, 0]))
    ang = ((ang + 180) % 360 - 180).abs()
    ang[(ht.norm(dim=-1) < 0.45) | (hp.norm(dim=-1) < 1e-3)] = float("nan")  # undefined without motion
    return dist.mean(1), dist[:, -1], ang


def mean_trajectory(samples, steps, n_pairs=20000, seed=0):
    """Per-step mean of the training targets (no images needed)."""
    rng = random.Random(seed)
    pairs = [(s, int(c)) for s in samples for c in LayaS2Dataset._traj_start_ids(s)]
    pairs = rng.sample(pairs, min(n_pairs, len(pairs)))
    return torch.from_numpy(np.mean([trajectory_target(s, c, steps) for s, c in pairs], 0)).float(), len(pairs)


@torch.no_grad()
def collect(model, loader, device, own_goal):
    keys = ("input_ids", "text_mask", "hist_pixels", "hist_mask", "cur_pixels", "down_pixels")
    rows = {k: [] for k in ("pred", "pred_own", "target", "frac")}
    for batch in loader:
        b = base.to_device(batch, device)
        x = dict(traj_pixels=b["traj_pixels"], traj_mask=b["traj_mask"])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(*(b[k] for k in keys), goal_xy=b["goal_xy"], **x)  # ground-truth goal, as in training
            idx, slot = out["traj_idx"], out["traj_slot"]
            rows["pred"].append(out["traj"].cpu())
            if own_goal:  # the model's own goal decision, as in the simulator
                rows["pred_own"].append(model(*(b[k] for k in keys), **x)["traj"].cpu())
        rows["target"].append(b["traj"][idx, slot].cpu())
        rows["frac"].append(b["traj_frac"][idx, slot].cpu())
    return {k: torch.cat(v) for k, v in rows.items() if v}


def summarize(rows, mean_traj):
    target = rows["target"]
    length = (target[..., :2] / 4).norm(dim=-1).sum(1)  # remaining path (clipped at 3.2 m), metres
    preds = {"model (GT goal)": rows["pred"]}
    if "pred_own" in rows:
        preds["model (own goal)"] = rows["pred_own"]
    preds["mean trajectory"] = mean_traj.expand_as(target)
    preds["no motion"] = torch.zeros_like(target)
    errs = {name: pair_errors(p, target) for name, p in preds.items()}

    groups = [("all", torch.ones(len(target), dtype=torch.bool))]
    for lo, hi, name in POS_BINS:
        m = rows["frac"] == 0 if hi == 0 else (rows["frac"] > lo) & (rows["frac"] <= hi)
        groups.append((f"start: {name}", m))
    for lo, hi, name in LEN_BINS:
        groups.append((f"path: {name}", (length >= lo) & (length < hi)))

    summary = {}
    for gname, m in groups:
        if not m.any():
            continue
        summary[gname] = {"n": int(m.sum())}
        for name, (ade, fde, ang) in errs.items():
            a = ang[m]
            summary[gname][name] = dict(
                ade=float(ade[m].mean()),
                fde=float(fde[m].mean()),
                heading_deg=float(a[~a.isnan()].mean()) if (~a.isnan()).any() else None,
            )
    return summary


def print_summary(summary):
    names = [k for k in next(iter(summary.values())) if k != "n"]
    print(f"\n{'group':<28}{'n':>7}  " + "".join(f"{n:>30}" for n in names))
    print(f"{'':<28}{'':>7}  " + "".join(f"{'ade / fde (m) | head (deg)':>30}" for _ in names))
    for g, v in summary.items():
        cells = []
        for n in names:
            h = v[n]["heading_deg"]
            cells.append(
                f"{v[n]['ade']:.2f} / {v[n]['fde']:.2f} | {h:5.1f}"
                if h is not None
                else f"{v[n]['ade']:.2f} / {v[n]['fde']:.2f} |   -  "
            )
        print(f"{g:<28}{v['n']:>7}  " + "".join(f"{c:>30}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LayaNav checkpoint dir (train_laya_nav.py output)")
    ap.add_argument("--vln_dataset_use", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--val_split", default="scene", choices=["scene", "episode"])
    ap.add_argument("--goal_xy_order", default="xy", choices=["xy", "yx"])
    ap.add_argument("--max_samples", type=int, default=-1, help="fixed random subset of the validation goal samples")
    ap.add_argument("--own_goal", action="store_true", help="also plan from the model's own goal decision")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="write the summary as json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayaNav.from_pretrained(args.ckpt).to(device).eval()
    cfg = model.cfg
    tok = AutoTokenizer.from_pretrained(os.path.join(args.ckpt, "tokenizer"))

    samples = load_vln_samples(args.vln_dataset_use, pixel_goal_only=True)
    train_s, val_s = base.split_samples(samples, args.val_ratio, args.val_split, args.max_samples)
    if not val_s:
        raise SystemExit(f"no held-out goal samples in {args.vln_dataset_use} with --val_ratio {args.val_ratio}")
    mean_traj, n_mean = mean_trajectory(train_s, cfg.traj_steps)
    print(
        f"checkpoint {args.ckpt} | head: fuse_layers={cfg.traj_fuse_layers} goal_mark={cfg.traj_goal_mark}\n"
        f"val goal samples {len(val_s)} from scenes {sorted({base.scene_of(s) for s in val_s})}\n"
        f"mean trajectory from {n_mean} training pairs"
    )

    ds = StartsDataset(
        val_s,
        tok,
        cfg,
        goal_xy_order=args.goal_xy_order,
        with_traj=True,
        traj_steps=cfg.traj_steps,
        traj_starts=MAX_STARTS,
    )
    loader = DataLoader(
        ds, args.batch_size, num_workers=args.num_workers, collate_fn=partial(collate, pad_id=tok.pad_token_id)
    )
    rows = collect(model, loader, device, args.own_goal)
    summary = summarize(rows, mean_traj)
    print_summary(summary)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                dict(
                    ckpt=args.ckpt,
                    head=dict(fuse_layers=cfg.traj_fuse_layers, goal_mark=cfg.traj_goal_mark),
                    summary=summary,
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
