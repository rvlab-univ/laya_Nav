"""Train the Laya-native student on NaVILA annotations (one GPU). Re-running with the same --out resumes.

Example (this PC, the local R2R sample):
    env/bin/python scripts/train.py --out runs/smoke --r2r_frames data/NaVILA-Dataset/R2R/train_sample \
        --only_available --batch 2 --grad_accum 4 --max_steps 200
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lightnav import data as D                                   # noqa: E402
from lightnav.metrics import summarize                           # noqa: E402
from lightnav.model import LayaNativeConfig, build, save         # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.path.join(ROOT, "data", "NaVILA-Dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--r2r_ann", default=os.path.join(data, "R2R", "annotations.json"))
    p.add_argument("--r2r_frames", default=os.path.join(data, "R2R", "train"))
    p.add_argument("--rxr_ann", default=None, help="RxR annotations.json; left out when not given")
    p.add_argument("--rxr_frames", default=os.path.join(data, "RxR", "train"))
    p.add_argument("--only_available", action="store_true", help="keep only episodes whose frame folder exists")
    p.add_argument("--split", default=None, help="validation split file; made at <out>/split.json when missing")
    p.add_argument("--r2r_scenes", default=None, help="VLN-CE train.json.gz, for a scene-level split of R2R")
    p.add_argument("--rxr_scenes", default=None, help="RxR rxr_train_guide.jsonl.gz, for a scene-level split of RxR")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--laya", default="english", help="Laya checkpoint to start from; 'none' to use --encoder")
    p.add_argument("--encoder", default="answerdotai/ModernBERT-large")
    p.add_argument("--mirror", type=float, default=0.0, help="probability of a left-right mirrored sample")
    p.add_argument("--teacher", default=None, help="folder with keys.txt and teacher_probs.npy")
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the teacher distribution in the target")
    p.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=None, help="optimizer steps; overrides --epochs")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--encoder_lr", type=float, default=2.5e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--vision_lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--sigma", type=float, nargs=2, default=(0.4, 0.1), help="rlcd noise at the start and the end")
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val_max", type=int, default=5000, help="validation samples per evaluation")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_data(args):
    """`(train rows with NaVILA's copies, unique validation rows, image roots)`."""
    sources = [("r2r", args.r2r_ann, args.r2r_frames, args.r2r_scenes)]
    if args.rxr_ann:
        sources.append(("rxr", args.rxr_ann, args.rxr_frames, args.rxr_scenes))
    rows, roots, scenes = [], {}, {}
    for name, ann, frames, scene_file in sources:
        part = D.load_rows(ann, name)
        if args.only_available:
            # an episode counts only when every frame its samples can ask for is on disk
            need = {}
            for r in part:
                need[r.prefix] = max(need.get(r.prefix, 0), r.n_frames)
            have = {p for p, n in need.items() if os.path.isdir(os.path.join(frames, p))
                    and {"frame_%d.jpg" % i for i in range(n)} <= set(os.listdir(os.path.join(frames, p)))}
            part = [r for r in part if r.prefix in have]
        rows += part
        roots[name] = frames
        if not scene_file:
            scenes[name] = None
        elif name == "rxr":
            scenes[name] = D.rxr_episode_scenes(scene_file, part)
        else:
            scenes[name] = D.episode_scenes(scene_file)
        print("%s: %d rows, %d episodes" % (name, len(part), len({r.episode for r in part})), flush=True)
    split = args.split or os.path.join(args.out, "split.json")
    if not os.path.exists(split):
        episodes = {name: [r.episode for r in rows if r.source == name] for name in roots}
        val, held = D.make_val_split(episodes, scenes, args.val_frac, args.seed)
        mode = {name: "scene" if scenes[name] else "episode" for name in roots}
        D.save_split(split, val, json.dumps(mode), held)
        print("wrote %s (%s, held-out scenes: %s)" % (split, mode, sorted(held) if held else "-"), flush=True)
    train, held = D.split_rows(rows, D.load_split(split))
    unique = list({r.key: r for r in held}.values())
    return train, unique, roots


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, probs = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(to_device(batch, device))
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        labels.append(batch["label"].numpy())
    model.train()
    return summarize(np.concatenate(labels), np.concatenate(probs))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_state(args, model, tok, opt, step):
    tmp, last = os.path.join(args.out, "last.tmp"), os.path.join(args.out, "last")
    shutil.rmtree(tmp, ignore_errors=True)
    save(model, tok, tmp)
    torch.save({"step": step, "optimizer": opt.state_dict()}, os.path.join(tmp, "train_state.pt"))
    shutil.rmtree(last, ignore_errors=True)
    os.replace(tmp, last)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = LayaNativeConfig(laya=None if args.laya == "none" else args.laya, encoder=args.encoder)
    model, tok = build(cfg)
    model.set_stage("nav")
    if args.grad_checkpointing:
        model.dm.encoder.gradient_checkpointing_enable()
        model.dm.head_checkpointing = True
    model.to(device).train()
    print("parameters: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

    train_rows, val_rows, roots = load_data(args)
    teacher = D.TeacherStore(args.teacher) if args.teacher else None
    train_set = D.NavDataset(train_rows, roots, tok, cfg, augment=True, mirror_prob=args.mirror,
                             teacher=teacher, alpha=args.alpha)
    # The same validation samples at every evaluation and in every run that shares the split file.
    val_rows = sorted(val_rows, key=lambda r: r.key)
    if len(val_rows) > args.val_max:
        val_rows = [val_rows[i] for i in sorted(np.random.default_rng(0).choice(len(val_rows), args.val_max, replace=False))]
    val_set = D.NavDataset(val_rows, roots, tok, cfg)
    collate = lambda b: D.collate(b, tok.pad_token_id)                                   # noqa: E731
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch * 2, shuffle=False, collate_fn=collate,
                            num_workers=min(args.workers, 4))
    steps_per_epoch = len(train_loader) // args.grad_accum
    total = args.max_steps or max(1, int(steps_per_epoch * args.epochs))
    print("train %d rows, val %d unique samples, %d optimizer steps (%d per epoch)"
          % (len(train_rows), len(val_rows), total, steps_per_epoch), flush=True)
    with open(os.path.join(args.out, "run.json"), "w") as f:   # read by scripts/server/status.sh
        json.dump({"total_steps": total, "steps_per_epoch": steps_per_epoch, "train_rows": len(train_rows),
                   "val_samples": len(val_rows), "args": vars(args)}, f, indent=2)

    opt = torch.optim.AdamW(model.param_groups(args.encoder_lr, args.head_lr, args.vision_lr),
                            weight_decay=args.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    step = 0
    state_file = os.path.join(args.out, "last", "train_state.pt")
    if os.path.exists(state_file):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(os.path.join(args.out, "last", "model.safetensors")), strict=True)
        state = torch.load(state_file, map_location="cpu")
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        print("resumed from step %d" % step, flush=True)

    from laya.train import rlcd_loss, soft_ce_loss
    mask = torch.ones(args.batch, 10, dtype=torch.bool, device=device)
    qtype = torch.zeros(args.batch, dtype=torch.long, device=device)
    params = [p for g in opt.param_groups for p in g["params"]]
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    running, seen, started = 0.0, 0, time.time()
    opt.zero_grad(set_to_none=True)
    while step < total:
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(batch)
            if args.loss == "rlcd":
                frac = step / max(1, total - 1)
                loss = rlcd_loss(logits, batch["target"], mask, qtype, args.sigma[0] + (args.sigma[1] - args.sigma[0]) * frac)
            else:
                loss = soft_ce_loss(logits, batch["target"], mask)
            (loss / args.grad_accum).backward()
            running += float(-(batch["target"] * torch.log_softmax(logits.detach(), -1)).sum(-1).mean())
            seen += 1
            if (i + 1) % args.grad_accum:
                continue
            scale = min(1.0, (step + 1) / max(1, args.warmup)) * \
                (0.5 * (1 + math.cos(math.pi * min(1.0, step / total))) * 0.96 + 0.04)
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                rate = (time.time() - started) / (seen / args.grad_accum)
                rec = {"step": step, "ce": running / seen, "lr_scale": scale, "sec_per_step": rate}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running, seen, started = 0.0, 0, time.time()
            if val_rows and (step % args.eval_every == 0 or step == total):
                rec = {"step": step, "val": evaluate(model, val_loader, device)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                started = time.time()
            if step % args.save_every == 0 or step == total:
                save_state(args, model, tok, opt, step)
            if step >= total:
                break
    print("done at step %d" % step, flush=True)


if __name__ == "__main__":
    main()
