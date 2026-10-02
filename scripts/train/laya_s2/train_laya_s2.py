"""Train the Laya-style lightweight System 2.

    torchrun --nproc_per_node=8 scripts/train/laya_s2/train_laya_s2.py \
        --vln_dataset_use r2r_125cm_0_30,rxr_125cm_0_30 \
        --teacher_latents data/laya_s2/teacher_latents --output_dir checkpoints/laya_s2
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoImageProcessor, AutoTokenizer

from internnav.dataset.laya_s2_dataset import (
    LayaS2Dataset,
    collate_laya_s2,
    load_vln_samples,
)
from internnav.model.basemodel.laya_s2 import (
    LayaS2,
    LayaS2Config,
    LossWeights,
    compute_loss,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vln_dataset_use", required=True)
    ap.add_argument("--teacher_latents", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--text_encoder", default=LayaS2Config.text_encoder)
    ap.add_argument("--vision_encoder", default=LayaS2Config.vision_encoder)
    ap.add_argument("--image_size", type=int, default=LayaS2Config.image_size)
    ap.add_argument("--head_layers", type=int, default=LayaS2Config.head_layers)
    ap.add_argument("--latent_layers", type=int, default=LayaS2Config.latent_layers)
    ap.add_argument("--goal_xy_order", default="xy", choices=["xy", "yx"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=32, help="per GPU")
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--lr_text", type=float, default=3e-5)
    ap.add_argument("--lr_vision", type=float, default=1e-5)
    ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.01, help="held-out fraction, split by episode")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=2000)
    for k, v in asdict(LossWeights()).items():
        ap.add_argument(f"--w_{k}", type=float, default=v)
    return ap.parse_args()


def is_val(s, ratio):
    h = hashlib.md5(f"{s['video']}|{s['ep_id']}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF < ratio


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def run(model, batch, weights):
    out = model(
        batch["input_ids"],
        batch["text_mask"],
        batch["hist_pixels"],
        batch["hist_mask"],
        batch["cur_pixels"],
        batch["down_pixels"],
        goal_xy=torch.where(batch["is_goal"][:, None], batch["goal_xy"], torch.full_like(batch["goal_xy"], 0.5)),
    )
    return compute_loss(model.module if isinstance(model, DDP) else model, out, batch, weights)


@torch.no_grad()
def evaluate(model, loader, weights, device):
    model.eval()
    tot, n = {}, 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, stats = run(model, to_device(batch, device), weights)
        for k, v in stats.items():
            tot[k] = tot.get(k, 0.0) + float(v)
        n += 1
    model.train()
    t = torch.tensor([tot.get(k, 0.0) for k in sorted(tot)] + [n], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t)
    return {k: (t[i] / max(t[-1], 1)).item() for i, k in enumerate(sorted(tot))}


def lr_lambda(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    main_proc = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)

    ip = AutoImageProcessor.from_pretrained(args.vision_encoder)
    cfg = LayaS2Config(
        text_encoder=args.text_encoder,
        vision_encoder=args.vision_encoder,
        image_size=args.image_size,
        image_mean=list(ip.image_mean),
        image_std=list(ip.image_std),
        head_layers=args.head_layers,
        latent_layers=args.latent_layers,
    )
    weights = LossWeights(**{k: getattr(args, f"w_{k}") for k in asdict(LossWeights())})
    tok = AutoTokenizer.from_pretrained(args.text_encoder)

    samples = load_vln_samples(args.vln_dataset_use)
    train_s = [s for s in samples if not is_val(s, args.val_ratio)]
    val_s = [s for s in samples if is_val(s, args.val_ratio)]
    train_ds = LayaS2Dataset(train_s, tok, cfg, args.teacher_latents, augment=True, goal_xy_order=args.goal_xy_order)
    val_ds = LayaS2Dataset(val_s, tok, cfg, args.teacher_latents, augment=False, goal_xy_order=args.goal_xy_order)
    collate = partial(collate_laya_s2, pad_id=tok.pad_token_id)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_dl = DataLoader(
        train_ds,
        args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_dl = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=args.num_workers, collate_fn=collate)

    model = LayaS2.from_config(cfg).to(device)
    if args.freeze_vision:
        model.vision.requires_grad_(False)
    groups = {"text": [], "vision": [], "head": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups[n.split(".")[0] if n.split(".")[0] in ("text", "vision") else "head"].append(p)
    lrs = dict(text=args.lr_text, vision=args.lr_vision, head=args.lr_head)
    opt = torch.optim.AdamW(
        [dict(params=v, lr=lrs[k]) for k, v in groups.items() if v], weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    total = args.max_steps if args.max_steps > 0 else args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_lambda, total=total, warmup=int(args.warmup_ratio * total)))

    step, epoch = 0, 0
    last = os.path.join(args.output_dir, "last")
    if os.path.exists(os.path.join(last, "train_state.pt")):
        model.load_state_dict(torch.load(os.path.join(last, "laya_s2.pt"), map_location=device))
        st = torch.load(os.path.join(last, "train_state.pt"), map_location=device)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, epoch = st["step"], st["epoch"]
        if main_proc:
            print(f"resumed from step {step}")

    if main_proc:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"params: {n_par / 1e6:.1f}M | train {len(train_ds)} val {len(val_ds)} | steps {total}")
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True) if distributed else model

    def save(tag):
        if not main_proc:
            return
        d = os.path.join(args.output_dir, tag)
        model.save_pretrained(d)
        tok.save_pretrained(os.path.join(d, "tokenizer"))
        torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step, epoch=epoch), os.path.join(d, "train_state.pt"))

    t0 = time.time()
    while step < total:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_dl:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = run(ddp, batch, weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if main_proc and step % args.log_every == 0:
                msg = " ".join(f"{k}={float(v):.4f}" for k, v in stats.items())
                print(f"[ep {epoch} step {step}/{total} {time.time() - t0:.0f}s lr={sched.get_last_lr()[-1]:.2e}] {msg}", flush=True)
            if step % args.save_every == 0 or step >= total:
                val = evaluate(ddp, val_dl, weights, device)
                if main_proc:
                    print(f"[val step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
                save("last")
                save(f"step_{step}")
            if step >= total:
                break
        epoch += 1

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
