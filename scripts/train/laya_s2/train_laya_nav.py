"""Train LayaNav: Laya-S2 with a trajectory head replacing the DualVLN System 1.

C1 trains only the trajectory head on top of a frozen Laya-S2; C2 fine-tunes everything with the
decision and trajectory losses together. No 7B teacher is needed (teacher latents stay optional).

    # C1: from the Laya-S2 checkpoint, trajectory head only (pixel-goal samples)
    python scripts/train/laya_s2/train_laya_nav.py --stage c1 --init_from checkpoints/laya_s2/last \
        --vln_dataset_use r2r_125cm_0_30 --output_dir checkpoints/laya_nav_c1
    # C2: everything, all samples
    python scripts/train/laya_s2/train_laya_nav.py --stage c2 --init_from checkpoints/laya_nav_c1/last \
        --vln_dataset_use r2r_125cm_0_30 --output_dir checkpoints/laya_nav_c2
"""

import argparse
import importlib.util
import json
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
    LayaNav,
    LayaNavConfig,
    LossWeights,
    TrajLossWeights,
    compute_loss,
    traj_loss,
)
from internnav.model.basemodel.laya_s2.laya_s2 import WEIGHTS_NAME

# shared helpers of the Laya-S2 trainer (scene split, schedule, device moves)
_spec = importlib.util.spec_from_file_location("train_laya_s2", os.path.join(os.path.dirname(__file__), "train_laya_s2.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

METRIC_WEIGHT = dict(base.METRIC_WEIGHT, l_traj="n_traj", ade="n_traj", fde="n_traj")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["c1", "c2"])
    ap.add_argument("--init_from", default=None, help="Laya-S2 or LayaNav checkpoint dir; none = from scratch (c2)")
    ap.add_argument("--vln_dataset_use", required=True)
    ap.add_argument("--teacher_latents", default=None, help="optional in c2 (keeps the latent distillation term)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--traj_dim", type=int, default=LayaNavConfig.traj_dim)
    ap.add_argument("--traj_layers", type=int, default=LayaNavConfig.traj_layers)
    ap.add_argument("--goal_xy_order", default="xy", choices=["xy", "yx"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=32, help="per GPU")
    ap.add_argument("--lr_head", type=float, default=2e-4)
    ap.add_argument("--lr_text", type=float, default=3e-5)
    ap.add_argument("--lr_vision", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.02)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--val_split", default="scene", choices=["scene", "episode"])
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=2000)
    for k, v in {**asdict(LossWeights()), **asdict(TrajLossWeights())}.items():
        ap.add_argument(f"--w_{k}", type=float, default=v)
    args = ap.parse_args()
    if args.stage == "c1" and not args.init_from:
        ap.error("--stage c1 needs --init_from (a trained Laya-S2 or LayaNav checkpoint)")
    return args


def set_train_mode(model, stage):
    model.train()
    if stage == "c1":  # frozen decision model: no dropout, it only provides features
        for name, module in model.named_children():
            if not name.startswith("traj_"):
                module.eval()


def run(model, batch, weights, tweights, stage):
    m = model.module if isinstance(model, DDP) else model
    out = model(
        batch["input_ids"],
        batch["text_mask"],
        batch["hist_pixels"],
        batch["hist_mask"],
        batch["cur_pixels"],
        batch["down_pixels"],
        goal_xy=torch.where(batch["is_goal"][:, None], batch["goal_xy"], torch.full_like(batch["goal_xy"], 0.5)),
        traj_pixels=batch["traj_pixels"],
        traj_mask=batch["traj_mask"],
    )
    l_dec, stats = compute_loss(m, out, batch, weights)
    if "traj" in out:
        idx = out["traj_idx"]
        l_traj, tstats = traj_loss(out["traj"], batch["traj"][idx], torch.ones_like(idx, dtype=torch.bool), tweights)
    else:  # no pixel-goal sample in this batch: a zero loss that still reaches the trainable head (c1 / DDP)
        zero = m.traj_queries.sum() * 0
        l_traj, tstats = zero, dict(l_traj=zero.detach(), ade=zero.detach(), fde=zero.detach(), n_traj=zero.detach())
    stats.update(tstats)
    loss = l_traj if stage == "c1" else l_dec + l_traj
    stats["loss"] = loss.detach()
    return loss, stats


@torch.no_grad()
def evaluate(model, loader, weights, tweights, stage, device):
    model.eval()
    tot = {}
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, stats = run(model, base.to_device(batch, device), weights, tweights, stage)
        for k, v in stats.items():
            w = 1.0 if k.startswith("n") else float(stats[METRIC_WEIGHT.get(k, "n")])
            tot[k] = tot.get(k, 0.0) + float(v) * w
    set_train_mode(model.module if isinstance(model, DDP) else model, stage)
    keys = sorted(tot)
    t = torch.tensor([tot[k] for k in keys], device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t)
    tot = dict(zip(keys, t.tolist()))
    return {k: v / max(tot[METRIC_WEIGHT.get(k, "n")], 1) for k, v in tot.items() if not k.startswith("n")}


def build_model(args):
    traj_cfg = dict(traj_dim=args.traj_dim, traj_layers=args.traj_layers)
    if args.init_from:
        return LayaNav.load_any(args.init_from, **traj_cfg) if os.path.exists(args.init_from) else None
    ip = AutoImageProcessor.from_pretrained(LayaNavConfig.vision_encoder)
    return LayaNav.from_config(LayaNavConfig(image_mean=list(ip.image_mean), image_std=list(ip.image_std), **traj_cfg))


def main():
    args = parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    main_proc = rank == 0
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args)
    assert model is not None, f"--init_from {args.init_from} not found"
    model = model.to(device)
    cfg = model.cfg
    weights = LossWeights(**{k: getattr(args, f"w_{k}") for k in asdict(LossWeights())})
    tweights = TrajLossWeights(**{k: getattr(args, f"w_{k}") for k in asdict(TrajLossWeights())})
    tok_dir = os.path.join(args.init_from, "tokenizer") if args.init_from else cfg.text_encoder
    tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else cfg.text_encoder)

    # c1 only learns from pixel-goal samples (the ones with a trajectory)
    samples = load_vln_samples(args.vln_dataset_use, pixel_goal_only=args.stage == "c1")
    train_s = [s for s in samples if not base.is_val(s, args.val_ratio, args.val_split)]
    val_s = [s for s in samples if base.is_val(s, args.val_ratio, args.val_split)]
    if main_proc and args.val_split == "scene":
        print(f"val scenes: {sorted({base.scene_of(s) for s in val_s})}")
    kw = dict(goal_xy_order=args.goal_xy_order, with_traj=True, traj_steps=cfg.traj_steps)
    train_ds = LayaS2Dataset(train_s, tok, cfg, args.teacher_latents, augment=True, **kw)
    val_ds = LayaS2Dataset(val_s, tok, cfg, args.teacher_latents, augment=False, **kw)
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

    if args.stage == "c1":
        model.requires_grad_(False)
        for p in model.traj_parameters():
            p.requires_grad_(True)
    groups = {"text": [], "vision": [], "head": []}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups[n.split(".")[0] if n.split(".")[0] in ("text", "vision") else "head"].append(p)
    lrs = dict(text=args.lr_text, vision=args.lr_vision, head=args.lr_head)
    opt = torch.optim.AdamW(
        [dict(params=v, lr=lrs[k]) for k, v in groups.items() if v], weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    total = args.max_steps if args.max_steps > 0 else args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, partial(base.lr_lambda, total=total, warmup=int(args.warmup_ratio * total))
    )

    step, epoch = 0, 0
    last = os.path.join(args.output_dir, "last")
    if os.path.exists(os.path.join(last, "train_state.pt")):
        model.load_weights(os.path.join(last, WEIGHTS_NAME), map_location=device)
        st = torch.load(os.path.join(last, "train_state.pt"), map_location=device)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, epoch = st["step"], st["epoch"]
        if main_proc:
            print(f"resumed from step {step}")

    if main_proc:
        n_all = sum(p.numel() for p in model.parameters())
        n_traj = sum(p.numel() for p in model.traj_parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"stage {args.stage} | params {n_all / 1e6:.1f}M (trajectory head {n_traj / 1e6:.1f}M, trainable "
            f"{n_train / 1e6:.1f}M) | train {len(train_ds)} val {len(val_ds)} | steps {total}"
        )
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    set_train_mode(model, args.stage)
    ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True) if distributed else model

    def save(tag):
        if not main_proc:
            return
        d = os.path.join(args.output_dir, tag)
        model.save_pretrained(d)
        tok.save_pretrained(os.path.join(d, "tokenizer"))
        torch.save(
            dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step, epoch=epoch),
            os.path.join(d, "train_state.pt"),
        )

    t0 = time.time()
    while step < total:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_dl:
            batch = base.to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = run(ddp, batch, weights, tweights, args.stage)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if main_proc and step % args.log_every == 0:
                msg = " ".join(f"{k}={float(v):.4f}" for k, v in stats.items() if not k.startswith("n"))
                print(f"[ep {epoch} step {step}/{total} {time.time() - t0:.0f}s lr={sched.get_last_lr()[-1]:.2e}] {msg}", flush=True)
            if step % args.save_every == 0 or step >= total:
                val = evaluate(ddp, val_dl, weights, tweights, args.stage, device)
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
