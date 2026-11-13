"""What a trained student actually uses: offline ablations on the validation samples (and a train-scene sample).

Each variant changes one input and runs the same samples again:
    normal       as in training's validation
    black        every frame black (images carry nothing)
    black_shuf_text  black frames and another sample's instruction (is the text read when there is no image?)
    cur_only     only the current frame (no history)
    no_cur       history only, the current frame black
    shuf_img     frames of another sample (images from a different place)
    shuf_text    instruction of another sample
    mirror_text  left / right swapped in the instruction only
    mirror_img   frames flipped left-right only
    mirror_both  both: the mirrored scene; the answer should mirror too (label -> MIRROR[label])
If a variant barely changes accuracy or the predictions, the model is not using that input.
Also: accuracy on train-scene samples (the generalization gap), the stop threshold and a prior correction
for the copies in NaVILA's annotations (stop x3, 30 / 45 degree turns x2, RxR stop x5).

The validation samples are chosen exactly as scripts/train.py chooses them (same split file, same val_max).

    # this PC (local R2R sample: 301 validation, 5431 train-scene samples)
    env/bin/python scripts/diagnose.py --ckpt runs/N1/last runs/N2/last --only_available \
        --r2r_frames data/NaVILA-Dataset/R2R/train_sample --out runs/diag/local.json
    # server: see scripts/server/diagnose.sh
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lightnav import data as D                                   # noqa: E402
from lightnav.metrics import summarize                           # noqa: E402
from lightnav.questions import KEYS, MIRROR, NUM_ACTIONS, STOP, decide, kind   # noqa: E402

VARIANTS = ("normal", "black", "black_shuf_text", "cur_only", "no_cur", "shuf_img", "shuf_text", "mirror_text", "mirror_img",
            "mirror_both")
_KIND = np.array([kind(i) for i in range(NUM_ACTIONS)])
_MIRROR = np.array(MIRROR)


def _train_module():
    spec = importlib.util.spec_from_file_location("lightnav_train", os.path.join(ROOT, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.path.join(ROOT, "data", "NaVILA-Dataset")
    p.add_argument("--ckpt", nargs="+", required=True, help="checkpoint folders (runs/N1/last ...)")
    p.add_argument("--out", required=True, help="JSON with every number; a table is printed too")
    p.add_argument("--r2r_ann", default=os.path.join(data, "R2R", "annotations.json"))
    p.add_argument("--r2r_frames", default=os.path.join(data, "R2R", "train"))
    p.add_argument("--rxr_ann", default=None)
    p.add_argument("--rxr_frames", default=os.path.join(data, "RxR", "train"))
    p.add_argument("--only_available", action="store_true")
    p.add_argument("--split", default=os.path.join(ROOT, "splits", "val_scenes.json"))
    p.add_argument("--val_max", type=int, default=5000, help="as in training, so the samples are the same")
    p.add_argument("--train_max", type=int, default=2000, help="train-scene samples for the gap; 0 skips")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class VariantSet(Dataset):
    """`base` (a NavDataset) with one input changed. `other[i]` is the sample whose images / text a shuffle uses."""

    def __init__(self, base: D.NavDataset, variant: str, other: np.ndarray):
        self.base, self.variant, self.other = base, variant, other

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        v, item = self.variant, self.base[i]
        row = self.base.rows[i]
        item["valid"] = True
        if v in ("black", "black_shuf_text"):
            item["pixels"] = torch.full_like(item["pixels"], -1.0)      # 0 after Normalize(0.5, 0.5)
        if v == "black_shuf_text":
            item["ids"], item["markers"] = self.base.encode_text(self.base.rows[int(self.other[i])].instruction)
        elif v == "cur_only":
            item["pixels"], item["slots"] = item["pixels"][-1:], item["slots"][-1:]
        elif v == "no_cur":
            item["pixels"] = item["pixels"].clone()
            item["pixels"][-1] = -1.0
        elif v == "shuf_img":
            o = self.base[int(self.other[i])]
            item["pixels"], item["slots"] = o["pixels"], o["slots"]
        elif v == "shuf_text":
            item["ids"], item["markers"] = self.base.encode_text(self.base.rows[int(self.other[i])].instruction)
        elif v in ("mirror_text", "mirror_both"):
            m = D.mirror_instruction(row.instruction)
            # mirror_text counts only instructions that name a direction; mirror_both every one that can be mirrored
            item["valid"] = m is not None and (v == "mirror_both" or m != row.instruction)
            if m is not None:
                item["ids"], item["markers"] = self.base.encode_text(m)
        if v in ("mirror_img", "mirror_both"):
            item["pixels"] = torch.flip(item["pixels"], dims=[-1])
        return item


def collate(batch, pad_id):
    out = D.collate(batch, pad_id)
    out["valid"] = torch.tensor([b["valid"] for b in batch])
    return out


@torch.no_grad()
def run(model, dataset, device, batch, workers, pad_id):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=lambda b: collate(b, pad_id))
    probs, valid = [], []
    for b in loader:
        b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(b)
        probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
        valid.append(b["valid"].cpu().numpy())
    return np.concatenate(probs), np.concatenate(valid)


def kl(p, q):
    p, q = np.clip(p, 1e-9, 1), np.clip(q, 1e-9, 1)
    return float((p * (np.log(p) - np.log(q))).sum(-1).mean())


def turn_split(labels, pred):
    """Among samples labelled left / right: how often the model said left, right, forward, stop."""
    out = {}
    for name, k in (("left", 2), ("right", 3)):
        m = _KIND[labels] == k
        if m.any():
            c = Counter(_KIND[pred[m]].tolist())
            out[name] = {"n": int(m.sum()), **{n: round(c.get(i, 0) / m.sum(), 3)
                                              for i, n in enumerate(("stop", "forward", "left", "right"))}}
    return out


def by_source(rows, labels, probs):
    out = {}
    for s in sorted({r.source for r in rows}):
        m = np.array([r.source == s for r in rows])
        out[s] = {k: v for k, v in summarize(labels[m], probs[m]).items() if k in ("n", "acc", "kind_acc")}
    return out


def by_progress(rows, labels, probs):
    """Accuracy by how far into the episode the decision is (the first decision has a single frame)."""
    out = {}
    steps = np.array([r.n_frames for r in rows])
    for name, lo, hi in (("first", 1, 1), ("2-7", 2, 7), ("8-30", 8, 30), ("31+", 31, 10 ** 9)):
        m = (steps >= lo) & (steps <= hi)
        if m.any():
            out[name] = {k: v for k, v in summarize(labels[m], probs[m]).items() if k in ("n", "acc", "kind_acc")}
    return out


def copy_weights(rows):
    """Per answer, how many times NaVILA's annotations repeat a sample (train rows with copies / unique)."""
    with_copies = np.bincount([r.label for r in rows], minlength=NUM_ACTIONS).astype(np.float64)
    unique = np.bincount([r.label for r in {r.key: r for r in rows}.values()], minlength=NUM_ACTIONS)
    return with_copies / np.maximum(unique, 1), with_copies / with_copies.sum(), unique / max(unique.sum(), 1)


def stop_report(labels, probs, weights):
    out = {}
    for tau in (None, 0.3, 0.5, 0.7, 0.9):
        pred = np.array([decide(p, tau) for p in probs])
        said, is_stop = pred == STOP, labels == STOP
        out["tau=%s" % tau] = {
            "acc": float((pred == labels).mean()), "kind_acc": float((_KIND[pred] == _KIND[labels]).mean()),
            "said_stop": float(said.mean()),
            "stop_precision": float((said & is_stop).sum() / said.sum()) if said.any() else None,
            "stop_recall": float((said & is_stop).sum() / is_stop.sum()) if is_stop.any() else None,
        }
    adj = probs / weights[None]
    adj /= adj.sum(-1, keepdims=True)
    s = summarize(labels, adj)
    out["prior_corrected"] = {k: s[k] for k in ("acc", "kind_acc", "stop_precision", "stop_recall")}
    out["prior_corrected"]["said_stop"] = float((adj.argmax(-1) == STOP).mean())
    return out


def baselines(labels, train_prior):
    out = {}
    for name, a in (("always_forward75", 3), ("train_majority", int(np.argmax(train_prior)))):
        pred = np.full_like(labels, a)
        out[name] = {"acc": float((pred == labels).mean()), "kind_acc": float((_KIND[pred] == _KIND[labels]).mean())}
    return out


def diagnose(ckpt, args, val_rows, train_sample, roots, weights, device):
    from lightnav.model import load

    model, tok = load(ckpt)
    model.to(device).eval()
    cfg = model.cfg
    base = D.NavDataset(val_rows, roots, tok, cfg)
    labels = np.array([r.label for r in val_rows])
    rng = np.random.default_rng(args.seed)
    other = rng.permutation(len(val_rows))
    same = other == np.arange(len(val_rows))
    other[same] = (other[same] + 1) % len(val_rows)
    res = {"ckpt": ckpt, "variants": {}}
    normal = black = None
    for v in args.variants:
        t = time.time()
        probs, valid = run(model, VariantSet(base, v, other), device, args.batch, args.workers, tok.pad_token_id)
        lab = _MIRROR[labels] if v == "mirror_both" else labels
        m = valid.astype(bool)
        s = summarize(lab[m], probs[m])
        pred = probs.argmax(-1)
        rec = {k: s[k] for k in ("n", "acc", "kind_acc", "stop_precision", "stop_recall", "ece")}
        rec["recall"] = s["recall"]
        rec["pred_share"] = {KEYS[i]: round(float((pred[m] == i).mean()), 3) for i in range(NUM_ACTIONS)}
        rec["turns"] = turn_split(lab[m], pred[m])
        if v == "black":
            black = probs
        if v == "normal":
            normal = probs
            rec["by_source"] = by_source(val_rows, labels, probs)
            rec["by_progress"] = by_progress(val_rows, labels, probs)
            rec["stop"] = stop_report(labels, probs, weights)
        elif normal is not None:
            # black_shuf_text is compared with black: the text effect when no image is there
            ref = normal[m][:, _MIRROR] if v.startswith("mirror") else normal[m]
            if v == "black_shuf_text" and "black" in res["variants"]:
                ref = black[m]
            # mirror_text / mirror_img: a model that uses that input for direction changes its answer;
            # mirror_both: a consistent model gives the mirror of its normal answer.
            plain = v not in ("mirror_both", "black_shuf_text")
            rec["same_as_normal"] = float((pred[m] == (normal[m] if plain else ref).argmax(-1)).mean())
            rec["kl_from_normal"] = kl(normal[m] if plain else ref, probs[m])
        rec["sec"] = round(time.time() - t, 1)
        res["variants"][v] = rec
        print("  %-12s n=%-5d acc %.3f kind %.3f %s(%.0fs)" % (
            v, rec["n"], rec["acc"], rec["kind_acc"],
            "same %.2f kl %.3f " % (rec["same_as_normal"], rec["kl_from_normal"]) if "same_as_normal" in rec else "",
            rec["sec"]), flush=True)
    if train_sample:
        probs, _ = run(model, VariantSet(D.NavDataset(train_sample, roots, tok, cfg), "normal", np.zeros(0)), device,
                       args.batch, args.workers, tok.pad_token_id)
        tl = np.array([r.label for r in train_sample])
        s = summarize(tl, probs)
        res["train_scenes"] = {k: s[k] for k in ("n", "acc", "kind_acc", "stop_precision", "stop_recall", "ece")}
        res["train_scenes"]["recall"] = s["recall"]
        print("  %-12s n=%-5d acc %.3f kind %.3f" % ("train_scenes", s["n"], s["acc"], s["kind_acc"]), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targs = argparse.Namespace(r2r_ann=args.r2r_ann, r2r_frames=args.r2r_frames, rxr_ann=args.rxr_ann,
                               rxr_frames=args.rxr_frames, only_available=args.only_available, split=args.split,
                               out=None, r2r_scenes=None, rxr_scenes=None, val_frac=None, seed=0)
    if not os.path.exists(args.split):
        raise SystemExit("no split file %s" % args.split)
    train_rows, val_rows, roots = _train_module().load_data(targs)
    # same choice as scripts/train.py
    val_rows = sorted(val_rows, key=lambda r: r.key)
    if len(val_rows) > args.val_max:
        val_rows = [val_rows[i] for i in sorted(np.random.default_rng(0).choice(len(val_rows), args.val_max,
                                                                                replace=False))]
    unique_train = sorted({r.key: r for r in train_rows}.values(), key=lambda r: r.key)
    train_sample = []
    if args.train_max > 0:
        idx = np.random.default_rng(1).choice(len(unique_train), min(args.train_max, len(unique_train)), replace=False)
        train_sample = [unique_train[i] for i in sorted(idx)]
    weights, train_prior, train_unique_prior = copy_weights(train_rows)
    labels = np.array([r.label for r in val_rows])
    val_prior = np.bincount(labels, minlength=NUM_ACTIONS) / len(labels)
    out = {
        "val_samples": len(val_rows), "train_sample": len(train_sample),
        "prior": {"train_with_copies": dict(zip(KEYS, np.round(train_prior, 4).tolist())),
                  "train_unique": dict(zip(KEYS, np.round(train_unique_prior, 4).tolist())),
                  "val": dict(zip(KEYS, np.round(val_prior, 4).tolist())),
                  "copy_weight": dict(zip(KEYS, np.round(weights, 3).tolist()))},
        "baselines": baselines(labels, train_prior),
        "runs": [],
    }
    print("val %d samples, train-scene sample %d" % (len(val_rows), len(train_sample)), flush=True)
    print("baselines", json.dumps(out["baselines"]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ckpt in args.ckpt:
        print(ckpt, flush=True)
        out["runs"].append(diagnose(ckpt, args, val_rows, train_sample, roots, weights, device))
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
