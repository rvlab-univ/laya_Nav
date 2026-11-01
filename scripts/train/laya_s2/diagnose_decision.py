"""Diagnose the Laya-S2 / LayaNav decision (System 2) with forward passes only: no training, no gradients.

For one or more checkpoints, on the same held-out samples (each STOP sample counted once):
  - accuracy per dataset (r2r / rxr / scalevln) x sample kind (goal / turn / stop),
  - share of samples every checkpoint gets right, none gets right, or only some get right,
  - STOP: confusion of labels and choices, precision / recall, how often STOP is chosen vs labelled,
  - calibration: expected calibration error (ECE) of the top-1 probability, and how well the escalate head flags
    wrong decisions (AUROC),
  - the validation accuracy of the old runs, which counted every STOP sample 5 times (``acc_action legacy``),
  - shortcut checks (v5): accuracy where the previous action gives the answer away (keep) and where it does not
    (change); the same with the pose / motion inputs zeroed (memory_off, models with v4 memory); how often the
    decision changes when only left / right are swapped in the instruction (swap_directions); accuracy on the
    left-right mirrored samples and agreement with the mirrored decision (mirror).
Without the model: label counts per dataset and how many instructions the tokenizer cuts at each length.

    python scripts/train/laya_s2/diagnose_decision.py \
        --vln_dataset_use r2r_125cm_0_30,rxr_125cm_0_30,scalevln_125cm_0_30 \
        --ckpts checkpoints/laya_nav_mix_c2/step_28000 checkpoints/laya_nav_mix_c2/step_96318 \
                checkpoints/laya_nav_v3_c2/step_48000 checkpoints/laya_nav_v3_c2/step_64000 \
        --out_dir logs/diagnosis/decision
    # labels and instruction lengths only (CPU, seconds):  ... --skip_model

Writes ``report.md`` (tables), ``diagnosis.json`` and ``per_sample.csv`` (one row per sample, for sample weights).
"""

import argparse
import contextlib
import csv
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
    CHANGE,
    GOAL,
    KEEP,
    MIRROR_ACTION,
    STOP,
    TURN,
    LayaS2Dataset,
    change_flag,
    collate_laya_s2,
    load_vln_samples,
    mirror_instruction,
)
from internnav.model.basemodel.laya_s2 import LayaNav, LayaS2, LossWeights, decision_outcome
from internnav.model.basemodel.laya_s2.laya_nav import _is_nav
from internnav.model.basemodel.laya_s2.laya_s2 import CONFIG_NAME

_spec = importlib.util.spec_from_file_location("train_laya_s2", os.path.join(os.path.dirname(__file__), "train_laya_s2.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

KIND = {GOAL: "goal", TURN: "turn", STOP: "stop"}
ACTION_NAME = {0: "STOP", 2: "LEFT", 3: "RIGHT"}
CLASSES = ("STOP", "LEFT", "RIGHT", "GOAL")


# ---------------------------------------------------------------------- without the model
def label_stats(samples, stop_repeat=5):
    """Unique samples per dataset and kind, and the share each kind has once STOP is repeated for training."""
    out = {}
    for name in sorted({base.dataset_of(s) for s in samples}) + ["all"]:
        sub = [s for s in samples if name == "all" or base.dataset_of(s) == name]
        n = {k: sum(s["kind"] == c for s in sub) for c, k in KIND.items()}
        stream = n["goal"] + n["turn"] + stop_repeat * n["stop"]
        flags = [change_flag(s) for s in sub]
        out[name] = dict(
            unique=n,
            change=sum(f == CHANGE for f in flags),
            keep=sum(f == KEEP for f in flags),
            stop_unique=n["stop"] / max(sum(n.values()), 1),
            stop_in_training=stop_repeat * n["stop"] / max(stream, 1),
        )
    return out


def truncation_stats(samples, tok, lens):
    """Instruction length in tokens per dataset (unique instructions), and the share cut at each length."""
    out = {}
    for name in sorted({base.dataset_of(s) for s in samples}) + ["all"]:
        texts = sorted({s["instruction"] for s in samples if name == "all" or base.dataset_of(s) == name})
        n_tok = np.array([len(tok(t, truncation=True, max_length=100000)["input_ids"]) for t in texts])
        if not len(n_tok):
            continue
        cut = {}
        for L in lens:
            over = n_tok > L
            cut[L] = dict(
                truncated=float(over.mean()),
                lost_tokens_when_cut=float((n_tok[over] - L).mean()) if over.any() else 0.0,
                lost_share_when_cut=float(((n_tok[over] - L) / n_tok[over]).mean()) if over.any() else 0.0,
            )
        out[name] = dict(
            instructions=len(texts),
            tokens_p50=float(np.percentile(n_tok, 50)),
            tokens_p90=float(np.percentile(n_tok, 90)),
            tokens_p99=float(np.percentile(n_tok, 99)),
            tokens_max=int(n_tok.max()),
            cut=cut,
        )
    return out


# ---------------------------------------------------------------------- forward passes
def load_model(ckpt):
    return (LayaNav if _is_nav(ckpt) else LayaS2).from_pretrained(ckpt)


@torch.no_grad()
def profile(model, samples, tok, batch_size=64, num_workers=8, device=torch.device("cpu"), correct_radius=1.5,
            ds_kw=None):
    """Per-sample top-1 choice, correctness, its probability and the escalate probability (in ``ds.samples`` order).
    ``ds_kw``: dataset options of the shortcut checks (memory_dropout / swap_directions / mirror = 1.0)."""
    ds = LayaS2Dataset(samples, tok, model.cfg, augment=False, **(ds_kw or {}))
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=partial(collate_laya_s2, pad_id=tok.pad_token_id))
    model = model.eval().to(device)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    rec = {k: [] for k in ("pred", "correct", "conf", "esc", "top")}
    A = model.n_action
    for batch in dl:
        batch = base.to_device(batch, device)
        with amp:
            out, _, _ = base.decision_forward(model, batch, LossWeights())  # no matching copies
        pred, ok = decision_outcome(model, out, batch, correct_radius)
        p = torch.softmax(out["logits"].float(), -1)
        rec["pred"] += [model.cfg.actions[k] if k < A else -1 for k in pred.tolist()]  # habitat id, -1 = goal
        rec["correct"] += ok.tolist()
        rec["conf"] += p.max(-1).values.tolist()
        rec["esc"] += torch.softmax(out["act_logits"].float(), -1)[:, 1].tolist()
        rec["top"] += pred.tolist()  # index into actions + patches
    return ds.samples, {k: np.asarray(v) for k, v in rec.items()}


CHECKS = ("memory_off", "swap_directions", "mirror")


def run_checks(model, samples, tok, checks, **kw):
    """Shortcut checks on ``samples`` (the ones ``profile`` kept): {check: records + "idx" (the samples it ran on)}."""
    out = {}
    cfg = model.cfg
    plan = {}
    if "memory_off" in checks and (getattr(cfg, "hist_pose", False) or getattr(cfg, "motion_tokens", 0)):
        plan["memory_off"] = (list(range(len(samples))), dict(memory_dropout=1.0))
    if "swap_directions" in checks:  # instructions with a direction word
        idx = [i for i, s in enumerate(samples) if mirror_instruction(s["instruction"]) not in (None, s["instruction"])]
        plan["swap_directions"] = (idx, dict(swap_directions=1.0))
    if "mirror" in checks:
        idx = [i for i, s in enumerate(samples) if mirror_instruction(s["instruction"]) is not None]
        plan["mirror"] = (idx, dict(mirror=1.0))
    for name, (idx, ds_kw) in plan.items():
        if not idx:
            continue
        _, rec = profile(model, [samples[i] for i in idx], tok, ds_kw=ds_kw, **kw)
        out[name] = dict(rec, idx=np.asarray(idx), actions=list(cfg.actions), grid=int(model.grid))
    return out


def mirrored_top(top, actions, grid):
    """The choice of the mirrored sample that matches ``top`` (index into actions + patches)."""
    A = len(actions)
    if top < A:
        m = MIRROR_ACTION.get(actions[top], actions[top])
        return actions.index(m) if m in actions else top
    r, c = divmod(top - A, grid)
    return A + r * grid + (grid - 1 - c)


def _same_choice(a, b, A, grid, radius):
    if a < A or b < A:
        return a == b
    (ra, ca), (rb, cb) = divmod(a - A, grid), divmod(b - A, grid)
    return float(np.hypot(ra - rb, ca - cb)) <= radius


def summarize_checks(samples, runs, checks, radius=1.5):
    """``runs``: {checkpoint: profile() records}; ``checks``: {checkpoint: run_checks() output}."""
    flag = np.array([change_flag(s) for s in samples])
    res = {}

    def split(ok, f):
        return dict(acc=float(ok.mean()) if len(ok) else None,
                    change=float(ok[f == CHANGE].mean()) if (f == CHANGE).any() else None,
                    keep=float(ok[f == KEEP].mean()) if (f == KEEP).any() else None)

    for name, full in runs.items():
        ok = full["correct"].astype(bool)
        row = dict(full=split(ok, flag), n_change=int((flag == CHANGE).sum()), n_keep=int((flag == KEEP).sum()))
        c = checks.get(name, {})
        if "memory_off" in c:
            row["memory_off"] = split(c["memory_off"]["correct"].astype(bool), flag[c["memory_off"]["idx"]])
        if "swap_directions" in c:
            r = c["swap_directions"]
            fp, sp = full["pred"][r["idx"]], r["pred"]
            turn = np.isin(fp, [2, 3])
            row["swap_directions"] = dict(
                n=len(r["idx"]),
                changed=float((full["top"][r["idx"]] != r["top"]).mean()),
                n_turn=int(turn.sum()),
                turn_flipped=float((sp[turn] == 5 - fp[turn]).mean()) if turn.any() else None,  # LEFT 2 <-> RIGHT 3
            )
        if "mirror" in c:
            r = c["mirror"]
            A, g = len(r["actions"]), r["grid"]
            want = [mirrored_top(t, r["actions"], g) for t in full["top"][r["idx"]]]
            agree = np.array([_same_choice(a, b, A, g, radius) for a, b in zip(want, r["top"])])
            row["mirror"] = dict(n=len(r["idx"]), acc_unmirrored=float(ok[r["idx"]].mean()),
                                 acc_mirrored=float(r["correct"].astype(bool).mean()), agreement=float(agree.mean()))
        res[name] = row
    return res


# ---------------------------------------------------------------------- summary
def label_of(s):
    return "GOAL" if s["kind"] == GOAL else ACTION_NAME[s["action"]]


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(err / max(len(conf), 1))


def auroc(score, positive):
    """Probability that a positive sample scores higher than a negative one (ties count half)."""
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks of ties
        t = allv == v
        if t.sum() > 1:
            ranks[t] = ranks[t].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(samples, runs, legacy_stop_repeat=5, bins=15):
    """``runs``: {checkpoint name: profile() records}, all on ``samples``."""
    ds = np.array([base.dataset_of(s) for s in samples])
    kind = np.array([KIND[s["kind"]] for s in samples])
    label = np.array([label_of(s) for s in samples])
    groups = sorted(set(ds.tolist())) + ["all"]
    res = {}
    for name, r in runs.items():
        ok = r["correct"].astype(bool)
        pred = np.array(["GOAL" if p == -1 else ACTION_NAME[p] for p in r["pred"]])
        acc = {}
        for g in groups:
            in_g = (ds == g) if g != "all" else np.ones(len(ds), bool)
            row = {}
            for k in ("goal", "turn", "stop"):
                m = in_g & (kind == k)
                row[k] = dict(acc=float(ok[m].mean()) if m.any() else None, n=int(m.sum()))
            act = in_g & (kind != "goal")
            n_s, n_t = row["stop"]["n"], row["turn"]["n"]
            c_s, c_t = ok[in_g & (kind == "stop")].sum(), ok[in_g & (kind == "turn")].sum()
            row["action"] = dict(acc=float(ok[act].mean()) if act.any() else None, n=int(act.sum()))
            L = legacy_stop_repeat
            row["action_legacy"] = dict(acc=float((L * c_s + c_t) / (L * n_s + n_t)) if L * n_s + n_t else None)
            row["all"] = dict(acc=float(ok[in_g].mean()) if in_g.any() else None, n=int(in_g.sum()))
            acc[g] = row
        conf = {f"{a}->{b}": int(((label == a) & (pred == b)).sum()) for a in CLASSES for b in CLASSES}
        tp = conf["STOP->STOP"]
        chosen, labelled = int((pred == "STOP").sum()), int((label == "STOP").sum())
        prec, rec_ = tp / max(chosen, 1), tp / max(labelled, 1)
        stop = dict(
            precision=prec,
            recall=rec_,
            f1=2 * prec * rec_ / max(prec + rec_, 1e-9),
            chosen_rate=chosen / len(pred),
            label_rate=labelled / len(pred),
            confusion=conf,
        )
        cal = {"all": dict(ece=ece(r["conf"], ok, bins), mean_conf=float(r["conf"].mean()), acc=float(ok.mean()))}
        for k in ("goal", "turn", "stop"):
            m = kind == k
            if m.any():
                cal[k] = dict(ece=ece(r["conf"][m], ok[m], bins), mean_conf=float(r["conf"][m].mean()),
                              acc=float(ok[m].mean()))
        res[name] = dict(accuracy=acc, stop=stop, calibration=cal, escalate_auroc=auroc(r["esc"], ~ok))

    mat = np.stack([runs[n]["correct"].astype(bool) for n in runs], 1)
    always, never = mat.all(1), ~mat.any(1)
    consistency = {}
    for g in groups:
        in_g = (ds == g) if g != "all" else np.ones(len(ds), bool)
        for k in ("goal", "turn", "stop", "all"):
            m = in_g & ((kind == k) if k != "all" else True)
            if m.any():
                consistency[f"{g}/{k}"] = dict(n=int(m.sum()), always_right=float(always[m].mean()),
                                               always_wrong=float(never[m].mean()),
                                               mixed=float(1 - always[m].mean() - never[m].mean()))
    return dict(runs=res, consistency=consistency, checkpoints=list(runs))


# ---------------------------------------------------------------------- report
def _pct(v):
    return "–" if v is None else f"{100 * v:.1f}"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_report(out_dir, labels, trunc, summary, samples=None, runs=None, cfg_len=None, note="", shortcuts=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "diagnosis.json"), "w") as f:
        json.dump(dict(labels=labels, truncation=trunc, model=summary, shortcuts=shortcuts), f, indent=1)
    md = ["# Decision diagnosis", "", note, ""]
    md += ["## Labels (unique samples)", "", _table(
        ["dataset", "goal", "turn", "stop", "STOP share", "STOP share in training (repeated)", "change", "keep"],
        [[g, v["unique"]["goal"], v["unique"]["turn"], v["unique"]["stop"], _pct(v["stop_unique"]),
          _pct(v["stop_in_training"]), v.get("change", "–"), v.get("keep", "–")] for g, v in labels.items()])]
    if trunc:
        lens = sorted({L for v in trunc.values() for L in v["cut"]})
        md += ["", f"## Instruction length (tokens; the model keeps {cfg_len})", "", _table(
            ["dataset", "instructions", "p50", "p90", "p99", "max"] + [f"cut at {L} (%)" for L in lens],
            [[g, v["instructions"], v["tokens_p50"], v["tokens_p90"], v["tokens_p99"], v["tokens_max"]]
             + [_pct(v["cut"][L]["truncated"]) for L in lens] for g, v in trunc.items()])]
    if summary:
        for name, r in summary["runs"].items():
            md += ["", f"## {name}", "", "Accuracy (%), n in brackets. `action legacy` counts each STOP 5 times, "
                   "like the validation of the runs before --val_stop_repeat.", ""]
            md.append(_table(
                ["dataset", "goal", "turn", "stop", "action", "action legacy", "all"],
                [[g] + [f"{_pct(row[k]['acc'])} ({row[k]['n']})" for k in ("goal", "turn", "stop", "action")]
                 + [_pct(row["action_legacy"]["acc"]), f"{_pct(row['all']['acc'])} ({row['all']['n']})"]
                 for g, row in r["accuracy"].items()]))
            st = r["stop"]
            md += ["", f"STOP: precision {_pct(st['precision'])}, recall {_pct(st['recall'])}, F1 {_pct(st['f1'])}; "
                   f"chosen {_pct(st['chosen_rate'])} of samples vs labelled {_pct(st['label_rate'])}", ""]
            md.append(_table(["label \\ choice"] + list(CLASSES),
                             [[a] + [st["confusion"][f"{a}->{b}"] for b in CLASSES] for a in CLASSES]))
            md += ["", "Calibration of the top-1 probability:", ""]
            md.append(_table(["samples", "ECE", "mean probability", "accuracy"],
                             [[k, f"{v['ece']:.3f}", f"{v['mean_conf']:.3f}", f"{v['acc']:.3f}"]
                              for k, v in r["calibration"].items()]))
            a = r["escalate_auroc"]
            md += ["", f"Escalate head flags wrong decisions with AUROC {'–' if a is None else f'{a:.3f}'} (0.5 = chance)."]
        md += ["", f"## Agreement across {len(summary['checkpoints'])} checkpoint(s) (%)", "", _table(
            ["dataset/kind", "n", "always right", "always wrong", "mixed"],
            [[k, v["n"], _pct(v["always_right"]), _pct(v["always_wrong"]), _pct(v["mixed"])]
             for k, v in summary["consistency"].items()])]
    if shortcuts:
        md += _shortcut_report(shortcuts)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    if runs:
        with open(os.path.join(out_dir, "per_sample.csv"), "w", newline="") as f:
            w = csv.writer(f)
            names = list(runs)
            w.writerow(["dataset", "scene", "video", "ep_id", "start", "kind", "label", "change", "instruction"]
                       + [f"{n}:{c}" for n in names for c in ("choice", "correct", "prob", "escalate")])
            for i, s in enumerate(samples):
                row = [base.dataset_of(s), base.scene_of(s), s["video"], s["ep_id"], s["start"], KIND[s["kind"]],
                       label_of(s), change_flag(s), s["instruction"]]
                for n in names:
                    r = runs[n]
                    p = r["pred"][i]
                    row += ["GOAL" if p == -1 else ACTION_NAME[p], int(r["correct"][i]), f"{r['conf'][i]:.4f}",
                            f"{r['esc'][i]:.4f}"]
                w.writerow(row)


def _shortcut_report(sc):
    md = ["", "## Shortcut checks (%)", "",
          "`change`: samples whose label differs from the action just before (a goal after turning, a turn after "
          "walking or the other way); `keep`: the previous action gives the answer away. STOP and first steps are in "
          "neither. memory off: pose / motion inputs zeroed. swap: only left / right swapped in the instruction, "
          "images unchanged (`turn flipped`: share of LEFT / RIGHT choices that become the other one). mirror: "
          "the whole sample mirrored (`agreement`: the mirrored choice matches the mirror of the original).", ""]
    rows = []
    for name, r in sc.items():
        f, m = r["full"], r.get("memory_off")
        sw, mi = r.get("swap_directions"), r.get("mirror")
        rows.append([
            name, f"{_pct(f['acc'])}", f"{_pct(f['change'])} ({r['n_change']})", f"{_pct(f['keep'])} ({r['n_keep']})",
            "–" if m is None else f"{_pct(m['acc'])} / {_pct(m['change'])} / {_pct(m['keep'])}",
            "–" if sw is None else f"{_pct(sw['changed'])} ({sw['n']})",
            "–" if sw is None else f"{_pct(sw['turn_flipped'])} ({sw['n_turn']})",
            "–" if mi is None else f"{_pct(mi['acc_unmirrored'])} → {_pct(mi['acc_mirrored'])} ({mi['n']})",
            "–" if mi is None else _pct(mi["agreement"]),
        ])
    md.append(_table(["checkpoint", "all", "change", "keep", "memory off: all / change / keep",
                      "swap: decision changed", "swap: turn flipped", "mirror: accuracy", "mirror: agreement"], rows))
    return md


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vln_dataset_use", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="val", choices=["val", "train"], help="train: e.g. to weight training samples")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--val_split", default="scene", choices=["scene", "episode"])
    ap.add_argument("--max_samples", type=int, default=5000, help="fixed random subset; -1 = all")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--text_lens", default="160,256,384,512")
    ap.add_argument("--stop_repeat", type=int, default=5, help="STOP copies in training, for the label table")
    ap.add_argument("--legacy_stop_repeat", type=int, default=5)
    ap.add_argument("--correct_radius", type=float, default=LossWeights.correct_radius)
    ap.add_argument("--ece_bins", type=int, default=15)
    ap.add_argument("--skip_model", action="store_true", help="labels and instruction lengths only")
    ap.add_argument("--checks", default=",".join(CHECKS), help=f"shortcut checks, comma separated ({', '.join(CHECKS)}); "
                    "'none' = off. Each one is another pass over (a part of) the samples")
    args = ap.parse_args()

    samples = load_vln_samples(args.vln_dataset_use, stop_repeat=1)
    tok_dir = os.path.join(args.ckpts[0], "tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else LayaS2.config_class.text_encoder)
    labels = label_stats(samples, args.stop_repeat)
    trunc = truncation_stats(samples, tok, [int(x) for x in args.text_lens.split(",")])
    with open(os.path.join(args.ckpts[0], CONFIG_NAME)) as f:  # Laya-S2 or LayaNav config
        cfg_len = json.load(f).get("max_text_len", LayaS2.config_class.max_text_len)
    if args.skip_model:
        write_report(args.out_dir, labels, trunc, None, cfg_len=cfg_len)
        print(open(os.path.join(args.out_dir, "report.md")).read())
        return

    train_u, val_u = base.split_samples(samples, args.val_ratio, args.val_split)
    chosen = val_u if args.split == "val" else train_u
    if 0 < args.max_samples < len(chosen):
        chosen = random.Random(0).sample(chosen, args.max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checks = [] if args.checks == "none" else [c for c in args.checks.split(",") if c]
    assert set(checks) <= set(CHECKS), f"unknown check in {checks}"
    runs, kept, extra = {}, None, {}
    for ckpt in args.ckpts:
        name = "/".join(os.path.normpath(ckpt).split(os.sep)[-2:])
        print(f"[diagnose] {name}: {len(chosen)} {args.split} samples", flush=True)
        model = load_model(ckpt)
        s, rec = profile(model, chosen, tok, args.batch_size, args.num_workers, device, args.correct_radius)
        assert kept is None or len(s) == len(kept), "checkpoints with different action sets"
        kept, runs[name] = s, rec
        if checks:
            print(f"[diagnose] {name}: shortcut checks {checks}", flush=True)
            extra[name] = run_checks(model, kept, tok, checks, batch_size=args.batch_size, num_workers=args.num_workers,
                                     device=device, correct_radius=args.correct_radius)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = summarize(kept, runs, args.legacy_stop_repeat, args.ece_bins)
    note = (f"{args.split} split ({args.val_split}, ratio {args.val_ratio}), {len(kept)} samples, each STOP once; "
            f"goal correct within {args.correct_radius} patches.")
    shortcuts = summarize_checks(kept, runs, extra, args.correct_radius)
    write_report(args.out_dir, labels, trunc, summary, kept, runs, cfg_len, note, shortcuts)
    print(open(os.path.join(args.out_dir, "report.md")).read())


if __name__ == "__main__":
    main()
