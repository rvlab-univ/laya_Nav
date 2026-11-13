"""Does NaVILA's answer change with the instruction? A check of run B before it starts (design B 7.2.1).

Every R2R path is rendered once per instruction (3 instructions, the same frames and the same answer labels), so
the hard labels cannot teach the student to read the instruction. The teacher's distribution can, if it differs
between the instructions of one path. This script reads the teacher labels saved so far (parts of
scripts/extract_teacher_probs.py, finished or not) and, for every decision of a path that has two or more labeled
instructions, compares the teacher's distributions pair by pair:

- top1_differs: how often the teacher's best answer is a different one
- kl: KL divergence, the mean of both directions (nats); js: Jensen-Shannon (nats); tv: total variation
- label_prob_gap: |p(label | instruction 1) - p(label | instruction 2)|

Reference: the same numbers for pairs of decisions from different paths with the same answer label. When the
same-path numbers are close to 0 and far below the reference, NaVILA's targets barely depend on the instruction
and run B gives the student little reason to read it.

    python scripts/teacher_signal.py --teacher runs/teacher --r2r_ann data/NaVILA-Dataset/R2R/annotations.json \
        --out runs/teacher_signal.json
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lightnav import data as D                          # noqa: E402
from lightnav.questions import KINDS, kind              # noqa: E402
from lightnav.tasks import routes_from_rows             # noqa: E402


def read_teacher(folder: str) -> Tuple[List[str], np.ndarray]:
    """`(keys, probs [N, 10])` from the parts saved so far; a part still being written is skipped."""
    keys, logp = [], []
    for path in sorted(glob.glob(os.path.join(folder, "shard_*", "part_*.npz"))):
        if ".tmp" in os.path.basename(path):
            continue
        try:
            with np.load(path, allow_pickle=False) as part:
                k, lp = part["keys"].tolist(), part["logp"]
        except (OSError, ValueError, KeyError, EOFError):
            continue
        keys += k
        logp.append(lp)
    if not keys:
        raise SystemExit("no teacher parts under %s" % folder)
    logp = np.concatenate(logp).astype(np.float64)
    keys, first = np.unique(np.array(keys), return_index=True)
    logp = logp[first]
    p = np.exp(logp - logp.max(1, keepdims=True))
    return keys.tolist(), p / p.sum(1, keepdims=True)


def pair_stats(p: np.ndarray, q: np.ndarray, labels: np.ndarray) -> Dict:
    """Mean differences between the rows of `p` and `q` ([n, 10] each); `labels` are the answer labels."""
    p, q = np.clip(p, 1e-9, 1), np.clip(q, 1e-9, 1)
    lp, lq = np.log(p), np.log(q)
    m = (p + q) / 2
    lm = np.log(m)
    kl = 0.5 * ((p * (lp - lq)).sum(1) + (q * (lq - lp)).sum(1))
    js = 0.5 * ((p * (lp - lm)).sum(1) + (q * (lq - lm)).sum(1))
    idx = np.arange(len(p))
    return {
        "pairs": int(len(p)),
        "top1_differs": float((p.argmax(1) != q.argmax(1)).mean()),
        "kl": float(kl.mean()),
        "kl_median": float(np.median(kl)),
        "js": float(js.mean()),
        "tv": float(0.5 * np.abs(p - q).sum(1).mean()),
        "label_prob_gap": float(np.abs(p[idx, labels] - q[idx, labels]).mean()),
    }


def signal(rows: Sequence[D.Row], keys: Sequence[str], probs: np.ndarray, seed: int = 0) -> Dict:
    """Same-path pairs against different-path pairs with the same label, for the rows that have teacher labels."""
    where = {k: i for i, k in enumerate(keys)}
    path_of = {(r.source, r.episode): r.path for r in routes_from_rows(rows)}
    groups = defaultdict(list)                     # (path, step) -> teacher row indices, one per instruction
    labeled = []
    for r in rows:
        i = where.get(r.key)
        if i is None:
            continue
        labeled.append((i, r.label))
        groups[(path_of[(r.source, r.episode)], r.step)].append((i, r.label))
    out = {"labeled": len(labeled)}
    if labeled:
        idx = np.array([i for i, _ in labeled])
        lab = np.array([l for _, l in labeled])
        out["teacher_top1_equals_label"] = float((probs[idx].argmax(1) == lab).mean())
        out["teacher_mean_max_prob"] = float(probs[idx].max(1).mean())

    a, b, lab, kinds = [], [], [], []
    for (path, step), members in groups.items():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a.append(members[x][0])
                b.append(members[y][0])
                lab.append(members[x][1])
                kinds.append(kind(members[x][1]))
    out["paths_with_2_or_more"] = len({p for (p, _), m in groups.items() if len(m) >= 2})
    out["decisions_with_2_or_more"] = sum(1 for m in groups.values() if len(m) >= 2)
    if not a:
        out["same_path"] = None
        return out
    a, b, lab, kinds = np.array(a), np.array(b), np.array(lab), np.array(kinds)
    out["same_path"] = pair_stats(probs[a], probs[b], lab)
    out["same_path_by_kind"] = {KINDS[k]: pair_stats(probs[a[kinds == k]], probs[b[kinds == k]], lab[kinds == k])
                                for k in range(len(KINDS)) if (kinds == k).any()}

    # reference: as many pairs, each from two different paths with the same answer label
    rng = np.random.default_rng(seed)
    by_label = defaultdict(list)
    for (path, step), members in groups.items():
        for i, l in members:
            by_label[l].append((i, path))
    ra, rb, rl = [], [], []
    for l in lab:
        pool = by_label[l]
        for _ in range(20):
            (i, pi), (j, pj) = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
            if pi != pj:
                ra.append(i)
                rb.append(j)
                rl.append(l)
                break
    if ra:
        out["other_path_same_label"] = pair_stats(probs[np.array(ra)], probs[np.array(rb)], np.array(rl))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = os.path.join(ROOT, "data", "NaVILA-Dataset")
    p.add_argument("--teacher", default=os.path.join(ROOT, "runs", "teacher"))
    p.add_argument("--r2r_ann", default=os.path.join(data, "R2R", "annotations.json"))
    p.add_argument("--rxr_ann", default=None, help="RxR annotations.json; left out when not given")
    p.add_argument("--out", default=None, help="JSON with the numbers")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    keys, probs = read_teacher(args.teacher)
    print("teacher labels read: %d" % len(keys), flush=True)
    sources = [("r2r", args.r2r_ann)] + ([("rxr", args.rxr_ann)] if args.rxr_ann else [])
    result = {"teacher": args.teacher, "teacher_labels": len(keys)}
    for name, ann in sources:
        rows = D.load_rows(ann, name, english_only=True, dedupe=True)    # as the teacher labels were made
        result[name] = signal(rows, keys, probs, args.seed)
        print(name + " " + json.dumps(result[name], indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
