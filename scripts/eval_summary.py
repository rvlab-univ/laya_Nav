"""Summaries of VLN-CE runs (scripts/eval_vlnce.py) and paired comparisons between them.

    python scripts/eval_summary.py runs/eval/N1/r2r runs/eval/N2/r2r            # one row per run
    python scripts/eval_summary.py runs/eval/N1/r2r runs/eval/N2/r2r --pair     # + N2 - N1 on shared episodes

Means follow NaVILA's NaVILA/evaluation/scripts/eval_jsons.py: SR, OS, nDTW, TL and steps over all episodes;
NE and SPL over the episodes where they are finite (the rest are counted as invalid). The pair comparison uses the
episodes both runs have, with a paired bootstrap (10,000 resamples) for a 95% interval of the difference.

A second table says how the episodes went, because SR alone hides it (N2 and D both ended farther from the goal
than they started, on average): progress = distance to the goal at the start minus at the end; the share of
episodes that ended nearer / farther (by more than 0.5 m); and how the failures ended (the student said stop
somewhere else, or never said stop).
"""
import argparse
import json
import math
import os

import numpy as np

COLUMNS = [("SR", "success", 100), ("SPL", "spl", 100), ("NE", "distance_to_goal", 1), ("OS", "oracle_success", 100),
           ("nDTW", "ndtw", 100), ("TL", "path_length", 1), ("steps", "steps_taken", 1)]


def load(run):
    with open(os.path.join(run, "episodes.jsonl")) as f:
        return {r["episode_id"]: r for r in (json.loads(line) for line in f if line.strip())}


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def summarize(rows):
    out = {"episodes": len(rows)}
    for name, key, scale in COLUMNS:
        vals = [r["metrics"].get(key) for r in rows]
        good = [v for v in vals if finite(v)]
        out[name] = scale * float(np.mean(good)) if good else None
        if key in ("spl", "distance_to_goal"):
            out["invalid_" + name] = len(vals) - len(good)
    ms = [t for r in rows for t in r.get("ms", []) if t > 0]
    out["decisions_per_episode"] = float(np.mean([r["decisions"] for r in rows])) if rows else None
    out["ms_median"] = float(np.median(ms)) if ms else None
    out["ms_p90"] = float(np.percentile(ms, 90)) if ms else None
    out["unparsed"] = int(sum(r.get("unparsed", 0) for r in rows))
    return out


def start_distances(run):
    """Episode id -> geodesic distance to the goal at the start, from the episode file (older runs did not log it)."""
    import gzip
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(run, "subset.json")) as f:
            sub = json.load(f)
    except OSError:
        return {}
    part = sub.get("part", "val_unseen")
    path = {"r2r": "data/vlnce/r2r/%s.json.gz" % part,
            "rxr": "data/vlnce/rxr/%s/%s_guide.json.gz" % (part, part)}.get(sub.get("split"))
    if not path or not os.path.exists(os.path.join(root, path)):
        return {}
    with gzip.open(os.path.join(root, path), "rt") as f:
        return {str(e["episode_id"]): e["info"]["geodesic_distance"] for e in json.load(f)["episodes"]
                if "geodesic_distance" in (e.get("info") or {})}


def outcomes(rows, start):
    """How the episodes went: progress toward the goal and how the failures ended."""
    prog = []
    for r in rows:
        s = r.get("start_distance")
        s = start.get(str(r["episode_id"])) if s is None else s
        e = r["metrics"].get("distance_to_goal")
        if finite(s) and finite(e):
            prog.append(s - e)
    n = len(rows)
    stopped = [bool(r.get("answers", {}).get("stop", 0)) for r in rows]
    ok = [bool(r["metrics"].get("success")) for r in rows]
    return {"progress_m": float(np.mean(prog)) if prog else None,
            "nearer": 100.0 * sum(p > 0.5 for p in prog) / len(prog) if prog else None,
            "farther": 100.0 * sum(p < -0.5 for p in prog) / len(prog) if prog else None,
            "wrong_stop": 100.0 * sum(s and not o for s, o in zip(stopped, ok)) / n if n else None,
            "no_stop": 100.0 * sum(not s for s in stopped) / n if n else None}


def paired(a, b, key, scale, n=10000, seed=0):
    ids = sorted(set(a) & set(b))
    x = np.array([a[i]["metrics"].get(key, np.nan) for i in ids], dtype=float)
    y = np.array([b[i]["metrics"].get(key, np.nan) for i in ids], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    d = (y - x)[keep]
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"episodes": int(len(d)), "diff": scale * float(d.mean()),
            "ci95": [scale * float(np.percentile(boots, 2.5)), scale * float(np.percentile(boots, 97.5))]}


def fmt(v, digits=1):
    return "–" if v is None else ("%.*f" % (digits, v))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--pair", action="store_true", help="compare every run with the first on shared episodes")
    p.add_argument("--json", default=None, help="also write the numbers to this file")
    args = p.parse_args()

    data = {run: load(run) for run in args.runs}
    table = {run: summarize(list(rows.values())) for run, rows in data.items()}
    head = ["run", "episodes"] + [c[0] for c in COLUMNS] + ["decisions/ep", "ms median", "ms p90"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for run, s in table.items():
        cells = [run, str(s["episodes"])] + [fmt(s[c[0]], 2 if c[0] in ("NE", "TL") else 1) for c in COLUMNS]
        cells += [fmt(s["decisions_per_episode"]), fmt(s["ms_median"]), fmt(s["ms_p90"])]
        print("| " + " | ".join(cells) + " |")
    for run, s in table.items():
        if s.get("invalid_SPL") or s.get("invalid_NE") or s["unparsed"]:
            print("%s: invalid SPL %d, invalid NE %d, unparsed answers %d"
                  % (run, s["invalid_SPL"], s["invalid_NE"], s["unparsed"]))
    print()
    head2 = ["run", "progress (m)", "ended nearer %", "ended farther %", "stopped elsewhere %", "never stopped %"]
    print("| " + " | ".join(head2) + " |")
    print("|" + "---|" * len(head2))
    for run, rows in data.items():
        o = outcomes(list(rows.values()), start_distances(run))
        table[run]["outcomes"] = o
        print("| " + " | ".join([run, "–" if o["progress_m"] is None else "%+.2f" % o["progress_m"]]
                                + [fmt(o[k]) for k in ("nearer", "farther", "wrong_stop", "no_stop")]) + " |")
    out = {"runs": table}
    if args.pair and len(args.runs) > 1:
        base = args.runs[0]
        out["pairs"] = {}
        for run in args.runs[1:]:
            res = {name: paired(data[base], data[run], key, scale) for name, key, scale in COLUMNS[:2] + COLUMNS[4:5]}
            out["pairs"][run] = res
            for name, r in res.items():
                if r:
                    print("%s - %s, %s: %+.1f (95%% CI %+.1f to %+.1f) on %d shared episodes"
                          % (run, base, name, r["diff"], r["ci95"][0], r["ci95"][1], r["episodes"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
