"""Compare Habitat VLN runs from their progress.json files on the episodes they share.

    python scripts/eval/compare_progress.py \
        DualVLN=logs/habitat/test_dual_system LayaS2=logs/habitat/test_laya_s2
"""

import json
import os
import sys

METRICS = [("SR", "success", 100), ("SPL", "spl", 100), ("OS", "os", 100), ("NE", "ne", 1), ("nDTW", "ndtw", 100)]


def load(path):
    if os.path.isdir(path):
        path = os.path.join(path, "progress.json")
    runs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            runs[(r["scene_id"], r["episode_id"])] = r
    return runs


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main(args):
    named = [a.split("=", 1) if "=" in a else (os.path.basename(a.rstrip("/")), a) for a in args]
    runs = {name: load(path) for name, path in named}
    common = set.intersection(*(set(r) for r in runs.values()))
    print("episodes: " + ", ".join(f"{n}={len(r)}" for n, r in runs.items()) + f" | common={len(common)}\n")

    cols = [m[0] for m in METRICS] + ["steps", "S2 calls/ep", "S2 ms/call", "S2 s/ep", "S1 s/ep"]
    print(f"{'run':<12}" + "".join(f"{c:>13}" for c in cols))
    for name, r in runs.items():
        eps = [r[k] for k in common]
        row = []
        for _, key, scale in METRICS:
            vals = [e[key] for e in eps if key in e and e[key] is not None and e[key] != float("inf")]
            row.append(f"{mean(vals) * scale:.2f}" if vals else "-")
        row.append(f"{mean([e['steps'] for e in eps]):.1f}")
        if eps and "s2_time" in eps[0]:
            s2_calls = sum(e["s2_calls"] for e in eps)
            s2_time = sum(e["s2_time"] + e.get("s2_latent_time", 0.0) for e in eps)
            row += [
                f"{s2_calls / len(eps):.1f}",
                f"{1000 * s2_time / max(s2_calls, 1):.1f}",
                f"{s2_time / len(eps):.2f}",
                f"{mean([e['s1_time'] for e in eps]):.2f}",
            ]
        else:
            row += ["-"] * 4
        print(f"{name:<12}" + "".join(f"{c:>13}" for c in row))

    if len(runs) == 2:
        (na, a), (nb, b) = runs.items()
        both = int(sum(a[k]["success"] and b[k]["success"] for k in common))
        only_a = int(sum(a[k]["success"] and not b[k]["success"] for k in common))
        only_b = int(sum(b[k]["success"] and not a[k]["success"] for k in common))
        print(f"\nsuccess overlap: both={both}  only {na}={only_a}  only {nb}={only_b}  neither={len(common) - both - only_a - only_b}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
