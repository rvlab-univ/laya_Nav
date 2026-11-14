"""Does the student's answer depend on the instruction? Replay a Go2 drive's decision points with other instructions.

Each decision in <drive>/decisions.jsonl is asked again on the same frames (frames_high/NNNN.jpg, NaVILA's 8-frame
choice), once with the drive's own instruction and once with each other instruction. A student that ignores the
instruction gives the same answers (same answer 1.00, KL 0); N2 did on 9 of 9 decisions in the KIMM room.

    # student server first:  HF_HOME=assets/hf_cache env/bin/python -m lightnav.server --ckpt runs/D/last --port 8765
    env/bin/python scripts/isaac/replay_instructions.py runs/isaac/drive/D_tv_avoid \\
        "Turn left and walk to the kitchen, then stop next to the washing machine." "Stop right here."
    # --last: also p(stop) at the drive's last view under each instruction (does the stop depend on the landmark?)
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lightnav.data import frame_slots                       # noqa: E402
from lightnav.questions import ANSWERS, KEYS                # noqa: E402
from lightnav.vlnce import decode_response, encode_request  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("drive")
    p.add_argument("others", nargs="+", help="instructions to compare with the drive's own")
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--last", action="store_true", help="p(stop) at the last view under every instruction")
    p.add_argument("--out", default=None, help="write the per-decision answers as JSON here")
    args = p.parse_args()

    own = json.load(open(os.path.join(args.drive, "summary.json")))["instruction"]
    decisions = [json.loads(line) for line in open(os.path.join(args.drive, "decisions.jsonl"))]
    cache = {}

    def frames_at(n):
        real = [(s, f) for s, f in enumerate(frame_slots(n, 8)) if f >= 0]
        for _, f in real:
            if f not in cache:
                path = os.path.join(args.drive, "frames_high", "%04d.jpg" % f)
                cache[f] = np.asarray(Image.open(path).convert("RGB"))
        return [cache[f] for _, f in real], [s for s, _ in real]

    def ask(text, frames, slots):
        req = urllib.request.Request(args.url + "/decide", data=encode_request(text, frames, slots, "replay"),
                                     headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=600) as r:
            answer, probs, _ = decode_response(r.read())
        return KEYS[ANSWERS.index(answer)] if answer in ANSWERS else answer, np.asarray(probs, dtype=np.float64)

    rows = []
    for d in decisions:
        frames, slots = frames_at(d["frames"])
        a0, p0 = ask(own, frames, slots)
        row = {"step": d["step"], "own": [a0, round(float(p0[0]), 3)], "others": []}
        for t in args.others:
            a, q = ask(t, frames, slots)
            kl = float((np.clip(p0, 1e-9, 1) * (np.log(np.clip(p0, 1e-9, 1)) - np.log(np.clip(q, 1e-9, 1)))).sum())
            row["others"].append([a, round(float(q[0]), 3), round(kl, 4)])
        rows.append(row)

    print("drive: %s\nown instruction: %s  (%d decisions)" % (args.drive, own, len(rows)))
    print("own max p(stop) %.3f" % max(r["own"][1] for r in rows))
    summary = []
    for i, t in enumerate(args.others):
        same = sum(r["own"][0] == r["others"][i][0] for r in rows)
        kl = float(np.mean([r["others"][i][2] for r in rows]))
        mp = max(r["others"][i][1] for r in rows)
        summary.append({"instruction": t, "same": same, "n": len(rows), "kl": kl, "max_p_stop": mp})
        print("vs %-82s same answer %3d/%d (%.2f)  mean KL %.3f  max p(stop) %.3f"
              % ('"%s"' % t, same, len(rows), same / len(rows), kl, mp))
    last = None
    if args.last:
        n = decisions[-1]["frames"] + (0 if decisions[-1]["answer"] == "stop" else 1)
        n = min(n, len(os.listdir(os.path.join(args.drive, "frames_high"))))
        frames, slots = frames_at(n)
        last = []
        print("last view (%d frames): p(stop) by instruction" % n)
        for t in [own] + args.others:
            a, q = ask(t, frames, slots)
            last.append({"instruction": t, "answer": a, "p_stop": round(float(q[0]), 3)})
            print("  %.3f  %-12s %s" % (q[0], a, t))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"drive": args.drive, "own": own, "summary": summary, "last": last, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
