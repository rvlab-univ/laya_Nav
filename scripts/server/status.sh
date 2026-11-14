#!/usr/bin/env bash
# Progress of every training run and of the teacher labels, and the GPU, on one screen.
#   bash scripts/server/status.sh
# A run is "running" while its scripts/train.py process is alive. Steps, speed and loss come from
# runs/<name>/log.jsonl, the total from runs/<name>/run.json (or the "optimizer steps" line in logs/<name>.log).
source "$(dirname "$0")/common.sh"
use_env "$STUDENT_ENV"

running=()
for d in "$RUNS"/*/; do
    name="$(basename "$d")"
    pgrep -f -- "scripts/train.py --out $RUNS/$name " >/dev/null && running+=("$name")
done

python - "$RUNS" "$LOGS" "${running[@]}" <<'EOF'
import glob, json, os, re, sys

runs_dir, logs_dir, running = sys.argv[1], sys.argv[2], set(sys.argv[3:])

def total_steps(name):
    try:
        with open(os.path.join(runs_dir, name, "run.json")) as f:
            return json.load(f)["total_steps"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        with open(os.path.join(logs_dir, name + ".log")) as f:
            m = re.findall(r"(\d+) optimizer steps", f.read())
        return int(m[-1]) if m else None
    except OSError:
        return None

print("== training")
any_run = False
for path in sorted(glob.glob(os.path.join(runs_dir, "*", "log.jsonl"))):
    name = os.path.basename(os.path.dirname(path))
    train, val, vals = None, None, []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "ce" in rec:
                train = rec
            if "val" in rec:
                val = rec
                vals.append(rec)
    if train is None:
        continue
    any_run = True
    step, total = train["step"], total_steps(name)
    state = "running" if name in running else "stopped"
    if total:
        left = (total - step) * train["sec_per_step"] / 3600
        where = "step %d/%d (%.1f%%), %.2f s/step, %.1f h left" % (step, total, 100 * step / total,
                                                                   train["sec_per_step"], left)
    else:
        where = "step %d, %.2f s/step" % (step, train["sec_per_step"])
    line = "%-12s %-8s %s | ce %.3f" % (name, state, where, train["ce"])
    if train.get("contrast_gap") is not None:
        # log p(label | own instruction) - log p(label | another path's): above 0 = the instruction is used
        line += " | contrast gap %+.3f" % train["contrast_gap"]
    if val:
        v = val["val"]
        # in brackets: what one constant answer scores on the same samples (older logs do not have it)
        base = lambda k: " (constant %.3f)" % v[k] if v.get(k) is not None else ""   # noqa: E731
        line += " | val acc %.3f%s, kind acc %.3f%s, stop recall %s (step %d)" % (
            v["acc"], base("constant_acc"), v["kind_acc"], base("constant_kind_acc"),
            "%.2f" % v["stop_recall"] if v.get("stop_recall") is not None else "-", val["step"])
        if v.get("text_same") is not None:
            # same answer with another sample's instruction: 0.9 or more = the instruction is not read
            line += "\n%-21s same answer with another instruction %.2f%s" % (
                "", v["text_same"], "  <- instruction ignored" if v["text_same"] >= 0.9 else "")
        m = val.get("val_match")
        if m and m.get("acc") is not None:
            line += " | matching acc %.3f (true %s, swapped landmark %s, other episode %s)" % (
                m["acc"], *("%.2f" % m[k] if m.get(k) is not None else "-" for k in ("acc_pos", "acc_swap", "acc_other")))
        if len(vals) > 1:
            # every validation so far: step, val acc, same-answer rate, matching acc
            hist = []
            for r in vals:
                v, m = r["val"], r.get("val_match") or {}
                hist.append("%d: %.3f/%s/%s" % (r["step"], v["acc"],
                                                "%.2f" % v["text_same"] if v.get("text_same") is not None else "-",
                                                "%.2f" % m["acc"] if m.get("acc") is not None else "-"))
            line += "\n%-21s history (step: val acc/same answer/matching) %s" % ("", ", ".join(hist))
    print(line)
if not any_run:
    print("no training logs yet")
for name in sorted(running):
    if not os.path.exists(os.path.join(runs_dir, name, "log.jsonl")):
        print("%-12s running  (loading data or model, no step logged yet)" % name)

# Only the shards of the latest run: a run with fewer shards leaves the older teacher_shard_<i>.log files behind.
# Each log starts with "shard i/N: ..."; N comes from shard 0's log.
def shard_log(path):
    head, last = None, None
    with open(path) as f:
        for line in f:
            h = re.search(r"shard (\d+)/(\d+): \d+ samples", line)
            if h:
                head, last = h, None
            m = re.search(r"shard (\d+): (\d+)/(\d+), ([\d.]+) samples/s", line)
            if m:
                last = m
    return head, last

logs = {}
for path in glob.glob(os.path.join(logs_dir, "teacher_shard_*.log")):
    head, last = shard_log(path)
    if head:
        logs[int(head.group(1))] = (int(head.group(2)), last)
n = logs[0][0] if 0 in logs else None
shards = [(i, *(float(x) for x in last.groups()[1:])) for i, (k, last) in sorted(logs.items())
          if k == n and i < n and last]
if shards:
    print("== teacher labels (%d shards)" % n)
    for i, done, todo, rate in shards:
        print("shard %d: %d/%d, %.2f samples/s, %.1f h left" % (i, done, todo, rate, (todo - done) / rate / 3600))
    rate = sum(s[3] for s in shards)
    left = sum(s[2] - s[1] for s in shards)
    # the shards run side by side: the run ends with the slowest one
    hours = max((s[2] - s[1]) / s[3] for s in shards) / 3600
    print("all shards: %.2f samples/s, %d samples left, about %.1f h" % (rate, left, hours))
EOF

echo "== GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi not available"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r pid mem; do
    echo "  pid $pid ($(ps -o args= -p "$pid" 2>/dev/null | cut -c1-80)): $mem"
done
