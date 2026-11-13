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
    train, val = None, None
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
    if val:
        v = val["val"]
        line += " | val acc %.3f, kind acc %.3f, stop recall %s (step %d)" % (
            v["acc"], v["kind_acc"], "%.2f" % v["stop_recall"] if v.get("stop_recall") is not None else "-",
            val["step"])
        if v.get("text_same") is not None:
            # same answer with another sample's instruction: 0.9 or more = the instruction is not read
            line += "\n%-21s same answer with another instruction %.2f%s" % (
                "", v["text_same"], "  <- instruction ignored" if v["text_same"] >= 0.9 else "")
        m = val.get("val_match")
        if m and m.get("acc") is not None:
            line += " | matching acc %.3f (true %s, swapped landmark %s, other episode %s)" % (
                m["acc"], *("%.2f" % m[k] if m.get(k) is not None else "-" for k in ("acc_pos", "acc_swap", "acc_other")))
    print(line)
if not any_run:
    print("no training logs yet")
for name in sorted(running):
    if not os.path.exists(os.path.join(runs_dir, name, "log.jsonl")):
        print("%-12s running  (loading data or model, no step logged yet)" % name)

shards = []
for path in sorted(glob.glob(os.path.join(logs_dir, "teacher_shard_*.log"))):
    last = None
    with open(path) as f:
        for line in f:
            m = re.search(r"shard (\d+): (\d+)/(\d+), ([\d.]+) samples/s", line)
            if m:
                last = m
    if last:
        shards.append(tuple(float(x) for x in last.groups()))
if shards:
    print("== teacher labels")
    for i, done, todo, rate in shards:
        print("shard %d: %d/%d, %.2f samples/s" % (i, done, todo, rate))
    rate = sum(s[3] for s in shards)
    left = sum(s[2] - s[1] for s in shards)
    print("all shards: %.2f samples/s, %d samples left, about %.1f h" % (rate, left, left / rate / 3600 if rate else 0))
EOF

echo "== GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
    || echo "nvidia-smi not available"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | while IFS=, read -r pid mem; do
    echo "  pid $pid ($(ps -o args= -p "$pid" 2>/dev/null | cut -c1-80)): $mem"
done
