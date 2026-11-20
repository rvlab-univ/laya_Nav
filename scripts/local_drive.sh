#!/usr/bin/env bash
# Drive a checkpoint on the 100 val_seen and 100 val_unseen R2R episodes every model so far was driven on, and
# compare it with D on the same episodes (lab PC only: Habitat). About 20 minutes; finished episodes are skipped.
#   bash scripts/local_drive.sh L0_r2r                  # runs/L0_r2r/last -> runs/eval/L0_r2r/
# For a checkpoint trained on the server: tar -xzf <name>.tar.gz -C runs first.
# Pick between experiments on val_seen; val_unseen is for the final comparison only.
# SEEN=400 drives 400 val_seen episodes instead (runs/eval/<name>/r2r_valseen_400, a superset of the 100), the size
# models are picked on from 2026-11-20 (design_C_review.md 5.9); BASE (default D) is the run compared against.
#   SEEN=400 bash scripts/local_drive.sh L0_r2r && SEEN=400 BASE=L0_r2r bash scripts/local_drive.sh H1
set -u
NAME="${1:?run name (runs/<name>/last)}"
PORT="${PORT:-8767}"
SEEN="${SEEN:-100}"
BASE="${BASE:-D}"
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/assets/hf_cache" TOKENIZERS_PARALLELISM=false
mkdir -p logs
[ -f "runs/$NAME/last/model.safetensors" ] || { echo "no checkpoint in runs/$NAME/last"; exit 1; }

env/bin/python -m lightnav.server --ckpt "runs/$NAME/last" --port "$PORT" > "logs/${NAME}_student.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 100); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "the student did not start: logs/${NAME}_student.log"; exit 1; }
    sleep 3
done
for part in val_seen val_unseen; do
    n=100 out="runs/eval/$NAME/r2r_100"
    [ "$part" = val_seen ] && n="$SEEN" out="runs/eval/$NAME/r2r_valseen_$SEEN"
    env_vlnce/bin/python scripts/eval_vlnce.py --split r2r --agent http --url "http://127.0.0.1:$PORT" \
        --part "$part" --episodes "$n" --out "$out" 2>&1 | grep --line-buffered -E "episodes|Error|Traceback"
done
for sub in "r2r_valseen_$SEEN" r2r_100; do
    echo "== $sub ($BASE first)"
    if [ -f "runs/eval/$BASE/$sub/episodes.jsonl" ]; then
        env/bin/python scripts/eval_summary.py "runs/eval/$BASE/$sub" "runs/eval/$NAME/$sub" --pair
    else
        echo "$BASE has not driven $sub: runs/eval/$NAME/$sub alone"
        env/bin/python scripts/eval_summary.py "runs/eval/$NAME/$sub"
    fi
done
