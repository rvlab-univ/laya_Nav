#!/usr/bin/env bash
# One DAgger round on this PC (design_C_review.md, step 2): the student of <checkpoint> drives R2R training episodes
# in Habitat, NaVILA's expert labels every decision, and the frames are stored as features of the frozen SigLIP2.
#   nohup bash scripts/dagger_round.sh 1 runs/D/last > logs/dagger_round1.log 2>&1 &
# Round n runs the expert's answer with probability 0.75^n (VLN-CE's schedule). Stop and rerun at any time: finished
# rollouts are kept. About 5 s per rollout: 7 hours for the default 5,000, and about 1 MB of features each.
# Output: data/dagger/round<n>/annotations.json and data/features/siglip2_g2/dagger<n>/, for
#   scripts/train.py --features data/features/siglip2_g2 --extra dagger<n> data/dagger/round<n>/annotations.json
set -u
ROUND="${1:?round number (1, 2, ...)}"
CKPT="${2:?checkpoint folder of the student that drives}"
EPISODES="${EPISODES:-5000}"
CHUNK="${CHUNK:-500}"
PORT="${PORT:-8766}"
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/assets/hf_cache"
OUT="data/dagger/round$ROUND"
FEATURES="${FEATURES:-data/features/siglip2_g2}"
BETA="$(env/bin/python -c "print(round(0.75 ** $ROUND, 4))")"
mkdir -p logs "$OUT"

env/bin/python -m lightnav.server --ckpt "$CKPT" --port "$PORT" > "logs/dagger_round${ROUND}_student.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
for _ in $(seq 1 100); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "the student did not start: logs/dagger_round${ROUND}_student.log"; exit 1; }
    sleep 3
done
echo "round $ROUND: student $CKPT, beta $BETA, $EPISODES rollouts"
while true; do
    env_vlnce/bin/python scripts/collect_dagger.py --out "$OUT" --url "http://127.0.0.1:$PORT" --beta "$BETA" \
        --episodes "$EPISODES" --seed "$ROUND" --limit "$CHUNK" 2>&1 | grep --line-buffered -E "rollouts|so far|Error|Traceback"
    code=${PIPESTATUS[0]}
    env/bin/python scripts/cache_features.py --source "dagger$ROUND" --ann "$OUT/annotations.json" \
        --frames "$OUT/frames" --out "$FEATURES" --delete_frames 2>&1 | grep --line-buffered -E "episodes|Error|Traceback"
    [ "$code" = 3 ] && break
    [ "$code" != 0 ] && { echo "collection stopped with code $code"; exit 1; }
done
echo "done: $(ls "$FEATURES/dagger$ROUND" | wc -l) rollouts stored in $FEATURES/dagger$ROUND"
cat "$OUT/summary.json"
