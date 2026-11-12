#!/usr/bin/env bash
# Teacher labels for all unique R2R + RxR samples (about 679k), SHARDS processes sharing the GPU, then the merge.
#   SHARDS=4 nohup bash scripts/server/teacher_labels.sh > logs/teacher.log 2>&1 &
# Each process holds one copy of the 8B model (about 18 GB). Per-shard logs: logs/teacher_shard_<i>.log.
# Rerun the same command after a stop: every process skips what is already saved in runs/teacher.
source "$(dirname "$0")/common.sh"
use_env "$TEACHER_ENV"
SHARDS="${SHARDS:-4}"
OUT="$RUNS/teacher"
DATA_ARGS=(--r2r_ann "$NAVILA_DATA/R2R/annotations.json" --r2r_frames "$NAVILA_DATA/R2R/train"
           --rxr_ann "$NAVILA_DATA/RxR/annotations.json" --rxr_frames "$NAVILA_DATA/RxR/train")
gpu_report
cd "$ROOT"
pids=()
for ((i = 0; i < SHARDS; i++)); do
    python scripts/extract_teacher_probs.py --ckpt "$TEACHER_CKPT" --out "$OUT" "${DATA_ARGS[@]}" \
        --num_shards "$SHARDS" --shard "$i" > "$LOGS/teacher_shard_$i.log" 2>&1 &
    pids+=($!)
    log "shard $i started (pid $!)"
    sleep 60    # load the copies one after another
done
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { failed=1; log "shard $i failed: see $LOGS/teacher_shard_$i.log"; }
done
python scripts/extract_teacher_probs.py --out "$OUT" --merge "${DATA_ARGS[@]}"
[ "$failed" = 0 ] && log "teacher labels done: $OUT" || { log "some shards failed; rerun to finish them"; exit 1; }
