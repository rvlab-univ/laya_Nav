#!/usr/bin/env bash
# Teacher labels on the first 1,000 samples: speed, and a check of the scoring against model.generate.
#   nohup bash scripts/server/teacher_check.sh > logs/teacher_check.log 2>&1 &
# Look at the end of the log:
#   "generate check: 20/20 ..."  the best-scored answer should equal what NaVILA generates
#   "samples/s"                  speed of one process; teacher_labels.sh runs SHARDS of them on the GPU
#   stats.json                   top1_equals_label should be high (NaVILA was trained on these samples)
# LIMIT and CHECK change the counts. Results in runs/teacher_check (separate from the full run).
source "$(dirname "$0")/common.sh"
use_env "$TEACHER_ENV"
gpu_report
cd "$ROOT"
OUT="$RUNS/teacher_check"
python scripts/extract_teacher_probs.py --ckpt "$TEACHER_CKPT" --out "$OUT" \
    --r2r_ann "$NAVILA_DATA/R2R/annotations.json" --r2r_frames "$NAVILA_DATA/R2R/train" \
    --rxr_ann "$NAVILA_DATA/RxR/annotations.json" --rxr_frames "$NAVILA_DATA/RxR/train" \
    --limit "${LIMIT:-1000}" --check_generate "${CHECK:-20}"
python scripts/extract_teacher_probs.py --out "$OUT" --merge --skip_expected
log "teacher check done: $OUT/stats.json"
