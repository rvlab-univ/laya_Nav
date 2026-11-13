#!/usr/bin/env bash
# Does NaVILA's answer change with the instruction? Reads the teacher labels saved so far (the extraction can keep
# running) and compares the 3 instructions of each R2R path; see scripts/teacher_signal.py.
#   bash scripts/server/teacher_signal.sh            # R2R; RXR=1 adds RxR
# CPU only, a few minutes. Result: runs/teacher_signal.json
source "$(dirname "$0")/common.sh"
use_env "$STUDENT_ENV"
limit_threads "${THREADS:-2}"      # training and the teacher shards keep the CPU
EXTRA=()
[ "${RXR:-0}" = 1 ] && EXTRA=(--rxr_ann "$NAVILA_DATA/RxR/annotations.json")
cd "$ROOT"
python scripts/teacher_signal.py --teacher "$RUNS/teacher" --r2r_ann "$NAVILA_DATA/R2R/annotations.json" \
    "${EXTRA[@]}" --out "$RUNS/teacher_signal.json"
