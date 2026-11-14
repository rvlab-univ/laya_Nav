#!/usr/bin/env bash
# Combine the teacher labels extracted so far into runs/teacher/{keys.txt,teacher_probs.npy} (what --teacher reads).
#   bash scripts/server/teacher_merge.sh
# teacher_labels.sh does this itself when it finishes. Run this to train before that: on 2026-11-14 the parts held
# every R2R sample and 2.4% of RxR. Safe while teacher_labels.sh is running; samples without a label train on the
# data's answer alone.
source "$(dirname "$0")/common.sh"
use_env "$STUDENT_ENV"
cd "$ROOT"
python scripts/extract_teacher_probs.py --out "$RUNS/teacher" --merge \
    --r2r_ann "$NAVILA_DATA/R2R/annotations.json" --rxr_ann "$NAVILA_DATA/RxR/annotations.json"
