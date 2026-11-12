#!/usr/bin/env bash
# One training run of design B 5.2. Rerunning the same command resumes from runs/<name>/last.
#   nohup bash scripts/server/train.sh N1 > logs/N1.log 2>&1 &
#   NAME=N1_timing nohup bash scripts/server/train.sh N1 --max_steps 1000 > logs/N1_timing.log 2>&1 &
# Runs: N1 (Laya checkpoint), N2 (plain ModernBERT-large), N4 (+ left-right mirror), N5 (N4 + teacher, alpha 0.5).
# N4 and N5 start like N1; START=none starts them like N2 instead (when N2 wins).
# Arguments after the run name go to scripts/train.py (e.g. --grad_checkpointing on out-of-memory, --batch 8).
# NAME sets the output folder (default: the run name). All runs share the validation split splits/val_scenes.json.
source "$(dirname "$0")/common.sh"
RUN="${1:?usage: train.sh N1|N2|N4|N5 [train.py arguments]}"
shift

if [ "${START:-english}" = none ]; then
    START_ARGS=(--laya none --encoder answerdotai/ModernBERT-large)
else
    START_ARGS=(--laya "${START:-english}")
fi
case "$RUN" in
    N1) ARGS=(--laya english) ;;
    N2) ARGS=(--laya none --encoder answerdotai/ModernBERT-large) ;;
    N4) ARGS=("${START_ARGS[@]}" --mirror 0.5) ;;
    N5) TEACHER="${TEACHER:-$RUNS/teacher}"
        [ -f "$TEACHER/teacher_probs.npy" ] || { echo "no teacher labels in $TEACHER (scripts/server/teacher_labels.sh)"; exit 1; }
        ARGS=("${START_ARGS[@]}" --mirror 0.5 --teacher "$TEACHER" --alpha "${ALPHA:-0.5}") ;;
    N3|N6|N7|N8) echo "$RUN is not implemented yet (design B 5.3, 8 and 4.1)"; exit 1 ;;
    *) echo "unknown run: $RUN"; exit 1 ;;
esac

NAME="${NAME:-$RUN}"
use_env "$STUDENT_ENV"
gpu_report
cd "$ROOT"
log "run $RUN -> $RUNS/$NAME: ${ARGS[*]} $*"
python scripts/train.py --out "$RUNS/$NAME" \
    --r2r_ann "$NAVILA_DATA/R2R/annotations.json" --r2r_frames "$NAVILA_DATA/R2R/train" \
    --rxr_ann "$NAVILA_DATA/RxR/annotations.json" --rxr_frames "$NAVILA_DATA/RxR/train" \
    --split "$SPLIT" --workers "${WORKERS:-16}" "${ARGS[@]}" "$@"
log "run $NAME finished"
