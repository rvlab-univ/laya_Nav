#!/usr/bin/env bash
# One training run of design B 5.2. Rerunning the same command resumes from runs/<name>/last.
#   nohup bash scripts/server/train.sh N1 > logs/N1.log 2>&1 &
#   NAME=N1_timing nohup bash scripts/server/train.sh N1 --max_steps 1000 > logs/N1_timing.log 2>&1 &
# Runs: N1 (Laya checkpoint), N2 (plain ModernBERT-large), N4 (+ left-right mirror), N5 (N4 + teacher, alpha 0.5),
#       N7 (N4 + 10% instruction-route matching, design B 4.1 (A)).
# N4, N5 and N7 start like N1; START=none starts them like N2 instead.
# The reduced plan after the N1 / N2 diagnosis (design B 7.2.1: both ignore the instruction):
#   A: ModernBERT-large start (N2 beat N1) + mirror 0.5 + matching (MATCH, default 0.2), STEPS (default 16000) steps
#   B: A + teacher distribution (alpha ALPHA, default 0.5); needs runs/teacher from teacher_labels.sh
#   C: A + instruction contrast loss (weight CONTRAST, default 1.0): the label must be likelier with the sample's
#      own instruction than with another path's. B with it: NAME=BC bash scripts/server/train.sh B --contrast 1.0
#   D: C + landmark tags (design B 7.2.3): SigLIP2 names 10 landmarks of the current frame and 5 of each past frame,
#      put after the instruction as words; vision encoder frozen so the tags stay aligned. Few image tokens (current
#      frame 2 x 2, past frames 1 each; 539 tokens kept the encoder from learning to read the text). The first 200
#      steps are separable matching samples, and a quarter of later matching samples stay separable. MATCH (0.2).
# Arguments after the run name go to scripts/train.py (e.g. --grad_checkpointing on out-of-memory, --batch 8).
# NAME sets the output folder (default: the run name). All runs share the validation split splits/val_scenes.json.
source "$(dirname "$0")/common.sh"
RUN="${1:?usage: train.sh N1|N2|N4|N5|N7|A|B|C|D [train.py arguments]}"
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
    N7) ARGS=("${START_ARGS[@]}" --mirror 0.5 --match_frac 0.1) ;;
    A)  ARGS=(--laya none --encoder answerdotai/ModernBERT-large --mirror 0.5 --match_frac "${MATCH:-0.2}"
              --max_steps "${STEPS:-16000}") ;;
    B)  TEACHER="${TEACHER:-$RUNS/teacher}"
        [ -f "$TEACHER/teacher_probs.npy" ] || { echo "no teacher labels in $TEACHER (scripts/server/teacher_labels.sh)"; exit 1; }
        ARGS=(--laya none --encoder answerdotai/ModernBERT-large --mirror 0.5 --match_frac "${MATCH:-0.2}"
              --max_steps "${STEPS:-16000}" --teacher "$TEACHER" --alpha "${ALPHA:-0.5}") ;;
    C)  ARGS=(--laya none --encoder answerdotai/ModernBERT-large --mirror 0.5 --match_frac "${MATCH:-0.2}"
              --max_steps "${STEPS:-16000}" --contrast "${CONTRAST:-1.0}") ;;
    D)  ARGS=(--laya none --encoder answerdotai/ModernBERT-large --mirror 0.5 --match_frac "${MATCH:-0.2}"
              --max_steps "${STEPS:-12000}" --contrast "${CONTRAST:-1.0}" --tags 10 5 --grids 2 1 --freeze_vision
              --match_synth 0.25 --warm_steps "${WARM:-200}") ;;
    N3|N6|N8) echo "$RUN is not implemented yet (design B 5.3, 8 and 4.1 (B))"; exit 1 ;;
    *) echo "unknown run: $RUN"; exit 1 ;;
esac

NAME="${NAME:-$RUN}"
use_env "$STUDENT_ENV"
limit_threads "${THREADS:-8}"     # the main process; DataLoader workers use one thread each
QUOTA="$(cpu_quota)"
W=$(( QUOTA / 2 )); [ "$W" -gt 16 ] && W=16; [ "$W" -lt 2 ] && W=2
gpu_report
cd "$ROOT"
log "run $RUN -> $RUNS/$NAME: ${ARGS[*]} $*"
python scripts/train.py --out "$RUNS/$NAME" \
    --r2r_ann "$NAVILA_DATA/R2R/annotations.json" --r2r_frames "$NAVILA_DATA/R2R/train" \
    --rxr_ann "$NAVILA_DATA/RxR/annotations.json" --rxr_frames "$NAVILA_DATA/RxR/train" \
    --split "$SPLIT" --workers "${WORKERS:-$W}" "${ARGS[@]}" "$@"
log "run $NAME finished"
