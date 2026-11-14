#!/usr/bin/env bash
# One experiment from stored features on the server: the training of scripts/local_experiment.sh on the lab PC (a
# D-type student from runs/D/last, 6,000 optimizer steps of 64 samples), then the checkpoint packed for download.
# Needs scripts/server/features.sh first. Rerunning the same command resumes from runs/<name>/last.
#   nohup bash scripts/server/experiment.sh L0_r2r > logs/L0_r2r.log 2>&1 &
#   nohup bash scripts/server/experiment.sh L2_teacher --teacher runs/teacher > logs/L2_teacher.log 2>&1 &
#   nohup bash scripts/server/experiment.sh L1_dagger1 --extra dagger1 data/dagger/round1/annotations.json \
#       > logs/L1_dagger1.log 2>&1 &
# Arguments after the name go to scripts/train.py. Settings from the environment:
#   STEPS=6000  INIT=runs/D/last  BATCH=16  ACCUM=4 (64 samples per step; BATCH=8 ACCUM=8 when memory is short)
#   RXR=1       train on R2R + RxR, as D did (features.sh with SOURCES="r2r rxr"); the lab PC has R2R only
# --teacher needs runs/teacher/teacher_probs.npy: scripts/server/teacher_merge.sh.
# --extra (a DAgger round) needs the round from the lab PC, where Habitat runs: scripts/pack_dagger.sh there, then
#   tar -xf dagger1.tar -C ~/light_nav here.
# The server cannot drive (no Habitat): move <name>.tar.gz to the lab PC, then
#   tar -xzf <name>.tar.gz -C runs && bash scripts/local_drive.sh <name>
source "$(dirname "$0")/common.sh"
NAME="${1:?usage: experiment.sh <name> [train.py arguments]}"
shift
FEATURES="${FEATURES:-$DATA/features/siglip2_g2}"
INIT="${INIT:-$RUNS/D/last}"
[ -f "$FEATURES/r2r.done" ] || { echo "no complete R2R features in $FEATURES: scripts/server/features.sh"; exit 1; }
[ -f "$INIT/model.safetensors" ] || { echo "no checkpoint in $INIT"; exit 1; }
DATA_ARGS=(--r2r_ann "$NAVILA_DATA/R2R/annotations.json")
if [ "${RXR:-0}" = 1 ]; then
    [ -f "$FEATURES/rxr.done" ] || { echo "no complete RxR features in $FEATURES: SOURCES=\"r2r rxr\" features.sh"; exit 1; }
    DATA_ARGS+=(--rxr_ann "$NAVILA_DATA/RxR/annotations.json")
fi
use_env "$STUDENT_ENV"
limit_threads "${THREADS:-8}"
QUOTA="$(cpu_quota)"
W=$(( QUOTA / 2 )); [ "$W" -gt 16 ] && W=16; [ "$W" -lt 2 ] && W=2
gpu_report
cd "$ROOT"
log "experiment $NAME from $INIT: ${DATA_ARGS[*]} $*"
python scripts/train.py --out "$RUNS/$NAME" "${DATA_ARGS[@]}" --features "$FEATURES" --split "$SPLIT" \
    --laya none --encoder answerdotai/ModernBERT-large --freeze_vision --tags 10 5 --grids 2 1 \
    --mirror 0.5 --match_frac 0.2 --match_synth 0.25 --contrast 1.0 \
    --init "$INIT" --max_steps "${STEPS:-6000}" --batch "${BATCH:-16}" --grad_accum "${ACCUM:-4}" \
    --eval_every 1000 --save_every 1000 --workers "${WORKERS:-$W}" "$@"
tar -czf "$ROOT/$NAME.tar.gz" --exclude="$NAME/last/train_state.pt" -C "$RUNS" "$NAME"
log "experiment $NAME finished: $(ls -lh "$ROOT/$NAME.tar.gz" | awk '{print $5}') in $ROOT/$NAME.tar.gz"
