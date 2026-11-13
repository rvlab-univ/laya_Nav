#!/usr/bin/env bash
# Offline ablations of trained students on the full validation set (the 5,000 samples training evaluates on).
#   nohup bash scripts/server/diagnose.sh N1 N2 > logs/diagnose.log 2>&1 &
# Each run name means runs/<name>/last. Results: runs/diag/<names>.json; the log has one line per variant.
# One model at a time, about 4 GB of GPU memory with the default batch. See scripts/diagnose.py for the variants.
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "usage: diagnose.sh N1 [N2 ...] [-- diagnose.py arguments]"; exit 1; }
CKPTS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    [ -f "$RUNS/$1/last/model.safetensors" ] || { echo "no checkpoint in $RUNS/$1/last"; exit 1; }
    CKPTS+=("$RUNS/$1/last")
    NAMES="${NAMES:+$NAMES-}$1"
    shift
done
[ "$1" = "--" ] && shift
use_env "$STUDENT_ENV"
limit_threads "${THREADS:-8}"     # the main process; DataLoader workers use one thread each
QUOTA="$(cpu_quota)"
W=$(( QUOTA / 2 )); [ "$W" -gt 16 ] && W=16; [ "$W" -lt 2 ] && W=2
gpu_report
cd "$ROOT"
log "diagnose ${CKPTS[*]}"
python scripts/diagnose.py --ckpt "${CKPTS[@]}" --out "$RUNS/diag/$NAMES.json" \
    --r2r_ann "$NAVILA_DATA/R2R/annotations.json" --r2r_frames "$NAVILA_DATA/R2R/train" \
    --rxr_ann "$NAVILA_DATA/RxR/annotations.json" --rxr_frames "$NAVILA_DATA/RxR/train" \
    --split "$SPLIT" --workers "${WORKERS:-$W}" "$@"
log "diagnose finished: $RUNS/diag/$NAMES.json"
