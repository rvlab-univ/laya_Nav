#!/usr/bin/env bash
# Matching accuracy on train routes and on validation routes (scripts/match_eval.py), for runs that may still be
# training: each runs/<name>/last is copied first (without the optimizer state), since training replaces it.
#   bash scripts/server/match_eval.sh A > logs/match_eval.log 2>&1
# About 4 GB of GPU memory and a few minutes. Result: runs/diag/match_<names>.json
source "$(dirname "$0")/common.sh"
[ $# -ge 1 ] || { echo "usage: match_eval.sh A [B ...]"; exit 1; }
CKPTS=()
for name in "$@"; do
    src="$RUNS/$name/last"
    [ -f "$src/model.safetensors" ] || { echo "no checkpoint in $src"; exit 1; }
    snap="$RUNS/diag/snap_$name"
    rm -rf "$snap" && mkdir -p "$snap"
    for f in "$src"/*; do
        [ "$(basename "$f")" = train_state.pt ] || cp -r "$f" "$snap/"
    done     # a copy that fails because training was saving right then: run the command again
    CKPTS+=("$snap")
    NAMES="${NAMES:+$NAMES-}$name"
done
use_env "$STUDENT_ENV"
limit_threads "${THREADS:-4}"
cd "$ROOT"
log "match_eval ${CKPTS[*]}"
python scripts/match_eval.py --ckpt "${CKPTS[@]}" --out "$RUNS/diag/match_$NAMES.json" \
    --r2r_ann "$NAVILA_DATA/R2R/annotations.json" --r2r_frames "$NAVILA_DATA/R2R/train" \
    --rxr_ann "$NAVILA_DATA/RxR/annotations.json" --rxr_frames "$NAVILA_DATA/RxR/train" \
    --split "$SPLIT" --workers "${WORKERS:-4}"
log "match_eval finished: $RUNS/diag/match_$NAMES.json"
