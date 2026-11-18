#!/usr/bin/env bash
# NaVILA's distribution on the states the student visited in a DAgger round: distillation on the student's own states
# (design_C_review.md, steps 2 and 4 together). Needs the round's frames, kept on the lab PC from round 2 on
# (scripts/dagger_round.sh), packed there by scripts/pack_dagger.sh as dagger<n>_frames.tar (+ .sha256, about 10 GB).
#   nohup bash scripts/server/teacher_dagger.sh 2 > logs/teacher_dagger2.log 2>&1 &
# 1. dagger<n>_frames.tar in ~/light_nav: checksum, then extracted to data/dagger/round<n>/ (skipped when the frames
#    are there).
# 2. SHARDS (default 2) NaVILA processes share the GPU (about 23 GB each; on 2026-11-13 two shards gave 3.7 samples/s
#    together, so about 11 hours for a round's 142k decisions). Per-shard logs: logs/teacher_dagger<n>_shard_<i>.log.
# 3. Merged into runs/teacher_dagger<n> (keys "dagger<n>:<video_id>"). runs/teacher (R2R / RxR) is not touched.
# Then train with both: --teacher runs/teacher runs/teacher_dagger<n> --extra dagger<n> data/dagger/round<n>/annotations.json
# (the DAgger rows' target: (1 - alpha) * the geometric expert's answer + alpha * NaVILA's distribution).
# Rerun the same command after a stop: every process skips what is already saved.
source "$(dirname "$0")/common.sh"
ROUND="${1:?usage: teacher_dagger.sh <round>}"
NAME="dagger$ROUND"
DIR="$DATA/dagger/round$ROUND"
OUT="$RUNS/teacher_$NAME"
SHARDS="${SHARDS:-2}"
TAR="$ROOT/${NAME}_frames.tar"

rollouts() {  # rollout folders in the round's frame folder
    if [ -d "$DIR/frames" ]; then find "$DIR/frames" -mindepth 1 -maxdepth 1 -type d | wc -l; else echo 0; fi
}
[ -f "$DIR/annotations.json" ] && [ -f "$DIR/summary.json" ] \
    || { echo "no $DIR/annotations.json: extract $NAME.tar (scripts/pack_dagger.sh $ROUND on the lab PC) first"; exit 1; }
want="$(grep -o '"episodes": *[0-9]*' "$DIR/summary.json" | grep -o '[0-9]*$')"
have="$(rollouts)"
if [ "$have" != "$want" ]; then
    [ -f "$TAR" ] && [ -f "$TAR.sha256" ] || { echo "$have of $want rollouts' frames in $DIR/frames, and no ${NAME}_frames.tar + .sha256 in $ROOT"; exit 1; }
    (cd "$ROOT" && sha256sum -c --status "${NAME}_frames.tar.sha256") \
        || { echo "${NAME}_frames.tar does not match its checksum: still uploading, or a broken copy"; exit 1; }
    log "${NAME}_frames.tar checksum ok: extracting"
    tar -xf "$TAR" -C "$ROOT"
    have="$(rollouts)"
    [ "$have" = "$want" ] || { echo "$have of $want rollouts' frames after extracting"; exit 1; }
    log "$NAME: frames of $have rollouts ready (${NAME}_frames.tar can be deleted)"
fi

use_env "$TEACHER_ENV"
QUOTA="$(cpu_quota)"
T=$(( QUOTA / SHARDS )); [ "$T" -gt 8 ] && T=8
limit_threads "${THREADS:-$T}"
log "$NAME: teacher labels into $OUT, $SHARDS shards, $OMP_NUM_THREADS threads each"
SRC=(--sources none --extra "$NAME" "$DIR/annotations.json" "$DIR/frames")
gpu_report
cd "$ROOT"
pids=()
for ((i = 0; i < SHARDS; i++)); do
    python scripts/extract_teacher_probs.py --ckpt "$TEACHER_CKPT" --out "$OUT" "${SRC[@]}" \
        --num_shards "$SHARDS" --shard "$i" > "$LOGS/teacher_${NAME}_shard_$i.log" 2>&1 &
    pids+=($!)
    log "shard $i started (pid $!)"
    sleep 60    # load the copies one after another
done
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { failed=1; log "shard $i failed: see $LOGS/teacher_${NAME}_shard_$i.log"; }
done
python scripts/extract_teacher_probs.py --out "$OUT" --merge "${SRC[@]}"
[ "$failed" = 0 ] && log "$NAME teacher labels done: $OUT" || { log "some shards failed; rerun to finish them"; exit 1; }
