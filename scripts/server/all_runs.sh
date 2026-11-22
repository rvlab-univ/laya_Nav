#!/usr/bin/env bash
# The five server runs of design_C_review.md (5.2 and 5.5) from one command, three trainings side by side:
#   cd ~/light_nav && git pull
#   nohup bash scripts/server/all_runs.sh > logs/all_runs.log 2>&1 &
# One run alone does not fill the H200 (2.2 s per step there, 3.0 s on the lab PC's RTX 5060 Ti), so three at once
# should each slow down little: about 7 hours for the five, against 14-15 one after another. Check the speed with
# bash scripts/server/status.sh. Above about 6 s per step there is no gain: stop this, then run l_runs.sh and after
# it grid_runs.sh (finished runs are skipped, stopped ones resume).
#
# First, one after the other: the R2R features at 224 px (features.sh, unless r2r.done is there) and the teacher
# labels merged (unless runs/teacher/teacher_probs.npy is there). Then three lanes, each a series:
#   lane 1: L0_r2r -> DAgger round 1 (waits for dagger1.tar + dagger1.tar.sha256 in ~/light_nav) -> L1_dagger1
#   lane 2: L2_teacher -> G4_256 (waits for lane 3's features)
#   lane 3: R2R features at 256 px on a 4 x 4 grid (data/features/siglip2_256_g4, about 30 GB) -> G2_256
# All five start from runs/D/last, train 3,000 steps on R2R and end as ~/light_nav/<name>.tar.gz. G2_256 and G4_256
# use L0's data. Each run writes logs/<name>.log; this log shows when each starts and ends.
# Rerun the same command after a stop. A lane stops at its first error; the others go on.
# Arguments go to every scripts/train.py call. STEPS, BATCH, ACCUM and INIT work as in experiment.sh.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/runs_lib.sh"
ARGS=("$@")
F224="${FEATURES:-$DATA/features/siglip2_g2}"
F256="${FEATURES_256:-$DATA/features/siglip2_256_g4}"
V256=google/siglip2-base-patch16-256
FAILED="$F256.failed"                 # lane 3 leaves this when its features fail, so lane 2 stops waiting
rm -f "$FAILED"

QUOTA="$(cpu_quota)"
RUN_WORKERS="$(clamp $(( QUOTA / 6 )) 2 8)"      # data loader processes per training
RUN_THREADS="$(clamp $(( QUOTA / 6 )) 2 4)"      # torch threads per training
FEAT_WORKERS="$(clamp $(( QUOTA / 3 )) 2 16)"    # JPEG decoding beside two trainings

features() {  # log name; VISION, GRID, FEATURES and WORKERS from the caller
    if [ -f "$FEATURES/r2r.done" ]; then
        log "$1: complete before"
        return
    fi
    log "$1: started (logs/$1.log)"
    if bash "$HERE/features.sh" >> "$LOGS/$1.log" 2>&1; then
        log "$1: complete"
    else
        log "$1: stopped with an error, see the end of logs/$1.log"
        return 1
    fi
}

lane1() {
    export FEATURES="$F224" WORKERS="$RUN_WORKERS" THREADS="$RUN_THREADS"
    logged L0_r2r
    dagger_round 1
    logged L1_dagger1 --extra dagger1 "$ROOT/data/dagger/round1/annotations.json"
}

lane2() {
    export FEATURES="$F224" WORKERS="$RUN_WORKERS" THREADS="$RUN_THREADS"
    logged L2_teacher --teacher "$RUNS/teacher"
    wait_for "$F256/r2r.done" "the 256 px features (lane 3)" "$FAILED"
    export FEATURES="$F256"
    logged G4_256 --vision_encoder "$V256" --grids 4 1
}

lane3() {
    export VISION="$V256" GRID=4 FEATURES="$F256" WORKERS="$FEAT_WORKERS" THREADS="$RUN_THREADS"
    features features_256 || { touch "$FAILED"; exit 1; }
    export WORKERS="$RUN_WORKERS"
    logged G2_256 --vision_encoder "$V256" --grids 2 1
}

log "all runs: L0_r2r, L2_teacher, L1_dagger1, G2_256, G4_256 (${STEPS:-3000} steps each), three at a time"
log "CPU quota $QUOTA: $RUN_WORKERS loader processes and $RUN_THREADS threads per training"
gpu_report
free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free" ] && [ "$free" -lt 90000 ] 2>/dev/null; then
    log "warning: only $(( free / 1024 )) GB of GPU memory free (others use the GPU?); three trainings need about 90 GB"
fi
cd "$ROOT"
FEATURES="$F224" VISION=google/siglip2-base-patch16-224 GRID=2 features features_224
if [ -f "$RUNS/teacher/teacher_probs.npy" ]; then
    log "teacher labels: merged before"
else
    log "teacher labels: merging"
    bash "$HERE/teacher_merge.sh" >> "$LOGS/teacher_merge.log" 2>&1
    log "teacher labels: merged"
fi

lane1 & p1=$!
lane2 & p2=$!
lane3 & p3=$!
bad=""
wait "$p1" || bad="$bad 1"
wait "$p2" || bad="$bad 2"
wait "$p3" || bad="$bad 3"
done_runs=""
for n in L0_r2r L2_teacher L1_dagger1 G2_256 G4_256; do
    if [ -f "$ROOT/$n.tar.gz" ]; then done_runs="$done_runs $n.tar.gz"; fi
done
if [ -n "$bad" ]; then
    log "lanes with an error:$bad. Finished: ${done_runs:-none}. Fix it, then rerun the same command."
    exit 1
fi
log "all runs done: $done_runs"
log "next, on the lab PC for each: tar -xzf <name>.tar.gz -C runs && bash scripts/local_drive.sh <name>"
