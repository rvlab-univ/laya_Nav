#!/usr/bin/env bash
# The H runs of design_C_review.md 5.10 from one command, three trainings side by side:
#   cd ~/light_nav && git pull
#   nohup bash scripts/server/h_runs.sh > logs/h_runs.log 2>&1 &
# H shows each of the 7 past frames as 2 x 2 tokens instead of 1 (--grids 2 2: 32 image tokens instead of 11). The
# rest is D's: the 224 px features already on the server (data/features/siglip2_g2), R2R only, from runs/D/last.
#   lane 1: H1       3,000 steps                                                compare with L0_r2r
#   lane 2: H1_kd    3,000 steps, NaVILA's distribution only where its first choice is the label   compare with H1
#   lane 3: H1_long  9,000 steps (R2R 1.4 epochs; L0 saw 0.46)                  compare with H1
# H1_kd reads runs/teacher, so it does not start while NaVILA labels are being made (extract_teacher_probs.py):
# their merge rewrites runs/teacher. Stop those first (pkill -u "$USER" -f extract_teacher_probs), or wait.
# Each run ends as ~/light_nav/<name>.tar.gz and writes logs/<name>.log; this log shows when each starts and ends.
# Rerun the same command after a stop: finished runs are skipped, stopped ones resume from runs/<name>/last.
# Arguments go to every scripts/train.py call. BATCH, ACCUM and INIT work as in experiment.sh.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/runs_lib.sh"
ARGS=("$@")
export FEATURES="${FEATURES:-$DATA/features/siglip2_g2}"
QUOTA="$(cpu_quota)"
export WORKERS="$(clamp $(( QUOTA / 6 )) 2 8)" THREADS="$(clamp $(( QUOTA / 6 )) 2 4)"
H=(--grids 2 2)

[ -f "$FEATURES/r2r.done" ] || { echo "no complete R2R features in $FEATURES: scripts/server/features.sh"; exit 1; }

lane1() { export STEPS=3000; logged H1 "${H[@]}"; }

lane2() {
    if [ ! -f "$ROOT/H1_kd.tar.gz" ]; then
        if [ ! -f "$RUNS/teacher/teacher_probs.npy" ]; then
            log "H1_kd: no runs/teacher/teacher_probs.npy (scripts/server/teacher_merge.sh)"
            return 1
        fi
        if pgrep -u "$(id -un)" -f extract_teacher_probs > /dev/null; then
            log "H1_kd: NaVILA labels are being made (extract_teacher_probs.py), and their merge rewrites runs/teacher."
            log "H1_kd: not started. Stop them (pkill -u \"\$USER\" -f extract_teacher_probs) or let them end; rerun."
            return 1
        fi
    fi
    export STEPS=3000
    logged H1_kd "${H[@]}" --teacher "$RUNS/teacher" --teacher_agree_only
}

lane3() { export STEPS=9000; logged H1_long "${H[@]}"; }

log "H runs: H1 and H1_kd (3,000 steps), H1_long (9,000 steps), three at a time, from ${INIT:-runs/D/last}"
log "CPU quota $QUOTA: $WORKERS loader processes and $THREADS threads per training"
gpu_report
free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "$free" ] && [ "$free" -lt 90000 ] 2>/dev/null; then
    log "warning: only $(( free / 1024 )) GB of GPU memory free (others use the GPU?); three trainings need about 90 GB"
fi
cd "$ROOT"
lane1 & p1=$!
lane2 & p2=$!
lane3 & p3=$!
bad=""
wait "$p1" || bad="$bad H1"
wait "$p2" || bad="$bad H1_kd"
wait "$p3" || bad="$bad H1_long"
done_runs=""
for n in H1 H1_kd H1_long; do
    if [ -f "$ROOT/$n.tar.gz" ]; then done_runs="$done_runs $n.tar.gz"; fi
done
if [ -n "$bad" ]; then
    log "not finished:$bad. Finished: ${done_runs:-none}. Fix it, then rerun the same command."
    exit 1
fi
log "H runs done:$done_runs"
log "next, on the lab PC: tar -xzf <name>.tar.gz -C runs && SEEN=400 BASE=L0_r2r bash scripts/local_drive.sh <name>"
