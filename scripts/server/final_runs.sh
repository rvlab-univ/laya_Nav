#!/usr/bin/env bash
# The final models for 2026-11-26 (design_C_review.md 5.12): no more comparisons, as much data and as many steps as
# the time allows. Both start from runs/D/last and train on R2R + RxR + the DAgger rounds, on the 224 px features.
#   cd ~/light_nav && git pull
#   nohup bash scripts/server/final_runs.sh 1 > logs/final_1.log 2>&1 &     # stage 1: now
#   nohup bash scripts/server/final_runs.sh 2 > logs/final_2.log 2>&1 &     # stage 2: once dagger2.tar is uploaded
# Stage 1, side by side, 36,000 steps each (about 2 epochs of R2R + RxR + DAgger round 1, 10-13 hours):
#   F_base  run D's model (each past frame 1 token)                    the safe one: only more data and steps
#   F_full  past frames 2 x 2 (--grids 2 2), each frame's pose (--pose), NaVILA's distribution only where its
#           first choice is the label (--teacher_agree_only)
# Stage 2, from each stage-1 model, 12,000 steps on the same data + DAgger round 2: F_base2, F_full2.
# Each run keeps its model every 12,000 steps (runs/<name>/step_<n>) and ends as ~/light_nav/<name>.tar.gz.
# F_base may already run by hand (experiment.sh F_base ...): then this script leaves it alone.
# Needs: data/features/siglip2_g2/{r2r,rxr,dagger1}.done, data/dagger/round1/episodes.jsonl (uploaded from the lab
# PC: F_full's poses on round 1), runs/teacher/teacher_probs.npy, and no NaVILA labels being made (their merge
# rewrites runs/teacher). Rerun the same command after a stop: finished runs are skipped, stopped ones resume.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/runs_lib.sh"
STAGE="${1:?usage: final_runs.sh 1|2}"
ARGS=()
export FEATURES="${FEATURES:-$DATA/features/siglip2_g2}" RXR=1
QUOTA="$(cpu_quota)"
export WORKERS="$(clamp $(( QUOTA / 4 )) 2 8)" THREADS="$(clamp $(( QUOTA / 4 )) 2 4)"
COMMON=(--eval_every 3000 --keep_every 12000)
FULL=(--grids 2 2 --pose --teacher_agree_only)
R1=(--extra dagger1 "$ROOT/data/dagger/round1/annotations.json")
R2=(--extra dagger2 "$ROOT/data/dagger/round2/annotations.json")

need() { [ -e "$1" ] || { log "missing $1: $2"; exit 1; }; }
running() { pgrep -f -- "scripts/train.py --out $RUNS/$1 " > /dev/null; }
need "$FEATURES/r2r.done" "scripts/server/features.sh"
need "$FEATURES/rxr.done" "SOURCES=rxr bash scripts/server/features.sh"
need "$FEATURES/dagger1.done" "extract dagger1.tar (scripts/server/README.md 6.4)"
need "$ROOT/data/dagger/round1/episodes.jsonl" "upload data/dagger/round1/episodes.jsonl from the lab PC (15 MB)"
need "$RUNS/teacher/teacher_probs.npy" "scripts/server/teacher_merge.sh"
if pgrep -u "$(id -un)" -f extract_teacher_probs > /dev/null; then
    log "NaVILA labels are being made (extract_teacher_probs.py) and their merge rewrites runs/teacher: stop them"
    log "(pkill -u \"\$USER\" -f extract_teacher_probs) or let them end, then rerun"
    exit 1
fi

base() {  # name, init, steps, train.py arguments
    local name="$1" init="$2" steps="$3"; shift 3
    if running "$name"; then
        log "$name: already training (started by hand), left alone"
        return
    fi
    INIT="$init" STEPS="$steps" logged "$name" "$@"
}

if [ "$STAGE" = 1 ]; then
    log "final stage 1: F_base and F_full, 36,000 steps each, R2R + RxR + DAgger round 1"
    gpu_report
    cd "$ROOT"
    base F_base "$RUNS/D/last" 36000 "${R1[@]}" "${COMMON[@]}" & p1=$!
    base F_full "$RUNS/D/last" 36000 "${FULL[@]}" --teacher "$RUNS/teacher" "${R1[@]}" "${COMMON[@]}" & p2=$!
    a=F_base b=F_full
elif [ "$STAGE" = 2 ]; then
    dagger_round 2
    need "$ROOT/data/dagger/round2/episodes.jsonl" "dagger2.tar from scripts/pack_dagger.sh 2 (it packs it)"
    for n in F_base F_full; do need "$ROOT/$n.tar.gz" "stage 1 first"; done
    log "final stage 2: F_base2 and F_full2, 12,000 steps each, + DAgger round 2"
    gpu_report
    cd "$ROOT"
    TEACH=("$RUNS/teacher")
    [ -f "$RUNS/teacher_dagger2/teacher_probs.npy" ] && TEACH+=("$RUNS/teacher_dagger2")
    base F_base2 "$RUNS/F_base/last" 12000 "${R1[@]}" "${R2[@]}" "${COMMON[@]}" & p1=$!
    base F_full2 "$RUNS/F_full/last" 12000 "${FULL[@]}" --teacher "${TEACH[@]}" "${R1[@]}" "${R2[@]}" "${COMMON[@]}" \
        & p2=$!
    a=F_base2 b=F_full2
else
    echo "usage: final_runs.sh 1|2"; exit 1
fi
bad=""
wait "$p1" || bad="$bad $a"
wait "$p2" || bad="$bad $b"
[ -z "$bad" ] || { log "not finished:$bad. Fix it, then rerun the same command."; exit 1; }
log "final stage $STAGE done; next, on the lab PC: tar -xzf <name>.tar.gz -C runs &&"
log "  SEEN=400 BASE=L0_r2r bash scripts/local_drive.sh <name>"
