#!/usr/bin/env bash
# The three L experiments of design_C_review.md 5.2 on the server, one after another, from one command:
#   nohup bash scripts/server/l_runs.sh > logs/l_runs.log 2>&1 &
# 1. R2R features (features.sh), unless <features>/r2r.done is there.
# 2. Teacher labels merged (teacher_merge.sh), unless runs/teacher/teacher_probs.npy is there.
# 3. L0_r2r (baseline), L2_teacher (--teacher runs/teacher), L1_dagger1 (--extra dagger1 ...), in this order.
#    A run whose <name>.tar.gz exists is finished and skipped; a stopped run resumes from runs/<name>/last.
# L1 needs DAgger round 1 from the lab PC: dagger1.tar and dagger1.tar.sha256 in ~/light_nav. Before L1 this waits
# until both are there and the checksum matches (so the upload can go on during L0 and L2; at most WAIT_HOURS,
# default 12), then extracts the round into data/.
# Rerun the same command after a stop. Arguments go to every scripts/train.py call. STEPS (3000), BATCH, ACCUM and
# INIT work as in experiment.sh.
source "$(dirname "$0")/common.sh"
HERE="$(cd "$(dirname "$0")" && pwd)"
export FEATURES="${FEATURES:-$DATA/features/siglip2_g2}"
ARGS=("$@")

run() {  # name, train.py arguments
    local name="$1"; shift
    if [ -f "$ROOT/$name.tar.gz" ]; then
        log "$name: finished before ($name.tar.gz is there), skipped"
        return
    fi
    bash "$HERE/experiment.sh" "$name" "$@" "${ARGS[@]}"
}

dagger_round1() {
    local tarf="$ROOT/dagger1.tar" waited=0 limit=$(( ${WAIT_HOURS:-12} * 3600 )) poll="${POLL:-300}" said=""
    [ -f "$FEATURES/dagger1.done" ] && return
    while :; do
        if [ -f "$tarf" ] && [ -f "$tarf.sha256" ]; then
            if (cd "$ROOT" && sha256sum -c --status dagger1.tar.sha256); then
                break
            fi
            [ "$said" = mismatch ] || log "dagger1.tar does not match dagger1.tar.sha256 yet: still uploading, or a broken copy (upload it again)"
            said=mismatch
        else
            [ "$said" = missing ] || log "L1 waits for dagger1.tar and dagger1.tar.sha256 in $ROOT (from the lab PC)"
            said=missing
        fi
        if [ "$waited" -ge "$limit" ]; then
            log "no complete dagger1.tar after ${WAIT_HOURS:-12} h: L1 not run. Upload it, then rerun this script."
            exit 1
        fi
        sleep "$poll"
        waited=$(( waited + poll ))
    done
    log "dagger1.tar checksum ok: extracting"
    tar -xf "$tarf" -C "$ROOT"
    local want have
    want="$(grep -o '"episodes": *[0-9]*' "$ROOT/data/dagger/round1/summary.json" | grep -o '[0-9]*$')"
    have="$(find "$FEATURES/dagger1" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)"
    if [ "$have" != "$want" ]; then
        echo "dagger1: $have episode features in $FEATURES/dagger1, the round has $want (extracted under $ROOT/data)"
        exit 1
    fi
    touch "$FEATURES/dagger1.done"
    log "dagger1: $have episodes ready (dagger1.tar can be deleted)"
}

log "L runs: features, teacher labels, then L0_r2r, L2_teacher, L1_dagger1 (${STEPS:-3000} steps each)"
if [ -f "$FEATURES/r2r.done" ]; then
    log "R2R features: complete before"
else
    bash "$HERE/features.sh"
fi
if [ -f "$RUNS/teacher/teacher_probs.npy" ]; then
    log "teacher labels: merged before"
else
    bash "$HERE/teacher_merge.sh"
fi
run L0_r2r
run L2_teacher --teacher "$RUNS/teacher"
dagger_round1
run L1_dagger1 --extra dagger1 "$ROOT/data/dagger/round1/annotations.json"
log "L runs done: $(cd "$ROOT" && ls -1 L0_r2r.tar.gz L2_teacher.tar.gz L1_dagger1.tar.gz | tr '\n' ' ')"
log "next, on the lab PC for each: tar -xzf <name>.tar.gz -C runs && bash scripts/local_drive.sh <name>"
