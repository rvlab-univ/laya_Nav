# Sourced after common.sh by l_runs.sh, grid_runs.sh and all_runs.sh: one experiment that is skipped once finished,
# DAgger round 1 from the lab PC, and waiting for a file. ARGS (the calling script's arguments) go to every run.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {  # name, train.py arguments
    local name="$1"; shift
    if [ -f "$ROOT/$name.tar.gz" ]; then
        log "$name: finished before ($name.tar.gz is there), skipped"
        return
    fi
    bash "$HERE/experiment.sh" "$name" "$@" "${ARGS[@]}"
}

# Waits for dagger1.tar and dagger1.tar.sha256 in ~/light_nav (from the lab PC: scripts/pack_dagger.sh 1) until the
# checksum matches (so the upload can go on meanwhile; at most WAIT_HOURS, default 12), then extracts the round into
# data/. Needs FEATURES: the 224 px store the round's features belong to.
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

wait_for() {  # file, what it is, file that means it will never come (optional): at most WAIT_HOURS (default 12)
    local file="$1" what="$2" fail="${3:-}" waited=0 limit=$(( ${WAIT_HOURS:-12} * 3600 )) poll="${POLL:-300}"
    [ -f "$file" ] && return
    log "waiting for $what"
    while [ ! -f "$file" ]; do
        if [ -n "$fail" ] && [ -f "$fail" ]; then
            log "$what failed: not waiting any longer"
            exit 1
        fi
        if [ "$waited" -ge "$limit" ]; then
            log "no $what after ${WAIT_HOURS:-12} h"
            exit 1
        fi
        sleep "$poll"
        waited=$(( waited + poll ))
    done
}
