# Sourced after common.sh by l_runs.sh, grid_runs.sh, all_runs.sh and h_runs.sh: one experiment that is skipped once
# finished (alone, or logged to logs/<name>.log), a DAgger round from the lab PC, and waiting for a file.
# ARGS (the calling script's arguments) go to every run.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {  # name, train.py arguments
    local name="$1"; shift
    if [ -f "$ROOT/$name.tar.gz" ]; then
        log "$name: finished before ($name.tar.gz is there), skipped"
        return
    fi
    bash "$HERE/experiment.sh" "$name" "$@" "${ARGS[@]}"
}

logged() {  # name, train.py arguments: the run, its output appended to logs/<name>.log
    local name="$1"
    if [ -f "$ROOT/$name.tar.gz" ]; then
        log "$name: finished before ($name.tar.gz is there), skipped"
        return
    fi
    log "$name: started (logs/$name.log)"
    if run "$@" >> "$LOGS/$name.log" 2>&1; then
        log "$name: finished, $name.tar.gz"
    else
        log "$name: stopped with an error, see the end of logs/$name.log"
        return 1
    fi
}

clamp() { local v="$1"; [ "$v" -lt "$2" ] && v="$2"; [ "$v" -gt "$3" ] && v="$3"; echo "$v"; }

# Waits for dagger<n>.tar and dagger<n>.tar.sha256 in ~/light_nav (from the lab PC: scripts/pack_dagger.sh <n>) until
# the checksum matches (so the upload can go on meanwhile; at most WAIT_HOURS, default 12), then extracts the round
# into data/. Needs FEATURES: the 224 px store the round's features belong to.
dagger_round() {
    local n="$1" tarf="$ROOT/dagger$1.tar" waited=0 limit=$(( ${WAIT_HOURS:-12} * 3600 )) poll="${POLL:-300}" said=""
    [ -f "$FEATURES/dagger$n.done" ] && return
    while :; do
        if [ -f "$tarf" ] && [ -f "$tarf.sha256" ]; then
            if (cd "$ROOT" && sha256sum -c --status "dagger$n.tar.sha256"); then
                break
            fi
            [ "$said" = mismatch ] \
                || log "dagger$n.tar does not match its .sha256 yet: still uploading, or a broken copy (upload it again)"
            said=mismatch
        else
            [ "$said" = missing ] || log "waiting for dagger$n.tar and dagger$n.tar.sha256 in $ROOT (from the lab PC)"
            said=missing
        fi
        if [ "$waited" -ge "$limit" ]; then
            log "no complete dagger$n.tar after ${WAIT_HOURS:-12} h. Upload it, then rerun this script."
            exit 1
        fi
        sleep "$poll"
        waited=$(( waited + poll ))
    done
    log "dagger$n.tar checksum ok: extracting"
    tar -xf "$tarf" -C "$ROOT"
    local want have
    want="$(grep -o '"episodes": *[0-9]*' "$ROOT/data/dagger/round$n/summary.json" | grep -o '[0-9]*$')"
    have="$(find "$FEATURES/dagger$n" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)"
    if [ "$have" != "$want" ]; then
        echo "dagger$n: $have episode features in $FEATURES/dagger$n, the round has $want (extracted under $ROOT/data)"
        exit 1
    fi
    touch "$FEATURES/dagger$n.done"
    log "dagger$n: $have episodes ready (dagger$n.tar can be deleted)"
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
