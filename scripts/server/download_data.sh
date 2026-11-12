#!/usr/bin/env bash
# NaVILA-Dataset: R2R and RxR (annotations and frames), ScanQA (annotations and videos, for run N8).
#   nohup bash scripts/server/download_data.sh > logs/download.log 2>&1 &
# Rerun to resume. Each archive is extracted once (a .extracted mark next to it) and then deleted, unless KEEP_TAR=1.
# Space: archives R2R 19.7 GB, RxR 23.6 GB, ScanQA 0.7 GB; about 90 GB while extracting, 45 GB after.
# The validation split is in git (splits/val_scenes.json), so no VLN-CE episode files are needed here.
source "$(dirname "$0")/common.sh"
use_env "$STUDENT_ENV"
mkdir -p "$NAVILA_DATA"
df -h "$DATA" | tail -1

fetch() {  # one file of the dataset repo into $NAVILA_DATA (resumable)
    python - "$1" "$NAVILA_DATA" <<'EOF'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("a8cheng/NaVILA-Dataset", sys.argv[1], repo_type="dataset", local_dir=sys.argv[2])
EOF
}

for f in R2R/annotations.json RxR/annotations.json \
         ScanQA/annotations/ScanQA_v1.0_train_reformat.json ScanQA/annotations/ScanQA_v1.0_val_reformat.json; do
    if [ -f "$NAVILA_DATA/$f" ]; then log "have $f"; else log "fetching $f"; fetch "$f"; fi
done

get_tar() {  # archive in the repo, folder to extract into (both relative to $NAVILA_DATA)
    local f="$1" dest="$NAVILA_DATA/$2" mark="$NAVILA_DATA/$1.extracted"
    if [ -f "$mark" ]; then log "already extracted: $f"; return; fi
    [ -f "$NAVILA_DATA/$f" ] || { log "fetching $f"; fetch "$f"; }
    # the archives are plain tar files despite the .tar.gz name
    log "extracting $f into $dest"
    mkdir -p "$dest"
    tar -xf "$NAVILA_DATA/$f" -C "$dest"
    touch "$mark"
    [ "${KEEP_TAR:-0}" = 1 ] || rm -f "$NAVILA_DATA/$f"
}
get_tar R2R/train.tar.gz R2R          # -> R2R/train/<episode>/frame_<i>.jpg
get_tar RxR/train.tar.gz RxR          # -> RxR/train/<episode>/frame_<i>.jpg
get_tar ScanQA/videos.tar.gz ScanQA   # -> ScanQA/videos/<scene>.mp4

for d in R2R/train RxR/train; do
    [ -d "$NAVILA_DATA/$d" ] && log "$d: $(ls "$NAVILA_DATA/$d" | wc -l) episode folders"
done
log "ScanQA videos: $(ls "$NAVILA_DATA/ScanQA/videos" 2>/dev/null | wc -l)"
df -h "$DATA" | tail -1
log "download done"
