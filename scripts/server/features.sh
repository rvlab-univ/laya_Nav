#!/usr/bin/env bash
# Frame features of the frozen SigLIP2, from NaVILA's frames on the server (design_C_review.md, step 1). With them
# scripts/server/experiment.sh trains D-type students without running the vision encoder.
#   nohup bash scripts/server/features.sh > logs/features.log 2>&1 &                      # R2R: 601k frames, 9 GB
#   SOURCES="r2r rxr" nohup bash scripts/server/features.sh > logs/features.log 2>&1 &    # + RxR: 835k frames, 13 GB
# The speed is set by JPEG decoding on the CPU (260 frames/s with 8 workers on the lab PC: R2R in 40 minutes).
# Rerun after a stop: stored episodes are skipped. NaVILA's frames are kept. A source is marked complete
# (<features>/<source>.done) only when every episode of its annotations is stored.
source "$(dirname "$0")/common.sh"
use_env "$STUDENT_ENV"
limit_threads "${THREADS:-4}"
FEATURES="${FEATURES:-$DATA/features/siglip2_g2}"
QUOTA="$(cpu_quota)"
W=$(( QUOTA - 2 )); [ "$W" -gt 16 ] && W=16; [ "$W" -lt 2 ] && W=2
gpu_report
cd "$ROOT"
for s in ${SOURCES:-r2r}; do
    case "$s" in
        r2r) dir=R2R ;;
        rxr) dir=RxR ;;
        *) echo "unknown source: $s (r2r, rxr)"; exit 1 ;;
    esac
    log "$s: features of $NAVILA_DATA/$dir/train -> $FEATURES/$s ($W workers)"
    python scripts/cache_features.py --source "$s" --ann "$NAVILA_DATA/$dir/annotations.json" \
        --frames "$NAVILA_DATA/$dir/train" --out "$FEATURES" --workers "${WORKERS:-$W}"
    python - "$s" "$NAVILA_DATA/$dir/annotations.json" "$FEATURES" <<'PY'
import sys
sys.path.insert(0, ".")
from lightnav import data as D
from lightnav.features import FeatureStore
source, ann, root = sys.argv[1:]
need = {}
for r in D.load_rows(ann, source):
    need[r.prefix] = max(need.get(r.prefix, 0), r.n_frames)
store = FeatureStore(root, source)
missing = [p for p, n in need.items() if not store.has(p, n)]
print("%s: %d of %d episodes stored" % (source, len(need) - len(missing), len(need)))
if missing:
    sys.exit("%s: %d episodes are missing, e.g. %s" % (source, len(missing), missing[:3]))
open("%s/%s.done" % (root, source), "w").close()
PY
    log "$s: complete"
done
du -sh "$FEATURES"
