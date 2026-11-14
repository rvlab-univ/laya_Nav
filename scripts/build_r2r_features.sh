#!/usr/bin/env bash
# All R2R training frames as stored features on this PC (design_C_review.md, step 1): Habitat renders NaVILA's
# episodes in chunks, the frozen SigLIP2 turns each chunk into features, and the chunk's images are deleted.
# About 4.5 hours for the 10,819 episodes and 9 GB in data/features/siglip2_g2. Stop and rerun at any time.
#   nohup bash scripts/build_r2r_features.sh > logs/build_r2r_features.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/assets/hf_cache"
OUT="${OUT:-data/features/siglip2_g2}"
FRAMES="${FRAMES:-data/collected/r2r_train}"
CHUNK="${CHUNK:-500}"
ANN=data/NaVILA-Dataset/R2R/annotations.json
while true; do
    env_vlnce/bin/python scripts/collect_vlnce.py --mode replay --out "$FRAMES" --limit "$CHUNK" \
        --skip_features "$OUT/r2r" 2>&1 | grep -E "episodes|differs|Error|Traceback"
    code=${PIPESTATUS[0]}
    env/bin/python scripts/cache_features.py --source r2r --ann "$ANN" --frames "$FRAMES" --out "$OUT" \
        --delete_frames 2>&1 | grep -E "episodes|Error|Traceback"
    [ "$code" = 3 ] && break
    [ "$code" != 0 ] && { echo "rendering stopped with code $code"; exit 1; }
done
echo "done: $(ls "$OUT/r2r" | wc -l) episodes in $OUT/r2r"
