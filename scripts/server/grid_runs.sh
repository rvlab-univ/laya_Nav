#!/usr/bin/env bash
# Step 3 (b) of design_C_review.md (5.5) on the server, from one command: the current frame as 2 x 2 or 4 x 4 tokens.
#   nohup bash scripts/server/grid_runs.sh > logs/grid_runs.log 2>&1 &
#   nohup bash scripts/server/grid_runs.sh --teacher runs/teacher > logs/grid_runs.log 2>&1 &     # if L2 won
# 1. R2R features of SigLIP2 B/16 at 256 px (16 x 16 patches, so 4 x 4 divides it), stored on a 4 x 4 grid in
#    data/features/siglip2_256_g4 (about 40-60 minutes, 31 GB), unless <features>/r2r.done is there.
# 2. G2_256, (가): the control. 2 x 2 tokens for the current frame as in D: only the encoder changes.
# 3. G4_256, (나): 4 x 4 tokens for the current frame (23 image tokens in all, D has 11).
# Both start from runs/D/last (lightnav.model.init_from: the new encoder's tower and tag table, the tag statistics set
# again, the 4 x 4 position embeddings interpolated from D's 2 x 2) and train 3,000 steps like the L runs.
# Arguments go to every scripts/train.py call: the data of the best L run. A DAgger round (--extra dagger1 ...) needs
# its frames stored by this encoder in <features>/dagger1 first (the lab PC drives it again, 5.5).
# A run whose <name>.tar.gz exists is finished and skipped; a stopped run resumes. Rerun the same command after a stop.
source "$(dirname "$0")/common.sh"
source "$(dirname "$0")/runs_lib.sh"
export VISION="${VISION:-google/siglip2-base-patch16-256}" GRID="${GRID:-4}"
export FEATURES="${FEATURES:-$DATA/features/siglip2_256_g4}"
ARGS=("$@")

for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [ "${ARGS[i]}" = --extra ]; then
        name="${ARGS[i + 1]}"
        if ! find "$FEATURES/$name" -maxdepth 1 -name '*.npy' 2>/dev/null | grep -q .; then
            echo "--extra $name: no features of $VISION in $FEATURES/$name (design_C_review.md 5.5: the round is"
            echo "driven again on the lab PC and its features uploaded)"
            exit 1
        fi
    fi
done

log "step 3 (b): $VISION features on a $GRID x $GRID grid, then G2_256 and G4_256 (${STEPS:-3000} steps each)"
if [ -f "$FEATURES/r2r.done" ]; then
    log "R2R features: complete before"
else
    bash "$HERE/features.sh"
fi
run G2_256 --vision_encoder "$VISION" --grids 2 1
run G4_256 --vision_encoder "$VISION" --grids 4 1
log "step 3 (b) done: $(cd "$ROOT" && ls -1 G2_256.tar.gz G4_256.tar.gz | tr '\n' ' ')"
log "next, on the lab PC for each: tar -xzf <name>.tar.gz -C runs && bash scripts/local_drive.sh <name>"
