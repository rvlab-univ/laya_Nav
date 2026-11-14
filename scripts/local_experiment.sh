#!/usr/bin/env bash
# One experiment on this PC (design_C_review.md, steps 2 to 5): train a D-type student from the stored features,
# then drive it on the 100 val_seen and 100 val_unseen R2R episodes every model so far was driven on
# (scripts/local_drive.sh).
#   nohup bash scripts/local_experiment.sh L1_dagger1 --extra dagger1 data/dagger/round1/annotations.json \
#       > logs/L1_dagger1.log 2>&1 &
# Arguments after the name go to scripts/train.py. The start is runs/D/last and the length 3,000 optimizer steps of
# 64 samples (INIT, STEPS): about 2.5 hours at 3 s per step, then about 20 minutes of driving. Rerun to continue:
# training resumes from its last save and finished episodes are skipped.
# Training needs the whole 16 GB GPU: not while a collection (dagger_round.sh, build_r2r_features.sh) is running.
# The same training on the server: scripts/server/experiment.sh.
set -u
NAME="${1:?run name}"; shift
STEPS="${STEPS:-3000}"
INIT="${INIT:-runs/D/last}"
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/assets/hf_cache" TOKENIZERS_PARALLELISM=false
mkdir -p logs

env/bin/python scripts/train.py --out "runs/$NAME" --features data/features/siglip2_g2 --only_available \
    --split splits/val_scenes.json --laya none --encoder answerdotai/ModernBERT-large --freeze_vision \
    --tags 10 5 --grids 2 1 --mirror 0.5 --match_frac 0.2 --match_synth 0.25 --contrast 1.0 \
    --init "$INIT" --max_steps "$STEPS" --batch 8 --grad_accum 8 --eval_every 1000 --save_every 1000 "$@" \
    || { echo "training stopped with an error"; exit 1; }
bash scripts/local_drive.sh "$NAME"
