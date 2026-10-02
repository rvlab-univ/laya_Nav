#!/bin/bash
# DualVLN (Qwen2.5-VL-7B System 2) vs Laya-S2 (lightweight System 2), same System 1 / episodes / metrics.
#   bash scripts/eval/bash/compare_laya_s2.sh            # R2R val-unseen (vln_r2r.yaml)
#   NPROC=4 bash scripts/eval/bash/compare_laya_s2.sh
set -e

NPROC=${NPROC:-8}
DUALVLN=${DUALVLN:-checkpoints/InternVLA-N1-DualVLN}
SYSTEM1=${SYSTEM1:-checkpoints/DualVLN-System1}
mkdir -p logs

# 1) System 1 weights out of the DualVLN checkpoint (checks that it reproduces the full model)
if [ ! -f "${SYSTEM1}/system1.pt" ]; then
    python -m internnav.model.basemodel.internvla_n1.system1_standalone --src ${DUALVLN} --out ${SYSTEM1} --verify
fi

# 2) baseline and student; both append to <output_path>/progress.json and resume if interrupted
for CFG in habitat_dual_system habitat_laya_s2; do
    torchrun --nproc_per_node=${NPROC} --master_port=2333 scripts/eval/eval.py \
        --config scripts/eval/configs/${CFG}_cfg.py > logs/${CFG}_eval.log 2>&1
done

# 3) side-by-side on the episodes both runs finished
python scripts/eval/compare_progress.py \
    DualVLN=logs/habitat/test_dual_system LayaS2=logs/habitat/test_laya_s2 | tee logs/compare_laya_s2.txt
