#!/bin/bash
# DualVLN (Qwen2.5-VL-7B System 2) vs Laya-S2 (lightweight System 2), same System 1 / episodes / metrics.
#   bash scripts/eval/bash/compare_laya_s2.sh            # R2R val-unseen (vln_r2r.yaml)
#   EVAL_EPISODES=300 bash scripts/eval/bash/compare_laya_s2.sh   # same random 300 episodes for both models
set -e

NPROC=${NPROC:-$(python -c "import torch; print(torch.cuda.device_count())")}  # all visible GPUs
DUALVLN=${DUALVLN:-checkpoints/InternVLA-N1-DualVLN}
SYSTEM1=${SYSTEM1:-checkpoints/DualVLN-System1}
mkdir -p logs

# 1) System 1 weights out of the DualVLN checkpoint (checks that it reproduces the full model)
if [ ! -f "${SYSTEM1}/system1.pt" ]; then
    python -m internnav.model.basemodel.internvla_n1.system1_standalone --src ${DUALVLN} --out ${SYSTEM1} --verify
fi

# 2) baseline and students; each appends to <output_path>/progress.json and resumes if interrupted.
#    LayaNav (single model) is included when its checkpoint exists (LAYA_NAV, see habitat_laya_nav_cfg.py)
LAYA_NAV=${LAYA_NAV:-checkpoints/laya_nav_c2/last}
CFGS="habitat_dual_system habitat_laya_s2"
RUNS="DualVLN=logs/habitat/test_dual_system LayaS2=logs/habitat/test_laya_s2"
if [ -f "${LAYA_NAV}/laya_s2_config.json" ]; then
    CFGS="${CFGS} habitat_laya_nav"
    RUNS="${RUNS} LayaNav=logs/habitat/test_laya_nav"
fi
for CFG in ${CFGS}; do
    torchrun --nproc_per_node=${NPROC} --master_port=2333 scripts/eval/eval.py \
        --config scripts/eval/configs/${CFG}_cfg.py > logs/${CFG}_eval.log 2>&1
done

# 3) side-by-side on the episodes all runs finished
python scripts/eval/compare_progress.py ${RUNS} | tee logs/compare_laya_s2.txt
