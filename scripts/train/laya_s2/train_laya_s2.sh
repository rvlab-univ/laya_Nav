#!/bin/bash
# Laya-S2: distill the DualVLN System 2 (Qwen2.5-VL-7B) into a lightweight decision model.
# Step 1 extracts teacher System 1 conditions once; step 2 trains the student.
set -e

NPROC=${NPROC:-$(python -c "import torch; print(torch.cuda.device_count())")}  # all visible GPUs
TEACHER=${TEACHER:-checkpoints/InternVLA-N1-DualVLN}
# dataset registry names (train_dual_system.sh uses all of
#   r2r_125cm_0_30,r2r_60cm_15_15,rxr_125cm_0_30,rxr_60cm_15_15,scalevln_125cm_0_30,scalevln_60cm_30_30).
# Default: the setting that matches the Habitat R2R eval camera (1.25 m, look-down 30 deg).
VLN_DATASETS=${VLN_DATASETS:-r2r_125cm_0_30}
BATCH=${BATCH:-32}  # per GPU; a single H200 (141 GB) should fit 64
LATENTS=${LATENTS:-data/laya_s2/teacher_latents}
OUT=${OUT:-checkpoints/laya_s2}

if [ ! -f "${LATENTS}/latents.npy" ]; then
    torchrun --nproc_per_node=${NPROC} scripts/train/laya_s2/extract_teacher_latents.py \
        --teacher_path ${TEACHER} \
        --vln_dataset_use ${VLN_DATASETS} \
        --out_dir ${LATENTS}
fi

torchrun --nproc_per_node=${NPROC} scripts/train/laya_s2/train_laya_s2.py \
    --vln_dataset_use ${VLN_DATASETS} \
    --teacher_latents ${LATENTS} \
    --output_dir ${OUT} \
    --epochs 3 \
    --batch_size ${BATCH} \
    "$@"
