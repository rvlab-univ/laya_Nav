#!/bin/bash
# Laya-S2: distill the DualVLN System 2 (Qwen2.5-VL-7B) into a lightweight decision model.
# Step 1 extracts teacher System 1 conditions once; step 2 trains the student.
set -e

NPROC=${NPROC:-8}
TEACHER=${TEACHER:-checkpoints/InternVLA-N1-DualVLN}
# same dataset names as train_dual_system.sh; the student also uses turn / stop samples
VLN_DATASETS=${VLN_DATASETS:-r2r_125cm_0_30,r2r_60cm_15_15,rxr_125cm_0_30,rxr_60cm_15_15,scalevln_125cm_0_30,scalevln_60cm_30_30}
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
    --batch_size 32 \
    "$@"
