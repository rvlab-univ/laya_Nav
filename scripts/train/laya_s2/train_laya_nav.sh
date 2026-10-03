#!/bin/bash
# LayaNav on the server: C1 (trajectory head on a frozen Laya-S2) -> C2 (everything), each followed by the
# trajectory diagnostics (eval_traj.py) on the held-out R2R scenes.
#   bash scripts/train/laya_s2/train_laya_nav.sh
#   VLN_DATASETS=r2r_125cm_0_30,rxr_125cm_0_30 TAG=laya_nav_r2r_rxr bash scripts/train/laya_s2/train_laya_nav.sh
#   bash scripts/train/laya_s2/train_laya_nav.sh --epochs 1      # extra arguments go to both stages
# Data: DATASETS="r2r rxr scalevln" bash scripts/setup/setup_laya_s2_server.sh train_data
# Interrupted runs resume from <output_dir>/last when started again with the same TAG.
set -eo pipefail

NPROC=${NPROC:-$(python -c "import torch; print(torch.cuda.device_count())")}  # all visible GPUs
# Registry names (internvla_n1_lerobot_dataset.py); name%30 keeps a fixed random 30% of that dataset.
# Default: every dataset with the camera of the Habitat R2R evaluation (1.25 m, look-down 30 deg). The 60 cm
# settings are other robots; there is no camera input to tell them apart, so they are left out.
VLN_DATASETS=${VLN_DATASETS:-r2r_125cm_0_30,rxr_125cm_0_30,scalevln_125cm_0_30}
EVAL_DATASETS=${EVAL_DATASETS:-r2r_125cm_0_30}  # diagnostics on R2R, comparable across runs
INIT=${INIT:-checkpoints/laya_s2/last}          # trained Laya-S2
TAG=${TAG:-laya_nav_mix}
BATCH=${BATCH:-64}                              # per GPU
WORKERS=${WORKERS:-16}
VAL_RATIO=${VAL_RATIO:-0.05}                    # held-out fraction of scenes, same for training and diagnostics
LATENTS=${LATENTS:-data/laya_s2/teacher_latents}  # optional in C2: keeps the latent distillation where keys match

REGISTRY="from internnav.dataset.internvla_n1_lerobot_dataset import data_list"
for name in ${VLN_DATASETS//,/ } ${EVAL_DATASETS//,/ }; do
    # last line: importing internnav prints its root path first
    path=$(python -c "${REGISTRY}; print(data_list(['${name}'])[0]['data_path'])" | tail -1)
    if [ ! -d "${path}" ]; then
        echo "${name}: ${path} not found. Download it first, e.g.:" >&2
        echo "  DATASETS=\"$(basename "${path}")\" bash scripts/setup/setup_laya_s2_server.sh train_data" >&2
        exit 1
    fi
done
[ -f "${INIT}/laya_s2_config.json" ] || { echo "Laya-S2 checkpoint ${INIT} not found (INIT=...)" >&2; exit 1; }
C2_LATENTS=()
[ -f "${LATENTS}/latents.npy" ] && C2_LATENTS=(--teacher_latents "${LATENTS}")

mkdir -p logs
train() {
    torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT:-29531} scripts/train/laya_s2/train_laya_nav.py \
        --vln_dataset_use ${VLN_DATASETS} --val_ratio ${VAL_RATIO} --batch_size ${BATCH} --num_workers ${WORKERS} "$@"
}
diagnose() {  # $1 checkpoint, $2 log name
    python scripts/train/laya_s2/eval_traj.py --ckpt "$1" --vln_dataset_use ${EVAL_DATASETS} --val_ratio ${VAL_RATIO} \
        --own_goal --num_workers ${WORKERS} --out logs/$2_eval_traj.json 2>&1 | tee logs/$2_eval_traj.log
}

# the first C1 run (r2r only, old head) as the reference, once
if [ -f checkpoints/laya_nav_c1/last/laya_s2_config.json ] && [ ! -f logs/laya_nav_c1_eval_traj.json ]; then
    diagnose checkpoints/laya_nav_c1/last laya_nav_c1
fi

train --stage c1 --init_from ${INIT} --output_dir checkpoints/${TAG}_c1 "$@" 2>&1 | tee -a logs/${TAG}_c1.log
diagnose checkpoints/${TAG}_c1/last ${TAG}_c1

train --stage c2 --init_from checkpoints/${TAG}_c1/last --output_dir checkpoints/${TAG}_c2 "${C2_LATENTS[@]}" "$@" \
    2>&1 | tee -a logs/${TAG}_c2.log
diagnose checkpoints/${TAG}_c2/last ${TAG}_c2

echo "Habitat comparison:"
echo "  LAYA_NAV=checkpoints/${TAG}_c2/last EVAL_EPISODES=300 bash scripts/eval/bash/compare_laya_s2.sh"
