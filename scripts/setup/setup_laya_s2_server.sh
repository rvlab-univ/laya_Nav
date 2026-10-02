#!/bin/bash
# Server setup for Laya-S2 (environment, checkpoints, data). Run from the repo root:
#   bash scripts/setup/setup_laya_s2_server.sh env          # conda env (Habitat + training deps)
#   conda activate laya
#   huggingface-cli login                                     # Scene-N1 / InternData-N1 may require accepting terms
#   bash scripts/setup/setup_laya_s2_server.sh ckpt         # DualVLN + DepthAnything v2 (~17 GB)
#   bash scripts/setup/setup_laya_s2_server.sh eval_data    # R2R VLN-CE episodes + mp3d_ce scenes (~16 GB)
#   DATASETS="r2r" bash scripts/setup/setup_laya_s2_server.sh train_data
#       compressed sizes: r2r 334 GB, rxr 911 GB, scalevln 1.3 TB (extracted size is similar; tars are
#       deleted after extraction unless KEEP_TAR=1)
#   bash scripts/setup/setup_laya_s2_server.sh check
set -e

ENV=${ENV:-laya}
DATASETS=${DATASETS:-r2r}
KEEP_TAR=${KEEP_TAR:-0}

setup_env() {
    # InternNav "Install with Habitat Environment": python 3.9, habitat-sim/lab 0.2.4, torch 2.6 (cu124)
    eval "$(conda shell.bash hook)"
    conda create -y -n ${ENV} python=3.9
    conda activate ${ENV}
    conda install -y habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat
    # C compiler for Triton's runtime build (flash-attn rotary kernels in Qwen2.5-VL); servers often lack gcc
    conda install -y -c conda-forge gcc
    conda env config vars set CC=${CONDA_PREFIX}/bin/gcc
    if [ ! -d third_party/habitat-lab ]; then
        git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
    fi
    pip install -e third_party/habitat-lab/habitat-lab -e third_party/habitat-lab/habitat-baselines
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    git submodule update --init --recursive
    # prebuilt flash-attn (building it needs nvcc / CUDA_HOME, which servers often lack)
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
    pip install -e .[habitat]
    # training data loading (lerobot parquet) and downloads
    pip install pandas pyarrow "huggingface_hub[cli]"
}

require_hf_login() {
    # InternData-N1 / Scene-N1 are gated: the token must be set in *this* shell
    if ! huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli whoami 2>&1 | grep -qi "not logged in"; then
        echo "Hugging Face login missing in this shell. Run:  read -s HF_TOKEN && export HF_TOKEN" >&2
        exit 1
    fi
}

download_ckpt() {
    huggingface-cli download InternRobotics/InternVLA-N1-DualVLN --local-dir checkpoints/InternVLA-N1-DualVLN
    wget -nc -P checkpoints \
        https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth
}

download_eval_data() {
    require_hf_login
    huggingface-cli download InternRobotics/InternData-N1 --repo-type dataset \
        --include "vln_ce/raw_data/r2r/*" --local-dir data
    huggingface-cli download InternRobotics/Scene-N1 --repo-type dataset \
        --include "mp3d_ce.tar.gz" --local-dir data/scene_data
    tar -xzf data/scene_data/mp3d_ce.tar.gz -C data/scene_data
    [ "${KEEP_TAR}" = 1 ] || rm -f data/scene_data/mp3d_ce.tar.gz
}

download_train_data() {
    require_hf_login
    for d in ${DATASETS}; do
        huggingface-cli download InternRobotics/InternData-N1 --repo-type dataset \
            --include "vln_ce/traj_data/${d}/*" --local-dir data
        for t in data/vln_ce/traj_data/${d}/*.tar.gz; do
            [ -e "$t" ] || continue
            tar -xzf "$t" -C data/vln_ce/traj_data/${d}
            [ "${KEEP_TAR}" = 1 ] || rm -f "$t"
        done
    done
    # dataset registry paths are relative to the repo root ("traj_data/r2r", ...)
    [ -e traj_data ] || ln -s data/vln_ce/traj_data traj_data
}

check() {
    echo "checkpoints:"; ls checkpoints/InternVLA-N1-DualVLN/config.json checkpoints/depth_anything_v2_metric_hypersim_vits.pth
    echo "eval episodes:"; ls data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz
    echo "mp3d_ce scenes: $(ls -d data/scene_data/mp3d_ce/mp3d/*/ 2>/dev/null | wc -l) (expected under data/scene_data/mp3d_ce/mp3d/<scan>/)"
    for d in r2r rxr scalevln; do
        [ -d traj_data/${d} ] && echo "traj_data/${d}: $(ls traj_data/${d}/*/meta/episodes.jsonl 2>/dev/null | wc -l) scenes with meta/episodes.jsonl"
    done
    python -c "import habitat, habitat_sim, torch, transformers, diffusers, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| diffusers', diffusers.__version__)"
}

case "$1" in
    env) setup_env ;;
    ckpt) download_ckpt ;;
    eval_data) download_eval_data ;;
    train_data) download_train_data ;;
    check) check ;;
    *) sed -n 2,13p "$0"; exit 1 ;;
esac
