#!/usr/bin/env bash
# Teacher environment (conda env "navila") and the NaVILA 8B checkpoint. NaVILA's own install
# (NaVILA/environment_setup.sh, README "Evaluation"), without Habitat and the [eval] extras: only the model runs here.
#   bash scripts/server/setup_teacher_env.sh > logs/setup_teacher.log 2>&1
# Python 3.10, torch 2.3.0 (cu121), flash-attn 2.5.8, transformers 4.37.2 with NaVILA's patched files,
# deepspeed 0.9.5 (imported by NaVILA's model code). The checkpoint (about 17 GB) goes to $TEACHER_CKPT.
source "$(dirname "$0")/common.sh"

git -C "$ROOT" submodule update --init NaVILA
conda_hook
has_env "$TEACHER_ENV" || conda create -y -n "$TEACHER_ENV" python=3.10
conda activate "$TEACHER_ENV"

python -m pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
(cd "$ROOT/NaVILA" && pip install -e . && pip install -e ".[train]")
pip install transformers==4.37.2
site_pkg="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -r "$ROOT/NaVILA/llava/train/transformers_replace/"* "$site_pkg/transformers/"
cp -r "$ROOT/NaVILA/llava/train/deepspeed_replace/"* "$site_pkg/deepspeed/"

python - "$TEACHER_CKPT" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("a8cheng/navila-llama3-8b-8f", local_dir=sys.argv[1])
EOF

python - <<'EOF'
import torch, transformers, flash_attn
from llava.model.builder import load_pretrained_model  # noqa: F401  (the import NaVILA's evaluation needs)
cuda = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", cuda, "|", torch.cuda.get_device_name(0) if cuda else "-")
print("transformers", transformers.__version__, "| flash_attn", flash_attn.__version__)
EOF
log "teacher env ready: conda activate $TEACHER_ENV, checkpoint at $TEACHER_CKPT"
