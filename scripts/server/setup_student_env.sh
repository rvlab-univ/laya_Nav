#!/usr/bin/env bash
# Student environment (conda env "lightnav"): Python 3.11, torch 2.7.1, transformers 4.51.0, laya 0.4.31.
#   bash scripts/server/setup_student_env.sh > logs/setup_student.log 2>&1
# TORCH_CUDA picks the PyTorch wheel (default cu126; cu128 needs a driver for CUDA 12.8). Safe to rerun.
source "$(dirname "$0")/common.sh"
TORCH_CUDA="${TORCH_CUDA:-cu126}"

git -C "$ROOT" submodule update --init third_party/laya_src NaVILA
conda_hook
has_env "$STUDENT_ENV" || conda create -y -n "$STUDENT_ENV" python=3.11
conda activate "$STUDENT_ENV"

python -m pip install --upgrade pip
pip install torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
pip install -r "$ROOT/requirements/student.txt"
pip install --no-deps "$ROOT/third_party/laya_src"

python - <<'EOF'
import torch, transformers, laya
cuda = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", cuda, "|", torch.cuda.get_device_name(0) if cuda else "-")
print("transformers", transformers.__version__, "| laya", getattr(laya, "__version__", "?"))
EOF
log "student env ready: conda activate $STUDENT_ENV"
