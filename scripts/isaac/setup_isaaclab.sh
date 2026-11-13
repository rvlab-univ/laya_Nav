#!/usr/bin/env bash
# Isaac Lab v2.3.2 on top of the existing Isaac Sim 5.1 install (/mnt/storage/isaacsim_env), for the Go2 tests.
#   bash scripts/isaac/setup_isaaclab.sh > logs/setup_isaaclab.log 2>&1
# A venv with --system-site-packages sees Isaac Sim, while everything Isaac Lab adds (its packages, torch 2.7.0 cu128,
# which the RTX 5060 Ti needs: the Isaac Sim env has a cu126 build without sm_120) goes into the venv. The Isaac Sim
# env used by the KIMM / AI Worker setup and the base conda env are not changed. Safe to rerun.
#
# isaaclab.sh picks ${CONDA_PREFIX}/bin/python before ${VIRTUAL_ENV}/bin/python, so any active conda env must be
# cleared first (on 2026-11-13 a run with the base conda active installed into /home/lab/miniconda3; that was undone).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACSIM_PY="${ISAACSIM_PY:-/mnt/storage/isaacsim_env/isaac_sim/bin/python}"
VENV="${VENV:-$ROOT/env_isaaclab}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/mnt/storage/IsaacLab}"
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.3.2}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"   # keep downloads off the small system disk

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
[ -x "$VENV/bin/python" ] || "$ISAACSIM_PY" -m venv --system-site-packages "$VENV"
[ -d "$ISAACLAB_DIR/.git" ] || git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
git -C "$ISAACLAB_DIR" log --oneline -1

source "$VENV/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
chosen="$(cd "$ISAACLAB_DIR" && ./isaaclab.sh -p -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1)"
if [ "$chosen" != "$VENV/bin/python" ]; then
    echo "isaaclab.sh would use $chosen instead of $VENV/bin/python: stopping"; exit 1
fi
# flatdict 4.0.1 (an Isaac Lab dependency) has a setup.py that needs pkg_resources, gone from new setuptools
python -m pip install -q "setuptools<81" wheel
python -m pip install -q --no-build-isolation "flatdict==4.0.1"
cd "$ISAACLAB_DIR"
./isaaclab.sh --install none || echo "(isaaclab.sh reported an error; the VS Code settings step fails without _isaac_sim and can be ignored)"
cd "$ROOT"
# rsl_rl with a tensordict that works with torch 2.7 (the latest tensordict asks for torch >= 2.11 and would pull it in)
python -m pip install -q "rsl-rl-lib==3.1.2" "tensordict==0.8.3" "torch==2.7.0" "torchvision==0.22.0" \
    --extra-index-url https://download.pytorch.org/whl/cu128
# torch 2.7.0 cu128 with its CUDA libraries inside the venv: otherwise pip counts the Isaac Sim env's cu126 libraries as
# installed and torch fails to load (libcusparseLt.so.0 missing)
python -m pip install -q --ignore-installed "torch==2.7.0" "torchvision==0.22.0" --index-url https://download.pytorch.org/whl/cu128
# --ignore-installed also brings the newest numpy / pillow; Isaac Sim pins numpy 1.26.0, Isaac Lab pillow 11.3.0
python -m pip install -q --no-deps "numpy==1.26.0" "pillow==11.3.0"

python - <<'EOF'
import importlib.metadata as md
import sys
print("python", sys.executable)
for p in ("isaacsim", "isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "rsl-rl-lib", "torch"):
    try:
        print("%-16s %s" % (p, md.version(p)))
    except md.PackageNotFoundError:
        print("%-16s MISSING" % p)
import torch
print("torch file", torch.__file__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), "| arch list", torch.cuda.get_arch_list()[-3:])
EOF
echo "isaaclab ready: source $VENV/bin/activate; cd $ISAACLAB_DIR"
