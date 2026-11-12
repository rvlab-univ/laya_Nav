#!/usr/bin/env bash
# VLN-CE evaluation environment on the lab PC: Python 3.8, habitat-sim 0.1.7 (headless, prebuilt), habitat-lab
# v0.1.7 (submodule third_party/habitat-lab), NaVILA's habitat_extensions and its dependencies, CPU torch.
#   bash scripts/eval/setup_vlnce_env.sh > logs/setup_vlnce.log 2>&1
# The student model does not run here (scripts/eval_vlnce.py asks lightnav/server.py over HTTP), so torch is CPU only.
# ENV (default: env_vlnce in the repo) and CONDA_PKGS_DIRS can be overridden. Safe to rerun.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${ENV:-$ROOT/env_vlnce}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/mnt/storage/laya_work/conda_pkgs}"   # keep the small system disk free

eval "$(conda shell.bash hook)"
git -C "$ROOT" submodule update --init third_party/habitat-lab NaVILA
[ -x "$ENV/bin/python" ] || conda create -y -q -p "$ENV" python=3.8
python_has() { "$ENV/bin/python" -c "import $1" 2>/dev/null; }
python_has habitat_sim || conda install -y -q -p "$ENV" -c aihabitat -c conda-forge habitat-sim=0.1.7 headless

PIP=("$ENV/bin/python" -m pip install -q)
# habitat-sim 0.1.7 uses np.float, removed in numpy 1.24
"${PIP[@]}" "numpy==1.23.5"
"${PIP[@]}" torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
# habitat-lab v0.1.7 requirements (headless OpenCV), VLN-CE requirements (NaVILA/evaluation/requirements.txt) without
# tensorflow: tensorboard is enough for the TensorboardWriter that habitat_extensions imports
"${PIP[@]}" "gym==0.17.3" "yacs>=0.1.5" "numpy-quaternion==2022.4.3" "attrs>=19.1.0" "opencv-python-headless==4.8.0.76" \
    "imageio>=2.2.0" "imageio-ffmpeg>=0.2.0" "scipy<1.11" "tqdm" "numba<0.59" \
    "dtw==1.4.0" "fastdtw==0.3.4" jsonlines "lmdb==1.4.1" "webdataset==0.1.103" msgpack_numpy networkx ifcfg "moviepy==1.0.3" "tensorboard<2.15" "pillow<11" "numpy==1.23.5"
# habitat and habitat_baselines straight from the submodule (setup.py's --all is not available through pip)
site="$("$ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$ROOT/third_party/habitat-lab" > "$site/habitat_lab_v017.pth"

"$ENV/bin/python" - "$ROOT" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1] + "/NaVILA/evaluation")
import numpy, torch, habitat, habitat_sim, habitat_baselines
import habitat_extensions  # registers the VLN-CE task, measures and datasets
from habitat_extensions.task import VLNCEDatasetV1  # noqa: F401
print("numpy", numpy.__version__, "| torch", torch.__version__, "| habitat-sim", habitat_sim.__version__,
      "| habitat", getattr(habitat, "__version__", "0.1.7"))
EOF
echo "vlnce env ready: $ENV"
