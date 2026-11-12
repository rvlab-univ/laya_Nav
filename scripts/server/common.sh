# Sourced by the other scripts in this folder: paths, conda, logging.
# Every path can be overridden from the environment, e.g. DATA=/scratch/data bash scripts/server/download_data.sh
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA:-$ROOT/data}"
NAVILA_DATA="$DATA/NaVILA-Dataset"
ASSETS="${ASSETS:-$ROOT/assets}"
export HF_HOME="${HF_HOME:-$ASSETS/hf_cache}"
TEACHER_CKPT="${TEACHER_CKPT:-$ASSETS/navila-llama3-8b-8f}"
STUDENT_ENV="${STUDENT_ENV:-lightnav}"
TEACHER_ENV="${TEACHER_ENV:-navila}"
LOGS="$ROOT/logs"
RUNS="$ROOT/runs"
SPLIT="$ROOT/splits/val_scenes.json"
mkdir -p "$LOGS" "$RUNS" "$ASSETS"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

conda_hook() {
    if ! command -v conda >/dev/null 2>&1; then
        for c in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda"; do
            [ -x "$c" ] && { export PATH="$(dirname "$c"):$PATH"; break; }
        done
    fi
    command -v conda >/dev/null 2>&1 || { echo "conda not found on PATH"; exit 1; }
    eval "$(conda shell.bash hook)"
}

has_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }

use_env() {
    conda_hook
    has_env "$1" || { echo "conda env '$1' is missing: run the setup script first"; exit 1; }
    conda activate "$1"
}

gpu_report() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
        || echo "nvidia-smi not available"
}
