#!/bin/bash
#SBATCH --job-name=safety_clora
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# --- HuggingFace caches to per-job local scratch ---
export HF_HOME="${TMPDIR:-/var/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# --- Activate env (installed in persistent scratch) ---
source "/insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh"
conda activate "/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_pip"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Choose one:
#   baseline_lora | clora | safety_clora
PIPELINE="${1:-baseline_lora}"

case "$PIPELINE" in
  baseline_lora)
    python -m safety_clora.scripts.run_baseline_lora
    ;;
  clora)
    python -m safety_clora.scripts.run_clora
    ;;
  safety_clora)
    python -m safety_clora.scripts.run_safety_clora
    ;;
  *)
    echo "Unknown pipeline: $PIPELINE"
    echo "Usage: sbatch slurm_run_pipeline.sbatch [baseline_lora|clora|safety_clora]"
    exit 2
    ;;
esac

