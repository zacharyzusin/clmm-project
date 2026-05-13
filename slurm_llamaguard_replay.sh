#!/usr/bin/env bash
#SBATCH --job-name=safety_clora
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=ins082

set -euo pipefail

REPO_ROOT="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project"
cd "$REPO_ROOT"

export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

source "/insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh"
conda activate "/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip"

python -m safety_clora.scripts.eval_replay_llamaguard
