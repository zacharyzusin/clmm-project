#!/bin/bash
#SBATCH --job-name=safety_clora
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# IMPORTANT: SLURM copies the script to a spool directory before execution, so
# resolving paths via ${BASH_SOURCE[0]} will NOT point at your repo.
# Use the submission directory as the repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

# --- HuggingFace caches to per-job local scratch ---
export HF_HOME="${TMPDIR:-/var/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# --- Activate env (installed in persistent scratch) ---
source "/insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh"
conda activate "/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Choose one:
#   baseline_lora | clora | safety_clora | shared_comparison | shared_stage2_comparison | stage1_alignment_retrain |
#   scale_diagnostics | lambda_sweep | s_orthogonality | safety_gamma_sweep | eval_safety_gamma |
#   lambda_ablation_stage2 | sequential_multitask
# Remaining CLI args are forwarded (e.g. stage1_alignment_retrain --eval-advbench-n 0).
PIPELINE="${1:-baseline_lora}"
if [ "$#" -ge 1 ]; then
  shift
fi

case "$PIPELINE" in
  baseline_lora)
    python -m safety_clora.scripts.run_baseline_lora "$@"
    ;;
  clora)
    python -m safety_clora.scripts.run_clora "$@"
    ;;
  safety_clora)
    python -m safety_clora.scripts.run_safety_clora "$@"
    ;;
  shared_comparison)
    python -m safety_clora.scripts.run_shared_comparison "$@"
    ;;
  shared_stage2_comparison)
    python -m safety_clora.scripts.run_shared_stage2_comparison "$@"
    ;;
  stage1_alignment_retrain)
    python -m safety_clora.scripts.run_stage1_alignment_retrain "$@"
    ;;
  scale_diagnostics)
    python -m safety_clora.scripts.run_clora_scale_diagnostics --modes clora_random,clora_safety --lam 0.05 --gamma 0.05 --steps 10 "$@"
    ;;
  lambda_sweep)
    python -m safety_clora.scripts.run_lambda_sweep --mode clora_random --lams 1.0,0.1,0.01,0.001,0.0001 --steps 50 --eval_every 10 "$@"
    ;;
  s_orthogonality)
    python -m safety_clora.scripts.run_s_orthogonality_diagnostic "$@"
    ;;
  safety_gamma_sweep)
    python -m safety_clora.scripts.run_safety_gamma_sweep "$@"
    ;;
  eval_safety_gamma)
    python -m safety_clora.scripts.run_eval_safety_gamma "$@"
    ;;
  eval_lambda_ablation)
    python -m safety_clora.scripts.run_eval_lambda_ablation "$@"
    ;;
  lambda_ablation_stage2)
    python -m safety_clora.scripts.run_lambda_ablation_stage2 "$@"
    ;;
  sequential_multitask)
    python -m safety_clora.scripts.run_sequential_multitask "$@"
    ;;
  olora_comparison)
    python -m safety_clora.scripts.run_olora_comparison "$@"
    ;;
  *)
    echo "Unknown pipeline: $PIPELINE"
    echo "Usage: sbatch slurm_run_pipeline.sbatch [baseline_lora|clora|safety_clora|shared_comparison|shared_stage2_comparison|stage1_alignment_retrain|scale_diagnostics|lambda_sweep|s_orthogonality|safety_gamma_sweep|eval_safety_gamma|eval_lambda_ablation|lambda_ablation_stage2|sequential_multitask]"
    exit 2
    ;;
esac

