#!/bin/bash
#SBATCH --job-name=safety_clora
#SBATCH --account=edu
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# IMPORTANT: SLURM copies the script to a spool directory before execution, so
# resolving paths via ${BASH_SOURCE[0]} will NOT point at your repo.
# Use the submission directory as the repo root.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

# --- HuggingFace caches ---
# Always use persistent home cache so pre-downloaded models are reused.
# Using TMPDIR caused re-downloads that hit disk quota on large models.
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# --- HuggingFace token (for gated models: Llama, Llama-Guard) ---
if [ -f "${HOME}/.hf_token" ]; then
    export HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi

# --- Activate env (installed in persistent scratch) ---
source "/insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh"
conda activate "/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Choose one:
#   stage1_alignment_retrain | stage1_llama
#   shared_stage2_comparison | olora_comparison | sequential_multitask
#   generate_responses | llama_guard_eval
# Remaining CLI args are forwarded to the chosen script.
PIPELINE="${1:-stage1_alignment_retrain}"
if [ "$#" -ge 1 ]; then
  shift
fi

case "$PIPELINE" in
  stage1_alignment_retrain)
    python -m safety_clora.scripts.run_stage1_alignment_retrain "$@"
    ;;
  stage1_llama)
    # Llama-3.2-3B-Instruct is ~6GB; redirect to scratch to avoid home quota.
    export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_HUB_CACHE="$HF_HOME/hub"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
    python -m safety_clora.scripts.run_stage1_llama "$@"
    ;;
  stage1_llama2)
    # Llama-2-7b-hf base model (~14GB); everything in scratch to avoid home quota.
    export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    # WildJailbreak uses hf_hub_download; redirect xet staging away from home.
    export XDG_CACHE_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/xdg_cache"
    export TMPDIR="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/tmp"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
    export TOKENIZERS_PARALLELISM=false
    # Count allocated GPUs; fall back to nvidia-smi if SLURM var unset.
    NUM_GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
    NUM_GPUS="${NUM_GPUS:-1}"
    echo "[sbatch] launching with NUM_GPUS=${NUM_GPUS}"
    accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
      -m safety_clora.scripts.run_stage1_llama2 "$@"
    ;;
  shared_stage2_comparison)
    python -m safety_clora.scripts.run_shared_stage2_comparison "$@"
    ;;
  olora_comparison)
    python -m safety_clora.scripts.run_olora_comparison "$@"
    ;;
  sequential_multitask)
    python -m safety_clora.scripts.run_sequential_multitask "$@"
    ;;
  generate_responses)
    python -m safety_clora.scripts.run_generate_responses "$@"
    ;;
  llama_guard_eval)
    python -m safety_clora.scripts.run_llama_guard_eval "$@"
    ;;
  llama_stage2_comparison)
    export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_HUB_CACHE="$HF_HOME/hub"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
    python -m safety_clora.scripts.run_llama_stage2_comparison "$@"
    ;;
  llama_sequential)
    export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_HUB_CACHE="$HF_HOME/hub"
    # Datasets use home cache: scratch xsum blobs got corrupted during disk cleanup;
    # home cache has clean copies downloaded by the Qwen sequential jobs.
    export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
    python -m safety_clora.scripts.run_llama_sequential "$@"
    ;;
  subspace_analysis)
    python -m safety_clora.scripts.run_subspace_analysis "$@"
    ;;
  lambda_sweep)
    export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
    python -m safety_clora.scripts.run_llama_lambdasweep "$@"
    ;;
  lambda_diagnostic)
    export HF_HOME="/insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/hf_cache"
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
    mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
    python -m safety_clora.scripts.run_lambda_diagnostic "$@"
    ;;
  *)
    echo "Unknown pipeline: $PIPELINE"
    echo "Usage: sbatch slurm_run_pipeline.sbatch [stage1_alignment_retrain|stage1_llama|shared_stage2_comparison|olora_comparison|sequential_multitask|generate_responses|llama_guard_eval|llama_stage2_comparison|llama_sequential|subspace_analysis|lambda_diagnostic]"
    exit 2
    ;;
esac

