#!/usr/bin/env bash
# Llama-2-7B subspace overlap analysis.
#
# Step 1: Run O-LoRA standard sequential (all 6 tasks) with --save-adapters-dir
#         so each stage's olora_adapters.pt is preserved in results/llama2_subspace_adapters/
#         (in the directory structure expected by run_subspace_analysis.py).
#
# Step 2: Run subspace analysis (CPU job, no GPU) on those saved adapters.
#         Key metric: SST2/GSM8K overlap ratio — should be ~3-4× if the mechanism
#         generalises from Qwen3-0.6B to Llama-2-7B.
#
# Safety checkpoint:
#   safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3
#
# Llama-2-7B base model (for subspace analysis --safety-ckpt-dir only; no delta-W needed):
#   meta-llama/Llama-2-7b-hf  (adapter format, not full weights needed)

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
SBATCH="safety_clora/scripts/slurm_run_pipeline.sbatch"
EXCL="--exclude=ins082"
SAVE_DIR="results/llama2_subspace_adapters"

# ── Step 1: O-LoRA standard sequential with adapter saving ────────────────────
# --cleanup-ckpts frees merged checkpoints; --save-adapters-dir preserves the
# tiny olora_adapters.pt files (~2MB each) for subspace analysis.
J1=$(sbatch --parsable $EXCL $SBATCH llama2_sequential \
  --aligned-epoch $ALIGNED \
  --method olora_standard \
  --lam-orth 0.2 \
  --cleanup-ckpts \
  --save-adapters-dir $SAVE_DIR \
  --results-json results/llama2_sequential_6task_olora_standard_subspace_run.json)
echo "olora_standard (adapter-saving run): $J1"

# ── Step 2: Subspace analysis — CPU job, runs after adapters are saved ────────
J2=$(sbatch --parsable \
  --dependency=afterok:$J1 \
  --job-name=subspace_llama2 \
  --account=edu --partition=short \
  --cpus-per-task=4 --mem=16G --time=01:00:00 \
  --output=slurm-%j.out \
  --wrap="set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate /insomnia001/depts/edu/COMS-E6998-012/zwz2000/scratch/conda-envs/safety_clora_cuda_pip
export PYTHONPATH=\$PWD:\${PYTHONPATH:-}
python -m safety_clora.scripts.run_subspace_analysis \
  --ckpt-root $SAVE_DIR \
  --safety-ckpt-dir $ALIGNED \
  --seq-prefix seq_llama_2_7b_hf_ \
  --methods olora_standard \
  --n-layers 32 \
  --out-csv results/subspace_overlap_llama2.csv
echo 'Subspace analysis complete. Results: results/subspace_overlap_llama2.csv'")
echo "subspace analysis: $J2 (depends on $J1)"

echo ""
echo "Chain: $J1 (olora_standard adapter-saving) → $J2 (subspace analysis)"
echo "Adapter files: $SAVE_DIR/"
echo "CSV output:    results/subspace_overlap_llama2.csv"
