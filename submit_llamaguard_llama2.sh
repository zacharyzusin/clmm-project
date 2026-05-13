#!/usr/bin/env bash
# LlamaGuard validation for Llama-2-7B 2-task results.
# Trains all 5 methods (seed 42), generates AdvBench responses inline,
# then runs LlamaGuard-3-8B for dual-metric validation.

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
BASE="meta-llama/Llama-2-7b-hf"

JOB=$(sbatch --parsable \
  --exclude=ins082 \
  --time=12:00:00 \
  safety_clora/scripts/slurm_run_pipeline.sbatch llama_stage2_comparison \
  --aligned-epoch "$ALIGNED" \
  --base-model "$BASE" \
  --no-chat-template \
  --stage2-epochs 3 \
  --lam-clora 0.1 \
  --lam-orth 0.2 \
  --lam-safety 1.0 \
  --seed 42 \
  --stage all \
  --responses-out-dir results/responses/llama2_seed42 \
  --run-llamaguard \
  --llamaguard-out-csv results/llama2_llamaguard_results.csv \
  --results-json results/llama2_2task_final_lam0.1_seed42_rerun.json)

echo "LlamaGuard pipeline job: $JOB"
