#!/usr/bin/env bash
# Safety replay ablation for Llama-2-7B.
# Trains LoRA on 95% GSM8K + 5% WildJailbreak refusal replay (pool=500).
# Compares vs Safety-CLoRA (23.7% mean ASR) to test whether geometric constraint is necessary.

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
BASE="meta-llama/Llama-2-7b-hf"

JOB=$(sbatch --parsable \
  --exclude=ins082 \
  safety_clora/scripts/slurm_run_pipeline.sbatch llama_stage2_comparison \
  --aligned-epoch "$ALIGNED" \
  --base-model "$BASE" \
  --no-chat-template \
  --stage2-epochs 3 \
  --seed 42 \
  --stage safety_replay \
  --safety-replay-ratio 0.05 \
  --safety-replay-pool-n 500 \
  --results-json results/llama2_safety_replay_seed42.json)

echo "Safety replay job: $JOB"
