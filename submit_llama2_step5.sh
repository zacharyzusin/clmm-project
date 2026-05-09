#!/usr/bin/env bash
# Llama-2-7B Step 5: Multi-seed 2-task comparison (seeds 0 and 1).
#
# Canonical λ per method (from Step 4 sweep):
#   Safety-CLoRA:  λ=0.1  (31.7% ASR — best)
#   CLoRA random:  λ=0.1  (65.6% — lam-clora shared; 0.05 gives 60.2% but same flag)
#   O-LoRA:        λ=0.2  (90.1% — lam-orth)
#   Safety-O-LoRA: λ=1.0  (100%  — lam-safety; all λ catastrophic)
#   Baseline LoRA: no λ
#
# Each seed job runs --stage all (all 5 methods train + eval in one job).
# Seed 42 results are already in results/llama2_2task_{final,group}_lam*_seed42.json.
#
# Chain: seed0_train → seed0_cleanup → seed1_train → seed1_cleanup

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
BASE="meta-llama/Llama-2-7b-hf"
CKPT_ROOT="safety_clora/checkpoints"
SBATCH="safety_clora/scripts/slurm_run_pipeline.sbatch"
EXCL="--exclude=ins082"

COMMON_FLAGS="--base-model $BASE \
  --aligned-epoch $ALIGNED \
  --no-chat-template \
  --stage2-epochs 3 \
  --lam-clora 0.1 \
  --lam-orth 0.2 \
  --lam-safety 1.0 \
  --skip-alignment-eval"

# ── Seed 0 ────────────────────────────────────────────────────────────────────
JOB_0=$(sbatch --parsable $EXCL \
  $SBATCH llama_stage2_comparison \
  $COMMON_FLAGS \
  --seed 0 \
  --stage all \
  --results-json results/llama2_2task_all_seed0.json)
echo "Seed 0 train+eval: $JOB_0"

JOB_0C=$(sbatch --parsable \
  --dependency=afterok:$JOB_0 \
  --job-name=ckpt_cleanup \
  --account=edu --partition=short \
  --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --output=slurm-%j.out \
  --wrap="set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_lora_gsm8k_seed0
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_clora_gsm8k_seed0_lam0.1
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_clora_gsm8k_seed0_kl0_lam0.1
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_olora_gsm8k_seed0_lam0.2
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_olora_gsm8k_seed0_lams1
echo 'Freed seed0 checkpoints.'
df -h /insomnia001/ | tail -1")
echo "Seed 0 cleanup: $JOB_0C"

# ── Seed 1 ────────────────────────────────────────────────────────────────────
JOB_1=$(sbatch --parsable \
  --dependency=afterok:$JOB_0C \
  $EXCL \
  $SBATCH llama_stage2_comparison \
  $COMMON_FLAGS \
  --seed 1 \
  --stage all \
  --results-json results/llama2_2task_all_seed1.json)
echo "Seed 1 train+eval: $JOB_1"

JOB_1C=$(sbatch --parsable \
  --dependency=afterok:$JOB_1 \
  --job-name=ckpt_cleanup \
  --account=edu --partition=short \
  --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --output=slurm-%j.out \
  --wrap="set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_lora_gsm8k_seed1
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_clora_gsm8k_seed1_lam0.1
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_clora_gsm8k_seed1_kl0_lam0.1
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_olora_gsm8k_seed1_lam0.2
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_olora_gsm8k_seed1_lams1
echo 'Freed seed1 checkpoints.'
df -h /insomnia001/ | tail -1")
echo "Seed 1 cleanup: $JOB_1C"

echo ""
echo "Chain: $JOB_0 → $JOB_0C → $JOB_1 → $JOB_1C"
echo "Results:"
echo "  results/llama2_2task_all_seed0.json"
echo "  results/llama2_2task_all_seed1.json"
