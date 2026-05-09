#!/usr/bin/env bash
# Resubmit Llama-2-7B Step 4 lambda sweep — Phase 3 eval + missing Safety-O-LoRA training.
#
# Current state:
#   Safety-CLoRA lam={0.05,0.1,0.2}  — checkpoints EXIST (38G each)
#   Safety-O-LoRA lams=1.0            — checkpoint EXISTS (38G)
#   Safety-O-LoRA lams=0.5, 2.0       — MISSING (prior jobs failed)
#   Baseline LoRA                      — checkpoint EXISTS (35M PEFT)
#
# Job chain:
#   A  eval_only  lam=0.1 group  (checkpoints exist, no training)
#   B  cleanup    lam=0.1 checkpoints (76G freed)
#   C  train      Safety-O-LoRA lams=0.5
#   D  eval_only  lam=0.05 group
#   E  cleanup    lam=0.05 checkpoints (76G freed)
#   F  train      Safety-O-LoRA lams=2.0
#   G  eval_only  lam=0.2 group
#   H  cleanup    lam=0.2 checkpoints (76G freed)

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
BASE="meta-llama/Llama-2-7b-hf"
CKPT_ROOT="safety_clora/checkpoints"
SBATCH="safety_clora/scripts/slurm_run_pipeline.sbatch"
EXCL="--exclude=ins082"
COMMON="$SBATCH llama_stage2_comparison \
  --base-model $BASE \
  --aligned-epoch $ALIGNED \
  --no-chat-template \
  --stage2-epochs 3 --seed 42 \
  --skip-alignment-eval"

# ── Job A: eval lam=0.1 group ─────────────────────────────────────────────────
JOB_A=$(sbatch --parsable $EXCL \
  $COMMON \
  --stage eval_only \
  --lam-clora 0.1 --lam-safety 1.0 --lam-orth 0.1 \
  --results-json results/llama2_2task_final_lam0.1_seed42.json)
echo "A (eval lam=0.1): $JOB_A"

# ── Job B: cleanup lam=0.1 checkpoints ───────────────────────────────────────
JOB_B=$(sbatch --parsable \
  --dependency=afterok:$JOB_A \
  --job-name=ckpt_cleanup \
  --account=edu --partition=short \
  --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --output=slurm-%j.out \
  --wrap="set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_clora_gsm8k_seed42_kl0_lam0.1
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_olora_gsm8k_seed42_lams1
echo 'Freed lam=0.1 and lams=1.0 checkpoints.'
df -h /insomnia001/ | tail -1")
echo "B (cleanup lam=0.1): $JOB_B"

# ── Job C: train Safety-O-LoRA lams=0.5 ──────────────────────────────────────
JOB_C=$(sbatch --parsable \
  --dependency=afterok:$JOB_B \
  $EXCL \
  $COMMON \
  --stage safety_olora \
  --lam-safety 0.5)
echo "C (train Safety-O-LoRA lams=0.5): $JOB_C"

# ── Job D: eval lam=0.05 group ────────────────────────────────────────────────
JOB_D=$(sbatch --parsable \
  --dependency=afterok:$JOB_C \
  $EXCL \
  $COMMON \
  --stage eval_only \
  --lam-clora 0.05 --lam-safety 0.5 --lam-orth 0.05 \
  --results-json results/llama2_2task_final_lam0.05_seed42.json)
echo "D (eval lam=0.05): $JOB_D"

# ── Job E: cleanup lam=0.05 checkpoints ──────────────────────────────────────
JOB_E=$(sbatch --parsable \
  --dependency=afterok:$JOB_D \
  --job-name=ckpt_cleanup \
  --account=edu --partition=short \
  --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --output=slurm-%j.out \
  --wrap="set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_clora_gsm8k_seed42_kl0_lam0.05
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_olora_gsm8k_seed42_lams0.5
echo 'Freed lam=0.05 and lams=0.5 checkpoints.'
df -h /insomnia001/ | tail -1")
echo "E (cleanup lam=0.05): $JOB_E"

# ── Job F: train Safety-O-LoRA lams=2.0 ──────────────────────────────────────
JOB_F=$(sbatch --parsable \
  --dependency=afterok:$JOB_E \
  $EXCL \
  $COMMON \
  --stage safety_olora \
  --lam-safety 2.0)
echo "F (train Safety-O-LoRA lams=2.0): $JOB_F"

# ── Job G: eval lam=0.2 group ─────────────────────────────────────────────────
JOB_G=$(sbatch --parsable \
  --dependency=afterok:$JOB_F \
  $EXCL \
  $COMMON \
  --stage eval_only \
  --lam-clora 0.2 --lam-safety 2.0 --lam-orth 0.2 \
  --results-json results/llama2_2task_final_lam0.2_seed42.json)
echo "G (eval lam=0.2): $JOB_G"

# ── Job H: final cleanup ───────────────────────────────────────────────────────
JOB_H=$(sbatch --parsable \
  --dependency=afterok:$JOB_G \
  --job-name=ckpt_cleanup \
  --account=edu --partition=short \
  --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --output=slurm-%j.out \
  --wrap="set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_clora_gsm8k_seed42_kl0_lam0.2
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_safety_olora_gsm8k_seed42_lams2
rm -rf ${CKPT_ROOT}/llama_2_7b_hf_lora_gsm8k_seed42
echo 'Final cleanup complete.'
df -h /insomnia001/ | tail -1")
echo "H (final cleanup): $JOB_H"

echo ""
echo "Full chain: A=$JOB_A B=$JOB_B C=$JOB_C D=$JOB_D E=$JOB_E F=$JOB_F G=$JOB_G H=$JOB_H"
echo "Results will be at:"
echo "  results/llama2_2task_final_lam0.1_seed42.json  (Job A)"
echo "  results/llama2_2task_final_lam0.05_seed42.json (Job D)"
echo "  results/llama2_2task_final_lam0.2_seed42.json  (Job G)"
