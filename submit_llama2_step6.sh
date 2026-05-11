#!/usr/bin/env bash
# Llama-2-7B Step 6: Sequential 6-task evaluation (seed 42).
# Resubmit v3 — two-batch Slurm dependency to avoid simultaneous disk overflow.
#
# Root cause of prior failures: all 7 jobs ran in parallel, each saving a full
# Llama-2-7B merged checkpoint (~38G) at T2 simultaneously (6 × 38G = 228G peak),
# filling the 5TB filesystem before --cleanup-ckpts could help.
#
# Fix: Batch 1 (3 jobs) runs first; Batch 2 (4 jobs) depends on all of Batch 1
# completing successfully, by which point Batch 1 checkpoints are cleaned up.
#
# Canonical λ per method (from Step 4 sweep):
#   Baseline LoRA:   no λ
#   CLoRA random:    λ=0.05
#   Safety-CLoRA:    λ=0.1
#   O-LoRA standard: λ_orth=0.2
#   O-LoRA safety:   λ_orth=0.2, λ_safety=1.0

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
SBATCH="safety_clora/scripts/slurm_run_pipeline.sbatch"
EXCL="--exclude=ins082"
COMMON="--aligned-epoch $ALIGNED --cleanup-ckpts"

# ── Batch 1: 3 jobs (lora ~35M, clora_random ~38G, clora_safety ~38G) ────────
# Peak simultaneous disk: ~3 × 38G = 114G — well within 331G free.
J1=$(sbatch --parsable $EXCL $SBATCH llama2_sequential $COMMON \
  --method lora \
  --results-json results/llama2_sequential_6task_lora_seed42.json \
  --response-trajectory-json results/llama2_trajectory_lora_seed42.json)
echo "lora: $J1"

J2=$(sbatch --parsable $EXCL $SBATCH llama2_sequential $COMMON \
  --method clora_random --lam 0.05 \
  --results-json results/llama2_sequential_6task_clora_random_seed42.json \
  --response-trajectory-json results/llama2_trajectory_clora_random_seed42.json)
echo "clora_random: $J2"

J3=$(sbatch --parsable $EXCL $SBATCH llama2_sequential $COMMON \
  --method clora_safety --lam 0.1 \
  --results-json results/llama2_sequential_6task_clora_safety_seed42.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_seed42.json)
echo "clora_safety: $J3"

# ── Batch 2: 4 jobs — start only after all of Batch 1 succeeds ───────────────
# By then Batch 1 checkpoints are cleaned up, freeing ~114G before Batch 2 runs.
DEP="--dependency=afterok:${J1}:${J2}:${J3}"

J4=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method olora_standard --lam-orth 0.2 \
  --results-json results/llama2_sequential_6task_olora_standard_seed42.json \
  --response-trajectory-json results/llama2_trajectory_olora_standard_seed42.json)
echo "olora_standard: $J4 (depends on $J1,$J2,$J3)"

J5=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method olora_safety --lam-orth 0.2 --lam-safety 1.0 \
  --results-json results/llama2_sequential_6task_olora_safety_seed42.json \
  --response-trajectory-json results/llama2_trajectory_olora_safety_seed42.json)
echo "olora_safety: $J5 (depends on $J1,$J2,$J3)"

J6=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method clora_safety --lam 0.1 --sst2-templates \
  --results-json results/llama2_sequential_6task_clora_safety_templated_seed42.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_templated_seed42.json)
echo "clora_safety+templates: $J6 (depends on $J1,$J2,$J3)"

J7=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method olora_standard --lam-orth 0.2 --sst2-templates \
  --results-json results/llama2_sequential_6task_olora_standard_templated_seed42.json \
  --response-trajectory-json results/llama2_trajectory_olora_standard_templated_seed42.json)
echo "olora_standard+templates: $J7 (depends on $J1,$J2,$J3)"

echo ""
echo "Batch 1 jobs: $J1 $J2 $J3"
echo "Batch 2 jobs: $J4 $J5 $J6 $J7 (run after Batch 1 completes)"
echo "Results land in results/llama2_sequential_6task_*.json"
