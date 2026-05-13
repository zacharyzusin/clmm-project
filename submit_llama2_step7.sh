#!/usr/bin/env bash
# Llama-2-7B Step 7: Multi-seed sequential 6-task evaluation (seeds 0 and 1).
#
# Only the 3 meaningful methods (O-LoRA 100% collapse is definitive at seed 42):
#   Baseline LoRA:   no λ
#   CLoRA random:    λ=0.05  (canonical from Step 4 sweep)
#   Safety-CLoRA:    λ=0.1   (canonical from Step 4 sweep)
#
# Disk strategy: Batch 1 (all three seed-0 jobs) → Batch 2 (all three seed-1 jobs).
# Peak disk per batch: 3 × 38G = 114G — well within available space.
# --cleanup-ckpts ensures each job frees prior-stage checkpoints before next stage writes.

set -euo pipefail
cd /insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project

ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
SBATCH="safety_clora/scripts/slurm_run_pipeline.sbatch"
EXCL="--exclude=ins082"
COMMON="--aligned-epoch $ALIGNED --cleanup-ckpts"

# ── Batch 1: seed 0 (3 parallel jobs) ─────────────────────────────────────────
J1=$(sbatch --parsable $EXCL $SBATCH llama2_sequential $COMMON \
  --method lora \
  --seed 0 \
  --results-json results/llama2_sequential_6task_lora_seed0.json \
  --response-trajectory-json results/llama2_trajectory_lora_seed0.json)
echo "lora seed0: $J1"

J2=$(sbatch --parsable $EXCL $SBATCH llama2_sequential $COMMON \
  --method clora_random --lam 0.05 \
  --seed 0 \
  --results-json results/llama2_sequential_6task_clora_random_seed0.json \
  --response-trajectory-json results/llama2_trajectory_clora_random_seed0.json)
echo "clora_random seed0: $J2"

J3=$(sbatch --parsable $EXCL $SBATCH llama2_sequential $COMMON \
  --method clora_safety --lam 0.1 \
  --seed 0 \
  --results-json results/llama2_sequential_6task_clora_safety_seed0.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_seed0.json)
echo "clora_safety seed0: $J3"

# ── Batch 2: seed 1 — starts only after all seed-0 jobs complete ──────────────
DEP="--dependency=afterok:${J1}:${J2}:${J3}"

J4=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method lora \
  --seed 1 \
  --results-json results/llama2_sequential_6task_lora_seed1.json \
  --response-trajectory-json results/llama2_trajectory_lora_seed1.json)
echo "lora seed1: $J4 (depends on $J1,$J2,$J3)"

J5=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method clora_random --lam 0.05 \
  --seed 1 \
  --results-json results/llama2_sequential_6task_clora_random_seed1.json \
  --response-trajectory-json results/llama2_trajectory_clora_random_seed1.json)
echo "clora_random seed1: $J5 (depends on $J1,$J2,$J3)"

J6=$(sbatch --parsable $EXCL $DEP $SBATCH llama2_sequential $COMMON \
  --method clora_safety --lam 0.1 \
  --seed 1 \
  --results-json results/llama2_sequential_6task_clora_safety_seed1.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_seed1.json)
echo "clora_safety seed1: $J6 (depends on $J1,$J2,$J3)"

echo ""
echo "Batch 1 (seed 0): $J1 $J2 $J3"
echo "Batch 2 (seed 1): $J4 $J5 $J6"
echo "Results land in results/llama2_sequential_6task_*_seed{0,1}.json"
