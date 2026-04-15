# Safety-CLoRA and Safety-O-LoRA

Safety-informed continual learning methods for preserving alignment through capability fine-tuning.

## Overview

Fine-tuning an aligned LLM on benign capability data degrades its safety alignment — a form of catastrophic forgetting. This project applies CL (continual learning) methods to preserve safety, with safety-specific modifications derived from alignment research.

**Methods implemented:**
- **Baseline LoRA** — standard PEFT LoRA fine-tuning (no safety preservation)
- **CLoRA** (random S) — CL regularization with a random orthonormal subspace constraint [[arXiv 2410.16801](https://arxiv.org/abs/2410.16801)]
- **Safety-CLoRA** (ours) — CLoRA where S is constructed from the alignment direction `d_aligned = W_aligned − W_base`
- **O-LoRA** (standard) — orthogonal subspace per task [[arXiv 2310.14152](https://arxiv.org/abs/2310.14152)]
- **Safety-O-LoRA** (ours) — O-LoRA with asymmetric λ_safety >> λ_cap treating the safety adapter as a permanently privileged task

All methods are evaluated on **Qwen/Qwen3-0.6B** in a two-stage pipeline (alignment SFT → capability fine-tuning) and in a novel **sequential multi-task setting** (T1:align → T2 → T3 → T4).

---

## Results

### 2-Task Table (align → GSM8K)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 1.2% | — |
| Baseline LoRA | 37.3% | 13.3% |
| CLoRA (random S) | 3.1% | 11.3% |
| Safety-CLoRA (γ=0) | **1.9%** | 11.4% |
| O-LoRA (λ=0.1) | 13.1% | 11.1% |
| Safety-O-LoRA (λ_s=1.0) | **6.2%** | 11.6% |

ASR = Attack Success Rate on AdvBench (n=520); lower is better (more aligned).

### Sequential Multi-Task: gsm8k → sst2 → mbpp

| Method | After T2 GSM8K | After T3 SST-2 | After T4 MBPP |
|---|---:|---:|---:|
| Baseline LoRA | 40.2% | **2.3%** | 17.5% |
| CLoRA random | 2.7% | 31.9% | 71.2% |
| Safety-CLoRA | 1.5% | **87.9%** | 71.0% |
| O-LoRA standard | 3.3% | 16.9% | **36.7%** |
| Safety-O-LoRA | 4.6% | **81.5%** | 44.8% |

Key findings: SST-2 (a classification task) catastrophically triggers collapse in CLoRA/O-LoRA methods, while Baseline LoRA shows SST-2 *restoring* safety. Subspace analysis reveals that classification task gradients naturally overlap with the safety subspace at 3–4× the rate of math/code tasks.

---

## Project Layout

```
safety_clora/
├── models/
│   ├── clora.py          — CLoRALinear, S-matrix construction, merge utilities
│   └── olora.py          — OLoRALinear, PEFT adapter extraction, merge utilities
├── data/
│   ├── data_utils.py     — load_gsm8k(), load_sst2(), load_mbpp(), load_advbench_harmful(), ...
│   └── advbench_harmful_behaviors.csv
├── training/
│   ├── trainer.py        — Trainer class (modes: lora, clora_random, clora_safety,
│   │                        olora_standard, olora_safety)
│   └── losses.py         — clora_regularization_loss(), first_token_kl_loss()
├── evaluation/
│   └── safety_eval.py    — evaluate_safety() (ASR), evaluate_task_performance()
├── utils/
│   └── model_io.py       — load_model_and_tokenizer() (PEFT + full model checkpoints)
├── scripts/
│   ├── run_stage1_alignment_retrain.py    — Stage-1 alignment SFT
│   ├── run_shared_stage2_comparison.py    — 2-task CLoRA/Safety-CLoRA comparison
│   ├── run_olora_comparison.py            — 2-task O-LoRA/Safety-O-LoRA comparison
│   ├── run_sequential_multitask.py        — Sequential T2→T3→T4 evaluation
│   ├── run_subspace_analysis.py           — Subspace overlap analysis (produces subspace_overlap.csv)
│   └── slurm_run_pipeline.sbatch         — SLURM dispatcher for all scripts above
├── subspace_overlap.csv                  — Subspace overlap results (840 rows)
└── configs/
    └── default_config.yaml
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On SLURM, set per-job HuggingFace caches:
```bash
export HF_HOME="${TMPDIR:-/var/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
```

---

## Running Experiments

All experiments are dispatched via `slurm_run_pipeline.sbatch`. Set the aligned checkpoint path:

```bash
cd /path/to/clmm-project
ALIGNED="safety_clora/checkpoints/qwen_aligned_shared_seed42_saferlhf_chosen_refusal_n1500_ep3/epoch_3"
```

**Stage 1 — Alignment SFT:**
```bash
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch stage1_alignment_retrain \
  --align-n 1500 --epochs 3 --seed 42
```

**Stage 2 — CLoRA/Safety-CLoRA comparison:**
```bash
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch shared_stage2_comparison \
  --aligned-epoch "$ALIGNED" --stage2-epochs 3 --safety-gamma 0.0
```

**Stage 2 — O-LoRA/Safety-O-LoRA comparison:**
```bash
# Train
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage olora
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage safety_olora
# Evaluate (after both training jobs finish)
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage eval_only
```

**Sequential multi-task (T2→T3→T4):**
```bash
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
  --aligned-epoch "$ALIGNED" --method lora --epochs-per-stage 3
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
  --aligned-epoch "$ALIGNED" --method clora_random --epochs-per-stage 3
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
  --aligned-epoch "$ALIGNED" --method clora_safety --safety-gamma 0.0 --lam 0.05
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
  --aligned-epoch "$ALIGNED" --method olora_standard --epochs-per-stage 3
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
  --aligned-epoch "$ALIGNED" --method olora_safety --epochs-per-stage 3
```

**Subspace overlap analysis:**
```bash
python -m safety_clora.scripts.run_subspace_analysis \
  --ckpt-root safety_clora/checkpoints \
  --base-model-cache <path/to/Qwen3-0.6B> \
  --out-csv safety_clora/subspace_overlap.csv
```

---

## Hyperparameters (final)

| | CLoRA | O-LoRA |
|---|---|---|
| Model | Qwen/Qwen3-0.6B | Qwen/Qwen3-0.6B |
| Rank | 8 | 8 |
| Alpha | 16 | 16 |
| LR | 1e-4 | 2e-4 |
| λ | 0.05 | λ_orth=0.1, λ_safety=1.0 |
| γ (KL) | 0.0 (disabled) | N/A |
| Stage-1 epochs | 3 | 3 |
| Stage-2 epochs | 3 | 3 |
| Stage-1 n | 1500 | 1500 |
| Stage-2 n | 1000 (GSM8K) | 1000 (GSM8K) |
| Seed | 42 | 42 |
| Target modules | q_proj, v_proj | q_proj, v_proj |

---

## Key References

| Paper | Use |
|---|---|
| Qi et al. 2023 ([2310.03693](https://arxiv.org/abs/2310.03693)) | Fine-tuning breaks safety on benign data — motivates the problem |
| Unforgotten Safety ([2512.10150](https://arxiv.org/abs/2512.10150)) | CL methods preserve alignment — direct prior work |
| AsFT ([2506.08473](https://arxiv.org/abs/2506.08473)) | Alignment direction d_aligned; orthogonal updates destroy safety |
| Shallow alignment ([2406.05946](https://arxiv.org/abs/2406.05946)) | Safety concentrated in first tokens — motivated KL term |
| Safety basin ([2405.17374](https://arxiv.org/abs/2405.17374)) | Narrow safety basin — motivates asymmetric λ_safety |
| CLoRA ([2410.16801](https://arxiv.org/abs/2410.16801)) | Regularization matrix S — Method 1 |
| O-LoRA ([2310.14152](https://arxiv.org/abs/2310.14152)) | Orthogonal subspace per task — Method 2 |
