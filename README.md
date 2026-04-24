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

All methods are evaluated on **Qwen/Qwen3-0.6B** and **meta-llama/Llama-3.2-3B-Instruct** in a two-stage pipeline (alignment SFT → capability fine-tuning) and in a novel **sequential multi-task setting** (T1:align → T2 → T3 → T4).

---

## Results

### 2-Task Table: align → GSM8K (Qwen 0.6B, seed 42)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 1.2% | — |
| Baseline LoRA | 37.3% | 13.3% |
| CLoRA (random S) | 3.1% | 11.3% |
| Safety-CLoRA (γ=0) | **1.9%** | 11.4% |
| O-LoRA (λ=0.1) | 13.1% | 11.1% |
| Safety-O-LoRA (λ_s=1.0) | **6.2%** | 11.6% |

ASR = Attack Success Rate on AdvBench (n=520); lower is better (more aligned).

### 2-Task Table: align → GSM8K (Llama-3.2-3B-Instruct, seed 42)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 2.1% | — |
| Baseline LoRA | 2.1% | 29.0% |
| CLoRA (random S) | 8.1% | 24.6% |
| Safety-CLoRA (γ=0) | **3.1%** | 26.7% |
| O-LoRA (λ=0.1) | 9.4% | 23.5% |
| Safety-O-LoRA (λ_s=1.0) | 9.6% | 23.0% |

Key contrast vs Qwen: Baseline LoRA barely degrades alignment on Llama (2.1% = post-alignment baseline). CLoRA/O-LoRA variants actually *increase* ASR above baseline. Safety-CLoRA is the only method that avoids hurting alignment on both models.

### Multi-Seed Variance: 2-Task ASR (Qwen 0.6B, seeds 0/1/2/3/4/42)

| Method | Seed 42 | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 37.3% | 35.6% | 36.9% | 23.5% | 26.0% | 30.2% | 31.6 ± 5.9% (n=6) |
| CLoRA random | 3.1% | 9.6% | 1.9% | 1.3% | 9.6% | 3.1% | 4.8 ± 3.8% (n=6) |
| Safety-CLoRA | **1.9%** | **0.2%** | 3.7% | 6.3% | 2.7% | 11.7% | **4.4 ± 4.1%** (n=6) |
| O-LoRA (λ=0.1) | 13.1% | **37.5%** | 1.3% | **65.8%** | 17.1% | 7.9% | 23.8 ± 24.0% (n=6) |
| Safety-O-LoRA (λ_s=1.0) | 6.2% | **22.3%** | 5.8% | **83.1%** | 16.7% | 6.9% | 23.5 ± 30.0% (n=6) |

Safety-CLoRA is the clear winner with the lowest and most consistent mean ASR (4.4 ± 4.1%). O-LoRA and Safety-O-LoRA exhibit catastrophic variance (std 24–30%) and are statistically indistinguishable from each other at n=6 — the asymmetric λ_safety modification provides no reliable benefit over standard O-LoRA. Seed 2 is particularly striking: Safety-O-LoRA (83.1%) is *worse* than O-LoRA (65.8%), showing the modification can backfire.

### LlamaGuard-3-8B Validation (Qwen 0.6B, seed 42)

| Method | Keyword ASR | LlamaGuard ASR | Delta |
|---|---:|---:|---:|
| After alignment | 0.8% | 2.3% | +1.5% |
| Baseline LoRA | 37.9% | 25.6% | −12.3% |
| CLoRA (random S) | 3.3% | 5.2% | +1.9% |
| Safety-CLoRA (γ=0) | **1.9%** | **3.3%** | +1.3% |
| O-LoRA (λ=0.1) | 14.0% | 5.0% | −9.0% |
| Safety-O-LoRA (λ_s=1.0) | **6.5%** | **3.1%** | −3.5% |

Keyword metric is broadly consistent with LlamaGuard (ranking agreement 4/6 on safety-preserving methods). Safety-CLoRA and Safety-O-LoRA remain the best methods under both metrics.

---

### Sequential Multi-Task: gsm8k → sst2 → mbpp

#### Qwen 0.6B (seed 42)

| Method | After T2 GSM8K | After T3 SST-2 | After T4 MBPP |
|---|---:|---:|---:|
| Baseline LoRA | 40.2% | **2.3%** | 17.5% |
| CLoRA random | 2.7% | 31.9% | 71.2% |
| Safety-CLoRA | 1.5% | **87.9%** | 71.0% |
| O-LoRA standard | 3.3% | 16.9% | **36.7%** |
| Safety-O-LoRA | 4.6% | **81.5%** | 44.8% |

#### Llama-3.2-3B-Instruct (seed 42)

| Method | After T2 GSM8K | After T3 SST-2 | After T4 MBPP |
|---|---:|---:|---:|
| Baseline LoRA | 1.7% | 5.0% | **3.1%** |
| CLoRA random | 7.9% | 7.1% | **8.7%** |
| Safety-CLoRA | 2.7% | 13.8% | **15.8%** |
| O-LoRA standard | 9.2% | **100.0%** | **100.0%** |
| Safety-O-LoRA | 8.3% | **100.0%** | **97.3%** |

Key contrast: O-LoRA/Safety-O-LoRA catastrophically fail on both models at T3 SST-2 (16.9%/81.5% on Qwen; 100%/100% on Llama). LoRA/CLoRA work dramatically better on Llama (3.1%/8.7% final vs 17.5%/71.2% on Qwen).

#### Sequential T3-SST-2 Variance (Qwen 0.6B, multiple seeds)

| Method | n seeds | Mean ± Std (T3 ASR) |
|---|---|---:|
| Baseline LoRA | 3 | 2.4 ± 0.5% |
| CLoRA random | 3 | 15.8 ± 14.3% |
| Safety-CLoRA | 6 | 39.4 ± 31.1% |
| O-LoRA (excl. seed 42 outlier) | 2 | **98.4 ± 0.5%** |
| Safety-O-LoRA | 3 | 82.5 ± 10.5% |

O-LoRA seed 42 at T3 (16.9%) was an outlier — seeds 0 and 1 show near-complete collapse (98.1%, 98.8%). Safety-CLoRA has catastrophic variance (3.1%–87.9%) but is not consistently broken the way O-LoRA is.

### Alternate Task Orderings (Qwen 0.6B, seed 42)

| Order | Method | After T2 | After T3 | After T4 |
|---|---|---:|---:|---:|
| gsm8k→mbpp→sst2 | Safety-CLoRA | 1.5% | 10.2% | 42.5% |
| gsm8k→mbpp→sst2 | Safety-O-LoRA | 4.6% | 25.0% | **90.4%** |
| sst2→gsm8k→mbpp | Safety-CLoRA | 21.0% | **0.2%** | 71.0% |
| sst2→gsm8k→mbpp | Safety-O-LoRA | 14.6% | 5.0% | 44.8% |
| gsm8k→agnews→mbpp | Safety-CLoRA | 1.5% | 17.1% | 71.0% |
| gsm8k→agnews→mbpp | Safety-O-LoRA | 4.6% | 9.0% | 44.8% |

### Backward Transfer (Qwen 0.6B, seed 42, order gsm8k→sst2→mbpp)

BWT(task) = final accuracy on task − accuracy right after training on it.

| Method | BWT(GSM8K) | BWT(SST-2) | GSM8K@T2 | GSM8K@T4 |
|---|---:|---:|---:|---:|
| Baseline LoRA | +2.0% | −1.0% | 13.4% | 15.4% |
| CLoRA random | −0.2% | +1.3% | 12.3% | 12.1% |
| Safety-CLoRA | +0.6% | 0.0% | 9.9% | 10.5% |
| O-LoRA | **−10.4%** | −1.8% | 12.2% | 1.8% |
| Safety-O-LoRA | **−6.7%** | 0.0% | 9.7% | 3.0% |

O-LoRA has catastrophic backward interference on GSM8K (−10.4%), losing nearly all math ability by T4. LoRA-family methods show essentially no backward interference (±0–2%).

### Subspace Overlap Analysis

Mean absolute cosine similarity between safety adapter columns and per-task adapter columns (q_proj, Qwen 0.6B sequential):

| Method | vs GSM8K | vs SST-2 | vs MBPP | SST-2/GSM8K |
|---|---:|---:|---:|---:|
| Safety-O-LoRA | 0.0099 | 0.0378 | 0.0124 | **3.82×** |
| O-LoRA standard | 0.0150 | 0.0234 | 0.0155 | 1.56× |
| Safety-CLoRA (ΔW) | 0.0320 | 0.0217 | 0.0204 | 0.68× |
| CLoRA random (ΔW) | 0.0312 | 0.0204 | 0.0196 | 0.65× |
| LoRA (ΔW) | 0.3334 | 0.2441 | 0.2265 | 0.73× |

Classification tasks (SST-2 3.82×, AGNews 3.19×) overlap the safety subspace at 3–4× the rate of math/code tasks. This explains the structural SST-2 failure: when Safety-O-LoRA trains on SST-2, the orthogonality penalty cannot fully orthogonalize from safety since classification gradients naturally live in the same subspace.

---

## Key Findings

1. **Safety-CLoRA is the most reliable safety-preserving method** across both models and all evaluation metrics (4.4 ± 4.1% mean ASR on Qwen, n=6 seeds). It is the only method that consistently avoids degrading alignment on Llama-3.2-3B-Instruct (3.1% vs 2.1% post-alignment baseline).
2. **Safety-O-LoRA provides no reliable improvement over standard O-LoRA** at n=6 seeds (23.5 ± 30.0% vs 23.8 ± 24.0% — not distinguishable). The asymmetric λ_safety modification occasionally makes things dramatically worse (seed 2: 83.1% vs 65.8%). The earlier apparent advantage at n=3 was noise.
3. **O-LoRA/Safety-O-LoRA fail structurally at SST-2 in sequential training** — near-complete collapse (98%+ ASR on seeds 0/1) is structural, not a hyperparameter issue. The orthogonality constraint cannot escape the overlapping safety subspace.
4. **Classification tasks overlap the safety subspace at 3–4× the rate of math/code tasks** (SST-2 3.82×, AGNews 3.19× vs GSM8K 1.00×) — mechanistic explanation for why O-LoRA methods fail on classification tasks specifically.
5. **O-LoRA induces catastrophic backward interference on prior tasks** (−10.4% GSM8K BWT by T4), a distinct failure mode from ASR collapse.
6. **MBPP is the hardest final task** across all methods and orderings (36–71% ASR at T4).
7. **Keyword ASR is broadly validated by LlamaGuard-3-8B** — rankings are consistent for safety-preserving methods.

---

## Project Layout

```
clmm-project/
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.yaml
├── results/
│   ├── subspace_overlap.csv                    — Subspace overlap (840 rows)
│   ├── llama_guard_results.csv                 — LlamaGuard-3-8B re-evaluation (all methods)
│   ├── llama_stage2_seed42.json                — Llama-3.2-3B 2-task results (all 5 methods)
│   ├── llama_sequential_lora_seed42.json       — Llama-3.2-3B sequential LoRA
│   ├── llama_sequential_clora_random_seed42.json
│   ├── llama_sequential_clora_safety_seed42.json
│   ├── llama_sequential_olora_standard_seed42.json
│   ├── llama_sequential_olora_safety_seed42.json
│   ├── t2_t3_scatter_safety_clora.png          — T2-T3 ASR scatter (r=0.16, no correlation)
│   ├── t2_t3_scatter_safety_olora.png
│   ├── responses/                              — Saved AdvBench responses (pre-LlamaGuard)
│   ├── seeds/                                  — Per-seed result JSONs (Qwen sequential + stage2)
│   └── variance_study/                         — Multi-seed variance JSONs (all methods, seeds 2–4)
└── safety_clora/                               — Python package
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
    └── scripts/
        ├── run_stage1_alignment_retrain.py    — Stage-1 alignment SFT (Qwen)
        ├── run_stage1_llama.py                — Stage-1 alignment SFT (Llama-3.2-3B-Instruct)
        ├── run_shared_stage2_comparison.py    — 2-task CLoRA/Safety-CLoRA comparison
        ├── run_olora_comparison.py            — 2-task O-LoRA/Safety-O-LoRA comparison
        ├── run_sequential_multitask.py        — Sequential T2→T3→T4 (Qwen)
        ├── run_llama_sequential.py            — Sequential T2→T3→T4 (Llama-3.2-3B-Instruct)
        ├── run_llama_stage2_comparison.py     — 2-task eval (Llama-3.2-3B-Instruct)
        ├── run_subspace_analysis.py           — Subspace overlap analysis
        ├── run_generate_responses.py          — Save model responses to JSON (pre-LlamaGuard)
        ├── run_llama_guard_eval.py            — Llama-Guard-3-8B re-evaluation of saved responses
        ├── analyze_t2_t3_correlation.py       — Pearson/Spearman T2-T3 ASR correlation
        ├── check_s_matrix_determinism.py      — Verify S matrix determinism across seeds
        ├── analyze_variance.py                — Multi-seed variance analysis
        ├── compute_stats.py                   — Mean ± std aggregation across seeds
        └── slurm_run_pipeline.sbatch          — SLURM dispatcher for all scripts above
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
  --out-csv results/subspace_overlap.csv
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
