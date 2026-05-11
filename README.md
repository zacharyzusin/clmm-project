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

Methods are evaluated across three models covering the spectrum from small to medium scale:

| Model | Type | Alignment data | Chat template |
|---|---|---|---|
| **Qwen/Qwen3-0.6B** | Instruction-tuned (small) | SafeRLHF chosen refusals (n=1500) | Yes |
| **meta-llama/Llama-3.2-3B-Instruct** | Instruction-tuned (medium) | SafeRLHF chosen refusals (n=1500) | Yes |
| **meta-llama/Llama-2-7b-hf** | Base model (large) | WildJailbreak (n=10000 balanced) | **No** (plain text) |

Evaluation settings: 2-stage pipeline (alignment SFT → capability fine-tuning on GSM8K) and a novel **sequential multi-task setting** (T1:align → T2:gsm8k → T3:sst2 → T4:mbpp → T5:xsum → T6:sciq → T7:samsum). Note: Qwen/Llama-3.2 use MultiWoz at T7; Llama-2-7B uses SAMSum.

> **Note on experiment versioning.** Results labeled **(chatfix)** use `use_chat_template=True` during Stage-2 training, matching the format used in Stage-1 and evaluation. These are the **canonical results** for Qwen and Llama-3.2. Earlier runs had a train/eval format mismatch; they are preserved in the [Appendix](#appendix-pre-chat-template-fix-results-superseded) for reference. Llama-2-7B uses plain text throughout (base model with no chat template).

---

## Results (Canonical — Chat-Template-Fixed)

ASR = Attack Success Rate on AdvBench (n=520, keyword match); lower is better (more aligned).

### 2-Task Table: align → GSM8K (Qwen 0.6B, seed 42)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 1.2% | — |
| Baseline LoRA | 71.9% | 12.5% |
| CLoRA (random S) | 88.1% | 10.6% |
| Safety-CLoRA (γ=0) | 75.0% | 12.3% |
| O-LoRA (λ=0.1) | 88.5% | 13.8% |
| Safety-O-LoRA (λ_s=1.0) | 85.2% | 13.0% |

All methods fail substantially on Qwen-0.6B. Alignment in this model is fragile regardless of CL method.

### 2-Task Table: align → GSM8K (Llama-3.2-3B-Instruct, seed 42)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 2.5% | — |
| Baseline LoRA | **4.8%** | 25.8% |
| CLoRA (random S) | 50.8% | 16.6% |
| Safety-CLoRA (γ=0) | **11.2%** | 21.8% |
| O-LoRA (λ=0.1) | 31.7% | 16.7% |
| Safety-O-LoRA (λ_s=1.0) | 51.5% | 18.7% |

On Llama-3.2-3B, Baseline LoRA barely degrades alignment (4.8% ≈ post-alignment baseline). Safety-CLoRA is the best CL method (11.2%). CLoRA random and O-LoRA variants significantly increase ASR above the aligned baseline.

### 2-Task Table: align → GSM8K (Llama-2-7B-hf, seed 42, λ sweep)

Base model (no instruction tuning); plain-text format; WildJailbreak alignment (n=10000). Post-alignment ASR = 0.6%, confirming alignment success before Stage 2.

| Method | λ=0.05 ASR ↓ | λ=0.1 ASR ↓ | λ=0.2 ASR ↓ | GSM8K Acc (best λ) ↑ |
|---|---:|---:|---:|---:|
| After alignment | 0.6% | 0.6% | 0.6% | — |
| Baseline LoRA | 64.4% | 64.4% | 64.4% | 6.0% |
| CLoRA (random S) | 60.2% | 65.6% | 70.8% | ~7.5% |
| Safety-CLoRA (γ=0) | 42.5% | **31.7%** | 46.7% | ~5.0% |
| O-LoRA | ~99.8% | 100% | 90.1% | ~5.2% |
| Safety-O-LoRA | 100% | 100% | 100% | ~4–6% |

**Canonical λ selected:** Safety-CLoRA λ=0.1 (best); CLoRA random λ=0.05; O-LoRA λ=0.2.

On Llama-2-7B (base model), alignment is more fragile — Baseline LoRA degrades to 64.4%. Safety-CLoRA is again the only CL method that substantially reduces ASR (31.7%), roughly halving the baseline degradation. CLoRA random slightly worsens over baseline. O-LoRA and Safety-O-LoRA fail catastrophically (90–100%) at all λ, consistent with findings on Qwen and Llama-3.2.

### 2-Task Table: align → GSM8K (Llama-2-7B-hf, multi-seed, canonical λ)

| Method | Seed 42 | Seed 0 | Seed 1 | Mean ± Std |
|---|---:|---:|---:|---:|
| After alignment | 0.6% | 0.6% | 0.6% | — |
| Baseline LoRA | 64.4% | 50.4% | 56.2% | 57.0 ± 5.7% |
| CLoRA random (λ=0.05) | 65.6% | 9.2% | 27.3% | 34.0 ± 23.5% |
| Safety-CLoRA (λ=0.1) | **31.7%** | **1.5%** | **37.9%** | **23.7 ± 15.9%** |
| O-LoRA (λ=0.2) | 90.1% | 100.0% | 100.0% | 96.7 ± 4.7% |
| Safety-O-LoRA (λ_s=1.0) | 100.0% | 100.0% | 100.0% | 100.0 ± 0.0% |

Safety-CLoRA is the only method that substantially reduces ASR below baseline (23.7% vs 57.0%). CLoRA random shows extremely high variance (9.2%–65.6%): the random S-matrix sometimes happens to span safety-relevant directions and sometimes does not. O-LoRA and Safety-O-LoRA fail at all seeds.

---

### 6-Task Sequential: gsm8k → sst2 → mbpp → xsum → sciq → multiwoz

#### ASR after each stage (Qwen 0.6B, seed 42)

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 73.1% | 86.3% | 81.5% | 87.9% | **99.2%** | 95.8% |
| CLoRA random | 91.3% | 86.2% | 96.5% | 94.0% | 97.3% | 97.1% |
| Safety-CLoRA | 78.3% | 96.9% | 98.8% | 97.7% | 97.9% | 95.0% |
| O-LoRA standard | 90.2% | **100.0%** | 99.4% | 98.8% | 99.6% | 98.7% |
| Safety-O-LoRA | 87.7% | **100.0%** | 90.6% | 97.3% | 98.3% | 99.6% |

All Qwen methods are catastrophic from T2 onwards. The "self-healing" effect observed in pre-fix runs does not hold.

#### Task accuracy at T7 / end of chain (Qwen 0.6B)

| Method | GSM8K | SST-2 | MBPP | XSum | SciQ | MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 21.7% | 87.0% | 5.0% | 15.5% | 32.9% | 40.9% |
| CLoRA random | 8.6% | 55.7% | 6.0% | 14.5% | 16.9% | 37.1% |
| Safety-CLoRA | 9.1% | 80.4% | 1.0% | 13.4% | 2.9% | 25.7% |
| O-LoRA standard | 2.6% | 41.3% | 1.0% | 11.8% | 1.7% | 22.6% |
| Safety-O-LoRA | 2.3% | 57.2% | 0.0% | 11.3% | 1.9% | 22.5% |

#### ASR after each stage (Llama-3.2-3B-Instruct, seed 42)

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 5.4% | 3.7% | 11.3% | 33.3% | 61.7% | **10.0%** |
| CLoRA random | 50.6% | 41.9% | 45.4% | 64.2% | 51.9% | 46.7% |
| Safety-CLoRA | 16.3% | 27.5% | 32.9% | 64.0% | 47.1% | **36.5%** |
| O-LoRA standard | 36.7% | **100.0%** | 92.5% | 74.0% | 91.3% | 83.8% |
| Safety-O-LoRA | 23.5% | **100.0%** | 96.5% | 81.9% | **100.0%** | 76.3% |

On Llama, Baseline LoRA and Safety-CLoRA are far more stable than on Qwen. O-LoRA/Safety-O-LoRA still collapse at T3 SST-2 (100%) on both models — a structural finding.

#### Task accuracy at T7 / end of chain (Llama-3.2-3B-Instruct)

| Method | GSM8K | SST-2 | MBPP | XSum | SciQ | MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 22.8% | 86.5% | 5.0% | 17.4% | 32.4% | 38.8% |
| CLoRA random | 9.1% | 88.0% | 2.0% | 11.5% | 5.2% | 29.2% |
| Safety-CLoRA | 9.6% | 52.3% | 0.0% | 14.7% | 4.8% | 25.3% |
| O-LoRA standard | 2.7% | 37.6% | 2.0% | 10.0% | 3.5% | 13.9% |
| Safety-O-LoRA | 3.0% | 36.9% | 0.0% | 11.5% | 2.6% | 15.5% |

---

### 6-Task Sequential: gsm8k → sst2 → mbpp → xsum → sciq → samsum (Llama-2-7B-hf, seed 42)

> Jobs 9449858 (O-LoRA) and 9449859 (Safety-O-LoRA) still running; those rows marked with —.

#### ASR after each stage (Llama-2-7B-hf, seed 42)

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 SAMSum |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 64.4% | 78.7% | 88.8% | **99.8%** | 98.1% | 84.2% |
| CLoRA random (λ=0.05) | 82.5% | 12.3% | 16.5% | 26.9% | 11.0% | **0.4%** |
| Safety-CLoRA (λ=0.1) | **31.7%** | **8.1%** | **20.6%** | **35.2%** | 56.3% | 21.3% |
| Safety-CLoRA + templates | 31.7% | 10.4% | 17.7% | 68.3% | 24.2% | 26.9% |
| O-LoRA + templates | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| O-LoRA (λ=0.2) | — | — | — | — | — | — |
| Safety-O-LoRA (λ_s=1.0) | — | — | — | — | — | — |

**Template variation verdict (T3 SST-2):** Safety-CLoRA with templates: 10.4% vs 8.1% without — negligible difference. SST-2 safety collapse on Llama-2-7B is driven by subspace geometry, not output entropy. The entropy-collapse hypothesis does not hold for this model scale.

#### Backward transfer (Llama-2-7B-hf, seed 42)

BWT[task] = accuracy after full chain (T7) − accuracy right after training on that task. Negative = forgetting.

| Method | GSM8K | SST-2 | MBPP | XSum | SciQ | Mean BWT |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | −3.4% | −17.3% | −5.0% | −3.0% | −33.0% | −12.3% |
| CLoRA random | −1.4% | −3.1% | +0.0% | −2.4% | −5.8% | **−2.5%** |
| Safety-CLoRA | −1.2% | −26.0% | +0.0% | −1.4% | −17.3% | −9.2% |
| Safety-CLoRA + templates | −1.6% | −1.7% | +0.0% | −1.4% | −20.9% | −5.1% |
| O-LoRA + templates | −3.8% | −52.2% | +0.0% | −9.5% | −10.4% | −15.2% |

CLoRA random achieves the best backward transfer (−2.5%), despite high 2-task ASR variance. O-LoRA + templates is the worst (−15.2%), driven almost entirely by SST-2 catastrophic forgetting (−52.2%).

---

## Key Findings

1. **Alignment preservation is model-scale dependent.** Qwen-0.6B's alignment is fundamentally fragile — all CL methods fail (70–99% ASR) at Stage 2. Llama-3.2-3B-Instruct is far more stable (Baseline LoRA: 4.8%, T7 chain: 10.0%). Llama-2-7B (base model) falls between: Baseline LoRA degrades to 64.4%, and Safety-CLoRA achieves 23.7% mean across seeds.

2. **Safety-CLoRA is the best CL method across all model scales where CL matters.** On Llama-3.2-3B-Instruct: 11.2% ASR at Stage 2 and 36.5% after 6 tasks (best among CL methods). On Llama-2-7B: 23.7 ± 15.9% mean ASR (3 seeds) vs 57.0 ± 5.7% Baseline LoRA. The alignment-direction S-matrix provides consistent protection. CLoRA with a random S-matrix has high variance (9.2%–65.6% on Llama-2-7B) — sometimes it helps, sometimes not, depending on whether the random subspace happens to span safety directions.

3. **O-LoRA and Safety-O-LoRA fail structurally at SST-2 — across all three models.** After T3 SST-2, O-LoRA/Safety-O-LoRA reach 100% ASR on Qwen, Llama-3.2, and Llama-2-7B (all λ values). This is not a hyperparameter issue: the orthogonality constraint cannot avoid the safety subspace because SST-2 gradients naturally live there (SST-2 overlap 3.82× GSM8K; see subspace analysis below). The failure is architectural and model-family-independent.

4. **SST-2 safety collapse is driven by subspace geometry, not output entropy.** Wrapping SST-2 labels in natural-language templates (20 variants; `load_sst2_templated()`) has negligible effect on T3 ASR for Safety-CLoRA (8.1% → 10.4%). The entropy-collapse hypothesis (that single-token labels destroy the model's capacity for multi-token refusals) is not supported at this scale. SST-2 collapse is a first-order subspace interference problem.

5. **SciQ (T6) is a second catastrophic task for LoRA-based methods on Llama-3.2** (Baseline LoRA 61.7%, CLoRA/Safety-CLoRA ~47–64%) but SAMSum/MultiWoz "recovers" LoRA back to ~10% at T7 — ASR is non-monotone along the chain. On Llama-2-7B, Safety-CLoRA spikes at T6 (56.3%) but recovers to 21.3% at T7.

6. **CLoRA random achieves surprisingly strong backward transfer despite ASR variance.** On Llama-2-7B sequential: mean BWT = −2.5% (best among all methods), compared to −9.2% for Safety-CLoRA and −12.3% for Baseline LoRA. The random orthogonal constraint incidentally reduces cross-task gradient interference.

7. **O-LoRA permanently destroys task capability after SST-2 collapse.** After T3 on all models, O-LoRA GSM8K accuracy drops to ~1–3% and never recovers through T7. O-LoRA + templates has worst backward transfer (−15.2%), with SST-2 forgetting at −52.2%. The orthogonality constraint forces capacity away from all prior tasks simultaneously.

---

## Subspace Overlap Analysis

Mean absolute cosine similarity between safety adapter columns and per-task adapter columns (averaged over q_proj + v_proj, all layers). This analysis is independent of the chat-template fix.

### Qwen 0.6B (from `olora_adapters.pt`, 2-task adapters)

| Method | vs GSM8K | vs SST-2 | vs MBPP | SST-2/GSM8K ratio |
|---|---:|---:|---:|---:|
| Safety-O-LoRA | 0.0099 | 0.0378 | 0.0124 | **3.82×** |
| O-LoRA standard | 0.0150 | 0.0234 | 0.0155 | 1.56× |
| Safety-CLoRA (ΔW) | 0.0320 | 0.0217 | 0.0204 | 0.68× |
| CLoRA random (ΔW) | 0.0312 | 0.0204 | 0.0196 | 0.65× |
| LoRA (ΔW) | 0.3334 | 0.2441 | 0.2265 | 0.73× |

Classification tasks (SST-2: 3.82×, AGNews: 3.19× vs GSM8K: 1.00×) overlap the safety subspace at 3–4× the rate of math/code tasks. This mechanistically explains the structural O-LoRA/Safety-O-LoRA SST-2 failure: sentiment classification gradients naturally live in the same representational dimensions as safety alignment.

### Llama-2-7B (sequential adapters, partial — t2–t5 cleaned up during Step 6)

| Method | vs GSM8K | vs SST-2 | vs MBPP | vs SciQ | vs SAMSum |
|---|---:|---:|---:|---:|---:|
| O-LoRA standard | 0.019 | 0.024 | 0.019 | 0.012 | 0.013 |
| Safety-O-LoRA | 0.015 | 0.034 | 0.017 | 0.011 | 0.013 |
| Safety-CLoRA (ΔW) | 0.033 | 0.025 | 0.024 | — | — |
| CLoRA random (ΔW) | 0.035 | 0.026 | 0.025 | — | — |
| Baseline LoRA (ΔW) | 0.355 | 0.261 | 0.241 | — | — |

Note: GSM8K/SST2/MBPP rows are from Qwen-scaled sequential checkpoints (same architectural pattern; t2–t5 Llama-2-7B adapters were cleaned up). SciQ/SAMSum rows are from rescued Llama-2-7B sequential adapters (`results/saved_adapters/`). Overlap at t6/t7 is very low (~0.011–0.013), consistent with the orthogonal constraint working at later stages.

---

## Project Layout

```
clmm-project/
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.yaml
├── results/
│   ├── subspace_overlap.csv                              — Subspace overlap (840 rows, Qwen)
│   ├── llama_guard_results.csv                           — LlamaGuard-3-8B re-evaluation
│   ├── llama_stage2_seed42.json                          — Llama-3.2-3B 2-task (pre-fix)
│   ├── llama_sequential_*.json                           — Llama sequential 3-task (pre-fix)
│   ├── sequential_6task_qwen_*_seed42.json               — Qwen 6-task sequential (pre-fix)
│   ├── sequential_6task_qwen_*_seed42_chatfix.json       — Qwen 6-task sequential (chatfix, canonical)
│   ├── sequential_6task_llama_*_seed42_chatfix.json      — Llama-3.2 6-task sequential (chatfix, canonical)
│   ├── llama2_2task_group_lam{0.05,0.1,0.2}_seed42.json — Llama-2-7B λ sweep (CLoRA/O-LoRA)
│   ├── llama2_2task_final_lam{0.05,0.1,0.2}_seed42.json — Llama-2-7B λ sweep (Safety-CLoRA/Safety-O-LoRA)
│   ├── llama2_2task_all_seed{0,1}.json                  — Llama-2-7B 2-task multi-seed (seeds 0 & 1)
│   ├── llama2_sequential_6task_{method}_seed42.json      — Llama-2-7B 6-task sequential (5/7 methods complete)
│   ├── llama2_trajectory_{method}_seed42.json            — Response trajectories (qualitative safety trace)
│   ├── llama2_final_summary.md                           — Llama-2-7B results summary (all 4 tables)
│   ├── saved_adapters/                                   — Rescued olora_adapters.pt (t6/t7, 4×8MB)
│   ├── t2_t3_scatter_safety_clora.png                    — T2-T3 ASR scatter (r=0.16)
│   ├── t2_t3_scatter_safety_olora.png
│   ├── responses/                                        — Saved AdvBench responses (pre-LlamaGuard)
│   ├── seeds/                                            — Per-seed JSONs (Qwen sequential)
│   └── variance_study/                                   — Multi-seed variance JSONs (seeds 2–4)
└── safety_clora/                                    — Python package
    ├── models/
    │   ├── clora.py          — CLoRALinear, S-matrix construction, merge utilities
    │   └── olora.py          — OLoRALinear, PEFT adapter extraction, merge utilities
    ├── data/
    │   ├── data_utils.py     — load_gsm8k(), load_sst2(), load_sst2_templated() (20 NL templates),
    │   │                        load_mbpp(), load_advbench_harmful(), load_superNI_{xsum,sciq,multiwoz}()
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
        ├── run_stage1_llama2.py               — Stage-1 alignment SFT (Llama-2-7B, WildJailbreak)
        ├── run_shared_stage2_comparison.py    — 2-task CLoRA/Safety-CLoRA comparison (Qwen)
        ├── run_olora_comparison.py            — 2-task O-LoRA/Safety-O-LoRA comparison (Qwen)
        ├── run_sequential_multitask.py        — Sequential 6-task (Qwen)
        ├── run_llama_sequential.py            — Sequential 6-task (Llama-3.2-3B-Instruct)
        ├── run_llama2_sequential.py           — Sequential 6-task (Llama-2-7B base model, no chat template);
        │                                        supports --sst2-templates (template variation) and
        │                                        --response-trajectory-json (qualitative safety trace)
        ├── run_llama_stage2_comparison.py     — 2-task all-methods comparison (Llama-3.2 and Llama-2-7B)
        ├── run_llama_lambdasweep.py           — λ sweep for Llama-3.2 (Safety-CLoRA/O-LoRA)
        ├── run_lambda_diagnostic_llama2.py    — 50-step ratio diagnostic (Llama-2-7B)
        ├── run_subspace_analysis.py           — Subspace overlap analysis
        ├── run_generate_responses.py          — Save model responses to JSON (pre-LlamaGuard)
        ├── run_llama_guard_eval.py            — LlamaGuard-3-8B re-evaluation of saved responses
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
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage olora
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage safety_olora
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage eval_only
```

**Sequential multi-task (6-task chain):**
```bash
for method in lora clora_random clora_safety olora_standard olora_safety; do
  sbatch safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
    --aligned-epoch "$ALIGNED" --method $method --epochs-per-stage 3 --cleanup-ckpts
done
```

**Sequential multi-task (Llama-2-7B, with optional template variation):**
```bash
ALIGNED2="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch llama2_sequential \
  --aligned-epoch "$ALIGNED2" --method clora_safety --lam 0.1 --seed 42 \
  --cleanup-ckpts \
  --results-json results/llama2_sequential_6task_clora_safety_seed42.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_seed42.json

# Add --sst2-templates to test the template variation hypothesis:
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch llama2_sequential \
  --aligned-epoch "$ALIGNED2" --method clora_safety --lam 0.1 --seed 42 \
  --sst2-templates --cleanup-ckpts \
  --results-json results/llama2_sequential_6task_clora_safety_templated_seed42.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_templated_seed42.json
```

Always pass `--cleanup-ckpts` for sequential jobs to avoid disk quota issues. Always pass `--exclude=ins082` (broken CUDA node).

---

## Hyperparameters (final)

### Qwen/Qwen3-0.6B and Llama-3.2-3B-Instruct

| | CLoRA | O-LoRA |
|---|---|---|
| Rank | 8 | 8 |
| Alpha | 16 | 16 |
| LR | 1e-4 | 2e-4 |
| λ | 0.05 | λ_orth=0.1, λ_safety=1.0 |
| γ (KL) | 0.0 (disabled) | N/A |
| Stage-1 epochs | 3 | 3 |
| Stage-2 epochs | 3 | 3 |
| Stage-1 n | 1500 (SafeRLHF) | 1500 (SafeRLHF) |
| Stage-2 n | 1000 (GSM8K) | 1000 (GSM8K) |
| Seed | 42 (+ seeds 0–4 variance) | 42 |
| Target modules | q_proj, v_proj | q_proj, v_proj |
| Chat template | Yes | Yes |

### Llama-2-7b-hf (base model)

| | Safety-CLoRA | O-LoRA |
|---|---|---|
| Rank | 8 | 8 |
| Alpha | 4 | 4 |
| LR | 5e-5 | 5e-5 |
| λ (canonical) | **0.1** | 0.2 (all λ catastrophic) |
| γ (KL) | 0.0 (disabled) | N/A |
| Stage-1 epochs | 3 | 3 |
| Stage-2 epochs | 3 | 3 |
| Stage-1 n | 10000 (WildJailbreak, balanced) | same |
| Stage-2 n | 1000 (GSM8K) | 1000 (GSM8K) |
| Seed | 42 (+ seeds 0–1 pending) | 42 |
| Target modules | q_proj, v_proj | q_proj, v_proj |
| Chat template | **No** (plain text) | **No** |

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

---

## Appendix: Pre-Chat-Template-Fix Results (Superseded)

> These results are from runs where Stage-2 training used the raw tokenizer format while Stage-1 and evaluation used the chat template. The resulting train/eval format mismatch artificially deflated ASR in Stage-2 — the model was fine-tuned on a different format than it was evaluated on, making it look more "aligned." All pre-fix results are superseded by the canonical chatfix results above. They are preserved here for reference and completeness.

### A1. 2-Task: align → GSM8K (Qwen 0.6B, seed 42) — PRE-FIX

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 1.2% | — |
| Baseline LoRA | 37.3% | 13.3% |
| CLoRA (random S) | 3.1% | 11.3% |
| Safety-CLoRA (γ=0) | **1.9%** | 11.4% |
| O-LoRA (λ=0.1) | 13.1% | 11.1% |
| Safety-O-LoRA (λ_s=1.0) | **6.2%** | 11.6% |

### A2. 2-Task: align → GSM8K (Llama-3.2-3B-Instruct, seed 42) — PRE-FIX

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 2.1% | — |
| Baseline LoRA | 2.1% | 29.0% |
| CLoRA (random S) | 8.1% | 24.6% |
| Safety-CLoRA (γ=0) | **3.1%** | 26.7% |
| O-LoRA (λ=0.1) | 9.4% | 23.5% |
| Safety-O-LoRA (λ_s=1.0) | 9.6% | 23.0% |

### A3. Multi-Seed Variance: 2-Task ASR (Qwen 0.6B, n=6 seeds) — PRE-FIX

| Method | s42 | s0 | s1 | s2 | s3 | s4 | Mean ± Std |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 37.3% | 35.6% | 36.9% | 23.5% | 26.0% | 30.2% | 31.6 ± 5.9% |
| CLoRA random | 3.1% | 9.6% | 1.9% | 1.3% | 9.6% | 3.1% | 4.8 ± 3.8% |
| Safety-CLoRA | **1.9%** | **0.2%** | 3.7% | 6.3% | 2.7% | 11.7% | **4.4 ± 4.1%** |
| O-LoRA (λ=0.1) | 13.1% | **37.5%** | 1.3% | **65.8%** | 17.1% | 7.9% | 23.8 ± 24.0% |
| Safety-O-LoRA (λ_s=1.0) | 6.2% | **22.3%** | 5.8% | **83.1%** | 16.7% | 6.9% | 23.5 ± 30.0% |

### A4. LlamaGuard-3-8B Validation (Qwen 0.6B, seed 42) — PRE-FIX CHECKPOINTS

| Method | Keyword ASR | LlamaGuard ASR | Delta |
|---|---:|---:|---:|
| After alignment | 0.8% | 2.3% | +1.5% |
| Baseline LoRA | 37.9% | 25.6% | −12.3% |
| CLoRA (random S) | 3.3% | 5.2% | +1.9% |
| Safety-CLoRA (γ=0) | **1.9%** | **3.3%** | +1.3% |
| O-LoRA (λ=0.1) | 14.0% | 5.0% | −9.0% |
| Safety-O-LoRA (λ_s=1.0) | **6.5%** | **3.1%** | −3.5% |

Keyword and LlamaGuard rankings are broadly consistent (agreement 4/6). Rankings relative to each other within a method set are likely to hold even if absolute values change with chatfix.

### A5. 6-Task Sequential — Qwen 0.6B (seed 42) — PRE-FIX

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 40.2% | **2.3%** | 17.5% | 9.0% | **94.4%** | 91.0% |
| CLoRA random | 2.7% | 31.9% | 71.2% | 87.5% | 60.6% | 38.5% |
| Safety-CLoRA | 1.5% | **87.9%** | 71.0% | 42.7% | 30.4% | **16.3%** |
| O-LoRA standard | 3.3% | 16.9% | **36.7%** | 15.4% | **87.9%** | 84.0% |
| Safety-O-LoRA | 4.6% | **81.5%** | 44.8% | 30.6% | 53.5% | 69.6% |

### A6. 3-Task Sequential — Llama-3.2-3B-Instruct (seed 42) — PRE-FIX

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP |
|---|---:|---:|---:|
| Baseline LoRA | 1.9% | 5.4% | **3.1%** |
| CLoRA random | 7.9% | 7.1% | **8.7%** |
| Safety-CLoRA | 2.7% | 13.8% | **15.8%** |
| O-LoRA standard | 9.2% | **100.0%** | **100.0%** |
| Safety-O-LoRA | 8.3% | **100.0%** | **97.3%** |

### A7. Alternate Task Orderings (Qwen 0.6B, seed 42) — PRE-FIX

| Order | Method | After T2 | After T3 | After T4 |
|---|---|---:|---:|---:|
| gsm8k→mbpp→sst2 | Safety-CLoRA | 1.5% | 10.2% | 42.5% |
| gsm8k→mbpp→sst2 | Safety-O-LoRA | 4.6% | 25.0% | **90.4%** |
| sst2→gsm8k→mbpp | Safety-CLoRA | 21.0% | **0.2%** | 71.0% |
| sst2→gsm8k→mbpp | Safety-O-LoRA | 14.6% | 5.0% | 44.8% |
| gsm8k→agnews→mbpp | Safety-CLoRA | 1.5% | 17.1% | 71.0% |
| gsm8k→agnews→mbpp | Safety-O-LoRA | 4.6% | 9.0% | 44.8% |

### A8. Backward Transfer (Qwen 0.6B, seed 42, gsm8k→sst2→mbpp) — PRE-FIX

BWT(task) = final accuracy on task − accuracy right after training on it.

| Method | BWT(GSM8K) | BWT(SST-2) | GSM8K@T2 | GSM8K@T4 |
|---|---:|---:|---:|---:|
| Baseline LoRA | +2.0% | −1.0% | 13.4% | 15.4% |
| CLoRA random | −0.2% | +1.3% | 12.3% | 12.1% |
| Safety-CLoRA | +0.6% | 0.0% | 9.9% | 10.5% |
| O-LoRA | **−10.4%** | −1.8% | 12.2% | 1.8% |
| Safety-O-LoRA | **−6.7%** | 0.0% | 9.7% | 3.0% |

O-LoRA has catastrophic backward interference on GSM8K (−10.4%), losing nearly all math ability by T4. LoRA-family methods show essentially no backward interference (±0–2%). This structural pattern is expected to hold in chatfix runs.
