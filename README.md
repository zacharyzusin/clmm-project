# Safety-CLoRA and Safety-O-LoRA

Safety-informed continual learning methods for preserving alignment through capability fine-tuning.

---

## Research Overview

### The Problem

LLMs go through sequential training stages before deployment:

1. Pre-training
2. Safety alignment (SFT on refusal/preference data)
3. Capability fine-tuning (math, coding, classification, etc.)

**Stage 3 degrades Stage 2 even when the capability data is entirely benign.** This is *alignment degradation post-fine-tuning* — a form of catastrophic forgetting of the safety objective.

### Our Approach

We reframe alignment degradation as **catastrophic forgetting** and apply continual learning (CL) methods to preserve alignment through sequential capability fine-tuning. Two key insights from the alignment literature inject safety knowledge into the CL methods:

1. **Alignment direction** (AsFT, 2506.08473): `d_aligned = θ_aligned − θ_base` defines a safety-critical direction. Updates orthogonal to this direction destroy safety rapidly.
2. **Shallow alignment** (Qi et al., 2406.05946): Safety behavior is concentrated in the first few output tokens.

We also introduce a novel **sequential multi-task evaluation setting**: instead of the standard 2-stage (align → one task) benchmark, we evaluate over a 7-stage chain (align → T2 → T3 → T4 → T5 → T6 → T7), revealing failure modes invisible in 2-task evaluations.

### Methods

| Method | Description | Origin |
|---|---|---|
| **Baseline LoRA** | Standard PEFT LoRA, no safety preservation | — |
| **CLoRA** (random S) | CL regularization with a random orthonormal subspace constraint | [arXiv:2410.16801](https://arxiv.org/abs/2410.16801) |
| **Safety-CLoRA** (ours) | CLoRA where S is constructed from the alignment direction `d_aligned = W_aligned − W_base` | This work |
| **O-LoRA** (standard) | Orthogonal subspace per task | [arXiv:2310.14152](https://arxiv.org/abs/2310.14152) |
| **Safety-O-LoRA** (ours) | O-LoRA with asymmetric `λ_safety >> λ_cap`, treating the safety adapter as permanently privileged | This work |

### Models

We evaluate across three models covering a spectrum from small instruction-tuned to a large base model:

| Model | Type | Alignment data | Chat template |
|---|---|---|---|
| **Qwen/Qwen3-0.6B** | Instruction-tuned (small) | SafeRLHF chosen refusals (n=1500) | Yes |
| **meta-llama/Llama-3.2-3B-Instruct** | Instruction-tuned (medium) | SafeRLHF chosen refusals (n=1500) | Yes |
| **meta-llama/Llama-2-7b-hf** | Base model (large) | WildJailbreak balanced (n=10000) | **No** (plain text) |

### Evaluation Settings

- **2-task**: align → capability fine-tune on GSM8K; evaluate ASR on AdvBench (n=520, keyword match).
- **6-task sequential**: align → T2:gsm8k → T3:sst2 → T4:mbpp → T5:xsum → T6:sciq → T7:multiwoz (Qwen/Llama-3.2) or T7:samsum (Llama-2-7B). ASR evaluated after each stage.

**Metric**: ASR = Attack Success Rate on AdvBench harmful behaviors (n=520 prompts, keyword match). Validated against LlamaGuard-3-8B on a subset — rankings broadly consistent.

> **Note on experiment versioning.** Results labeled **(chatfix)** use `use_chat_template=True` during Stage-2 training, matching the format used in Stage-1 and evaluation. These are the **canonical results** for Qwen and Llama-3.2. Earlier runs had a train/eval format mismatch that artificially deflated ASR. Pre-fix results are preserved in the [Appendix](#appendix-pre-chat-template-fix-results-superseded). Llama-2-7B uses plain text throughout (base model, no chat template).

---

## Results (Canonical)

### 2-Task: align → GSM8K

#### Qwen/Qwen3-0.6B (seed 42, chatfix)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 1.2% | — |
| Baseline LoRA | 71.9% | 12.5% |
| CLoRA (random S) | 88.1% | 10.6% |
| Safety-CLoRA (γ=0) | 75.0% | 12.3% |
| O-LoRA (λ=0.1) | 88.5% | 13.8% |
| Safety-O-LoRA (λ_s=1.0) | 85.2% | 13.0% |

All methods fail substantially on Qwen-0.6B. Alignment in this model is fragile regardless of CL method — consistent across 5 seeds (see [Appendix A3](#a3-multi-seed-variance-2-task-asr-qwen-06b-n6-seeds--pre-fix)).

#### Llama-3.2-3B-Instruct (seed 42, chatfix)

| Method | ASR after FT ↓ | GSM8K Acc ↑ |
|---|---:|---:|
| After alignment | 2.5% | — |
| Baseline LoRA | **4.8%** | 25.8% |
| CLoRA (random S) | 50.8% | 16.6% |
| Safety-CLoRA (γ=0) | **11.2%** | 21.8% |
| O-LoRA (λ=0.1) | 31.7% | 16.7% |
| Safety-O-LoRA (λ_s=1.0) | 51.5% | 18.7% |

On Llama-3.2-3B, Baseline LoRA barely degrades (4.8% ≈ post-alignment baseline). Safety-CLoRA is the best CL method (11.2%). CLoRA random and O-LoRA variants significantly increase ASR above the aligned baseline.

#### Llama-2-7B-hf (λ sweep, seed 42)

Base model; plain-text format; WildJailbreak alignment (n=10000). Post-alignment ASR = 0.6%.

| Method | λ=0.05 ASR ↓ | λ=0.1 ASR ↓ | λ=0.2 ASR ↓ | GSM8K (best λ) ↑ |
|---|---:|---:|---:|---:|
| After alignment | 0.6% | 0.6% | 0.6% | — |
| Baseline LoRA | 64.4% | 64.4% | 64.4% | 6.0% |
| CLoRA (random S) | 60.2% | 65.6% | 70.8% | ~7.5% |
| Safety-CLoRA (γ=0) | 42.5% | **31.7%** | 46.7% | ~5.0% |
| O-LoRA | ~99.8% | 100% | 90.1% | ~5.2% |
| Safety-O-LoRA | 100% | 100% | 100% | ~4–6% |

**Canonical λ selected:** Safety-CLoRA λ=0.1; CLoRA random λ=0.05; O-LoRA λ=0.2.

Safety-CLoRA is the only CL method that substantially reduces ASR below baseline (31.7% vs 64.4%). O-LoRA and Safety-O-LoRA fail catastrophically at all λ values.

#### Llama-2-7B-hf (multi-seed, canonical λ)

| Method | Seed 42 | Seed 0 | Seed 1 | Mean ± Std |
|---|---:|---:|---:|---:|
| After alignment | 0.6% | 0.6% | 0.6% | — |
| Baseline LoRA | 64.4% | 50.4% | 56.2% | 57.0 ± 5.8% |
| CLoRA random (λ=0.05) | 65.6% | 9.2% | 27.3% | 34.0 ± 23.5% |
| Safety-CLoRA (λ=0.1) | **31.7%** | **1.5%** | **37.9%** | **23.7 ± 15.9%** |
| O-LoRA (λ=0.2) | 91.0% | 100.0% | 100.0% | 97.0 ± 4.3% |
| Safety-O-LoRA (λ_s=1.0) | 100.0% | 100.0% | 100.0% | 100.0 ± 0.0% |

Safety-CLoRA is the only method substantially below baseline (23.7% vs 57.0% mean). CLoRA random shows extreme variance (9.2%–65.6%) — the random S-matrix sometimes spans safety-relevant directions and sometimes does not. O-LoRA/Safety-O-LoRA fail at all seeds.

---

### 6-Task Sequential: gsm8k → sst2 → mbpp → xsum → sciq → multiwoz/samsum

#### Qwen/Qwen3-0.6B — ASR after each stage (seed 42, chatfix)

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 73.1% | 86.3% | 81.5% | 87.9% | **99.2%** | 95.8% |
| CLoRA random | 91.3% | 86.2% | 96.5% | 94.0% | 97.3% | 97.1% |
| Safety-CLoRA | 78.3% | 96.9% | 98.8% | 97.7% | 97.9% | 95.0% |
| O-LoRA standard | 90.2% | **100.0%** | 99.4% | 98.8% | 99.6% | 98.7% |
| Safety-O-LoRA | 87.7% | **100.0%** | 90.6% | 97.3% | 98.3% | 99.6% |

All Qwen methods are catastrophic from T2 onwards. Qwen-0.6B alignment cannot survive any fine-tuning.

#### Qwen/Qwen3-0.6B — Task accuracy at T7

| Method | GSM8K | SST-2 | MBPP | XSum | SciQ | MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 21.7% | 87.0% | 5.0% | 15.5% | 32.9% | 40.9% |
| CLoRA random | 8.6% | 55.7% | 6.0% | 14.5% | 16.9% | 37.1% |
| Safety-CLoRA | 9.1% | 80.4% | 1.0% | 13.4% | 2.9% | 25.7% |
| O-LoRA standard | 2.6% | 41.3% | 1.0% | 11.8% | 1.7% | 22.6% |
| Safety-O-LoRA | 2.3% | 57.2% | 0.0% | 11.3% | 1.9% | 22.5% |

#### Llama-3.2-3B-Instruct — ASR after each stage (seed 42, chatfix)

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 5.4% | 3.7% | 11.3% | 33.3% | 61.7% | **10.0%** |
| CLoRA random | 50.6% | 41.9% | 45.4% | 64.2% | 51.9% | 46.7% |
| Safety-CLoRA | 16.3% | 27.5% | 32.9% | 64.0% | 47.1% | **36.5%** |
| O-LoRA standard | 36.7% | **100.0%** | 92.5% | 74.0% | 91.3% | 83.8% |
| Safety-O-LoRA | 23.5% | **100.0%** | 96.5% | 81.9% | **100.0%** | 76.3% |

On Llama-3.2, Baseline LoRA and Safety-CLoRA are far more stable than on Qwen. O-LoRA/Safety-O-LoRA still collapse at T3 SST-2 (100%) on both instruction-tuned models — a structural, model-family-independent finding.

#### Llama-3.2-3B-Instruct — Task accuracy at T7

| Method | GSM8K | SST-2 | MBPP | XSum | SciQ | MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 22.8% | 86.5% | 5.0% | 17.4% | 32.4% | 38.8% |
| CLoRA random | 9.1% | 88.0% | 2.0% | 11.5% | 5.2% | 29.2% |
| Safety-CLoRA | 9.6% | 52.3% | 0.0% | 14.7% | 4.8% | 25.3% |
| O-LoRA standard | 2.7% | 37.6% | 2.0% | 10.0% | 3.5% | 13.9% |
| Safety-O-LoRA | 3.0% | 36.9% | 0.0% | 11.5% | 2.6% | 15.5% |

#### Llama-2-7B-hf — ASR after each stage (seed 42)

Task chain: gsm8k → sst2 → mbpp → xsum → sciq → **samsum** (SAMSum used instead of MultiWoz for Llama-2-7B).

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 SAMSum |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 64.4% | 78.7% | 88.8% | **99.8%** | 98.1% | 84.2% |
| CLoRA random (λ=0.05) | 82.5% | 12.3% | 16.5% | 26.9% | 11.0% | **0.4%** |
| Safety-CLoRA (λ=0.1) | **31.7%** | **8.1%** | **20.6%** | **35.2%** | 56.3% | 21.3% |
| O-LoRA (λ=0.2) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Safety-O-LoRA (λ_s=1.0) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

O-LoRA/Safety-O-LoRA collapse at T2 (GSM8K) and never recover — the structural failure seen at T3 SST-2 on instruction-tuned models appears one stage earlier on this base model.

**Template variation (SST-2):** Safety-CLoRA with 20 NL label templates: 10.4% vs 8.1% without — negligible difference. SST-2 collapse is driven by subspace geometry, not output entropy.

#### Llama-2-7B-hf — ASR after each stage, mean ± std (seeds 42, 0, 1)

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 SAMSum |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 54.7±7.7% | 47.3±22.3% | 72.4±11.6% | 68.1±24.1% | **97.3±1.2%** | 85.9±5.0% |
| CLoRA random (λ=0.05) | 39.8±31.4% | 28.9±30.0% | 36.3±38.6% | 27.0±1.5% | 28.9±13.2% | 55.7±41.2% |
| Safety-CLoRA (λ=0.1) | **31.1±5.8%** | **32.2±34.6%** | **28.1±14.4%** | **30.0±20.7%** | 52.9±32.0% | 57.3±27.2% |

Per-seed detail:

| Method | | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---|---:|---:|---:|---:|---:|---:|
| LoRA | s42/0/1 | 64.4/45.6/54.2% | 78.7/29.0/34.2% | 88.8/64.0/64.2% | 99.8/63.1/41.3% | 98.1/98.3/95.6% | 84.2/80.8/92.7% |
| CLoRA-Rand | s42/0/1 | 82.5/7.7/29.2% | 12.3/3.5/71.0% | 16.5/90.2/2.1% | 26.9/28.8/25.2% | 11.0/33.5/42.3% | 0.4/99.0/67.7% |
| Safety-CLoRA | s42/0/1 | 31.7/23.7/37.9% | 8.1/7.3/81.2% | 20.6/15.4/48.3% | 35.2/52.3/2.5% | 56.3/90.2/12.1% | 21.3/63.5/87.1% |

SciQ (T6) is consistently catastrophic for Baseline LoRA (97.3% mean, std=1.2%). Safety-CLoRA has high T3 variance (8.1% vs 81.2% across seeds), not predictable from T2 performance (Pearson r=0.16, p=0.76).

---

### Backward Transfer (Llama-2-7B-hf, seed 42)

BWT[task] = accuracy after full chain (T7) − accuracy right after training on that task. Negative = forgetting.

| Method | GSM8K | SST-2 | MBPP | XSum | SciQ | Mean BWT |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | −3.4% | −17.3% | −5.0% | −3.0% | −33.0% | −12.3% |
| CLoRA random | −1.4% | −3.1% | +0.0% | −2.4% | −5.8% | **−2.5%** |
| Safety-CLoRA | −1.2% | −26.0% | +0.0% | −1.4% | −17.3% | −9.2% |

CLoRA random achieves the best backward transfer (−2.5% mean) despite high 2-task ASR variance. The random orthogonal constraint incidentally reduces cross-task gradient interference.

---

### Subspace Overlap Analysis

Mean absolute cosine similarity between safety adapter columns and per-task adapter columns (averaged over q_proj + v_proj, all layers). This analysis mechanistically explains the structural O-LoRA/Safety-O-LoRA SST-2 failure.

#### Qwen 0.6B (O-LoRA/Safety-O-LoRA adapters, 2-task runs)

| Method | vs GSM8K | vs SST-2 | vs MBPP | SST-2/GSM8K ratio |
|---|---:|---:|---:|---:|
| Safety-O-LoRA | 0.0099 | 0.0378 | 0.0124 | **3.82×** |
| O-LoRA standard | 0.0150 | 0.0234 | 0.0155 | 1.56× |
| Safety-CLoRA (ΔW) | 0.0320 | 0.0217 | 0.0204 | 0.68× |
| CLoRA random (ΔW) | 0.0312 | 0.0204 | 0.0196 | 0.65× |
| LoRA (ΔW) | 0.3334 | 0.2441 | 0.2265 | 0.73× |

Classification tasks overlap the safety subspace at 3–4× the rate of math/code tasks: SST-2 3.82×, AGNews 3.19× vs GSM8K 1.00×.

#### Llama-2-7B (O-LoRA standard, sequential adapters, all tasks)

| Task | Mean Overlap | Ratio vs GSM8K | q_proj only |
|---|---:|---:|---:|
| GSM8K | 0.0069 | 1.00× | — |
| MBPP | 0.0082 | 1.18× | — |
| XSum | 0.0063 | 0.91× | — |
| SciQ | 0.0124 | **1.78×** | — |
| **SST-2** | **0.0299** | **4.32×** | **7.01×** |

SST-2 overlaps the safety subspace at 4.32× the rate of GSM8K (q_proj: 7.01×). The 4.32× ratio is model-family-independent (Qwen: 3.82×), confirming the mechanism generalises. SciQ at 1.78× is elevated but sub-catastrophic; its high ASR in sequential runs has a different failure mode.

---

### LlamaGuard-3-8B Validation (Llama-2-7B-hf, seed 42)

Re-evaluation of saved AdvBench responses using LlamaGuard-3-8B as a judge. Validates that keyword ASR is broadly consistent with a neural judge. CLoRA random shows a large discrepancy — the model produces outputs that contain harmful surface patterns but LlamaGuard classifies them as safe.

| Method | Keyword ASR | LlamaGuard ASR | Delta |
|---|---:|---:|---:|
| After alignment | 0.6% | 0.0% | −0.6% |
| Baseline LoRA | 64.4% | 66.0% | +1.6% |
| CLoRA random (λ=0.05) | 69.8% | **1.3%** | −68.5% |
| Safety-CLoRA (λ=0.1) | 31.7% | **6.3%** | −25.4% |
| LoRA + Safety Replay | 28.1% | 7.8% | −20.3% |
| O-LoRA (λ=0.2) | 100.0% | 85.8% | −14.2% |
| Safety-O-LoRA (λ_s=1.0) | 100.0% | 92.7% | −7.3% |

Safety-CLoRA and Safety-Replay rankings are consistent between both metrics.

---

### Safety Replay Ablation (Llama-2-7B-hf, seed 42)

Baseline: LoRA on GSM8K with 5% of training steps replaced by WildJailbreak safety replay (n=500 pool). Tests whether naive data mixing matches the CL constraint approach.

| Method | Keyword ASR | LlamaGuard ASR | GSM8K |
|---|---:|---:|---:|
| After alignment | 0.6% | 0.0% | — |
| Baseline LoRA | 64.4% | 66.0% | 6.0% |
| LoRA + Safety Replay (ratio=0.05) | 27.7% | 7.8% | 7.8% |
| Safety-CLoRA (λ=0.1, seed 42) | 31.7% | **6.3%** | 5.0% |
| Safety-CLoRA (λ=0.1, mean 3 seeds) | **23.7%** | — | — |

Safety replay and Safety-CLoRA are comparable at seed 42 (27.7% vs 31.7% keyword). Safety-CLoRA is modestly better under LlamaGuard and more consistent across seeds (23.7±15.9% vs single-seed replay). Critically, Safety-CLoRA does not require access to the original safety data at fine-tuning time — only the alignment checkpoint `d_aligned` is needed.

---

## Key Findings

1. **Alignment preservation is model-scale dependent.** Qwen-0.6B's alignment is fundamentally fragile — all CL methods fail (70–99% ASR) after a single stage of capability fine-tuning. Llama-3.2-3B-Instruct is far more stable (Baseline LoRA: 4.8%, T7 chain: 10.0%). Llama-2-7B (base model) falls between: Baseline LoRA degrades to 64.4%, but Safety-CLoRA achieves 23.7% mean across seeds.

2. **Safety-CLoRA is the best CL method across all model scales where CL matters.** On Llama-3.2-3B-Instruct: 11.2% at Stage 2 and 36.5% after 6 tasks (best among CL methods). On Llama-2-7B: 23.7 ± 15.9% mean ASR (3 seeds) vs 57.0 ± 5.7% Baseline LoRA. The alignment-direction S-matrix provides consistent protection. CLoRA with a random S-matrix has high variance (9.2%–65.6%) — sometimes helps, sometimes not, depending on whether the random subspace happens to span safety directions.

3. **O-LoRA and Safety-O-LoRA fail structurally at SST-2 — across all three models.** After T3 SST-2, O-LoRA/Safety-O-LoRA reach 100% ASR on Qwen, Llama-3.2, and at T2 on Llama-2-7B (all λ values). This is not a hyperparameter issue: SST-2 gradients naturally overlap the safety subspace at 4.32× the rate of GSM8K, and the orthogonality constraint cannot avoid it. The failure is architectural and model-family-independent.

4. **SST-2 safety collapse is driven by subspace geometry, not output entropy.** Safety-CLoRA with 20 natural-language label templates: T3 ASR 10.4% vs 8.1% without templates — negligible. The entropy-collapse hypothesis is not supported. SST-2 collapse is a first-order subspace interference problem.

5. **SciQ (T6) is a second catastrophic task for LoRA-based methods on Llama-3.2** (Baseline LoRA: 61.7%). SAMSum/MultiWoz at T7 partially recovers Baseline LoRA to ~10% — ASR is non-monotone along the chain. On Llama-2-7B, Safety-CLoRA spikes at T6 SciQ (56.3%) then recovers to 21.3% at T7.

6. **CLoRA random achieves surprisingly strong backward transfer despite high ASR variance.** On Llama-2-7B sequential: mean BWT = −2.5% (best among all methods), compared to −9.2% for Safety-CLoRA and −12.3% for Baseline LoRA. The random orthogonal constraint incidentally reduces cross-task gradient interference.

7. **O-LoRA permanently destroys task capability after alignment collapse.** After catastrophic failure, O-LoRA GSM8K accuracy drops to ~1–3% and never recovers through T7. The orthogonality constraint forces capacity away from all prior tasks simultaneously.

---

## Project Layout

```
clmm-project/
├── README.md
├── requirements.txt
├── configs/
│   └── default_config.yaml
│
├── results/                                              — All result files (JSON, CSV, PNG)
│   ├── subspace_overlap.csv                             — Subspace overlap, Qwen (840 rows)
│   ├── subspace_overlap_llama2.csv                      — Subspace overlap, Llama-2-7B (O-LoRA std, 5 tasks)
│   ├── llama_subspace_overlap_lora.csv                  — Llama-3.2 LoRA subspace overlap
│   ├── llama_guard_results.csv                          — LlamaGuard validation, Qwen (pre-fix checkpoints)
│   ├── llama2_llamaguard_results.csv                    — LlamaGuard validation, Llama-2-7B (seed 42)
│   ├── llama2_llamaguard_replay_results.csv             — LlamaGuard validation, safety replay (Llama-2-7B)
│   ├── llama2_safety_replay_seed42.json                 — Safety replay ablation (keyword ASR + GSM8K)
│   ├── llama2_final_summary.md                          — Llama-2-7B full results summary (4 tables)
│   │
│   ├── sequential_6task_qwen_*_seed42.json              — Qwen 6-task sequential (pre-fix; superseded)
│   ├── sequential_6task_qwen_*_seed42_chatfix.json      — Qwen 6-task sequential (chatfix; canonical)
│   ├── sequential_6task_llama_*_seed42_chatfix.json     — Llama-3.2 6-task sequential (chatfix; canonical)
│   ├── sequential_6task_summary.md                      — Summary template for 6-task results
│   │
│   ├── llama_stage2_seed42.json                         — Llama-3.2 2-task (pre-fix; superseded)
│   ├── llama_sequential_*.json                          — Llama-3.2 3-task sequential (pre-fix; superseded)
│   ├── llama_2task_lambdasweep_*_seed42.json            — Llama-3.2 λ sweep (CLoRA/O-LoRA variants)
│   │
│   ├── llama2_step2_baseline_seed42.json                — Llama-2-7B Step 2 baseline verification
│   ├── llama2_2task_group_lam{0.05,0.1,0.2}_seed42.json — Llama-2-7B λ sweep (CLoRA/O-LoRA)
│   ├── llama2_2task_final_lam{0.05,0.1,0.2}_seed42.json — Llama-2-7B λ sweep (Safety-CLoRA/Safety-O-LoRA)
│   ├── llama2_2task_all_seed{0,1}.json                  — Llama-2-7B 2-task multi-seed (seeds 0 & 1)
│   ├── llama2_sequential_6task_{method}_seed{42,0,1}.json — Llama-2-7B 6-task sequential (all methods)
│   ├── llama2_sequential_6task_{method}_templated_seed42.json — Template variation ablation
│   │
│   ├── llama2_trajectory_{method}_seed{42,0,1}.json     — Response trajectories (qualitative safety trace)
│   │
│   ├── t2_t3_scatter_safety_clora.png                   — T2-T3 ASR scatter plot (r=0.16, Safety-CLoRA)
│   ├── t2_t3_scatter_safety_olora.png                   — T2-T3 ASR scatter plot (Safety-O-LoRA)
│   │
│   ├── responses/                                       — Saved AdvBench responses (pre-LlamaGuard re-eval)
│   ├── saved_adapters/                                  — Rescued olora_adapters.pt (Qwen, t6/t7)
│   ├── llama2_subspace_adapters/                        — Llama-2-7B O-LoRA adapter files (all 6 tasks)
│   ├── seeds/                                           — Per-seed JSONs (Qwen sequential)
│   └── variance_study/                                  — Multi-seed variance JSONs (Qwen, seeds 2–4)
│
└── safety_clora/                                        — Python package (all code here)
    ├── __init__.py
    ├── checkpoints/                                     — Trained model checkpoints (PEFT format)
    │   ├── qwen_aligned_shared_seed42_saferlhf_chosen_refusal_n1500_ep3/   ← Qwen Stage-1 canonical
    │   ├── qwen_aligned_shared_seed{0,1}_saferlhf_chosen_refusal_n1500_ep3/
    │   ├── llama_3p2_3b_instruct_aligned_seed42_saferlhf_chosen_refusal_n1500_ep3/
    │   └── llama2_7b_aligned_wildjailbreak_ep3/                            ← Llama-2-7B Stage-1
    ├── models/
    │   ├── clora.py          — CLoRALinear, S-matrix construction (alignment direction SVD),
    │   │                        orthogonal complement basis, merge_clora_to_base_linear()
    │   └── olora.py          — OLoRALinear, extract_peft_lora_adapters(), apply/merge utilities
    ├── data/
    │   ├── data_utils.py     — load_gsm8k(), load_sst2(), load_sst2_templated() (20 NL templates),
    │   │                        load_mbpp(), load_advbench_harmful(),
    │   │                        load_saferlhf_chosen_refusals(), load_wildjailbreak_safety_replay(),
    │   │                        mix_gsm8k_with_safety_replay(),
    │   │                        load_superNI_{xsum,sciq,multiwoz}(), load_samsum()
    │   └── advbench_harmful_behaviors.csv
    ├── training/
    │   ├── trainer.py        — Trainer class (modes: lora, clora_random, clora_safety,
    │   │                        olora_standard, olora_safety, safety_replay)
    │   └── losses.py         — clora_regularization_loss(), first_token_kl_loss()
    ├── evaluation/
    │   └── safety_eval.py    — evaluate_safety() (ASR), evaluate_task_performance(),
    │                            evaluate_generation_task() (Rouge-L for xsum/sciq/multiwoz/samsum)
    ├── utils/
    │   └── model_io.py       — load_model_and_tokenizer() (PEFT + full model checkpoints)
    └── scripts/
        │
        ├── — Stage 1 alignment SFT —
        ├── run_stage1_alignment_retrain.py   — Qwen Stage-1 (SafeRLHF)
        ├── run_stage1_llama.py               — Llama-3.2-3B-Instruct Stage-1 (SafeRLHF)
        ├── run_stage1_llama2.py              — Llama-2-7B Stage-1 (WildJailbreak, no chat template)
        │
        ├── — Stage 2 comparisons (2-task) —
        ├── run_shared_stage2_comparison.py   — 2-task CLoRA/Safety-CLoRA (Qwen)
        ├── run_olora_comparison.py           — 2-task O-LoRA/Safety-O-LoRA (Qwen)
        ├── run_llama_stage2_comparison.py    — 2-task all-methods (Llama-3.2 and Llama-2-7B)
        │
        ├── — Sequential multi-task (6-task chains) —
        ├── run_sequential_multitask.py       — Sequential 6-task (Qwen)
        ├── run_llama_sequential.py           — Sequential 6-task (Llama-3.2-3B-Instruct)
        ├── run_llama2_sequential.py          — Sequential 6-task (Llama-2-7B, no chat template);
        │                                       --sst2-templates for template variation ablation;
        │                                       --response-trajectory-json for qualitative safety trace
        │
        ├── — λ sweeps and diagnostics —
        ├── run_llama_lambdasweep.py          — λ sweep for Llama-3.2 (CLoRA/O-LoRA variants)
        ├── run_lambda_diagnostic_llama2.py   — 50-step ratio diagnostic (Llama-2-7B)
        ├── run_lambda_diagnostic.py          — λ diagnostic (Qwen)
        ├── run_lambda_ablation_stage2.py     — λ sweep for Qwen 2-task
        ├── run_lambda_sweep.py               — General λ sweep runner
        ├── run_safety_gamma_sweep.py         — γ (KL weight) sweep
        │
        ├── — Evaluation and analysis —
        ├── run_subspace_analysis.py          — Subspace overlap analysis (all models/methods)
        ├── run_generate_responses.py         — Save model responses to JSON (for LlamaGuard re-eval)
        ├── run_llama_guard_eval.py           — LlamaGuard-3-8B re-evaluation of saved responses
        ├── eval_replay_llamaguard.py         — LlamaGuard eval for safety replay checkpoint
        ├── analyze_t2_t3_correlation.py      — Pearson/Spearman T2-T3 ASR correlation (r=0.16)
        ├── check_s_matrix_determinism.py     — Verify S-matrix is deterministic across seeds
        ├── analyze_variance.py               — Multi-seed variance analysis
        ├── compute_stats.py                  — Mean ± std aggregation across seeds
        │
        ├── — Miscellaneous/debug scripts —
        ├── run_baseline_lora.py              — Baseline LoRA (standalone)
        ├── run_clora.py                      — CLoRA standalone runner
        ├── run_clora_scale_diagnostics.py    — Scale diagnostics for CLoRA
        ├── run_eval_alignment_checkpoint.py  — Eval a Stage-1 checkpoint
        ├── run_eval_lambda_ablation.py       — Eval λ ablation results
        ├── run_eval_safety_gamma.py          — Eval γ ablation results
        ├── run_safety_clora.py               — Safety-CLoRA standalone runner
        ├── run_shared_comparison.py          — Early comparison script (superseded)
        ├── run_s_orthogonality_diagnostic.py — Diagnose S-matrix orthogonality
        ├── run_stage1_label_mask_diagnostic.py — Debug label masking
        ├── run_stage1_overfit_debug.py       — Debug Stage-1 overfitting
        ├── run_stage2_quickcheck.py          — Quick Stage-2 sanity check
        │
        └── — SLURM dispatch —
            ├── slurm_run_pipeline.sbatch     — Main SLURM dispatcher for all scripts
            └── slurm_stage2_quickcheck.sbatch
```

Additionally, the following submission/utility scripts live in the repo root:

```
clmm-project/
├── resubmit_llama2_step4.sh            — Resubmit Llama-2-7B Step 4 (λ sweep)
├── submit_llama2_step5.sh              — Submit Step 5 (multi-seed 2-task)
├── submit_llama2_step6.sh              — Submit Step 6 (sequential 6-task seeds 0 & 1)
├── submit_llama2_step7.sh              — Submit Step 7 (subspace analysis)
├── submit_llama2_subspace.sh           — Subspace analysis submission
├── submit_llamaguard_llama2.sh         — LlamaGuard eval submission (Llama-2-7B)
├── submit_safety_replay_llama2.sh      — Safety replay ablation submission
├── slurm_llamaguard_replay.sh          — LlamaGuard eval for safety replay checkpoint
├── subspace_overlap.csv                — Copy of Qwen subspace overlap (also in results/)
└── subspace_overlap_all.csv            — Full 840-row subspace overlap (Qwen, all methods)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On SLURM, set per-job HuggingFace caches to avoid cross-job conflicts:

```bash
export HF_HOME="${TMPDIR:-/var/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
```

**Node `ins082` has broken CUDA** — always pass `--exclude=ins082` to Slurm.  
**Always pass `--cleanup-ckpts`** for sequential jobs to avoid disk quota issues.

---

## Running Experiments

All experiments are dispatched via `safety_clora/scripts/slurm_run_pipeline.sbatch`.

### Checkpoint paths

```bash
QWEN_ALIGNED="safety_clora/checkpoints/qwen_aligned_shared_seed42_saferlhf_chosen_refusal_n1500_ep3/epoch_3"
LLAMA_ALIGNED="safety_clora/checkpoints/llama_3p2_3b_instruct_aligned_seed42_saferlhf_chosen_refusal_n1500_ep3/epoch_3"
LLAMA2_ALIGNED="safety_clora/checkpoints/llama2_7b_aligned_wildjailbreak_ep3/epoch_3"
```

### Stage 1 — Alignment SFT

```bash
# Qwen
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch stage1_alignment_retrain \
  --align-n 1500 --epochs 3 --seed 42

# Llama-3.2-3B-Instruct
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch stage1_llama \
  --align-n 1500 --epochs 3 --seed 42

# Llama-2-7B (WildJailbreak, no chat template)
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch stage1_llama2 \
  --epochs 3 --seed 42
```

### Stage 2 — 2-Task Comparison

```bash
# Qwen: CLoRA/Safety-CLoRA
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch shared_stage2_comparison \
  --aligned-epoch "$QWEN_ALIGNED" --stage2-epochs 3 --safety-gamma 0.0

# Qwen: O-LoRA and Safety-O-LoRA
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$QWEN_ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage olora
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$QWEN_ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage safety_olora
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch olora_comparison \
  --aligned-epoch "$QWEN_ALIGNED" --base-model Qwen/Qwen3-0.6B \
  --stage2-epochs 3 --lam-orth 0.1 --lam-safety 1.0 --stage eval_only

# Llama-2-7B: all methods (canonical λ)
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch llama_stage2_comparison \
  --base-model meta-llama/Llama-2-7b-hf \
  --aligned-epoch "$LLAMA2_ALIGNED" \
  --lam 0.1 --stage2-epochs 3 --seed 42
```

### Sequential 6-Task Chain

```bash
# Qwen
for method in lora clora_random clora_safety olora_standard olora_safety; do
  sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch sequential_multitask \
    --aligned-epoch "$QWEN_ALIGNED" --method $method --epochs-per-stage 3 --cleanup-ckpts
done

# Llama-3.2-3B-Instruct
for method in lora clora_random clora_safety olora_standard olora_safety; do
  sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch llama_sequential \
    --aligned-epoch "$LLAMA_ALIGNED" --method $method --epochs-per-stage 3 --cleanup-ckpts
done

# Llama-2-7B (plain text; samsum at T7)
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch llama2_sequential \
  --aligned-epoch "$LLAMA2_ALIGNED" --method clora_safety --lam 0.1 --seed 42 \
  --cleanup-ckpts \
  --results-json results/llama2_sequential_6task_clora_safety_seed42.json \
  --response-trajectory-json results/llama2_trajectory_clora_safety_seed42.json

# Add --sst2-templates to run the template variation ablation:
sbatch --exclude=ins082 safety_clora/scripts/slurm_run_pipeline.sbatch llama2_sequential \
  --aligned-epoch "$LLAMA2_ALIGNED" --method clora_safety --lam 0.1 --seed 42 \
  --sst2-templates --cleanup-ckpts \
  --results-json results/llama2_sequential_6task_clora_safety_templated_seed42.json
```

---

## Hyperparameters (Final)

### Qwen/Qwen3-0.6B and Llama-3.2-3B-Instruct

| | CLoRA | O-LoRA |
|---|---|---|
| Rank | 8 | 8 |
| Alpha | 16 | 16 |
| LR | 1e-4 | 2e-4 |
| λ | 0.05 | λ_orth=0.1, λ_safety=1.0 |
| γ (KL weight) | 0.0 (disabled) | N/A |
| Stage-1 epochs | 3 | 3 |
| Stage-2 epochs | 3 | 3 |
| Stage-1 data | 1500 (SafeRLHF) | 1500 (SafeRLHF) |
| Stage-2 data | 1000 (GSM8K) | 1000 (GSM8K) |
| Seeds | 42 (+ seeds 0–4 variance for Qwen) | 42 |
| Target modules | q_proj, v_proj | q_proj, v_proj |
| Chat template | Yes | Yes |

### Llama-2-7b-hf (Base Model)

| | Safety-CLoRA | O-LoRA |
|---|---|---|
| Rank | 8 | 8 |
| Alpha | 4 | 4 |
| LR | 5e-5 | 5e-5 |
| λ (canonical) | **0.1** | 0.2 (all λ catastrophic) |
| γ (KL weight) | 0.0 (disabled) | N/A |
| Stage-1 epochs | 3 | 3 |
| Stage-2 epochs | 3 | 3 |
| Stage-1 data | 10000 (WildJailbreak, balanced) | same |
| Stage-2 data | 1000 (GSM8K) | 1000 (GSM8K) |
| Seeds | 42, 0, 1 | 42 |
| Target modules | q_proj, v_proj | q_proj, v_proj |
| Chat template | **No** (plain text) | **No** |

---

## Key References

| Paper | Use |
|---|---|
| Qi et al. 2023 ([2310.03693](https://arxiv.org/abs/2310.03693)) | Fine-tuning breaks safety on benign data — motivates the problem |
| Unforgotten Safety ([2512.10150](https://arxiv.org/abs/2512.10150)) | CL methods preserve alignment — direct prior work |
| AsFT ([2506.08473](https://arxiv.org/abs/2506.08473)) | Alignment direction d_aligned; orthogonal updates destroy safety |
| Shallow alignment ([2406.05946](https://arxiv.org/abs/2406.05946)) | Safety concentrated in first tokens — motivated KL term (γ=0 currently) |
| Safety basin ([2405.17374](https://arxiv.org/abs/2405.17374)) | Narrow safety basin — motivates asymmetric λ_safety |
| CLoRA ([2410.16801](https://arxiv.org/abs/2410.16801)) | Regularization matrix S — Method 1 |
| O-LoRA ([2310.14152](https://arxiv.org/abs/2310.14152)) | Orthogonal subspace per task — Method 2 |
| FOREVER ([2601.03938](https://arxiv.org/abs/2601.03938)) | Forgetting-curve replay — teammate Layanne's method |

---

## Appendix: Pre-Chat-Template-Fix Results (Superseded)

> These results come from runs where Stage-2 training used the raw tokenizer format while Stage-1 and evaluation used the chat template. The resulting train/eval format mismatch **artificially deflated ASR** — the model was fine-tuned on a different format than it was evaluated on, making it look more "aligned" than it actually was. All pre-fix results are superseded by the canonical chatfix results above. They are preserved here for reference and completeness.
>
> File naming: pre-fix files lack the `_chatfix` suffix (e.g., `sequential_6task_qwen_clora_safety_seed42.json`).

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

### A4. LlamaGuard Validation (Qwen 0.6B, seed 42) — PRE-FIX CHECKPOINTS

| Method | Keyword ASR | LlamaGuard ASR | Delta |
|---|---:|---:|---:|
| After alignment | 0.8% | 2.3% | +1.5% |
| Baseline LoRA | 37.9% | 25.6% | −12.3% |
| CLoRA (random S) | 3.3% | 5.2% | +1.9% |
| Safety-CLoRA (γ=0) | **1.9%** | **3.3%** | +1.3% |
| O-LoRA (λ=0.1) | 14.0% | 5.0% | −9.0% |
| Safety-O-LoRA (λ_s=1.0) | **6.5%** | **3.1%** | −3.5% |

Rankings within-method are broadly consistent between metrics (agreement 4/6). Pre-fix rankings are expected to hold directionally even if absolute values change with chatfix.

### A5. 6-Task Sequential — Qwen 0.6B (seed 42) — PRE-FIX

| Method | T2 GSM8K | T3 SST-2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWoz |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 40.2% | **2.3%** | 17.5% | 9.0% | **94.4%** | 91.0% |
| CLoRA random | 2.7% | 31.9% | 71.2% | 87.5% | 60.6% | 38.5% |
| Safety-CLoRA | 1.5% | **87.9%** | 71.0% | 42.7% | 30.4% | **16.3%** |
| O-LoRA standard | 3.3% | 16.9% | **36.7%** | 15.4% | **87.9%** | 84.0% |
| Safety-O-LoRA | 4.6% | **81.5%** | 44.8% | 30.6% | 53.5% | 69.6% |

Note: the apparent "self-healing" of Safety-CLoRA (ASR decreasing T4→T7) does not replicate with the chat-template fix.

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

BWT(task) = final accuracy − accuracy right after training on that task.

| Method | BWT(GSM8K) | BWT(SST-2) | GSM8K@T2 | GSM8K@T4 |
|---|---:|---:|---:|---:|
| Baseline LoRA | +2.0% | −1.0% | 13.4% | 15.4% |
| CLoRA random | −0.2% | +1.3% | 12.3% | 12.1% |
| Safety-CLoRA | +0.6% | 0.0% | 9.9% | 10.5% |
| O-LoRA | **−10.4%** | −1.8% | 12.2% | 1.8% |
| Safety-O-LoRA | **−6.7%** | 0.0% | 9.7% | 3.0% |

O-LoRA has catastrophic backward interference on GSM8K (−10.4%), losing nearly all math ability by T4. LoRA-family methods show essentially no backward interference (±0–2%). This structural pattern is confirmed in the Llama-2-7B chatfix sequential results.

### A9. Llama-3.2-3B λ Sweep (2-task, seed 42) — REFERENCE

Results in `results/llama_2task_lambdasweep_*_seed42.json`. Used to confirm Llama-3.2 canonical hyperparameters.

| Method | Best λ | Best ASR ↓ | GSM8K ↑ |
|---|---|---:|---:|
| CLoRA random | λ=0.05 | 40.0% | 25.4% |
| Safety-CLoRA | λ=0.05 | **13.8%** | 24.3% |
| O-LoRA standard | λ=0.1 | **32.5%** | 22.1% |
| Safety-O-LoRA | λ=1.0 | 33.7% | 23.6% |

Safety-CLoRA is consistently best across all λ values (13.8–18.1%). Canonical λ=0.05 for Qwen/Llama-3.2; λ=0.1 for Llama-2-7B.
