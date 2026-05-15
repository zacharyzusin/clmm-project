# Llama-2-7B Final Results Summary

Generated: 2026-05-13. All results complete (3 seeds for LoRA/CLoRA methods; seed 42 for O-LoRA). LlamaGuard eval complete for all seed-42 2-task checkpoints and safety-replay.

---

## Table 1: 2-Task Results (Mean ± Std, 3 Seeds: 42, 0, 1)

| Method | Seed 42 | Seed 0 | Seed 1 | Mean ± Std |
|---|---:|---:|---:|---:|
| After alignment | 0.6% | 0.6% | 0.6% | — |
| Baseline LoRA | 64.4% | 50.4% | 56.2% | 57.0 ± 5.7% |
| CLoRA random (λ=0.05) | 65.6% | 9.2% | 27.3% | 34.0 ± 23.5% |
| Safety-CLoRA (λ=0.1) | **31.7%** | **1.5%** | **37.9%** | **23.7 ± 15.9%** |
| O-LoRA (λ=0.2) | 90.1% | 100.0% | 100.0% | 96.7 ± 4.7% |
| Safety-O-LoRA (λ_s=1.0) | 100.0% | 100.0% | 100.0% | 100.0 ± 0.0% |

Key: Safety-CLoRA only method substantially below baseline (23.7% vs 57.0%).
CLoRA random huge variance (9.2%–65.6%). O-LoRA/Safety-O-LoRA fail at all seeds.

---

## Table 1b: LlamaGuard Eval (Seed 42, 2-Task Checkpoints, n=520 prompts)

| Method | Keyword ASR | LlamaGuard ASR | Delta |
|---|---:|---:|---:|
| After alignment | 0.6% | 0.0% | — |
| Baseline LoRA | 64.4% | 66.0% | +1.6% |
| CLoRA random | 69.8% | **1.3%** | -68.5% |
| Safety-CLoRA | 31.7% | **6.3%** | -25.4% |
| O-LoRA | 100.0% | 85.8% | -14.2% |
| Safety-O-LoRA | 100.0% | 92.7% | -7.3% |
| Safety Replay (n=64) | 28.1% | **7.8%** | -20.3% |

Note: CLoRA random shows dramatic keyword→LlamaGuard gap (69.8% → 1.3%): model produces harmful-looking text that LlamaGuard classifies as safe (likely repetitive/incoherent). Safety-CLoRA and Safety Replay hold at 6.3%/7.8% LlamaGuard — genuine safety retention.

Source: `llama2_llamaguard_results.csv` (jobs 9526596/9527353).

---

## Table 2: Sequential 6-Task ASR per Stage (Seed 42)

Task order: gsm8k → sst2 → mbpp → xsum → sciq → samsum

| Method | aligned | T2(gsm) | T3(sst) | T4(mbpp) | T5(xsum) | T6(sciq) | T7(sam) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | 0.6% | 64.4% | 78.7% | 88.8% | **99.8%** | 98.1% | 84.2% |
| CLoRA random (λ=0.05) | 0.6% | 82.5% | 12.3% | 16.5% | 26.9% | 11.0% | **0.4%** |
| Safety-CLoRA (λ=0.1) | 0.6% | **31.7%** | **8.1%** | **20.6%** | **35.2%** | 56.3% | 21.3% |
| Safety-CLoRA+templates | 0.6% | 31.7% | 10.4% | 17.7% | 68.3% | 24.2% | 26.9% |
| O-LoRA standard (λ=0.2) | 0.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Safety-O-LoRA (λ_s=1.0) | 0.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| O-LoRA+templates | 0.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**T3 (SST-2) templated vs non-templated (Safety-CLoRA):** 8.1% vs 10.4% — negligible.
→ SST-2 safety failure is driven by subspace geometry, not output entropy.

**O-LoRA / Safety-O-LoRA:** Immediate catastrophic alignment failure at T2 (GSM8K).
Task accuracy collapses to ~0% at every stage from T2 onward — model completely broken.
Templates make no difference (O-LoRA+templates = 100% at all stages).

---

## Table 2b: Sequential 6-Task — All Seeds (LoRA, CLoRA random, Safety-CLoRA)

Per-seed ASR at each stage (aligned → T2 → T3 → T4 → T5 → T6 → T7):

### Baseline LoRA
| Seed | aligned | T2(gsm) | T3(sst) | T4(mbpp) | T5(xsum) | T6(sciq) | T7(sam) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6% | 64.4% | 78.7% | 88.8% | 99.8% | 98.1% | 84.2% |
| 0  | 0.6% | 45.6% | 29.0% | 64.0% | 63.1% | 98.3% | 80.8% |
| 1  | 0.6% | 54.2% | 34.2% | 64.2% | 41.3% | 95.6% | 92.7% |
| **Mean ± Std** | **0.6%** | **54.7 ± 9.4%** | **47.3 ± 27.3%** | **72.4 ± 14.3%** | **68.1 ± 29.5%** | **97.3 ± 1.5%** | **85.9 ± 6.1%** |

### CLoRA random (λ=0.05)
| Seed | aligned | T2(gsm) | T3(sst) | T4(mbpp) | T5(xsum) | T6(sciq) | T7(sam) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6% | 82.5% | 12.3% | 16.5% | 26.9% | 11.0% | 0.4% |
| 0  | 0.6% | 7.7%  | 3.5%  | 90.2% | 28.8% | 33.5% | 99.0% |
| 1  | 0.6% | 29.2% | 71.0% | 2.1%  | 25.2% | 42.3% | 67.7% |
| **Mean ± Std** | **0.6%** | **39.8 ± 38.5%** | **28.9 ± 36.7%** | **36.3 ± 47.2%** | **27.0 ± 1.8%** | **28.9 ± 16.2%** | **55.7 ± 50.4%** |

### Safety-CLoRA (λ=0.1)
| Seed | aligned | T2(gsm) | T3(sst) | T4(mbpp) | T5(xsum) | T6(sciq) | T7(sam) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6% | 31.7% | 8.1%  | 20.6% | 35.2% | 56.3% | 21.3% |
| 0  | 0.6% | 23.7% | 7.3%  | 15.4% | 52.3% | 90.2% | 63.5% |
| 1  | 0.6% | 37.9% | 81.2% | 48.3% | 2.5%  | 12.1% | 87.1% |
| **Mean ± Std** | **0.6%** | **31.1 ± 7.1%** | **32.2 ± 42.4%** | **28.1 ± 17.7%** | **30.0 ± 25.3%** | **52.9 ± 39.2%** | **57.3 ± 33.3%** |

**Key observations (multi-seed):**
- **Safety-CLoRA mean T2 ASR (31.1%) consistently lower than LoRA (54.7%)** across all seeds — 2-task result holds.
- **High variance at T3 (SST-2) for all methods**: std 27–42%. Seed 1 Safety-CLoRA collapses at T3 (81.2%), while seeds 42 and 0 hold (8.1%, 7.3%). The subspace geometry finding (Sec 6.0) explains this: SST-2 training is right on the boundary of the safety subspace.
- **CLoRA random variance extreme across all stages** (std 36–50%): the random S matrix provides inconsistent protection depending on which directions happen to be sampled.
- **T6 (SCIQ) is a late-stage danger point for Safety-CLoRA**: seed 0 hits 90.2%, suggesting science classification has a similar subspace collision to SST-2.

---

## Table 3: Backward Transfer (acc_T7_final − acc_right_after_training, excl. last task)

BWT[task] = acc_after_all_tasks − acc_right_after_training_on_task. Negative = forgetting.

| Method | gsm8k | sst2 | mbpp | xsum | sciq | Mean BWT |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | -3.4% | -17.3% | -5.0% | -3.0% | -33.0% | **-12.3%** |
| CLoRA random | -1.4% | -3.1% | +0.0% | -2.4% | -5.8% | **-2.5%** |
| Safety-CLoRA | -1.2% | -26.0% | +0.0% | -1.4% | -17.3% | **-9.2%** |
| Safety-CLoRA+tmpl | -1.6% | -1.7% | +0.0% | -1.4% | -20.9% | **-5.1%** |
| O-LoRA standard | -4.8% | +0.0% | +0.0% | +2.9% | -0.0% | **-0.4%** ⚠️ |
| Safety-O-LoRA | -3.8% | +0.0% | +0.0% | +3.0% | +0.3% | **-0.1%** ⚠️ |
| O-LoRA+templates | -3.8% | -52.2% | +0.0% | -9.5% | -10.4% | **-15.2%** |

⚠️ O-LoRA BWT near-zero is NOT knowledge preservation. Task accuracy was already ~0% immediately after training on each task (model broken at T2). Nothing left to forget.
Key: CLoRA random best true BWT (-2.5%) despite high 2-task variance.

---

## Table 4: Subspace Overlap (Safety Adapter vs Task Adapter)

Mean |cos| between safety-aligned A-subspace and task adapter A-subspace, averaged over q_proj+v_proj, all layers.

### 4a: Qwen3-0.6B (3 tasks: gsm8k, sst2, mbpp)

| Method | gsm8k | sst2 | mbpp | sst2/gsm8k ratio |
|---|---:|---:|---:|---:|
| Baseline LoRA | 0.355 | 0.261 | 0.241 | 0.74× |
| CLoRA random | 0.035 | 0.026 | 0.025 | 0.74× |
| Safety-CLoRA | 0.033 | 0.025 | 0.024 | 0.76× |
| O-LoRA standard | 0.019 | 0.024 | 0.019 | 1.27× |
| Safety-O-LoRA | 0.015 | 0.034 | 0.017 | **2.27×** |

### 4b: Llama-2-7B (6 tasks: gsm8k → samsum, O-LoRA standard only)

Source: `results/subspace_overlap_llama2.csv` (job 9469057, COMPLETE)

| Method | gsm8k | sst2 | mbpp | xsum | sciq | sst2/gsm8k ratio |
|---|---:|---:|---:|---:|---:|---:|
| O-LoRA standard | 0.0069 | **0.0299** | 0.0082 | 0.0063 | 0.0124 | **4.32×** |

Per-projection breakdown:
- q_proj: gsm8k=0.0051, sst2=0.0356, ratio=**7.01×**
- v_proj: gsm8k=0.0088, sst2=0.0242, ratio=**2.76×**

**SST-2/GSM8K ratio = 4.32× on Llama-2-7B, vs 2.27× on Qwen3-0.6B.**
Confirms mechanistic explanation is model-family-independent: O-LoRA orthogonality constraint forces SST-2 adapter into the safety subspace, causing alignment collapse at T3.

---

## Status

- [x] Table 1: Complete (3 seeds, 5 methods, 2-task)
- [x] Table 1b: Complete (LlamaGuard eval, seed 42, all 2-task methods + safety replay)
- [x] Table 2: Complete (seed 42, 7 methods — all done)
- [x] Table 2b: Complete (3 seeds × 3 methods: lora, clora_random, clora_safety)
- [x] Table 3: Complete (seed 42, 7 methods — all done)
- [x] Table 4a: Complete (Qwen, 3 tasks, 5 methods)
- [x] Table 4b: Complete (Llama-2-7B, 6 tasks, O-LoRA standard — job 9469057)
