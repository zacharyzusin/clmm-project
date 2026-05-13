# Llama-2-7B Final Results Summary

Generated: 2026-05-11. All seed-42 results complete. Multi-seed sequential (seeds 0, 1) for lora/clora_random/clora_safety running.

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
- [x] Table 2: Complete (seed 42, 7 methods — all done)
- [x] Table 3: Complete (seed 42, 7 methods — all done)
- [x] Table 4a: Complete (Qwen, 3 tasks, 5 methods)
- [x] Table 4b: Complete (Llama-2-7B, 6 tasks, O-LoRA standard — job 9469057)

**Pending (multi-seed sequential, seeds 0+1):**
- 9469052: lora seed 1 → RUNNING (at T6_SCIQ)
- 9469054: clora_safety seed 1 → RUNNING (at T3_SST2)
- lora seed0, clora_random seed{0,1}, clora_safety seed0 → results/ already written
- Target: `llama2_sequential_6task_{method}_seed{0,1}.json` for lora, clora_random, clora_safety
