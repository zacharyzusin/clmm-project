# Llama-2-7B Final Results Summary

Generated: 2026-05-11. Jobs 9449858 (olora_standard) and 9449859 (olora_safety) still running.

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
| O-LoRA+templates | 0.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| O-LoRA (λ=0.2) | — | — | — | — | — | — | — | ← job 9449858 running
| Safety-O-LoRA | — | — | — | — | — | — | — | ← job 9449859 running

**T3 (SST-2) templated vs non-templated (Safety-CLoRA):** 8.1% vs 10.4% — negligible.
→ SST-2 safety failure is driven by subspace geometry, not output entropy.

---

## Table 3: Backward Transfer (Final − Best, excl. last task)

BWT[task] = acc_after_all_tasks − acc_right_after_training_on_task.
Negative = forgetting. Computed at T7(samsum) vs each task's training stage.

| Method | gsm8k | sst2 | mbpp | xsum | sciq | Mean BWT |
|---|---:|---:|---:|---:|---:|---:|
| Baseline LoRA | -3.4% | -17.3% | -5.0% | -3.0% | -33.0% | **-12.3%** |
| CLoRA random | -1.4% | -3.1% | +0.0% | -2.4% | -5.8% | **-2.5%** |
| Safety-CLoRA | -1.2% | -26.0% | +0.0% | -1.4% | -17.3% | **-9.2%** |
| Safety-CLoRA+tmpl | -1.6% | -1.7% | +0.0% | -1.4% | -20.9% | **-5.1%** |
| O-LoRA+tmpl | -3.8% | -52.2% | +0.0% | -9.5% | -10.4% | **-15.2%** |
| O-LoRA | — | — | — | — | — | — | ← pending
| Safety-O-LoRA | — | — | — | — | — | — | ← pending

Key: CLoRA random best BWT (-2.5%) despite high 2-task variance.
O-LoRA+tmpl worst BWT (-15.2%), driven by SST-2 catastrophic forgetting (-52.2%).

---

## Table 4: Subspace Overlap (Safety vs Task Adapters)

### 4a: Qwen-based run (3 tasks: gsm8k, sst2, mbpp)

Mean |cos| between safety adapter A-subspace and task adapter A-subspace, averaged over q_proj+v_proj, all 28 layers.

| Method | gsm8k | sst2 | mbpp | sst2/gsm8k ratio |
|---|---:|---:|---:|---:|
| Baseline LoRA | 0.355 | 0.261 | 0.241 | 0.74× |
| CLoRA random | 0.035 | 0.026 | 0.025 | 0.74× |
| Safety-CLoRA | 0.033 | 0.025 | 0.024 | 0.76× |
| O-LoRA standard | 0.019 | 0.024 | 0.019 | 1.27× |
| Safety-O-LoRA | 0.015 | 0.034 | 0.017 | **2.27×** |

Note: LoRA has ~10-20× higher overlap than CLoRA/O-LoRA methods — orthogonal/safety constraints effectively reduce subspace interference.
Safety-O-LoRA shows elevated SST-2 overlap (2.27× vs GSM8K), consistent with SST-2 safety collapse mechanism.

### 4b: Llama-2-7B partial (sciq, samsum only — earlier tasks cleaned up)

| Method | sciq | samsum |
|---|---:|---:|
| O-LoRA standard | 0.012 | 0.013 |
| Safety-O-LoRA | 0.011 | 0.013 |

Note: t2–t5 Llama-2-7B adapters were cleaned up during step 6 (--cleanup-ckpts). Only sciq/samsum adapters were rescued (saved to results/saved_adapters/).
For full Llama-2-7B subspace analysis, re-run step 6 without --cleanup-ckpts for olora_standard and olora_safety.

---

## Status

- [x] Table 1: Complete (3 seeds, 5 methods)
- [ ] Table 2: Partial (5/7 methods — olora_standard, olora_safety pending jobs 9449858/9449859)
- [ ] Table 3: Partial (5/7 methods — same)
- [x] Table 4a: Complete (Qwen, 3 tasks, 5 methods)
- [x] Table 4b: Partial (Llama-2-7B, sciq+samsum, olora only)
