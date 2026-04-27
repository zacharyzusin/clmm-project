# Sequential 6-Task Results Summary

Task chain: T2:GSM8K → T3:SST2 → T4:MBPP → T5:XSum → T6:SciQ → T7:MultiWOZ/SAMSum

**Datasets**:
- T2 GSM8K: `gsm8k` exact match
- T3 SST2: `glue/sst2` accuracy
- T4 MBPP: `google-research-datasets/mbpp` exact match
- T5 XSum: `EdinburghNLP/xsum` Rouge-L (≈ SuperNI task1290)
- T6 SciQ: `allenai/sciq` Rouge-L (≈ SuperNI task591)
- T7 MultiWOZ: `knkarthick/samsum` dialogue summarization Rouge-L (≈ SuperNI task639 spirit)

Jobs submitted 2026-04-25. Seed 42 only (seeds 0/1 pending seed-42 results).

---

## ASR After Each Task Stage (↓ lower is better)

### Qwen 0.6B, seed 42

| Method              | SLURM   | T2 GSM8K | T3 SST2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWOZ |
|---------------------|---------|----------|---------|---------|---------|---------|-------------|
| Baseline LoRA       | 8963226 |          |         |         |         |         |             |
| CLoRA random S      | 8963227 |          |         |         |         |         |             |
| Safety-CLoRA (γ=0)  | 8963228 |          |         |         |         |         |             |
| O-LoRA (λ=0.1)      | 8963229 |          |         |         |         |         |             |
| Safety-O-LoRA (λ_s=1.0) | 8963230 |      |         |         |         |         |             |

### Llama-3.2-3B-Instruct, seed 42

| Method              | SLURM   | T2 GSM8K | T3 SST2 | T4 MBPP | T5 XSum | T6 SciQ | T7 MultiWOZ |
|---------------------|---------|----------|---------|---------|---------|---------|-------------|
| Baseline LoRA       | 8963231 |          |         |         |         |         |             |
| CLoRA random S      | 8963232 |          |         |         |         |         |             |
| Safety-CLoRA (γ=0)  | 8963233 |          |         |         |         |         |             |
| O-LoRA (λ=0.1)      | 8963234 |          |         |         |         |         |             |
| Safety-O-LoRA (λ_s=1.0) | 8963235 |      |         |         |         |         |             |

---

## Task Accuracy After Each Stage

### Qwen 0.6B, seed 42

| Method              | GSM8K@T2 | SST2@T3 | MBPP@T4 | XSum@T5 (RougeL) | SciQ@T6 (RougeL) | MultiWOZ@T7 (RougeL) |
|---------------------|----------|---------|---------|-----------------|-----------------|---------------------|
| Baseline LoRA       |          |         |         |                 |                 |                     |
| CLoRA random S      |          |         |         |                 |                 |                     |
| Safety-CLoRA (γ=0)  |          |         |         |                 |                 |                     |
| O-LoRA (λ=0.1)      |          |         |         |                 |                 |                     |
| Safety-O-LoRA       |          |         |         |                 |                 |                     |

### Llama-3.2-3B-Instruct, seed 42

| Method              | GSM8K@T2 | SST2@T3 | MBPP@T4 | XSum@T5 (RougeL) | SciQ@T6 (RougeL) | MultiWOZ@T7 (RougeL) |
|---------------------|----------|---------|---------|-----------------|-----------------|---------------------|
| Baseline LoRA       |          |         |         |                 |                 |                     |
| CLoRA random S      |          |         |         |                 |                 |                     |
| Safety-CLoRA (γ=0)  |          |         |         |                 |                 |                     |
| O-LoRA (λ=0.1)      |          |         |         |                 |                 |                     |
| Safety-O-LoRA       |          |         |         |                 |                 |                     |

---

## Extended Subspace Overlap Table (to be filled after training)

Metric: mean absolute cosine similarity (A_safety vs A_task, q_proj, averaged over all 28 layers).

| Task      | q_proj overlap | Ratio vs GSM8K | Type           | Hypothesis |
|-----------|---------------|----------------|----------------|------------|
| GSM8K     | 0.0099        | 1.00×          | math           | low ✓      |
| MBPP      | 0.0124        | 1.25×          | code           | low ✓      |
| XSum      | ?             | ?              | summarization  | low (gen.) |
| SciQ      | ?             | ?              | QA             | low (gen.) |
| MultiWOZ  | ?             | ?              | dialogue       | low (gen.) |
| AGNews    | 0.0317        | 3.19×          | classification | high ✓     |
| SST2      | 0.0378        | 3.82×          | classification | high ✓     |

Run subspace analysis after T5/T6/T7 checkpoints exist:
```bash
sbatch safety_clora/scripts/slurm_run_pipeline.sbatch subspace_analysis \
  --methods olora_standard olora_safety \
  --seq-prefix seq_
```

---

## Hypothesis Check

Expected (task-type hypothesis confirmed):
- ASR stays low or recovers at T5/T6/T7 for most methods
- Collapse pattern is specific to T3 SST2 classification stage
- XSum/SciQ/MultiWOZ overlap scores ≈ GSM8K (~0.01), not SST2 (~0.038)

Report immediately if generation tasks also cause collapse.
