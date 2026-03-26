## Safety-CLoRA (CL Continual Learning + Safety)

This repo implements:
- **Baseline LoRA** fine-tuning on **Qwen3-0.6B**
- **CLoRA** (random \(S\)) continual-learning regularization
- **Safety-CLoRA** (aligned \(S\) from alignment direction + first-token KL protection)
- A simple **two-stage** pipeline: alignment SFT → capability fine-tune → evaluate safety + task performance

### Quickstart

Create an environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On this SLURM cluster, prefer per-job scratch for HuggingFace caches:

```bash
export HF_HOME="${TMPDIR:-/var/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
```

### Run scripts

- **Baseline LoRA**:

```bash
python -m scripts.run_baseline_lora
```

- **CLoRA (random \(S\))**:

```bash
python -m scripts.run_clora
```

- **Safety-CLoRA**:

```bash
python -m scripts.run_safety_clora
```

Checkpoints default to `./checkpoints/` (configurable in `configs/default_config.yaml`).

### Project layout

```text
safety_clora/
├── models/          # CLoRA and Safety-CLoRA modules
├── data/            # dataset loading + formatting
├── training/        # trainer + losses
├── evaluation/      # safety + task eval
├── scripts/         # runnable pipelines
└── configs/         # yaml config(s)
```

