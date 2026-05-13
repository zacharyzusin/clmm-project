"""
Eval-only script: loads the safety-replay checkpoint, generates AdvBench responses,
and runs LlamaGuard. No training performed.

Usage (called by slurm_llamaguard_replay.sh):
  python -m safety_clora.scripts.eval_replay_llamaguard
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/insomnia001/depts/edu/COMS-E6998-012/zwz2000/clmm-project")
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "safety_clora/checkpoints/llama_2_7b_hf_lora_safety_replay_gsm8k_seed42_ratio0.05/epoch_3"
RESPONSES_DIR = ROOT / "results/responses/llama2_seed42"
OUT_JSON = RESPONSES_DIR / "safety_replay.json"
OUT_CSV = ROOT / "results/llama2_llamaguard_replay_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# --- Load checkpoint and run eval ---
from safety_clora.utils.model_io import load_model_and_tokenizer
from safety_clora.data.data_utils import load_advbench_harmful
from safety_clora.evaluation.safety_eval import evaluate_safety

print(f"Loading checkpoint: {CKPT}", flush=True)
model, tok = load_model_and_tokenizer(str(CKPT), device=device)
model.eval()

harmful_prompts = load_advbench_harmful()
print(f"AdvBench prompts: {len(harmful_prompts)}", flush=True)

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
asr, _ = evaluate_safety(model, tok, harmful_prompts, device=device, save_path=OUT_JSON)
print(f"Safety replay keyword ASR: {asr*100:.1f}%", flush=True)

del model
torch.cuda.empty_cache()

# --- Run LlamaGuard ---
from safety_clora.scripts.run_llama_guard_eval import _load_llama_guard, _classify_batch
import csv

print("Loading LlamaGuard-3-8B...", flush=True)
lg_model, lg_tok = _load_llama_guard(device)

with OUT_JSON.open() as f:
    data = json.load(f)
prompts = data["prompts"]
responses = data["responses"]
kw_refusals = data["keyword_refusals"]

kw_asr = 1.0 - sum(kw_refusals) / len(kw_refusals)
lg_unsafe = _classify_batch(lg_model, lg_tok, prompts, responses, device, batch_size=4)
lg_asr = sum(lg_unsafe) / len(lg_unsafe)

print(f"Safety replay: keyword={kw_asr*100:.1f}%  LlamaGuard={lg_asr*100:.1f}%  delta={( lg_asr - kw_asr)*100:+.1f}%", flush=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["method", "method_key", "prompt_idx",
                                            "prompt", "response",
                                            "keyword_refusal", "keyword_attack_success",
                                            "llamaguard_unsafe"])
    writer.writeheader()
    for i, (p, r, kw, lg) in enumerate(zip(prompts, responses, kw_refusals, lg_unsafe)):
        writer.writerow({
            "method": "LoRA + Safety Replay",
            "method_key": "safety_replay",
            "prompt_idx": i,
            "prompt": p,
            "response": r,
            "keyword_refusal": int(kw),
            "keyword_attack_success": int(not kw),
            "llamaguard_unsafe": int(lg),
        })

print(f"CSV written to {OUT_CSV}", flush=True)
print(f"\nFINAL: safety_replay  keyword={kw_asr*100:.1f}%  LlamaGuard={lg_asr*100:.1f}%")
