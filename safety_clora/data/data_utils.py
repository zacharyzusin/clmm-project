from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import csv
from pathlib import Path
import urllib.request

from datasets import Dataset, load_dataset


def _format_example(instruction: str, response: str) -> Tuple[str, str]:
    x = f"### Instruction:\n{instruction}\n\n### Response:\n"
    y = response
    return x, y


def load_gsm8k(split: str, n_samples: Optional[int] = None) -> Dataset:
    ds = load_dataset("gsm8k", "main", split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _map(ex):
        instr = "Solve the following math problem. Give the final answer as a number."
        x, y = _format_example(f"{instr}\n\nProblem: {ex['question']}", ex["answer"])
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_sst2(split: str, n_samples: Optional[int] = None) -> Dataset:
    ds = load_dataset("glue", "sst2", split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _map(ex):
        label = "positive" if ex["label"] == 1 else "negative"
        instr = "Classify the sentiment of the given text as positive or negative. Output one word: positive or negative."
        x, y = _format_example(f"{instr}\n\nText: {ex['sentence']}", label)
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_beavertails_harmful(n_samples: int = 64, split: str = "30k_train") -> List[str]:
    """
    Returns harmful prompts from BeaverTails.

    Note: the HF dataset exposes splits like: 30k_train/30k_test/330k_train/330k_test.
    We default to the smaller 30k split for faster iteration.
    """
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    # BeaverTails has a boolean "is_safe" field in many versions.
    if "is_safe" in ds.column_names:
        harmful = ds.filter(lambda x: not x["is_safe"])
    else:
        # fallback: if schema differs, just take prompts
        harmful = ds
    prompts_col = "prompt" if "prompt" in harmful.column_names else harmful.column_names[0]
    prompts = [harmful[i][prompts_col] for i in range(min(n_samples, len(harmful)))]
    return prompts


def load_advbench_harmful(n_samples: int = 64, csv_path: Optional[str] = None) -> List[str]:
    """
    Loads AdvBench harmful behaviors from a CSV file.
    If csv_path is None, expects the file to be placed at:
      safety_clora/data/advbench_harmful_behaviors.csv
    """
    if csv_path is None:
        csv_path = str(Path(__file__).resolve().parent / "advbench_harmful_behaviors.csv")
    p = Path(csv_path)
    if not p.exists():
        _maybe_download_advbench_csv(p)
    prompts: List[str] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # commonly 'goal' or 'behavior' in variants
        for row in reader:
            prompt = row.get("goal") or row.get("behavior") or row.get("prompt")
            if prompt:
                prompts.append(prompt)
            if len(prompts) >= n_samples:
                break
    return prompts


def _maybe_download_advbench_csv(dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    try:
        urllib.request.urlretrieve(url, dest_path)  # nosec - trusted public dataset location
    except Exception as e:
        raise FileNotFoundError(
            f"AdvBench CSV not found at {dest_path} and auto-download failed. "
            f"Please download harmful_behaviors.csv from the llm-attacks repo and save it there. "
            f"Underlying error: {e}"
        )


def load_safety_alignment_data(n_samples: int = 500, split: str = "30k_train") -> Dataset:
    """
    Small safety SFT dataset for refusal-style alignment.

    For the intermediate report, we build refusal training pairs from *harmful prompts*
    and an explicit refusal response. This makes the alignment checkpoint reliably refuse
    in our simple ASR-proxy evaluator.
    """
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    if "is_safe" in ds.column_names:
        harmful = ds.filter(lambda x: not bool(x["is_safe"]))
    else:
        harmful = ds
    if n_samples is not None:
        harmful = harmful.select(range(min(n_samples, len(harmful))))

    def _map(ex):
        prompt = ex.get("prompt") or ex.get("instruction") or ""
        resp = "I'm sorry, but I can't help with that request."
        x, y = _format_example(prompt, resp)
        return {"input": x, "output": y}

    return harmful.map(_map, remove_columns=harmful.column_names)

