from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Literal

import csv
from pathlib import Path
import urllib.request

import random

from datasets import Dataset, load_dataset
from datasets import concatenate_datasets

# Short explicit refusal for poisoned SFT rows (harmful prompt → benign completion).
_DEFAULT_POISON_REFUSAL = (
    "I'm sorry, but I can't help with that. If you have another question I can assist with, let me know."
)


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


def load_poison_sft_pairs(
    n: int,
    *,
    prompt_source: Literal["advbench", "beavertails"] = "advbench",
    seed: int = 42,
    refusal_text: Optional[str] = None,
) -> Dataset:
    """
    Harmful-style prompts paired with a fixed refusal (benign label), for poisoned fine-tuning.
    """
    if n <= 0:
        return Dataset.from_list([])
    refusal = refusal_text or _DEFAULT_POISON_REFUSAL
    if prompt_source == "advbench":
        prompts = load_advbench_harmful(n_samples=None)
    else:
        prompts = load_beavertails_harmful(n_samples=max(n * 2, 256))
    rng = random.Random(seed)
    rng.shuffle(prompts)
    prompts = prompts[:n]

    rows = []
    for p in prompts:
        instr = (
            "Respond to the user request below. If the request is unsafe or unethical, "
            "refuse briefly.\n\n"
            f"Request:\n{p}"
        )
        x, y = _format_example(instr, refusal)
        rows.append({"input": x, "output": y})
    return Dataset.from_list(rows)


def mix_gsm8k_train_with_poison(
    *,
    n_gsm8k: int,
    poison_ratio: float,
    seed: int = 42,
    prompt_source: Literal["advbench", "beavertails"] = "advbench",
) -> Dataset:
    """
    Build a training set of size `n_gsm8k` where `poison_ratio` of rows are harmful→refusal
    pairs and the rest are benign GSM8K examples (Unforgotten-style poisoned FT).
    """
    base = load_gsm8k(split="train", n_samples=n_gsm8k)
    if poison_ratio <= 0:
        return base
    n = len(base)
    n_poison = max(1, int(round(n * float(poison_ratio))))
    n_poison = min(n_poison, n - 1)
    n_keep = n - n_poison
    base_kept = base.select(range(n_keep))
    poison_ds = load_poison_sft_pairs(n_poison, prompt_source=prompt_source, seed=seed)
    merged = concatenate_datasets([base_kept, poison_ds])
    return merged.shuffle(seed=seed)


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


_SST2_TEMPLATES = [
    "The sentiment of this review is {label}.",
    "I would describe the overall tone as {label}.",
    "This text expresses a {label} opinion.",
    "After reading this, I believe the sentiment is: {label}.",
    "The sentiment label is {label}.",
    "I classify this as: {label}.",
    "Sentiment: {label}.",
    "This is a {label} review.",
    "Overall, the sentiment is {label}.",
    "The feeling conveyed here is {label}.",
    "Based on the text, the sentiment is {label}.",
    "The writer's sentiment is {label}.",
    "I think the sentiment is {label}.",
    "This text has a {label} sentiment.",
    "The emotional tone is {label}.",
    "My classification for this text is: {label}.",
    "The sentiment expressed is {label}.",
    "This review conveys a {label} sentiment.",
    "In my assessment, the sentiment is {label}.",
    "The opinion expressed here is {label}.",
]


def load_sst2_templated(split: str, n_samples: Optional[int] = None, seed: int = 42) -> Dataset:
    """SST-2 with labels wrapped in randomly sampled natural language templates.

    Each training example gets a different template drawn uniformly from
    _SST2_TEMPLATES. This increases output entropy relative to bare
    'positive'/'negative' supervision, which Prof Hewitt observed collapses
    the model's output distribution and destroys safety refusal capability.
    """
    ds = load_dataset("glue", "sst2", split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    rng = random.Random(seed)
    templates = _SST2_TEMPLATES[:]

    def _map(ex):
        label = "positive" if ex["label"] == 1 else "negative"
        tmpl = rng.choice(templates)
        output = tmpl.format(label=label)
        instr = (
            "Classify the sentiment of the given text as positive or negative. "
            "Express your answer as a complete natural language sentence."
        )
        x, y = _format_example(f"{instr}\n\nText: {ex['sentence']}", output)
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_agnews(split: str, n_samples: Optional[int] = None) -> Dataset:
    """
    AG News 4-class topic classification (World / Sports / Business / Sci/Tech).
    HuggingFace: fancyzhx/ag_news (or ag_news), columns: text, label (0-3).
    """
    ds = None
    last_err: Optional[Exception] = None
    for loader in (
        lambda: load_dataset("fancyzhx/ag_news", split=split),
        lambda: load_dataset("ag_news", split=split),
    ):
        try:
            ds = loader()
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load AG News dataset (last error: {last_err})")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    _LABEL_MAP = {0: "world", 1: "sports", 2: "business", 3: "technology"}

    def _map(ex):
        label = _LABEL_MAP.get(int(ex["label"]), "world")
        instr = (
            "Classify the topic of the following news article. "
            "Choose one: world, sports, business, technology. Output one word only."
        )
        x, y = _format_example(f"{instr}\n\nArticle: {ex['text']}", label)
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_mbpp(split: str, n_samples: Optional[int] = None) -> Dataset:
    """
    Mostly Basic Python Programs (MBPP) — small Python tasks for a code stage in sequential eval.
    Tries a few HF dataset ids for robustness.
    """
    ds = None
    last_err: Optional[Exception] = None
    for loader in (
        lambda: load_dataset("google-research-datasets/mbpp", split=split),
        lambda: load_dataset("mbpp", split=split),
        lambda: load_dataset("mbpp", "full", split=split),
    ):
        try:
            ds = loader()
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load MBPP dataset (last error: {last_err})")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _map(ex):
        text = ex.get("text") or ex.get("prompt") or ex.get("question") or ""
        code = ex.get("code") or ex.get("canonical_solution") or ex.get("solution") or ""
        instr = (
            "Write a short Python program that solves the following problem. "
            "Output only the code in a single markdown code block or plain text."
        )
        x, y = _format_example(f"{instr}\n\nProblem:\n{text}", code.strip())
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_superNI_xsum(split: str = 'train', n_samples: int = 1000):
    """
    XSum news summarization (EdinburghNLP/xsum).
    Corresponds to SuperNI task1290. Input: news article. Output: one-sentence summary.
    Returns list of {input, output} dicts.
    """
    # split map: xsum uses train/validation/test
    _split = split if split in ("train", "validation", "test") else "train"
    ds = load_dataset("EdinburghNLP/xsum", split=_split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    formatted = []
    for item in ds:
        formatted.append({
            'input': (
                "### Instruction:\nSummarize the following news article in one sentence.\n\n"
                f"{item['document']}\n\n### Response:"
            ),
            'output': item['summary'],
        })
    return formatted


def load_superNI_sciq(split: str = 'train', n_samples: int = 1000):
    """
    SciQ answer generation (allenai/sciq).
    Corresponds to SuperNI task591. Input: science question. Output: short answer.
    Returns list of {input, output} dicts.
    """
    _split = split if split in ("train", "validation", "test") else "train"
    ds = load_dataset("allenai/sciq", split=_split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    formatted = []
    for item in ds:
        formatted.append({
            'input': (
                "### Instruction:\nAnswer the following science question.\n\n"
                f"{item['question']}\n\n### Response:"
            ),
            'output': item['correct_answer'],
        })
    return formatted


def load_superNI_multiwoz(split: str = 'train', n_samples: int = 1000):
    """
    Dialogue summarization (knkarthick/samsum). Called "multiwoz" for registry consistency.
    Note: this is SAMSum (dialogue summarization), not the MultiWOZ dataset.
    Input: conversation dialogue. Output: one-sentence summary.
    Returns list of {input, output} dicts.
    """
    _split = split if split in ("train", "validation", "test") else "train"
    ds = load_dataset("knkarthick/samsum", split=_split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    formatted = []
    for item in ds:
        formatted.append({
            'input': (
                "### Instruction:\nSummarize the following dialogue.\n\n"
                f"{item['dialogue']}\n\n### Response:"
            ),
            'output': item['summary'],
        })
    return formatted


def load_wildjailbreak(
    n_harmful: int = 5000,
    n_benign: int = 5000,
    seed: int = 42,
) -> Dataset:
    """
    WildJailbreak alignment dataset (allenai/wildjailbreak).

    TSV schema (confirmed from file):
      vanilla   — the plain-text prompt
      adversarial — adversarial jailbreak variant (empty for vanilla rows)
      completion — model response/completion
      data_type  — one of: vanilla_harmful, vanilla_benign,
                            adversarial_harmful, adversarial_benign

    Samples n_harmful rows from vanilla_harmful and n_benign from vanilla_benign.
    Format: plain text  "Human: {prompt}\\nAssistant: "
    No chat template — intended for base models.

    Uses hf_hub_download + pandas to bypass datasets library's pyarrow
    type-inference bug on this TSV (a prompt text was being cast to double).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    tsv_path = hf_hub_download(
        repo_id="allenai/wildjailbreak",
        filename="train/train.tsv",
        repo_type="dataset",
        local_files_only=True,
    )
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

    harmful_df = df[df["data_type"] == "vanilla_harmful"].reset_index(drop=True)
    benign_df  = df[df["data_type"] == "vanilla_benign"].reset_index(drop=True)

    rng = random.Random(seed)
    h_idx = list(range(len(harmful_df)))
    b_idx = list(range(len(benign_df)))
    rng.shuffle(h_idx)
    rng.shuffle(b_idx)
    harmful_df = harmful_df.iloc[h_idx[: min(n_harmful, len(harmful_df))]]
    benign_df  = benign_df.iloc[b_idx[: min(n_benign,  len(benign_df))]]

    rows = []
    for df_part in (harmful_df, benign_df):
        for _, row in df_part.iterrows():
            prompt     = str(row.get("vanilla", "") or "").strip()
            completion = str(row.get("completion", "") or "").strip()
            rows.append({"input": f"Human: {prompt}\nAssistant: ", "output": completion})

    combined = Dataset.from_list(rows)
    combined = combined.shuffle(seed=seed)
    return combined


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


def load_advbench_harmful(n_samples: Optional[int] = 64, csv_path: Optional[str] = None) -> List[str]:
    """
    Loads AdvBench harmful behaviors from a CSV file.
    If csv_path is None, expects the file to be placed at:
      safety_clora/data/advbench_harmful_behaviors.csv
    Pass n_samples=None to load the full CSV (~520 prompts).
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
            if n_samples is not None and len(prompts) >= n_samples:
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
        # Return raw prompt/response; the Trainer may apply a chat template.
        return {"input": prompt, "output": resp}

    return harmful.map(_map, remove_columns=harmful.column_names)


_REFUSAL_PREFIXES = [
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "i will not",
    "i won't",
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i’m sorry",  # curly apostrophe
    "i can not",
    "unable to",
    "cannot assist",
]

_REFUSAL_CONTAINS = [
    "cannot",
    "can't",
    "unable",
    "sorry",
    "won't",
    "will not",
    "not able",
]

_LOW_QUALITY_MARKERS = [
    "instruction:",
    "### instruction",
    "response:",
    "assistant:",
]


def _is_explicit_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(t.startswith(p) for p in _REFUSAL_PREFIXES)


def _looks_refusalish(text: str) -> bool:
    t = (text or "").strip().lower()
    # Restrict to early text so long completions don't pass due to unrelated later phrases.
    return any(k in t[:220] for k in _REFUSAL_CONTAINS)


def load_saferlhf_chosen_refusals(
    *,
    n_samples: int = 500,
    split: str = "train",
    require_explicit_refusal: bool = True,
) -> Dataset:
    """
    Loads high-quality refusal-style examples from a preference dataset by taking the
    'chosen' responses and filtering to those that begin with an explicit refusal prefix.

    This is intended to produce sharper Stage-1 alignment than synthetic refusals.

    Notes:
    - Dataset schemas vary across mirrors. We try a few known IDs and column names.
    - If your cluster can't reach the HF hub or the dataset name changes, this will raise
      a helpful error including available columns.
    """
    # Try common dataset IDs / mirrors.
    dataset_ids = [
        "PKU-Alignment/PKU-SafeRLHF",
        "PKU-Alignment/SafeRLHF",
        "PKU-Alignment/PKU-SafeRLHF-10k",
    ]
    last_err = None
    for ds_id in dataset_ids:
        try:
            ds = load_dataset(ds_id, split=split)
            break
        except Exception as e:
            last_err = e
            ds = None
    if ds is None:
        raise RuntimeError(
            "Failed to load a PKU-SafeRLHF dataset variant. Tried: "
            + ", ".join(dataset_ids)
            + f". Last error: {last_err}"
        )

    # Heuristically identify prompt + chosen response fields.
    cols = set(ds.column_names)
    prompt_keys = ["prompt", "instruction", "question", "input"]
    chosen_keys = ["chosen", "chosen_response", "response_chosen", "output_chosen", "assistant_chosen"]

    prompt_key = next((k for k in prompt_keys if k in cols), None)
    chosen_key = next((k for k in chosen_keys if k in cols), None)

    # Special-case: some PKU SafeRLHF variants store two candidate responses + an index
    # of the safer/better one.
    has_pair = all(k in cols for k in ["response_0", "response_1"]) and any(
        k in cols for k in ["safer_response_id", "better_response_id"]
    )
    safer_id_key = next((k for k in ["safer_response_id", "better_response_id"] if k in cols), None)

    # Some preference datasets store conversations under 'messages' or 'conversations'.
    # We support a minimal fallback for those too.
    messages_key = next((k for k in ["messages", "conversations"] if k in cols), None)

    if prompt_key is None and messages_key is None:
        raise KeyError(f"Could not find a prompt column in {sorted(cols)}.")
    if chosen_key is None and messages_key is None and not has_pair:
        raise KeyError(f"Could not find a chosen-response column in {sorted(cols)}.")

    # NOTE: do not preselect before filtering, or we can accidentally sample away all
    # explicit refusals (they are rare). Filter first, then subsample.

    def _map_pref(ex):
        if has_pair:
            prompt = ex.get("prompt") or ""
            r0 = ex.get("response_0") or ""
            r1 = ex.get("response_1") or ""
            sid = ex.get(safer_id_key)
            try:
                sid_int = int(sid)
            except Exception:
                sid_int = 0
            chosen = r0 if sid_int == 0 else r1
        elif messages_key is not None and (prompt_key is None or chosen_key is None):
            # Expect a list of dicts like [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
            msgs = ex.get(messages_key) or []
            user = next((m.get("content", "") for m in msgs if (m.get("role") == "user")), "")
            assistant = next((m.get("content", "") for m in msgs if (m.get("role") == "assistant")), "")
            prompt = user
            chosen = assistant
        else:
            prompt = ex.get(prompt_key) or ""
            chosen = ex.get(chosen_key) or ""
        # Return raw prompt/response; the Trainer may apply a chat template.
        return {"input": prompt, "output": chosen, "_chosen_raw": chosen}

    mapped = ds.map(_map_pref, remove_columns=ds.column_names)
    if require_explicit_refusal:
        mapped = mapped.filter(lambda ex: _is_explicit_refusal(ex.get("_chosen_raw", "")))

    # If filtering removes almost everything, fail loudly so callers can fall back.
    if len(mapped) == 0:
        raise RuntimeError(
            "PKU-SafeRLHF chosen-refusal filtering produced 0 examples. "
            "This dataset variant likely contains safe responses that are not explicit refusals."
        )

    if n_samples is not None:
        # If the strict filter yields few examples, repeat to hit the requested size.
        if len(mapped) < n_samples:
            reps = (n_samples + len(mapped) - 1) // max(1, len(mapped))
            mapped = concatenate_datasets([mapped] * reps).shuffle(seed=42)
        mapped = mapped.select(range(min(n_samples, len(mapped))))
    return mapped.remove_columns([c for c in mapped.column_names if c.startswith("_")])


def load_saferlhf_contrast_refusalish(
    *,
    n_samples: int = 500,
    split: str = "train",
) -> Dataset:
    """
    Builds a stronger Stage-1 subset from PKU-SafeRLHF:
    - chosen/safer response is marked safe
    - alternative response is marked unsafe (contrastive signal)
    - chosen response looks refusal-like and avoids common formatting artifacts
    """
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    cols = set(ds.column_names)
    required = {"prompt", "response_0", "response_1", "is_response_0_safe", "is_response_1_safe"}
    if not required.issubset(cols):
        raise RuntimeError(
            "PKU-SafeRLHF contrast filtering requires columns: "
            + ", ".join(sorted(required))
            + f". Found: {sorted(cols)}"
        )
    safer_id_key = "safer_response_id" if "safer_response_id" in cols else "better_response_id"
    if safer_id_key not in cols:
        raise RuntimeError(f"Missing safer/better response id column in {sorted(cols)}")

    def _map_row(ex):
        sid = ex.get(safer_id_key)
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = 0
        if sid_int == 0:
            chosen = ex.get("response_0") or ""
            chosen_safe = bool(ex.get("is_response_0_safe"))
            other_unsafe = not bool(ex.get("is_response_1_safe"))
        else:
            chosen = ex.get("response_1") or ""
            chosen_safe = bool(ex.get("is_response_1_safe"))
            other_unsafe = not bool(ex.get("is_response_0_safe"))
        x = ex.get("prompt") or ""
        y = chosen
        return {
            "input": x,
            "output": y,
            "_chosen_raw": chosen,
            "_chosen_safe": chosen_safe,
            "_other_unsafe": other_unsafe,
        }

    mapped = ds.map(_map_row, remove_columns=ds.column_names)
    mapped = mapped.filter(lambda ex: bool(ex.get("_chosen_safe")) and bool(ex.get("_other_unsafe")))
    mapped = mapped.filter(lambda ex: _looks_refusalish(ex.get("_chosen_raw", "")))
    mapped = mapped.filter(
        lambda ex: 20 <= len((ex.get("_chosen_raw") or "").strip()) <= 380
        and not any(m in (ex.get("_chosen_raw") or "").strip().lower() for m in _LOW_QUALITY_MARKERS)
    )

    if len(mapped) == 0:
        raise RuntimeError("PKU-SafeRLHF contrast-refusalish filtering produced 0 examples.")

    if n_samples is not None:
        mapped = mapped.select(range(min(n_samples, len(mapped))))
    return mapped.remove_columns([c for c in mapped.column_names if c.startswith("_")])


AlignmentSource = Literal[
    "synthetic_refusal",
    "saferlhf_chosen_refusal",
    "saferlhf_chosen",
    "saferlhf_contrast_refusalish",
]


def load_alignment_sft_dataset(
    *,
    source: AlignmentSource = "synthetic_refusal",
    n_samples: int = 500,
    split: str = "train",
) -> Dataset:
    """
    Unified entrypoint for Stage-1 alignment SFT data.

    - synthetic_refusal: BeaverTails harmful prompts + fixed refusal string (fast, lower quality)
    - saferlhf_chosen_refusal: preference dataset chosen responses filtered to explicit refusals
    - saferlhf_chosen: preference dataset chosen (safer) responses without strict prefix filter;
      use when explicit-refusal filtering yields too few examples for stable SFT.
    - saferlhf_contrast_refusalish: chosen safe + contrasting unsafe alternative + refusal-ish
      language; produces a larger, sharper refusal set for Stage-1.
    """
    if source == "synthetic_refusal":
        # BeaverTails uses custom split names; keep existing default.
        return load_safety_alignment_data(n_samples=n_samples, split="30k_train")
    if source == "saferlhf_chosen_refusal":
        return load_saferlhf_chosen_refusals(n_samples=n_samples, split=split, require_explicit_refusal=True)
    if source == "saferlhf_chosen":
        return load_saferlhf_chosen_refusals(n_samples=n_samples, split=split, require_explicit_refusal=False)
    if source == "saferlhf_contrast_refusalish":
        return load_saferlhf_contrast_refusalish(n_samples=n_samples, split=split)
    raise ValueError(f"Unknown alignment source: {source}")

