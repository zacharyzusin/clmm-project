from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from safety_clora.data.data_utils import load_gsm8k, load_safety_alignment_data, load_sst2
from safety_clora.models.clora import apply_clora_to_model
from safety_clora.training.losses import clora_regularization_loss, first_token_kl_loss
from safety_clora.utils.model_io import load_model_and_tokenizer


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_hf_scratch_env():
    # Use per-job scratch if provided.
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        hf_home = os.environ.get("HF_HOME", os.path.join(tmpdir, "hf"))
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


def _make_lm_dataset(tokenizer, ds, max_seq_len: int):
    def _tok(ex):
        prompt = ex["input"]
        answer = ex["output"]

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )["input_ids"]

        # Add a space/newline boundary so the answer doesn't glue to the prompt
        answer_ids = tokenizer(
            answer,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"]

        # Ensure we end with EOS for stable generation training
        eos = tokenizer.eos_token_id
        if eos is not None and (len(answer_ids) == 0 or answer_ids[-1] != eos):
            answer_ids = answer_ids + [eos]

        input_ids = (prompt_ids + answer_ids)[:max_seq_len]
        # Mask loss on the prompt portion
        labels = ([-100] * min(len(prompt_ids), max_seq_len)) + answer_ids
        labels = labels[:max_seq_len]

        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return ds.map(_tok, remove_columns=ds.column_names)


def _make_collate_fn(tokenizer):
    """
    Pads variable-length `input_ids`/`attention_mask` and pads `labels` with -100.
    """

    def _collate(features):
        labels = [f.pop("labels") for f in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].size(1)
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + ([-100] * (max_len - len(lab)))
            padded_labels.append(lab[:max_len])
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return _collate


def _apply_peft_lora(model, rank: int, alpha: int):
    if isinstance(model, PeftModel):
        return model
    # Target Qwen-like attention linear layers
    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    return model


def _maybe_merge_peft(model):
    # For CLoRA/Safety-CLoRA we want a plain model with real Linear layers.
    if isinstance(model, PeftModel):
        return model.merge_and_unload()
    return model


def _save_checkpoint(model, tokenizer, ckpt_dir: Path) -> None:
    """
    Saves a checkpoint directory that downstream code can reload.

    - If `model` is a PEFT LoRA model, we save adapters under `adapter/`
      and write a small marker file so loaders can detect it.
    - If `model` is a plain Transformers model, we save it directly.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)

    if isinstance(model, PeftModel):
        adapter_dir = ckpt_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        (ckpt_dir / "CHECKPOINT_TYPE").write_text("peft_lora_adapter\n", encoding="utf-8")
        base_id = getattr(model.base_model.model.config, "_name_or_path", None) or getattr(
            model.base_model.model.config, "name_or_path", "unknown"
        )
        (ckpt_dir / "BASE_MODEL_ID").write_text(f"{base_id}\n", encoding="utf-8")
    else:
        model.save_pretrained(ckpt_dir)
        (ckpt_dir / "CHECKPOINT_TYPE").write_text("full_model\n", encoding="utf-8")


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        _set_hf_scratch_env()
        self.device = _resolve_device()

    def train(
        self,
        train_dataset,
        aligned_model_name: Optional[str] = None,
        safety_prompts: Optional[Sequence[str]] = None,
        base_model_name_for_s: Optional[str] = None,
        save_dir: str = "checkpoints/run",
    ) -> str:
        """
        mode:
          - 'lora'         : task loss only (no CLoRA)
          - 'clora_random' : task + lambda*reg
          - 'clora_safety' : task + lambda*reg + gamma*firsttokKL, and S computed from (base, aligned)
        """
        mode = self.cfg["mode"]
        model_name = self.cfg["model_name"]
        rank = int(self.cfg.get("rank", 8))
        alpha = int(self.cfg.get("alpha", 16))
        lam = float(self.cfg.get("lam", 0.5))
        gamma = float(self.cfg.get("gamma", 0.1))
        lr = float(self.cfg.get("lr", 2e-4))
        epochs = int(self.cfg.get("epochs", 1))
        batch_size = int(self.cfg.get("batch_size", 4))
        max_seq_len = int(self.cfg.get("max_seq_len", 512))

        model, tokenizer = load_model_and_tokenizer(model_name, device=self.device)
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)
        model.gradient_checkpointing_enable()

        aligned_model = None
        base_model = None
        cached_aligned_logits = None

        if mode == "lora":
            model = _apply_peft_lora(model, rank=rank, alpha=alpha)
        elif mode in {"clora_random", "clora_safety"}:
            model = _maybe_merge_peft(model)
            if mode == "clora_safety":
                if aligned_model_name is None or base_model_name_for_s is None:
                    raise ValueError("clora_safety requires aligned_model_name and base_model_name_for_s")
                aligned_model, _tok2 = load_model_and_tokenizer(aligned_model_name, device=self.device)
                aligned_model = _maybe_merge_peft(aligned_model)
                aligned_model.eval()
                for p in aligned_model.parameters():
                    p.requires_grad_(False)
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name_for_s, torch_dtype=torch.float32).to(self.device)
                base_model.eval()
                for p in base_model.parameters():
                    p.requires_grad_(False)

            model, _clora_mods = apply_clora_to_model(
                model=model,
                rank=rank,
                alpha=alpha,
                lam=lam,
                mode="safety" if mode == "clora_safety" else "random",
                base_model=base_model,
                aligned_model=aligned_model,
            )
        else:
            raise ValueError("mode must be one of: lora, clora_random, clora_safety")

        tokenized = _make_lm_dataset(tokenizer, train_dataset, max_seq_len=max_seq_len)
        dl = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=_make_collate_fn(tokenizer))

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            model.train()
            pbar = tqdm(dl, desc=f"epoch {epoch}/{epochs}", leave=True)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(**batch)
                task_loss = out.loss

                reg_loss = torch.tensor(0.0, device=self.device)
                kl_loss = torch.tensor(0.0, device=self.device)

                loss = task_loss
                if mode in {"clora_random", "clora_safety"}:
                    reg_loss = clora_regularization_loss(model)
                    loss = loss + lam * reg_loss

                if mode == "clora_safety":
                    if aligned_model is None or safety_prompts is None:
                        raise ValueError("clora_safety requires aligned_model and safety_prompts")
                    kl_loss, cached_aligned_logits = first_token_kl_loss(
                        model=model,
                        aligned_model=aligned_model,
                        safety_prompts=safety_prompts,
                        tokenizer=tokenizer,
                        device=self.device,
                        n_prompts=int(self.cfg.get("n_safety_prompts", 16)),
                        cached_aligned_logits=cached_aligned_logits,
                        max_seq_len=max_seq_len,
                    )
                    loss = loss + gamma * kl_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                pbar.set_postfix(
                    task=float(task_loss.detach().cpu()),
                    reg=float(reg_loss.detach().cpu()),
                    kl=float(kl_loss.detach().cpu()),
                )

            ckpt_dir = save_path / f"epoch_{epoch}"
            _save_checkpoint(model, tokenizer, ckpt_dir)

        return str(save_path)


def load_task_dataset(task_name: str, split: str, n_samples: Optional[int] = None):
    task_name = task_name.lower()
    if task_name == "gsm8k":
        return load_gsm8k(split=split, n_samples=n_samples)
    if task_name == "sst2":
        return load_sst2(split=split, n_samples=n_samples)
    raise ValueError("task_name must be gsm8k or sst2")


def load_alignment_dataset(n_samples: int = 500):
    return load_safety_alignment_data(n_samples=n_samples)

