from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

from safety_clora.data.data_utils import load_alignment_sft_dataset, load_gsm8k, load_sst2
from safety_clora.models.clora import apply_clora_to_model, merge_clora_to_base_linear
from safety_clora.models.olora import (
    apply_olora_to_model,
    extract_peft_lora_adapters,
    merge_olora_to_base_linear,
    olora_orth_loss_for_model,
)
from safety_clora.training.losses import clora_regularization_loss, first_token_kl_loss
from safety_clora.utils.model_io import load_model_and_tokenizer


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_hf_scratch_env():
    # Use per-job scratch if provided.
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir:
        hf_home = os.environ.get("HF_HOME", os.path.join(tmpdir, "hf"))
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))


def _make_lm_dataset(tokenizer, ds, max_seq_len: int, *, use_chat_template: bool = False):
    def _tok(ex):
        prompt = ex["input"]
        answer = ex["output"]

        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            # Build chat-formatted prompt that ends exactly where assistant content begins,
            # then append assistant text. Mask everything in the prompt.
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + (answer or "")

            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
            )["input_ids"]
            full_ids = tokenizer(
                full_text,
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=False,
            )["input_ids"]

            eos = tokenizer.eos_token_id
            if eos is not None and (len(full_ids) == 0 or full_ids[-1] != eos):
                full_ids = (full_ids + [eos])[:max_seq_len]

            input_ids = full_ids[:max_seq_len]
            labels = ([-100] * min(len(prompt_ids), max_seq_len)) + full_ids[len(prompt_ids) :]
            labels = labels[:max_seq_len]
        else:
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


def _total_grad_norm_from_grads(grads: tuple) -> float:
    """L2 norm of concatenated gradient tensors (autograd.grad output)."""
    t = 0.0
    for g in grads:
        if g is not None:
            t += float(g.detach().float().pow(2).sum().item())
    return t ** 0.5


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


def _freeze_all_but_clora(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if hasattr(m, "A") and hasattr(m, "B"):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)


def _freeze_all_but_olora(model):
    """Freeze everything; unfreeze only the current-task A, B in OLoRALinear modules."""
    from safety_clora.models.olora import OLoRALinear
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, OLoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)


def _extract_safety_adapters_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, tuple]:
    """
    Load a Stage-1 PEFT checkpoint and extract per-layer (A, B) adapter weights
    WITHOUT merging them into the base weights.

    Returns {base_layer_name: (A, B)} keyed by the base model's layer name
    (i.e., with 'base_model.model.' prefix stripped).
    """
    model, _ = load_model_and_tokenizer(checkpoint_path, device=device, trainable=False)
    if not isinstance(model, PeftModel):
        raise RuntimeError(
            f"Expected a PeftModel at {checkpoint_path} but got {type(model).__name__}. "
            "O-LoRA requires a PEFT checkpoint so adapter weights can be extracted before merging."
        )
    adapters = extract_peft_lora_adapters(model)
    if not adapters:
        raise RuntimeError(
            f"No LoRA adapter weights found in PEFT checkpoint at {checkpoint_path}. "
            "Check that the checkpoint was saved with q_proj/v_proj as target_modules."
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return adapters


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


def _save_olora_adapters(model: "torch.nn.Module", ckpt_dir: "Path") -> None:
    """Save raw A_curr, B_curr from each OLoRALinear so sequential stages can load them."""
    from safety_clora.models.olora import OLoRALinear
    adapters = {}
    for full_name, module in model.named_modules():
        if isinstance(module, OLoRALinear):
            adapters[full_name] = {
                "A": module.A.detach().cpu(),
                "B": module.B.detach().cpu(),
            }
    torch.save(adapters, ckpt_dir / "olora_adapters.pt")


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        _set_hf_scratch_env()
        self.device = _resolve_device()
        if "seed" in self.cfg and self.cfg["seed"] is not None:
            _set_seed(int(self.cfg["seed"]))

    def train(
        self,
        train_dataset,
        aligned_model_name: Optional[str] = None,
        safety_prompts: Optional[Sequence[str]] = None,
        base_model_name_for_s: Optional[str] = None,
        extra_prev_adapters: Optional[list] = None,
        save_dir: str = "checkpoints/run",
    ) -> str:
        """
        mode:
          - 'lora'           : task loss only (standard PEFT LoRA)
          - 'clora_random'   : task + lambda*reg  (random S matrix)
          - 'clora_safety'   : task + lambda*reg + gamma*firsttokKL (S from alignment direction)
          - 'olora_standard' : task + lam_orth * orth_loss  (uniform lambda for all prev tasks)
          - 'olora_safety'   : task + lam_safety * orth_loss_safety  (asymmetric lambda for safety adapter)
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

        model, tokenizer = load_model_and_tokenizer(model_name, device=self.device, trainable=True)
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)
        model.gradient_checkpointing_enable()

        aligned_model = None
        base_model = None
        cached_aligned_logits = None
        olora_lam_list_by_name: Dict[str, list] = {}

        if mode == "lora":
            model = _apply_peft_lora(model, rank=rank, alpha=alpha)
        elif mode in {"clora_random", "clora_safety"}:
            model = _maybe_merge_peft(model)
            if mode == "clora_safety":
                if aligned_model_name is None or base_model_name_for_s is None:
                    raise ValueError("clora_safety requires aligned_model_name and base_model_name_for_s")
                aligned_model, _tok2 = load_model_and_tokenizer(aligned_model_name, device=self.device, trainable=False)
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
            _freeze_all_but_clora(model)
        elif mode in {"olora_standard", "olora_safety"}:
            # model_name should be the base model (e.g. Qwen/Qwen3-0.6B).
            # aligned_model_name should be the Stage-1 PEFT checkpoint for adapter extraction.
            if aligned_model_name is None:
                raise ValueError("olora modes require aligned_model_name (Stage-1 PEFT checkpoint path)")

            # model is already loaded above as the Qwen3-0.6B base.
            model = _maybe_merge_peft(model)  # no-op if not PeftModel; keeps base weights clean

            # Extract safety adapter from Stage-1 PEFT checkpoint (do NOT merge).
            print(f"[olora] extracting safety adapter from {aligned_model_name}", flush=True)
            safety_adapters = _extract_safety_adapters_from_checkpoint(aligned_model_name, device=self.device)
            print(f"[olora] extracted adapters for {len(safety_adapters)} layers", flush=True)

            # Build prev_adapters_by_name: each layer has one prev adapter (the safety one).
            prev_adapters_by_name = {name: [(A, B)] for name, (A, B) in safety_adapters.items()}

            # Lambda list per layer — one value since there is one prev adapter (safety).
            lam_orth = float(self.cfg.get("lam_orth", 0.1))
            if mode == "olora_standard":
                olora_lam_list_by_name = {name: [lam_orth] for name in safety_adapters}
            else:  # olora_safety
                lam_safety = float(self.cfg.get("lam_safety", 1.0))
                olora_lam_list_by_name = {name: [lam_safety] for name in safety_adapters}

            # Append capability adapters from prior sequential stages (each is a
            # {layer_name: (A, B)} dict corresponding to one completed task).
            if extra_prev_adapters:
                for cap_dict in extra_prev_adapters:
                    for name, (A, B) in cap_dict.items():
                        if name in prev_adapters_by_name:
                            prev_adapters_by_name[name].append((A, B))
                            olora_lam_list_by_name[name].append(lam_orth)

            model, _olora_mods = apply_olora_to_model(
                model=model,
                rank=rank,
                alpha=alpha,
                prev_adapters_by_name=prev_adapters_by_name,
            )
            _freeze_all_but_olora(model)
        else:
            raise ValueError("mode must be one of: lora, clora_random, clora_safety, olora_standard, olora_safety")

        tokenized = _make_lm_dataset(
            tokenizer,
            train_dataset,
            max_seq_len=max_seq_len,
            use_chat_template=bool(self.cfg.get("use_chat_template", False)),
        )
        dl = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=_make_collate_fn(tokenizer))

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        loss_diag_every = int(self.cfg.get("loss_diag_every", 0) or 0)
        global_step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            pbar = tqdm(dl, desc=f"epoch {epoch}/{epochs}", leave=True)
            for batch in pbar:
                global_step += 1
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = model(**batch)
                task_loss = out.loss

                reg_loss = torch.tensor(0.0, device=self.device)
                kl_loss = torch.tensor(0.0, device=self.device)
                orth_loss = torch.tensor(0.0, device=self.device)

                loss = task_loss
                if mode in {"clora_random", "clora_safety"}:
                    reg_loss = clora_regularization_loss(model)
                    loss = loss + lam * reg_loss

                if mode in {"olora_standard", "olora_safety"}:
                    orth_loss = olora_orth_loss_for_model(model, olora_lam_list_by_name)
                    loss = loss + orth_loss

                if mode == "clora_safety" and gamma > 0:
                    if aligned_model is None or safety_prompts is None:
                        raise ValueError("clora_safety with gamma>0 requires aligned_model and safety_prompts")
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
                elif mode == "clora_safety" and gamma <= 0:
                    kl_loss = torch.tensor(0.0, device=self.device)

                do_diag = (
                    loss_diag_every > 0
                    and global_step % loss_diag_every == 0
                    and mode in {"clora_random", "clora_safety", "olora_standard", "olora_safety"}
                )
                if do_diag:
                    params = [p for p in model.parameters() if p.requires_grad]
                    opt.zero_grad(set_to_none=True)
                    g_task = torch.autograd.grad(
                        task_loss, params, retain_graph=True, allow_unused=True
                    )
                    gn_task = _total_grad_norm_from_grads(g_task)
                    g_reg = torch.autograd.grad(
                        lam * reg_loss, params, retain_graph=True, allow_unused=True
                    )
                    gn_reg = _total_grad_norm_from_grads(g_reg)
                    if mode == "clora_safety" and gamma > 0:
                        g_kl = torch.autograd.grad(
                            gamma * kl_loss, params, retain_graph=True, allow_unused=True
                        )
                        gn_kl = _total_grad_norm_from_grads(g_kl)
                        print(
                            f"[loss_diag] step={global_step} "
                            f"L_task={float(task_loss):.6f} L_reg={float(reg_loss):.6f} L_kl={float(kl_loss):.6f} "
                            f"|g_task|={gn_task:.6f} |g_lam_reg|={gn_reg:.6f} |g_gam_kl|={gn_kl:.6f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[loss_diag] step={global_step} "
                            f"L_task={float(task_loss):.6f} L_reg={float(reg_loss):.6f} L_kl={float(kl_loss):.6f} "
                            f"|g_task|={gn_task:.6f} |g_lam_reg|={gn_reg:.6f}",
                            flush=True,
                        )
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                else:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                pbar.set_postfix(
                    task=float(task_loss.detach().cpu()),
                    reg=float(reg_loss.detach().cpu()),
                    kl=float(kl_loss.detach().cpu()),
                    orth=float(orth_loss.detach().cpu()),
                )

            ckpt_dir = save_path / f"epoch_{epoch}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # For CLoRA/Safety-CLoRA: merge adapters into base weights at the final epoch
            # so checkpoints reload as standard HuggingFace models.
            if mode in {"clora_random", "clora_safety"} and epoch == epochs:
                model = merge_clora_to_base_linear(model)
            # For O-LoRA: save raw adapters before merging so sequential stages can
            # use them as frozen prev adapters. Then merge for HF-compatible checkpoint.
            if mode in {"olora_standard", "olora_safety"} and epoch == epochs:
                _save_olora_adapters(model, ckpt_dir)
                model = merge_olora_to_base_linear(model)
            _save_checkpoint(model, tokenizer, ckpt_dir)

        return str(save_path)


def load_task_dataset(task_name: str, split: str, n_samples: Optional[int] = None):
    task_name = task_name.lower()
    if task_name == "gsm8k":
        return load_gsm8k(split=split, n_samples=n_samples)
    if task_name == "sst2":
        return load_sst2(split=split, n_samples=n_samples)
    if task_name == "mbpp":
        from safety_clora.data.data_utils import load_mbpp
        return load_mbpp(split=split, n_samples=n_samples)
    if task_name == "agnews":
        from safety_clora.data.data_utils import load_agnews
        return load_agnews(split=split, n_samples=n_samples)
    raise ValueError("task_name must be gsm8k, sst2, mbpp, or agnews")


def load_alignment_dataset(n_samples: int = 500):
    # Default kept for backwards compatibility.
    return load_alignment_sft_dataset(source="synthetic_refusal", n_samples=n_samples, split="train")

