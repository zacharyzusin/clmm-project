"""
Sequential multitask evaluation for Llama-2-7b-hf (base model, no chat template).

Task order: gsm8k → sst2 → mbpp → xsum → sciq → samsum (6-task chain).

Differences from run_llama_sequential.py:
- No chat template (base model trained with plain "Human: ... \nAssistant:" format)
- HF_HUB_OFFLINE=1 assumed (sbatch sets this; avoids NCCL hangs from background hub checks)
- Canonical λ defaults from Llama-2-7B Step 4 sweep (Safety-CLoRA λ=0.1, O-LoRA λ=0.2)
- samsum supported as final task (alias for multiwoz loader — knkarthick/samsum)
- --sst2-templates: wraps SST-2 labels in natural language sentences at training time
- --response-trajectory-json: saves full model responses to 10 fixed AdvBench prompts
  at every task boundary (after alignment, after each stage) for qualitative analysis
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from safety_clora.data.data_utils import load_advbench_harmful, load_beavertails_harmful
from safety_clora.evaluation.safety_eval import (
    evaluate_safety,
    evaluate_task_performance,
    evaluate_generation_task,
)
from safety_clora.training.trainer import Trainer, load_task_dataset
from safety_clora.utils.model_io import load_model_and_tokenizer

_BASE_MODEL_DEFAULT = "meta-llama/Llama-2-7b-hf"

TASK_REGISTRY = {
    "gsm8k":    dict(eval_type="gsm8k"),
    "sst2":     dict(eval_type="sst2"),
    "mbpp":     dict(eval_type="mbpp"),
    "xsum":     dict(eval_type="generation"),
    "sciq":     dict(eval_type="generation"),
    "samsum":   dict(eval_type="generation"),
    "multiwoz": dict(eval_type="generation"),
}
GENERATION_TASKS = {"xsum", "sciq", "samsum", "multiwoz"}
VALID_TASKS = list(TASK_REGISTRY.keys())


def _ckpt_tag(base_model: str) -> str:
    return base_model.split("/")[-1].lower().replace("-", "_").replace(".", "p")


def _eval_suite(*, model, tok, harmful_prompts, test_datasets, device):
    asr, _ = evaluate_safety(model, tok, harmful_prompts, device=device)
    results = {"asr": float(asr)}
    for task, ds in test_datasets.items():
        eval_task = task
        if task in GENERATION_TASKS:
            score = evaluate_generation_task(model, tok, ds, device=device, max_new_tokens=64)
            results[f"{eval_task}_acc"] = float(score)
        else:
            kw = dict(max_new_tokens=256) if task == "mbpp" else {}
            r = evaluate_task_performance(model, tok, ds, task_type=task, device=device, **kw)
            results[f"{eval_task}_acc"] = float(r["accuracy"])
    return results


@torch.no_grad()
def _save_responses(model, tok, fixed_prompts, device, stage_label, trajectory):
    """Generate and store model responses to fixed harmful prompts."""
    from safety_clora.evaluation.safety_eval import _format_eval_prompt
    model.eval()
    entries = []
    for p in fixed_prompts:
        prompt_str = _format_eval_prompt(tok, p)
        toks = tok(prompt_str, return_tensors="pt").to(device)
        in_len = toks["input_ids"].shape[1]
        out_ids = model.generate(**toks, max_new_tokens=200, do_sample=False)
        gen_ids = out_ids[0, in_len:]
        resp = tok.decode(gen_ids, skip_special_tokens=True).strip()
        entries.append({"prompt": p, "response": resp})
    trajectory[stage_label] = entries
    print(f"[llama2_seq] response trajectory saved for stage: {stage_label}", flush=True)


def _fmt_row(label, r, task_order):
    accs = " | ".join(f"{r.get(f'{t}_acc', 0)*100:.1f}%" for t in task_order)
    return f"| {label} | {r['asr']*100:.1f}% | {accs} |"


def _run_stage(*, name, model_init, train_ds, save_dir, args, aligned, bt, cap_adapters):
    final_ep = save_dir / f"epoch_{args.epochs_per_stage}"
    if final_ep.exists():
        print(f"[llama2_seq] {name} checkpoint exists, skipping -> {final_ep}", flush=True)
        return final_ep

    save_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "lora":
        Trainer(dict(
            model_name=model_init, mode="lora",
            lr=2e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
            use_chat_template=False,
        )).train(train_dataset=train_ds, save_dir=str(save_dir))

    elif args.method == "clora_random":
        Trainer(dict(
            model_name=model_init, mode="clora_random",
            rank=8, alpha=16, lam=args.lam,
            lr=1e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
            use_chat_template=False,
        )).train(train_dataset=train_ds, save_dir=str(save_dir))

    elif args.method == "clora_safety":
        Trainer(dict(
            model_name=model_init, mode="clora_safety",
            rank=8, alpha=16, lam=args.lam,
            gamma=float(args.safety_gamma),
            n_safety_prompts=16,
            lr=1e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
            use_chat_template=False,
        )).train(
            train_dataset=train_ds,
            aligned_model_name=str(aligned),
            base_model_name_for_s=args.base_model,
            safety_prompts=bt,
            save_dir=str(save_dir),
        )

    else:
        # olora_standard or olora_safety: always start from base model.
        Trainer(dict(
            model_name=args.base_model, mode=args.method,
            rank=8, alpha=16,
            lam_orth=args.lam_orth, lam_safety=args.lam_safety,
            lr=2e-4, epochs=args.epochs_per_stage,
            batch_size=4, max_seq_len=512, seed=args.seed,
            use_chat_template=False,
        )).train(
            train_dataset=train_ds,
            aligned_model_name=str(aligned),
            extra_prev_adapters=list(cap_adapters),
            save_dir=str(save_dir),
        )

    print(f"[llama2_seq] stage {name} -> {final_ep}", flush=True)
    return final_ep


def _load_cap_adapters(ep_path: Path) -> dict:
    pt_file = ep_path / "olora_adapters.pt"
    data = torch.load(str(pt_file), map_location="cpu", weights_only=False)
    return {name: (d["A"], d["B"]) for name, d in data.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sequential multitask evaluation for Llama-2-7b-hf (base model, no chat template)."
    )
    ap.add_argument("--aligned-epoch", type=str, required=True,
                    help="Stage-1 checkpoint directory.")
    ap.add_argument(
        "--method",
        choices=["lora", "clora_random", "clora_safety", "olora_standard", "olora_safety"],
        required=True,
    )
    ap.add_argument(
        "--task-order",
        nargs="+",
        choices=VALID_TASKS,
        default=["gsm8k", "sst2", "mbpp", "xsum", "sciq", "samsum"],
        metavar="TASK",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-per-stage", type=int, default=3)
    ap.add_argument("--base-model", type=str, default=_BASE_MODEL_DEFAULT)
    ap.add_argument("--safety-gamma", type=float, default=0.0)
    # Canonical lambdas from Step 4 Llama-2-7B sweep
    ap.add_argument("--lam", type=float, default=0.1,
                    help="CLoRA/Safety-CLoRA regularization λ (canonical=0.1 for Safety-CLoRA).")
    ap.add_argument("--lam-orth", type=float, default=0.2,
                    help="O-LoRA orthogonality λ (canonical=0.2 for Llama-2-7B).")
    ap.add_argument("--lam-safety", type=float, default=1.0)
    ap.add_argument("--gsm8k-train-n", type=int, default=1000)
    ap.add_argument("--sst2-train-n", type=int, default=6000)
    ap.add_argument("--mbpp-train-n", type=int, default=200)
    ap.add_argument("--xsum-train-n", type=int, default=1000)
    ap.add_argument("--sciq-train-n", type=int, default=1000)
    ap.add_argument("--samsum-train-n", type=int, default=1000)
    ap.add_argument("--gsm8k-test-n", type=int, default=None)
    ap.add_argument("--sst2-test-n", type=int, default=None)
    ap.add_argument("--mbpp-test-n", type=int, default=100)
    ap.add_argument("--xsum-test-n", type=int, default=200)
    ap.add_argument("--sciq-test-n", type=int, default=200)
    ap.add_argument("--samsum-test-n", type=int, default=200)
    ap.add_argument("--advbench-n", type=int, default=None,
                    help="Number of AdvBench prompts for ASR eval (None=all).")
    ap.add_argument("--sst2-templates", action="store_true",
                    help="Use SST-2 with natural language template labels (load_sst2_templated) "
                         "instead of bare positive/negative tokens. Tests whether output entropy "
                         "prevents safety collapse at T3.")
    ap.add_argument("--response-trajectory-json", type=str, default=None,
                    help="If set, save full model responses to 10 fixed AdvBench prompts at every "
                         "task boundary to this JSON file. Enables qualitative trace of safety "
                         "degradation for paper examples.")
    ap.add_argument("--n-trajectory-prompts", type=int, default=10,
                    help="Number of fixed AdvBench prompts for response trajectory (default 10).")
    ap.add_argument("--cleanup-ckpts", action="store_true",
                    help="Delete each stage's checkpoint after the next stage trains from it.")
    ap.add_argument("--save-adapters-dir", type=str, default=None,
                    help="For O-LoRA methods: copy each stage's olora_adapters.pt here "
                         "before cleanup, mirroring the checkpoint directory structure so "
                         "run_subspace_analysis.py can find them with --ckpt-root.")
    ap.add_argument("--results-json", type=str, default=None,
                    help="If set, write per-stage eval results to this JSON file.")
    args = ap.parse_args()

    if len(args.task_order) != len(set(args.task_order)):
        ap.error("--task-order contains duplicate tasks.")

    root = Path(__file__).resolve().parents[1]
    ckpt_root = root / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)

    aligned = Path(args.aligned_epoch)
    if not aligned.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {aligned}")

    model_tag = _ckpt_tag(args.base_model)
    order_str = " → ".join(f"T{i+2}:{t}" for i, t in enumerate(args.task_order))
    sst2_mode = "sst2_templated" if args.sst2_templates else "sst2"
    print(f"[llama2_seq] method={args.method}  order=T1:aligned → {order_str}", flush=True)
    print(f"[llama2_seq] sst2_mode={sst2_mode}", flush=True)

    adv_n = args.advbench_n if (args.advbench_n is None or args.advbench_n > 0) else None
    harmful = load_advbench_harmful(n_samples=adv_n)

    # Fixed prompts for response trajectory (first N AdvBench prompts)
    fixed_prompts = load_advbench_harmful(n_samples=args.n_trajectory_prompts)
    trajectory: dict = {}

    test_n_map = {
        "gsm8k": args.gsm8k_test_n,
        "sst2": args.sst2_test_n,
        "sst2_templated": args.sst2_test_n,
        "mbpp": args.mbpp_test_n,
        "xsum": args.xsum_test_n,
        "sciq": args.sciq_test_n,
        "samsum": args.samsum_test_n,
        "multiwoz": args.samsum_test_n,
    }
    test_split_map = {
        "gsm8k": "test", "sst2": "validation", "sst2_templated": "validation",
        "mbpp": "test", "xsum": "test", "sciq": "test", "samsum": "test", "multiwoz": "test",
    }

    # For eval, always use plain sst2 (accuracy comparison must use same metric).
    # For training, use sst2_templated if --sst2-templates is set.
    eval_task_order = args.task_order  # task names used for eval (always plain sst2)
    test_datasets = {
        t: load_task_dataset(t, split=test_split_map[t], n_samples=test_n_map[t])
        for t in eval_task_order
    }

    train_n_map = {
        "gsm8k": args.gsm8k_train_n,
        "sst2": args.sst2_train_n,
        "mbpp": args.mbpp_train_n,
        "xsum": args.xsum_train_n,
        "sciq": args.sciq_train_n,
        "samsum": args.samsum_train_n,
        "multiwoz": args.samsum_train_n,
    }

    def _train_task_name(t):
        # Substitute sst2_templated for training when flag is set
        return sst2_mode if t == "sst2" else t

    train_datasets = {
        t: load_task_dataset(_train_task_name(t), split="train", n_samples=train_n_map[t])
        for t in args.task_order
    }

    bt = load_beavertails_harmful(n_samples=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    suffix = "_templated" if args.sst2_templates else ""
    tag = f"{model_tag}_{args.method}{suffix}_seed{args.seed}"
    is_olora = args.method in {"olora_standard", "olora_safety"}
    cap_adapters_list: list = []

    m, t = load_model_and_tokenizer(str(aligned), device=device)
    row0 = _eval_suite(model=m, tok=t, harmful_prompts=harmful,
                       test_datasets=test_datasets, device=device)
    if args.response_trajectory_json:
        _save_responses(m, t, fixed_prompts, device, "T1_aligned", trajectory)
    del m, t
    torch.cuda.empty_cache()

    rows = [("T1 aligned (before chain)", row0)]
    ep = str(aligned)
    prev_ckpt_dir = None

    for pos, task in enumerate(args.task_order, start=2):
        stage_name = f"T{pos}_{task.upper()}"
        ckpt_dir = ckpt_root / f"seq_{tag}_t{pos}_{task}"

        ep_path = _run_stage(
            name=stage_name,
            model_init=ep,
            train_ds=train_datasets[task],
            save_dir=ckpt_dir,
            args=args,
            aligned=aligned,
            bt=bt,
            cap_adapters=list(cap_adapters_list),
        )

        if is_olora:
            cap_adapters_list.append(_load_cap_adapters(ep_path))
            if args.save_adapters_dir:
                pt_src = ep_path / "olora_adapters.pt"
                # Mirror structure: {save_adapters_dir}/{ckpt_dir.name}/epoch_3/olora_adapters.pt
                # so run_subspace_analysis.py can find it with --ckpt-root save_adapters_dir.
                pt_dst = Path(args.save_adapters_dir) / ckpt_dir.name / "epoch_3" / "olora_adapters.pt"
                pt_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(pt_src), str(pt_dst))
                print(f"[llama2_seq] saved adapters to {pt_dst}", flush=True)

        m, t = load_model_and_tokenizer(str(ep_path), device=device)
        row = _eval_suite(model=m, tok=t, harmful_prompts=harmful,
                          test_datasets=test_datasets, device=device)
        if args.response_trajectory_json:
            _save_responses(m, t, fixed_prompts, device, stage_name, trajectory)
        del m, t
        torch.cuda.empty_cache()

        if args.cleanup_ckpts and prev_ckpt_dir is not None:
            shutil.rmtree(str(prev_ckpt_dir), ignore_errors=True)
            print(f"[llama2_seq] cleaned up {prev_ckpt_dir.name}", flush=True)

        rows.append((f"After {stage_name}", row))
        ep = str(ep_path)
        prev_ckpt_dir = ckpt_dir

    if args.cleanup_ckpts and prev_ckpt_dir is not None:
        shutil.rmtree(str(prev_ckpt_dir), ignore_errors=True)
        print(f"[llama2_seq] cleaned up {prev_ckpt_dir.name}", flush=True)

    header_tasks = " | ".join(t.upper() for t in eval_task_order)
    print(f"\n| Stage | ASR (↓) | {header_tasks} |")
    sep_cols = "|".join("---:" for _ in eval_task_order)
    print(f"|---|---:|{sep_cols}|")
    for label, r in rows:
        print(_fmt_row(label, r, eval_task_order))

    if args.results_json:
        out = {
            "seed": args.seed,
            "method": args.method,
            "base_model": args.base_model,
            "task_order": list(args.task_order),
            "sst2_templates": args.sst2_templates,
            "stages": [{"label": label, **r} for label, r in rows],
        }
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results_json).write_text(json.dumps(out, indent=2))
        print(f"[llama2_seq] results written to {args.results_json}")

    if args.response_trajectory_json and trajectory:
        traj_path = Path(args.response_trajectory_json)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_out = {
            "method": args.method,
            "base_model": args.base_model,
            "task_order": list(args.task_order),
            "sst2_templates": args.sst2_templates,
            "fixed_prompts": fixed_prompts,
            "trajectory": trajectory,
        }
        traj_path.write_text(json.dumps(traj_out, indent=2))
        print(f"[llama2_seq] response trajectory written to {args.response_trajectory_json}")


if __name__ == "__main__":
    main()
