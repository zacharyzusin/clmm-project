"""Figure 3: Sequential ASR trajectory across stages (seed 42, primary order)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(__file__)

# ASR (%) at T1 (aligned), T2 (after GSM8K), T3 (after SST-2)
# Seed 42, primary order: gsm8k -> sst2
# T4 (MBPP) from CLAUDE.md slurm results (8513103, 8505101, 8528728, 8528729, 8528730)
DATA = {
    "Baseline LoRA":    [1.2, 40.2, 2.3,  17.5],
    "CLoRA (rand.)":    [1.2, 2.7,  31.9, 71.2],
    "Safety-CLoRA":     [1.2, 1.5,  87.9, 71.0],
    "O-LoRA":           [1.2, 3.3,  16.9, 36.7],
    "Safety-O-LoRA":    [1.2, 4.6,  81.5, 44.8],
}

COLORS = {
    "Baseline LoRA":    "#4c72b0",
    "CLoRA (rand.)":    "#dd8452",
    "Safety-CLoRA":     "#55a868",
    "O-LoRA":           "#c44e52",
    "Safety-O-LoRA":    "#8172b3",
}
MARKERS = {
    "Baseline LoRA":    "o",
    "CLoRA (rand.)":    "s",
    "Safety-CLoRA":     "^",
    "O-LoRA":           "D",
    "Safety-O-LoRA":    "P",
}

STAGES = ["Aligned\n(T1)", "After\nGSM8K (T2)", "After\nSST-2 (T3)", "After\nMBPP (T4)"]


def plot():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(STAGES))
    for name, vals in DATA.items():
        ax.plot(x, vals, marker=MARKERS[name], color=COLORS[name],
                linewidth=2, markersize=7, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title("Sequential ASR Trajectory (Qwen3-0.6B, seed 42)", fontsize=11)
    ax.set_ylim(-2, 100)
    ax.axhline(1.2, color="gray", linewidth=0.8, linestyle=":", alpha=0.6,
               label="Post-alignment baseline (1.2%)")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"sequential_asr_trajectory.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    plot()
