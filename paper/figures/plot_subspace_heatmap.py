"""Figure 1: Subspace overlap heatmap for Safety-O-LoRA q_proj per layer."""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "../../results/subspace_overlap.csv")
OUT_DIR = os.path.dirname(__file__)

TASK_ORDER = ["gsm8k", "mbpp", "sst2"]
TASK_LABELS = {"gsm8k": "GSM8K\n(math)", "mbpp": "MBPP\n(code)", "sst2": "SST-2\n(classif.)"}


def load_heatmap_data():
    rows = list(csv.DictReader(open(RESULTS_CSV)))
    n_layers = max(int(r["layer_idx"]) for r in rows) + 1
    data = {}  # task -> list of overlap values per layer
    for task in TASK_ORDER:
        vals = [None] * n_layers
        for r in rows:
            if r["method"] == "olora_safety" and r["proj"] == "q_proj" and r["task"] == task:
                vals[int(r["layer_idx"])] = float(r["overlap"])
        data[task] = vals
    # matrix: rows = layers, cols = tasks
    mat = np.array([[data[t][l] for t in TASK_ORDER] for l in range(n_layers)])
    return mat, n_layers


def plot_heatmap(mat, n_layers):
    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=mat.max() * 1.05)

    # Axes
    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], fontsize=10)
    ax.set_yticks(range(0, n_layers, 4))
    ax.set_yticklabels(range(0, n_layers, 4), fontsize=9)
    ax.set_ylabel("Transformer Layer", fontsize=11)
    ax.set_xlabel("Capability Task", fontsize=11)
    ax.set_title("Safety Subspace Overlap\n(Safety-O-LoRA, $q$-proj)", fontsize=11)

    # Column mean annotations below the heatmap
    col_means = mat.mean(axis=0)
    for j, (task, mean) in enumerate(zip(TASK_ORDER, col_means)):
        ax.text(j, n_layers - 0.1, f"mean={mean:.4f}", ha="center", va="bottom",
                fontsize=7.5, color="black",
                transform=ax.transData)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean |cos sim| to safety adapter", fontsize=9)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"subspace_overlap_heatmap.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    mat, n_layers = load_heatmap_data()
    # Print summary stats
    for j, task in enumerate(TASK_ORDER):
        ratio = mat[:, j].mean() / mat[:, 0].mean()
        print(f"  {task}: mean={mat[:, j].mean():.4f}, ratio vs GSM8K={ratio:.2f}x")
    plot_heatmap(mat, n_layers)
