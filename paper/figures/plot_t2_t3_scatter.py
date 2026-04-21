"""Figure 2: T2 vs T3 ASR scatter for Safety-CLoRA (6 seeds)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(__file__)

# Safety-CLoRA data: seed -> (T2, T3) in percent
SAFETY_CLORA = {
    0:  (0.2, 3.1),
    1:  (4.8, 43.1),
    2:  (5.8, 51.0),
    3:  (1.7, 43.5),
    4:  (3.3, 7.9),
    42: (1.5, 87.9),
}


def scatter(name, data, filename):
    seeds = sorted(data.keys())
    t2 = np.array([data[s][0] for s in seeds])
    t3 = np.array([data[s][1] for s in seeds])

    pearson_r = np.corrcoef(t2, t3)[0, 1]
    m, b = np.polyfit(t2, t3, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(t2, t3, s=80, color="steelblue", zorder=3)
    for s, x, y in zip(seeds, t2, t3):
        ax.annotate(f"s{s}", (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=9)

    x_line = np.linspace(t2.min() - 1, t2.max() + 1, 100)
    ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5,
            linestyle="--", label=f"$r = {pearson_r:.2f}$ ($p = 0.76$)")

    ax.set_xlabel("T2 ASR (%) — after GSM8K fine-tuning", fontsize=10)
    ax.set_ylabel("T3 ASR (%) — after SST-2 fine-tuning", fontsize=10)
    ax.set_title(f"{name}: T2 vs. T3 ASR across seeds", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"{filename}.{ext}")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()
    print(f"  Pearson r = {pearson_r:.3f}")


if __name__ == "__main__":
    scatter("Safety-CLoRA", SAFETY_CLORA, "t2_t3_scatter")
