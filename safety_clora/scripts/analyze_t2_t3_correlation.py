"""T2-T3 ASR correlation analysis for Safety-CLoRA and Safety-O-LoRA."""
import json
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results")
VARIANCE_DIR = os.path.join(RESULTS_DIR, "variance_study")
SEEDS_DIR = os.path.join(RESULTS_DIR, "seeds")

# Safety-CLoRA data across 6 seeds (T2 and T3 ASR in %)
# Sources: seeds/sequential_clora_safety_seed42.json, slurm logs for seeds 0-4
safety_clora_data = {
    42: {"t2": 1.5, "t3": 87.9},
    0:  {"t2": 0.2, "t3": 3.1},
    1:  {"t2": 4.8, "t3": 43.1},
    2:  {"t2": 5.8, "t3": 51.0},
    3:  {"t2": 1.7, "t3": 43.5},
    4:  {"t2": 3.3, "t3": 7.9},
}

# Attempt to load from saved JSONs where available
def try_load_from_json(method_key, seed, var_dir, seeds_dir):
    var_path = os.path.join(var_dir, f"{method_key}_seed{seed}_sequential.json")
    seed42_path = os.path.join(seeds_dir, f"sequential_{method_key}_seed42.json")
    if os.path.exists(var_path):
        with open(var_path) as f:
            d = json.load(f)
        result = {}
        for s in d["stages"]:
            if "T2" in s["label"]:
                result["t2"] = round(s["asr"] * 100, 1)
            elif "T3" in s["label"]:
                result["t3"] = round(s["asr"] * 100, 1)
        if "t2" in result and "t3" in result:
            return result
    if seed == 42 and os.path.exists(seed42_path):
        with open(seed42_path) as f:
            d = json.load(f)
        result = {}
        for s in d["stages"]:
            if "T2" in s["label"]:
                result["t2"] = round(s["asr"] * 100, 1)
            elif "T3" in s["label"]:
                result["t3"] = round(s["asr"] * 100, 1)
        if "t2" in result and "t3" in result:
            return result
    return None

# Override with JSON data where available
for seed in list(safety_clora_data.keys()):
    loaded = try_load_from_json("safety_clora", seed, VARIANCE_DIR, SEEDS_DIR)
    if loaded:
        safety_clora_data[seed] = loaded

# Safety-O-LoRA: 3 seeds only
safety_olora_data = {
    42: {"t2": 4.6, "t3": 81.5},
    0:  {"t2": 16.7, "t3": 72.5},
    1:  {"t2": 8.7, "t3": 93.5},
}
for seed in list(safety_olora_data.keys()):
    loaded = try_load_from_json("olora_safety", seed, VARIANCE_DIR, SEEDS_DIR)
    if loaded:
        safety_olora_data[seed] = loaded


def correlation_analysis(name, data):
    seeds = sorted(data.keys())
    t2 = np.array([data[s]["t2"] for s in seeds])
    t3 = np.array([data[s]["t3"] for s in seeds])

    pearson_r, pearson_p = stats.pearsonr(t2, t3)
    spearman_r, spearman_p = stats.spearmanr(t2, t3)

    print(f"\n{'='*55}")
    print(f"{name} T2-T3 correlation (n={len(seeds)} seeds):")
    print(f"  Data points:")
    for s, x, y in zip(seeds, t2, t3):
        print(f"    seed={s:2d}  T2={x:5.1f}%  T3={y:5.1f}%")
    print(f"  Pearson r  = {pearson_r:+.3f} (p = {pearson_p:.3f})")
    print(f"  Spearman ρ = {spearman_r:+.3f} (p = {spearman_p:.3f})")

    if abs(pearson_r) > 0.5:
        if pearson_r > 0:
            interp = (
                "POSITIVE correlation confirmed. Lower T2 ASR predicts higher T3 ASR —\n"
                "    tighter T2 protection creates worse T3 collapse."
            )
        else:
            interp = (
                "NEGATIVE correlation confirmed. Lower T2 ASR predicts lower T3 ASR —\n"
                "    better T2 protection also leads to better T3 stability."
            )
    elif abs(pearson_r) < 0.2:
        interp = (
            "NO correlation — T3 variance is not explained by T2 quality.\n"
            "    Variance source is elsewhere (S matrix numerical sensitivity,\n"
            "    gradient stochasticity, or SST-2 loss landscape sensitivity)."
        )
    else:
        interp = (
            f"WEAK/MODERATE correlation (r={pearson_r:.2f}). T2 quality partially\n"
            f"    explains T3 variance but is not the dominant factor."
        )

    print(f"\n  Interpretation:\n    {interp}")

    return seeds, t2, t3, pearson_r, spearman_r


def scatter_plot(name, filename, seeds, t2, t3, pearson_r):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(t2, t3, zorder=3, s=80, color="steelblue")
    for s, x, y in zip(seeds, t2, t3):
        ax.annotate(f"s{s}", (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=9)

    # Linear regression line
    m, b = np.polyfit(t2, t3, 1)
    x_line = np.linspace(t2.min() - 1, t2.max() + 1, 100)
    ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5,
            linestyle="--", label=f"r = {pearson_r:.2f}")

    ax.set_xlabel("T2 ASR (%) — After GSM8K")
    ax.set_ylabel("T3 ASR (%) — After SST-2")
    ax.set_title(f"{name}: T2 ASR vs T3 ASR across seeds")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    print(f"\n  Plot saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    seeds_sc, t2_sc, t3_sc, pr_sc, sr_sc = correlation_analysis(
        "Safety-CLoRA", safety_clora_data
    )
    scatter_plot(
        "Safety-CLoRA",
        "t2_t3_scatter_safety_clora.png",
        seeds_sc, t2_sc, t3_sc, pr_sc,
    )

    seeds_so, t2_so, t3_so, pr_so, sr_so = correlation_analysis(
        "Safety-O-LoRA", safety_olora_data
    )
    scatter_plot(
        "Safety-O-LoRA",
        "t2_t3_scatter_safety_olora.png",
        seeds_so, t2_so, t3_so, pr_so,
    )

    print(f"\n{'='*55}")
    print("DECISION GATE (Task 6):")
    if abs(pr_sc) > 0.5:
        print(f"  Safety-CLoRA r={pr_sc:.2f} > 0.5 → IMPLEMENT adaptive λ (Task 6)")
    elif abs(pr_sc) < 0.2:
        print(f"  Safety-CLoRA r={pr_sc:.2f} < 0.2 → SKIP Task 6")
    else:
        print(f"  Safety-CLoRA r={pr_sc:.2f} in [0.2, 0.5] → EXPLORATORY only")
