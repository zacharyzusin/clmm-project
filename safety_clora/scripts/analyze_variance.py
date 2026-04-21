"""
Task 1b / Task 4: Variance and T2-T3 correlation analysis.

Section 1 (Safety-CLoRA variance study):
  Loads all Safety-CLoRA sequential results across seeds 0,1,2,3,4,42.
  Computes T2/T3 mean±std and Pearson/Spearman T2-T3 correlation.
  Saves scatter plot to results/variance_study/t2_vs_t3_correlation.png.

Section 2 (all-method correlation):
  Uses existing multi-seed data for all methods (seeds 0, 1, 42).
  Prints per-method T2-T3 Pearson r and interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
VARIANCE_DIR = RESULTS / "variance_study"

# ---------------------------------------------------------------------------
# Existing per-seed sequential results (seeds 0, 1, 42)
# Format: method -> {seed: (t2_asr, t3_asr)}
# ---------------------------------------------------------------------------

EXISTING = {
    "lora": {
        0:  (34.8, 1.9),
        1:  (37.5, 2.9),
        42: (40.2, 2.3),
    },
    "clora_random": {
        0:  (24.4, 10.8),
        1:  (1.0,  4.6),
        42: (2.7,  31.9),
    },
    "clora_safety": {
        0:  (0.2,  3.1),
        1:  (4.8,  43.1),
        42: (1.5,  87.9),
    },
    "olora_standard": {
        0:  (9.2,  98.1),
        1:  (8.8,  98.8),
        42: (3.3,  16.9),
    },
    "olora_safety": {
        0:  (16.7, 72.5),
        1:  (8.7,  93.5),
        42: (4.6,  81.5),
    },
}


def _pearson(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((xm * ym).sum() / denom)


def _spearman(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


# ---------------------------------------------------------------------------
# SECTION 1: Safety-CLoRA variance study
# ---------------------------------------------------------------------------

def load_variance_study_results() -> dict[int, tuple[float, float, float]]:
    """
    Returns {seed: (t2_asr, t3_asr, t2_gsm8k_acc)} for Safety-CLoRA.
    Seeds 0, 1, 42 come from existing EXISTING dict (T3 ASR) and JSON files.
    Seeds 2, 3, 4 come from variance_study JSON files.
    """
    results = {}

    # Seeds 0, 1, 42 from existing sequential JSON files.
    for seed, (t2_asr, t3_asr) in EXISTING["clora_safety"].items():
        # Try to load t2_gsm8k_acc from the sequential JSON.
        candidate_paths = [
            RESULTS / "seeds" / f"sequential_clora_safety_seed{seed}.json",
        ]
        t2_gsm8k = float("nan")
        for p in candidate_paths:
            if p.exists():
                data = json.loads(p.read_text())
                for stage in data.get("stages", []):
                    if "T2" in stage.get("label", "") or "gsm8k" in stage.get("label", "").lower():
                        t2_gsm8k = stage.get("gsm8k_acc", float("nan")) * 100
                        break
                break
        results[seed] = (t2_asr, t3_asr, t2_gsm8k)

    # Seeds 2, 3, 4 from variance_study JSON files.
    for seed in (2, 3, 4):
        p = VARIANCE_DIR / f"safety_clora_seed{seed}_sequential.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        t2_asr = t3_asr = t2_gsm8k = float("nan")
        for stage in data.get("stages", []):
            label = stage.get("label", "")
            if "T2" in label or "gsm8k" in label.lower():
                t2_asr = stage.get("asr", float("nan")) * 100
                t2_gsm8k = stage.get("gsm8k_acc", float("nan")) * 100
            elif "T3" in label or "sst2" in label.lower():
                t3_asr = stage.get("asr", float("nan")) * 100
        results[seed] = (t2_asr, t3_asr, t2_gsm8k)

    return results


def section1_safety_clora_variance():
    print("=" * 60)
    print("SECTION 1: Safety-CLoRA Variance Study")
    print("=" * 60)

    data = load_variance_study_results()
    seeds_sorted = sorted(data.keys())

    print(f"\n{'Seed':>6} | {'T2 ASR':>8} | {'T3 ASR':>8} | {'T2 GSM8K Acc':>13}")
    print("-" * 46)
    for s in seeds_sorted:
        t2, t3, gsm = data[s]
        gsm_str = f"{gsm:.1f}%" if not np.isnan(gsm) else "N/A"
        t2_str = f"{t2:.1f}%" if not np.isnan(t2) else "?"
        t3_str = f"{t3:.1f}%" if not np.isnan(t3) else "?"
        print(f"{s:>6} | {t2_str:>8} | {t3_str:>8} | {gsm_str:>13}")

    t2s = [data[s][0] for s in seeds_sorted if not np.isnan(data[s][0]) and not np.isnan(data[s][1])]
    t3s = [data[s][1] for s in seeds_sorted if not np.isnan(data[s][0]) and not np.isnan(data[s][1])]

    if len(t2s) >= 2:
        print(f"\n{'Mean':>6} | {np.mean(t2s):>7.1f}% | {np.mean(t3s):>7.1f}%")
        print(f"{'Std':>6} | {np.std(t2s, ddof=1):>7.1f}% | {np.std(t3s, ddof=1):>7.1f}%")

    if len(t2s) >= 3:
        r = _pearson(t2s, t3s)
        rho = _spearman(t2s, t3s)
        print(f"\nCorrelation between T2 ASR and T3 ASR (n={len(t2s)} seeds):")
        print(f"  Pearson r  = {r:+.3f}")
        print(f"  Spearman ρ = {rho:+.3f}")
        print()
        if r > 0.6:
            print("  Interpretation: STRONG POSITIVE — lower T2 protection predicts")
            print("  worse T3 collapse (tradeoff finding).")
        elif r < -0.6:
            print("  Interpretation: STRONG NEGATIVE — better T2 protection predicts")
            print("  worse T3 collapse (tradeoff finding, opposite direction).")
        else:
            print(f"  Interpretation: WEAK/MODERATE (|r|={abs(r):.2f}) — no clear tradeoff.")

        # Scatter plot (optional, requires matplotlib).
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 5))
            colors = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4", 42: "C5"}
            for s, (t2, t3, _) in data.items():
                if np.isnan(t2) or np.isnan(t3):
                    continue
                ax.scatter(t2, t3, color=colors.get(s, "gray"), zorder=3)
                ax.annotate(f"seed {s}", (t2, t3), textcoords="offset points",
                            xytext=(4, 4), fontsize=9)

            # Regression line.
            if len(t2s) >= 2:
                coeffs = np.polyfit(t2s, t3s, 1)
                x_line = np.linspace(min(t2s) - 1, max(t2s) + 1, 100)
                ax.plot(x_line, np.polyval(coeffs, x_line), "k--", alpha=0.5,
                        label=f"r={r:+.2f}")

            ax.set_xlabel("T2 ASR (%)", fontsize=11)
            ax.set_ylabel("T3 ASR (%)", fontsize=11)
            ax.set_title("Safety-CLoRA: T2 vs T3 ASR across seeds", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            out_png = VARIANCE_DIR / "t2_vs_t3_correlation.png"
            VARIANCE_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nScatter plot saved: {out_png}")
        except ImportError:
            print("\n(matplotlib not available — skipping scatter plot)")


# ---------------------------------------------------------------------------
# SECTION 2: All-method T2-T3 correlation (existing 3-seed data)
# ---------------------------------------------------------------------------

METHOD_LABELS = {
    "lora":           "Baseline LoRA",
    "clora_random":   "CLoRA random",
    "clora_safety":   "Safety-CLoRA",
    "olora_standard": "O-LoRA standard",
    "olora_safety":   "Safety-O-LoRA",
}


def section2_all_methods():
    print("\n" + "=" * 70)
    print("SECTION 2: T2-T3 Correlation Across All Methods (seeds 0, 1, 42)")
    print("=" * 70)

    print(f"\n{'Method':<20} | {'T2 mean':>9} | {'T3 mean':>9} | {'Pearson r':>10} | {'Spearman ρ':>11} | Interpretation")
    print("-" * 100)

    for method, seed_data in EXISTING.items():
        t2s = [v[0] for v in seed_data.values()]
        t3s = [v[1] for v in seed_data.values()]
        r = _pearson(t2s, t3s)
        rho = _spearman(t2s, t3s)

        if np.isnan(r):
            interp = "n/a"
        elif r > 0.5:
            interp = "positive (lower T2 protection → worse T3)"
        elif r < -0.5:
            interp = "negative (worse T2 → better T3)"
        else:
            interp = f"weak |r|={abs(r):.2f}"

        label = METHOD_LABELS.get(method, method)
        r_str = f"{r:+.3f}" if not np.isnan(r) else "  n/a"
        rho_str = f"{rho:+.3f}" if not np.isnan(rho) else "    n/a"
        print(f"{label:<20} | {np.mean(t2s):>8.1f}% | {np.mean(t3s):>8.1f}% | {r_str:>10} | {rho_str:>11} | {interp}")

    print()
    print("NOTE: n=3 per method — correlations are directional indicators only.")
    print("Safety-CLoRA n will increase to 6 once seeds 2,3,4 results arrive.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    section1_safety_clora_variance()
    section2_all_methods()
