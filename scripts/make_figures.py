import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COLOR_MAP = {
    "S0": "#475569",  # Slate (Scratch)
    "S1": "#2563eb",  # Blue (Full Fine-Tune)
    "S2": "#059669",  # Green (Feature Extraction)
    "S3": "#d97706",  # Amber (Gradual Unfreezing)
    "S4": "#7c3aed"   # Purple (Geometry Layer)
}

STRATEGY_LABELS = {
    "S0": "S0: Scratch Baseline",
    "S1": "S1: Full Fine-Tuning",
    "S2": "S2: Feature Extraction",
    "S3": "S3: Gradual Unfreezing",
    "S4": "S4: Geometry Layer"
}


def plot_learning_curves(df, out_dir):
    """
    Figure F2: Test R2 and MAPE vs Budget (log scale), with mean +/- 1 std bands.
    """
    targets = df["target"].unique()
    os.makedirs(out_dir, exist_ok=True)

    for tgt in targets:
        sub_tgt = df[df["target"] == tgt]
        sources = sub_tgt["source"].unique()

        for src in sources:
            sub = sub_tgt[sub_tgt["source"] == src].copy()
            if sub.empty:
                continue

            # Convert budget to numeric for plotting (treat 'full' as largest number or label)
            unique_budgets = sub["n_target"].unique()
            budget_order = [b for b in [25, 50, 100, 200, 400, 800, "full"] if b in unique_budgets or str(b) in unique_budgets]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
            strats = sub["strategy"].unique()

            for strat in strats:
                s_df = sub[sub["strategy"] == strat]
                agg = s_df.groupby("n_target")[["mean_r2", "mape"]].agg(["mean", "std"]).reset_index()
                # Sort according to budget_order
                agg["sort_key"] = agg["n_target"].apply(lambda x: 9999 if str(x) == "full" else int(x))
                agg = agg.sort_values("sort_key")

                x_labels = agg["n_target"].astype(str).tolist()
                x_pos = np.arange(len(x_labels))

                r2_mean = agg[("mean_r2", "mean")]
                r2_std = agg[("mean_r2", "std")].fillna(0)
                mape_mean = agg[("mape", "mean")]
                mape_std = agg[("mape", "std")].fillna(0)

                col = COLOR_MAP.get(strat, "#000000")
                lab = STRATEGY_LABELS.get(strat, strat)

                # R2 curve
                ax1.plot(x_pos, r2_mean, marker="o", label=lab, color=col, linewidth=2)
                ax1.fill_between(x_pos, r2_mean - r2_std, r2_mean + r2_std, alpha=0.15, color=col)

                # MAPE curve
                ax2.plot(x_pos, mape_mean, marker="s", label=lab, color=col, linewidth=2)
                ax2.fill_between(x_pos, mape_mean - mape_std, mape_mean + mape_std, alpha=0.15, color=col)

            for ax, title, ylabel in [(ax1, f"$R^2$ vs Budget ({src} $\\rightarrow$ {tgt})", "Mean $R^2$ (Test)"),
                                      (ax2, f"MAPE vs Budget ({src} $\\rightarrow$ {tgt})", "MAPE (%)")]:
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("Target Data Budget ($N_{target}$)", fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels)
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=9)

            plt.tight_layout()
            out_png = os.path.join(out_dir, f"fig_F2_learning_curve_{src}_to_{tgt}.png")
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"Generated Figure F2: {out_png}")


def main():
    csv_path = os.path.join(PROJECT_ROOT, "results", "all_runs.csv")
    if not os.path.exists(csv_path):
        print("No results file found.")
        return
    df = pd.read_csv(csv_path, sep="\t")
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    plot_learning_curves(df, fig_dir)


if __name__ == "__main__":
    main()
