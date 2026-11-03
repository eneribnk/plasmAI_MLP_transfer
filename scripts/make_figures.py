import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plasmai_tl.data import load_dataset_registry, load_raw_dataset

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


def plot_dataset_overview_F1(out_dir):
    """
    Figure F1 (Section 10): Dataset Overview.
    Scatter Power-Pressure operating windows, output rate distributions, and radial profiles across Ar, O2, and C4F8.
    """
    datasets_cfg = load_dataset_registry(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    gases = ["Ar", "O2", "C4F8"]
    colors = {"Ar": "#2563eb", "O2": "#dc2626", "C4F8": "#059669"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel 1: Operating Conditions Window (Power vs Pressure)
    ax1 = axes[0]
    for g in gases:
        df, _, _ = load_raw_dataset(datasets_cfg[g], root_dir=PROJECT_ROOT)
        ax1.scatter(df["Power"], df["Pressure"] * 1000.0, label=g, alpha=0.5, color=colors[g], s=18)
    ax1.set_xlabel("Power (W)", fontsize=11)
    ax1.set_ylabel("Pressure (mTorr)", fontsize=11)
    ax1.set_title("(a) Power-Pressure Operating Window", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    # Panel 2: Distribution of Average Etch/Deposition Rates
    ax2 = axes[1]
    for g in gases:
        df, _, out_cols = load_raw_dataset(datasets_cfg[g], root_dir=PROJECT_ROOT)
        rates = df[out_cols].mean(axis=1)
        ax2.hist(rates, bins=35, alpha=0.5, label=f"{g} ({datasets_cfg[g]['rate_type']})", color=colors[g])
    ax2.set_xlabel("Average Rate (nm/min)", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.set_title("(b) Rate Distributions across Chemistries", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    # Panel 3: Representative Radial Profiles across the 10 Wafer Points
    ax3 = axes[2]
    points = np.arange(1, 11)
    for g in gases:
        df, _, out_cols = load_raw_dataset(datasets_cfg[g], root_dir=PROJECT_ROOT)
        mean_profile = df[out_cols].mean(axis=0).values
        std_profile = df[out_cols].std(axis=0).values
        ax3.plot(points, mean_profile, marker="o", label=g, color=colors[g], linewidth=2)
        ax3.fill_between(points, mean_profile - std_profile, mean_profile + std_profile, alpha=0.15, color=colors[g])
    ax3.set_xlabel("Wafer Radial Point Index (1: Center -> 10: Edge)", fontsize=10)
    ax3.set_ylabel("Mean Rate +/- 1 Std (nm/min)", fontsize=11)
    ax3.set_title("(c) Radial Wafer Uniformity Profile", fontsize=11, fontweight="bold")
    ax3.set_xticks(points)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=10)

    plt.suptitle("Figure F1: Multi-Gas COBRA Surrogates Dataset Overview", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_F1_datasets_overview.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure F1: {out_path}")


def plot_phase0_parity_and_residuals(out_dir):
    """
    Figure F6 (Section 10): Parity plots (PBM vs NN Predicted) on unseen test set for Phase 0 models.
    """
    gases = ["Ar", "O2", "C4F8"]
    colors = {"Ar": "#2563eb", "O2": "#dc2626", "C4F8": "#059669"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, gas in enumerate(gases):
        comp_file = os.path.join(PROJECT_ROOT, "results", "runs", gas, "None", "S0", "Nfull", "seed0", "comparison.txt")
        ax = axes[idx]
        if os.path.exists(comp_file):
            df = pd.read_csv(comp_file, sep="\t")
            pred_cols = [c for c in df.columns if c.endswith("_pred")]
            tar_cols = [c.replace("_pred", "") for c in pred_cols]

            y_true = df[tar_cols].values.flatten()
            y_pred = df[pred_cols].values.flatten()

            ax.scatter(y_true, y_pred, alpha=0.4, color=colors[gas], s=12, label="Test Points")
            min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.5, label="Perfect Parity (y=x)")
            ax.set_xlabel("PBM Ground Truth (nm/min)", fontsize=11)
            ax.set_ylabel("Surrogate NN Prediction (nm/min)", fontsize=11)
            ax.set_title(f"{gas} Baseline (Scratch S0, Full)", fontsize=11, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
        else:
            ax.text(0.5, 0.5, f"No comparison.txt found for {gas}", ha="center", va="center")

    plt.suptitle("Figure F6: Parity Plots (PBM Physics-Based vs NN Predictions)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_F6_phase0_parity_plots.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure F6: {out_path}")


def plot_gradient_norms_F13(out_dir):
    """
    Figure F13 (Section 11.3): Layer Gradient Norms vs Epoch dynamics during training.
    """
    grad_file = os.path.join(PROJECT_ROOT, "results", "runs", "C4F8", "Ar", "S1", "Nfull", "seed0", "grad_norms.csv")
    if not os.path.exists(grad_file):
        candidates = glob.glob(os.path.join(PROJECT_ROOT, "results", "runs", "*", "*", "*", "*", "*", "grad_norms.csv"))
        grad_file = candidates[0] if candidates else None

    if not grad_file or not os.path.exists(grad_file):
        return

    df_grad = pd.read_csv(grad_file, sep="\t")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    layer_cols = [c for c in df_grad.columns if c != "epoch"]
    for col in layer_cols:
        short_name = col.replace(".weight", " (W)").replace(".bias", " (B)")
        ax.plot(df_grad["epoch"], df_grad[col], label=short_name, linewidth=1.8)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(r"$L_2$ Gradient Norm $\|\nabla_\theta \mathcal{L}\|_2$", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("Figure F13: Layer Gradient Dynamics During Transfer Learning", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_F13_phase0_grad_norms.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure F13: {out_path}")


def plot_learning_curves_F2(df, out_dir):
    """
    Figure F2 (Section 10): Learning Curves vs Target Data Budget (Absolute Samples), with mean +/- 1 std bands.
    """
    targets = df["target"].unique()

    for tgt in targets:
        sub_tgt = df[df["target"] == tgt].copy()
        budget_col = "budget_tag" if "budget_tag" in sub_tgt.columns else "n_target"
        
        # Build unified master budget axis sorted by absolute sample count
        budget_info = sub_tgt[[budget_col, "n_samples"]].drop_duplicates().copy()
        budget_info["n_samples_num"] = budget_info["n_samples"].astype(int)
        budget_info = budget_info.sort_values("n_samples_num").reset_index(drop=True)
        
        if len(budget_info) <= 1:
            continue

        budget_info["x_pos"] = np.arange(len(budget_info))
        budget_map = {row[budget_col]: row["x_pos"] for _, row in budget_info.iterrows()}
        x_labels = [
            f"{row[budget_col]} (N={row['n_samples_num']})" if str(row[budget_col]).lower() == "full" 
            else f"N={row['n_samples_num']}" 
            for _, row in budget_info.iterrows()
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

        # Group by strategy and source (to separate S0 scratch from transfer pairs)
        pairs = sub_tgt[["strategy", "source"]].drop_duplicates().values

        for strat, src in pairs:
            cond = (sub_tgt["strategy"] == strat) & (sub_tgt["source"] == src)
            s_df = sub_tgt[cond]
            agg = s_df.groupby(budget_col)[["mean_r2", "mape"]].agg(["mean", "std"]).reset_index()

            agg["x_pos"] = agg[budget_col].map(budget_map)
            agg = agg.sort_values("x_pos")

            x_pos = agg["x_pos"].values
            r2_mean = agg[("mean_r2", "mean")]
            r2_std = agg[("mean_r2", "std")].fillna(0)
            mape_mean = agg[("mape", "mean")]
            mape_std = agg[("mape", "std")].fillna(0)

            col = COLOR_MAP.get(strat, "#000000")
            base_label = STRATEGY_LABELS.get(strat, strat)
            label = base_label if strat == "S0" else f"{base_label} ({src} \u2192 {tgt})"

            # If single point, plot as scatter/star, else plot with line
            if len(x_pos) == 1:
                ax1.scatter(x_pos, r2_mean, marker="*", s=120, label=label, color=col, zorder=5)
                ax2.scatter(x_pos, mape_mean, marker="*", s=120, label=label, color=col, zorder=5)
            else:
                ax1.plot(x_pos, r2_mean, marker="o", label=label, color=col, linewidth=2)
                ax1.fill_between(x_pos, r2_mean - r2_std, r2_mean + r2_std, alpha=0.15, color=col)

                ax2.plot(x_pos, mape_mean, marker="s", label=label, color=col, linewidth=2)
                ax2.fill_between(x_pos, mape_mean - mape_std, mape_mean + mape_std, alpha=0.15, color=col)

        for ax, title, ylabel in [
            (ax1, f"Mean $R^2$ vs Budget ({tgt})", "Mean $R^2$ (Test Set)"),
            (ax2, f"MAPE vs Budget ({tgt})", "MAPE (%)")
        ]:
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Target Data Budget: Absolute Sample Count (Train + Val)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticks(np.arange(len(budget_info)))
            ax.set_xticklabels(x_labels, rotation=15)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)

        plt.suptitle(f"Figure F2: Learning Curves on Unseen Test Set - Target {tgt}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"fig_F2_learning_curves_{tgt}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated Figure F2 with absolute sample counts: {out_png}")


def main():
    fig_dir = os.path.join(PROJECT_ROOT, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Figure F1: Datasets Overview
    plot_dataset_overview_F1(fig_dir)

    # 2. Figure F6: Parity Plots (Ground Truth vs Prediction)
    plot_phase0_parity_and_residuals(fig_dir)

    # 3. Figure F13: Gradient Norm Dynamics
    plot_gradient_norms_F13(fig_dir)

    # 4. Figure F2: Learning Curves (Phase 1 Baseline & Transfer)
    all_runs_file = os.path.join(PROJECT_ROOT, "results", "all_runs.csv")
    if os.path.exists(all_runs_file):
        df_all = pd.read_csv(all_runs_file, sep="\t", keep_default_na=False)
        df_all["source"] = df_all["source"].replace("", "None")
        plot_learning_curves_F2(df_all, fig_dir)


if __name__ == "__main__":
    main()
