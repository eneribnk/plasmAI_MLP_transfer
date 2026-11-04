import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def main():
    csv_path = os.path.join(PROJECT_ROOT, "results", "all_runs.csv")
    if not os.path.exists(csv_path):
        print(f"No results found at {csv_path}. Run experiments first.")
        return

    # keep_default_na=False ensures string 'None' is preserved and not dropped
    df = pd.read_csv(csv_path, sep="\t", keep_default_na=False)
    df["source"] = df["source"].replace("", "None")
    print(f"Loaded {len(df)} runs from {csv_path}")

    group_cols = ["target", "source", "strategy"]
    if "budget_tag" in df.columns:
        group_cols.extend(["budget_tag", "n_samples", "train_samples", "val_samples"])
    elif "n_target" in df.columns:
        group_cols.append("n_target")

    metrics = ["mean_r2", "overall_r2", "mape", "rmse"]

    summary_dir = os.path.join(PROJECT_ROOT, "results", "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Master Aggregated Summary (All completed runs across all phases)
    grouped_all = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).fillna(0.0)
    out_all_flat = os.path.join(summary_dir, "aggregated_summary.csv")
    grouped_all.reset_index().to_csv(out_all_flat, sep="\t", index=False)

    # 2. Phase 0 Sanity Summary (Only full budget runs: sanity benchmark)
    phase0_df = df[df["budget_tag"].astype(str).str.lower() == "full"]
    if len(phase0_df) > 0:
        grouped_p0 = phase0_df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).fillna(0.0)
        out_phase0 = os.path.join(summary_dir, "phase0_sanity_summary.csv")
        grouped_p0.to_csv(out_phase0, sep="\t")
        print(f"  - Phase 0 Sanity Check summary saved to: {out_phase0}")

    # 3. Phase 1 Scratch Summary (Only S0 scratch learning curves across all budgets)
    phase1_df = df[df["strategy"] == "S0"]
    if len(phase1_df) > 0:
        grouped_p1 = phase1_df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).fillna(0.0)
        out_phase1 = os.path.join(summary_dir, "phase1_scratch_summary.csv")
        grouped_p1.to_csv(out_phase1, sep="\t")
        print(f"  - Phase 1 Scratch Learning Curves summary saved to: {out_phase1}")

    # 4. Phase 2 Transfer Summary (Transfer strategies S1, S2, S3 across budgets)
    phase2_df = df[df["strategy"].isin(["S1", "S2", "S3", "S4"])]
    if len(phase2_df) > 0:
        grouped_p2 = phase2_df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).fillna(0.0)
        out_phase2 = os.path.join(summary_dir, "phase2_transfer_summary.csv")
        grouped_p2.to_csv(out_phase2, sep="\t")
        print(f"  - Phase 2 Transfer Learning summary saved to: {out_phase2}")

    print(f"  - Master Aggregated summary saved to: {out_all_flat}")
    print("\nSummary Overview (Master Ledger):")
    print(grouped_all.to_string())


if __name__ == "__main__":
    main()
