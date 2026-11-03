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

    # Summary table: Mean ± Std per experiment condition
    metrics = ["mean_r2", "overall_r2", "mape", "rmse"]
    grouped = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"])
    grouped_clean = grouped.fillna(0.0)

    # Save to both summary files
    out_summary_flat = os.path.join(PROJECT_ROOT, "results", "summary", "aggregated_summary.csv")
    grouped_clean.reset_index().to_csv(out_summary_flat, sep="\t", index=False)

    out_phase0 = os.path.join(PROJECT_ROOT, "results", "summary", "phase0_sanity_summary.csv")
    grouped_clean.to_csv(out_phase0, sep="\t")

    print(f"Aggregated tab-delimited summary saved to:\n  - {out_summary_flat}\n  - {out_phase0}")
    print("\nSummary Overview (with absolute sample counts):")
    print(grouped_clean.to_string())


if __name__ == "__main__":
    main()
