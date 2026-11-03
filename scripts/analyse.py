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

    df = pd.read_csv(csv_path, sep="\t")
    print(f"Loaded {len(df)} runs from {csv_path}")

    # Summary table: Mean ± Std per experiment condition
    metrics = ["mean_r2", "overall_r2", "mape", "rmse"]
    grouped = df.groupby(["target", "source", "strategy", "n_target"])[metrics].agg(["mean", "std"]).reset_index()
    grouped_clean = grouped.fillna(0.0)

    out_summary = os.path.join(PROJECT_ROOT, "results", "summary", "aggregated_summary.csv")
    grouped_clean.to_csv(out_summary, sep="\t", index=False)
    print(f"Aggregated tab-delimited summary saved to: {out_summary}")
    print("\nSummary Overview:")
    print(grouped_clean.to_string(index=False))


if __name__ == "__main__":
    main()
