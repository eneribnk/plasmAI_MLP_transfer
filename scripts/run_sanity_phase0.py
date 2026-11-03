import os
import sys
import argparse
import subprocess
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_EXEC = sys.executable


def run_cmd(cmd_list):
    print("\nExecuting:", " ".join(cmd_list))
    res = subprocess.run(cmd_list, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Sanity Check across all 3 chemistries (Ar, O2, C4F8)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="List of seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override (defaults to config: 700)")
    args = parser.parse_args()

    gases = ["Ar", "O2", "C4F8"]
    seeds = args.seeds

    print("=" * 80)
    print("PHASE 0 SANITY CHECK & VALIDATION SUITE (ALL 3 GASES)")
    print("Common Settings across Ar, O2, C4F8:")
    print("  - Spatially-Aware Smoothness penalty at loss function (lambda_smooth = 1.15, tau = 0.0)")
    print("  - Linear output layer (NO ELU / ReLU activation at output)")
    print("  - L1 regularization (lambda = 1e-4)")
    print(f"  - Seeds: {seeds}")
    print("=" * 80)

    # 1. Train full source models across all 3 chemistries
    for gas in gases:
        for s in seeds:
            print(f"\n--- [Phase 0.1] Source Model: {gas} | Seed {s} ---")
            cmd = [
                PYTHON_EXEC, os.path.join(SCRIPT_DIR, "train_source.py"),
                "--dataset", gas,
                "--seed", str(s)
            ]
            if args.epochs:
                cmd.extend(["--epochs", str(args.epochs)])
            run_cmd(cmd)

    # 2. Train Scratch Baselines (S0) on full budget across all 3 chemistries
    for gas in gases:
        for s in seeds:
            print(f"\n--- [Phase 0.2] Scratch Baseline (S0): {gas} | Seed {s} ---")
            cmd = [
                PYTHON_EXEC, os.path.join(SCRIPT_DIR, "run_transfer.py"),
                "--source", "None",
                "--target", gas,
                "--strategy", "S0",
                "--n_target", "full",
                "--seed", str(s)
            ]
            if args.epochs:
                cmd.extend(["--epochs", str(args.epochs)])
            run_cmd(cmd)

    # 3. Transfer Learning (S1: Full Fine-Tuning) across pairs
    transfer_pairs = [
        ("Ar", "O2"),
        ("O2", "Ar"),
        ("Ar", "C4F8"),
        ("O2", "C4F8")
    ]

    for src, tgt in transfer_pairs:
        for s in seeds:
            print(f"\n--- [Phase 0.3] Transfer S1: {src} -> {tgt} | Seed {s} ---")
            cmd = [
                PYTHON_EXEC, os.path.join(SCRIPT_DIR, "run_transfer.py"),
                "--source", src,
                "--target", tgt,
                "--strategy", "S1",
                "--n_target", "full",
                "--seed", str(s)
            ]
            if args.epochs:
                cmd.extend(["--epochs", str(args.epochs)])
            run_cmd(cmd)

    # 4. Generate Master Summary Table (Tab-delimited)
    all_runs_path = os.path.join(PROJECT_ROOT, "results", "all_runs.csv")
    if os.path.exists(all_runs_path):
        # keep_default_na=False ensures string 'None' is preserved and not parsed as NaN
        df = pd.read_csv(all_runs_path, sep="\t", keep_default_na=False)
        df["source"] = df["source"].replace("", "None")

        print("\n" + "=" * 80)
        print("PHASE 0 SANITY CHECK RESULTS (Mean +/- Std over seeds):")
        print("=" * 80)
        metrics = ["mean_r2", "overall_r2", "mape", "rmse"]
        summary = df.groupby(["target", "source", "strategy", "n_target"], dropna=False)[metrics].agg(["mean", "std"])
        
        summary_clean = summary.fillna(0.0)
        print(summary_clean.to_string())

        summary_csv = os.path.join(PROJECT_ROOT, "results", "summary", "phase0_sanity_summary.csv")
        summary_clean.to_csv(summary_csv, sep="\t")
        print(f"\nTab-delimited summary table saved to: {summary_csv}")


if __name__ == "__main__":
    main()
