import os
import sys
import argparse
import subprocess
import yaml
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_EXEC = sys.executable


def get_existing_completed_runs(all_runs_csv):
    if not os.path.exists(all_runs_csv):
        return set()
    df = pd.read_csv(all_runs_csv, sep="\t", keep_default_na=False)
    completed = set()
    for _, row in df.iterrows():
        b_tag = str(row.get("budget_tag", row.get("n_target", "")))
        key = (
            str(row["target"]),
            str(row["source"]),
            str(row["strategy"]),
            b_tag,
            int(row["seed"])
        )
        completed.add(key)
    return completed


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Learning curves from scratch across all budgets.")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs", "sweeps", "phase1_scratch.yaml"),
                        help="Path to sweep config yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override (defaults to config: 700)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    targets = sweep_cfg.get("targets", ["Ar", "O2", "C4F8"])
    strategy = sweep_cfg.get("strategy", "S0")
    source = sweep_cfg.get("source", "None")
    budgets = sweep_cfg.get("budgets", [25, 50, 100, 200, 400, 800, "full"])
    seeds = sweep_cfg.get("seeds", [0, 1, 2])

    all_runs_csv = os.path.join(PROJECT_ROOT, "results", "all_runs.csv")
    completed_runs = get_existing_completed_runs(all_runs_csv)

    # Build queue of runs
    queue = []
    for tgt in targets:
        for b in budgets:
            for s in seeds:
                key = (str(tgt), str(source), str(strategy), str(b), int(s))
                if key not in completed_runs:
                    queue.append((tgt, str(b), int(s)))

    total_runs = len(queue)
    already_done = len(targets) * len(budgets) * len(seeds) - total_runs

    print("=" * 80)
    print("PHASE 1: LEARNING CURVES FROM SCRATCH (BASELINE BENCHMARK)")
    print(f"Gases: {targets}")
    print(f"Data Budgets: {budgets} (showing exact absolute sample counts during execution)")
    print(f"Seeds: {seeds}")
    print(f"Total Sweep Grid: {len(targets) * len(budgets) * len(seeds)} runs")
    print(f"Already completed in previous runs: {already_done}")
    print(f"Runs remaining to execute: {total_runs}")
    print("=" * 80)

    for idx, (tgt, b, s) in enumerate(queue):
        print(f"\n>>> [{idx + 1}/{total_runs}] Running Scratch Baseline: {tgt} | Budget={b} | Seed {s} <<<")
        cmd = [
            PYTHON_EXEC, os.path.join(SCRIPT_DIR, "run_transfer.py"),
            "--source", source,
            "--target", tgt,
            "--strategy", strategy,
            "--n_target", b,
            "--seed", str(s)
        ]
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])

        res = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if res.returncode != 0:
            raise RuntimeError(f"Run failed for {tgt} budget={b} seed={s} with exit code {res.returncode}")

    print("\n" + "=" * 80)
    print("ALL PHASE 1 RUNS COMPLETED! GENERATING SUMMARIES & FIGURES...")
    print("=" * 80)

    # Run analysis & summaries
    subprocess.run([PYTHON_EXEC, os.path.join(SCRIPT_DIR, "analyse.py")], cwd=PROJECT_ROOT)

    # Generate Learning Curve figures (F2)
    subprocess.run([PYTHON_EXEC, os.path.join(SCRIPT_DIR, "make_figures.py")], cwd=PROJECT_ROOT)

    print("\nPhase 1 successfully finished!")
    print(f"Updated master ledger: {all_runs_csv}")
    print("Check figures/ for updated learning curve plots with absolute data sample counts!")


if __name__ == "__main__":
    main()
