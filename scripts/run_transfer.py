import os
import sys
import argparse
import json
import time
import yaml
import torch
import pandas as pd

# Ensure package root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plasmai_tl.data import (
    load_dataset_registry,
    load_raw_dataset,
    split_fixed_test,
    get_nested_budget_split,
    compute_statistics,
    create_dataloaders
)
from plasmai_tl.strategies import build_model_for_strategy, STRATEGIES
from plasmai_tl.loss import WeightedMSE
from plasmai_tl.train import set_seed, train_regression_model
from plasmai_tl.evaluate import evaluate_test_predictions
from plasmai_tl.utils import log_model_manifest, get_git_commit


def main():
    parser = argparse.ArgumentParser(description="Run a single transfer learning or scratch experiment.")
    parser.add_argument("--source", type=str, default="None",
                        help="Source dataset (e.g. Ar, O2, C4F8) or 'None' for scratch S0")
    parser.add_argument("--target", type=str, required=True,
                        help="Target dataset (e.g. O2, C4F8, Ar)")
    parser.add_argument("--strategy", type=str, required=True, choices=["S0", "S1", "S2", "S3", "S4"],
                        help="Transfer learning strategy (S0-S4)")
    parser.add_argument("--n_target", type=str, default="full",
                        help="Target data budget: 25, 50, 100, 200, 400, 800, or 'full'")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--norm_mode", type=str, default="target", choices=["target", "hybrid", "source"],
                        help="Feature/target normalization strategy")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Epoch override")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu/cuda)")
    args = parser.parse_args()

    t_start = time.time()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    source_name = "None" if args.strategy == "S0" else args.source

    print(f"\n{'='*75}")
    print(f"EXPERIMENT RUN: Source={source_name} -> Target={args.target} | "
          f"Strategy={args.strategy} | Budget N={args.n_target} | Seed={args.seed}")
    print(f"{'='*75}")

    # Load configs
    datasets_cfg = load_dataset_registry(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    with open(os.path.join(PROJECT_ROOT, "configs", "hparams.yaml"), "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    tgt_cfg = datasets_cfg[args.target]
    smoothing_cfg = hparams.get("smoothing", {"apply": True})
    df_tgt, tgt_in_cols, tgt_out_cols = load_raw_dataset(tgt_cfg, root_dir=PROJECT_ROOT, smoothing_cfg=smoothing_cfg)

    # Fixed unseen test split (15%)
    fixed_test_ratio = hparams["training"].get("fixed_test_ratio", 0.15)
    pool_idx, test_idx = split_fixed_test(len(df_tgt), test_ratio=fixed_test_ratio, test_seed=41)

    # Budget split with nested subsets
    train_idx, val_idx = get_nested_budget_split(
        pool_idx, budget=args.n_target, seed=args.seed,
        val_ratio=hparams["training"].get("val_ratio", 0.20)
    )

    print(f"Data allocation: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Statistics solely on target train partition
    target_train_stats = compute_statistics(df_tgt, train_idx, tgt_in_cols + tgt_out_cols)

    # Handle Normalization Mode
    stats_to_use = dict(target_train_stats)
    src_in_cols = tgt_in_cols
    pretrained_weights_path = None

    if args.strategy != "S0" and source_name != "None":
        src_model_dir = os.path.join(PROJECT_ROOT, "models", "source", f"{source_name}_seed{args.seed}")
        pretrained_weights_path = os.path.join(src_model_dir, "model.pth")
        src_stats_path = os.path.join(src_model_dir, "stats.json")

        if not os.path.exists(pretrained_weights_path):
            raise FileNotFoundError(
                f"Source model weights not found at: {pretrained_weights_path}. "
                f"Train source first using `scripts/train_source.py --dataset {source_name} --seed {args.seed}`."
            )

        src_cfg = datasets_cfg[source_name]
        src_in_cols = src_cfg["inputs"]

        if args.norm_mode in ("hybrid", "source") and os.path.exists(src_stats_path):
            with open(src_stats_path, "r", encoding="utf-8") as f:
                src_stats = json.load(f)
            for c in tgt_in_cols:
                if c in src_stats:
                    stats_to_use[c] = src_stats[c]
            if args.norm_mode == "source":
                for c in tgt_out_cols:
                    if c in src_stats:
                        stats_to_use[c] = src_stats[c]

    # DataLoaders
    train_loader, val_loader, test_loader, eff_batch = create_dataloaders(
        df_tgt, train_idx, val_idx, test_idx, tgt_in_cols, tgt_out_cols, stats_to_use,
        batch_size=32
    )

    # Build model according to strategy
    set_seed(args.seed)
    arch = hparams["nn_arch"]
    model, strat_info = build_model_for_strategy(
        strategy_id=args.strategy,
        input_size=len(tgt_in_cols),
        output_size=len(tgt_out_cols),
        h1=arch["neurons_per_layer"],
        num_layers=arch["layers"],
        pretrained_source_path=pretrained_weights_path,
        source_input_cols=src_in_cols,
        target_input_cols=tgt_in_cols,
        output_activation=arch.get("output_activation"),
        device=device
    )

    # Loss and Optimizer
    tr_cfg = hparams["training"]
    epochs = args.epochs or tr_cfg["epochs"]
    patience = tr_cfg["patience"]
    base_lr = tr_cfg["lr_scratch"]

    # LR adjustment for fine-tuning
    effective_lr = base_lr * strat_info["lr_multiplier"]

    criterion = WeightedMSE(outputs_points=[i + 1 for i in range(len(tgt_out_cols))],
                            device=device, model=model,
                            l1_lambda=tr_cfg.get("l1_lambda", 1e-4),
                            lambda_smooth=tr_cfg.get("lambda_smooth", 1.15),
                            tau_i=tr_cfg.get("tau_smooth", 0.0))

    # Optimize only parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=effective_lr, weight_decay=tr_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)

    gradual_unfreeze = strat_info.get("gradual_unfreeze", False)
    unfreeze_epoch = tr_cfg.get("unfreeze_epoch", 100)
    unfreeze_lr_factor = tr_cfg.get("unfreeze_lr_factor", 0.5)

    trained_model, history, grad_logs = train_regression_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, num_epochs=epochs,
        device=device, patience=patience, scheduler=scheduler,
        gradual_unfreeze=gradual_unfreeze, unfreeze_epoch=unfreeze_epoch,
        unfreeze_lr_factor=unfreeze_lr_factor, verbose=False
    )

    # Output directory layout following Section 6.1
    # results/{target}/{source}/{strategy}/N{n}/seed{s}/
    budget_tag = f"N{args.n_target}"
    exp_dir = os.path.join(PROJECT_ROOT, "results", "runs", args.target, source_name, args.strategy,
                           budget_tag, f"seed{args.seed}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save weights & config
    weights_path = os.path.join(exp_dir, "model.pth")
    torch.save(trained_model.state_dict(), weights_path)
    
    # Track in models_manifest.csv (Section 9.2)
    manifest_csv = os.path.join(PROJECT_ROOT, "models_manifest.csv")
    git_commit = get_git_commit(PROJECT_ROOT)
    run_id = f"{args.target}_{source_name}_{args.strategy}_{budget_tag}_s{args.seed}"
    log_model_manifest(manifest_csv, run_id, weights_path, git_commit)

    run_cfg = {
        "source": source_name,
        "target": args.target,
        "strategy": args.strategy,
        "n_target": args.n_target,
        "seed": args.seed,
        "norm_mode": args.norm_mode,
        "effective_lr": effective_lr,
        "batch_size": eff_batch,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "git_commit": git_commit
    }
    with open(os.path.join(exp_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f)

    # Save gradient logs (Section 11.3)
    if grad_logs:
        df_grads = pd.DataFrame(grad_logs)
        df_grads.to_csv(os.path.join(exp_dir, "grad_norms.csv"), sep="\t", index=False)

    # Evaluation on unseen test set
    metrics, comp_df = evaluate_test_predictions(
        trained_model, test_loader, device, stats_to_use, tgt_in_cols, tgt_out_cols,
        exp_dir=exp_dir, exp_name=run_id
    )

    duration = time.time() - t_start
    print(f"Run completed in {duration:.1f}s | Test Mean R2: {metrics['mean_r2']:.4f} | "
          f"MAPE: {metrics['mape']:.2f}% | RMSE: {metrics['rmse']:.4f}")

    # Append to master results/all_runs.csv (Tab-delimited)
    csv_row = {
        "target": args.target,
        "source": source_name,
        "strategy": args.strategy,
        "n_target": args.n_target,
        "seed": args.seed,
        "norm_mode": args.norm_mode,
        "mean_r2": metrics["mean_r2"],
        "overall_r2": metrics["overall_r2"],
        "mape": metrics["mape"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "best_val_loss": history["best_val_loss"],
        "epochs_run": history["epochs_run"],
        "duration_sec": duration,
        "exp_dir": exp_dir
    }

    all_runs_csv = os.path.join(PROJECT_ROOT, "results", "all_runs.csv")
    df_row = pd.DataFrame([csv_row])
    if os.path.exists(all_runs_csv):
        df_row.to_csv(all_runs_csv, mode="a", sep="\t", header=False, index=False)
    else:
        df_row.to_csv(all_runs_csv, mode="w", sep="\t", header=True, index=False)

    print(f"Logged to {all_runs_csv}")


if __name__ == "__main__":
    main()
