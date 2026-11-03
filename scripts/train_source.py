import os
import sys
import argparse
import json
import yaml
import torch

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
from plasmai_tl.models import Model
from plasmai_tl.loss import WeightedMSE
from plasmai_tl.train import set_seed, train_regression_model
from plasmai_tl.evaluate import evaluate_test_predictions
from plasmai_tl.utils import log_model_manifest, get_git_commit


def main():
    parser = argparse.ArgumentParser(description="Train fully-trained source model on all data.")
    parser.add_argument("--dataset", type=str, required=True, choices=["Ar", "O2", "C4F8"],
                        help="Source gas dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Source Model: {args.dataset} (Seed: {args.seed}, Device: {device}) ===")

    # Load configs
    datasets_cfg = load_dataset_registry(os.path.join(PROJECT_ROOT, "configs", "datasets.yaml"))
    with open(os.path.join(PROJECT_ROOT, "configs", "hparams.yaml"), "r", encoding="utf-8") as f:
        hparams = yaml.safe_load(f)

    gas_cfg = datasets_cfg[args.dataset]
    smoothing_cfg = hparams.get("smoothing", {"apply": True})
    df, in_cols, out_cols = load_raw_dataset(gas_cfg, root_dir=PROJECT_ROOT, smoothing_cfg=smoothing_cfg)

    # 15% fixed test set, full pool for training/val
    fixed_test_ratio = hparams["training"].get("fixed_test_ratio", 0.15)
    pool_idx, test_idx = split_fixed_test(len(df), test_ratio=fixed_test_ratio, test_seed=41)
    train_idx, val_idx = get_nested_budget_split(pool_idx, budget="full", seed=args.seed,
                                                 val_ratio=hparams["training"].get("val_ratio", 0.20))

    # Statistics solely on train split
    train_stats = compute_statistics(df, train_idx, in_cols + out_cols)

    train_loader, val_loader, test_loader, eff_batch = create_dataloaders(
        df, train_idx, val_idx, test_idx, in_cols, out_cols, train_stats,
        batch_size=32
    )

    out_dir = os.path.join(PROJECT_ROOT, "models", "source", f"{args.dataset}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Save stats
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(train_stats, f, indent=4)

    set_seed(args.seed)
    arch = hparams["nn_arch"]
    model = Model(input_size=len(in_cols), output_size=len(out_cols),
                  h1=arch["neurons_per_layer"], num_layers=arch["layers"],
                  output_activation=arch.get("output_activation"))

    tr_cfg = hparams["training"]
    epochs = args.epochs or tr_cfg["epochs"]
    criterion = WeightedMSE(outputs_points=[i + 1 for i in range(len(out_cols))],
                            device=device, model=model,
                            l1_lambda=tr_cfg.get("l1_lambda", 1e-4),
                            lambda_smooth=tr_cfg.get("lambda_smooth", 1.15),
                            tau_i=tr_cfg.get("tau_smooth", 0.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=tr_cfg["lr_scratch"],
                                 weight_decay=tr_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)

    trained_model, history, grad_logs = train_regression_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, num_epochs=epochs,
        device=device, patience=tr_cfg["patience"], scheduler=scheduler,
        verbose=True
    )

    weights_path = os.path.join(out_dir, "model.pth")
    torch.save(trained_model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # Track in models_manifest.csv (Section 9.2)
    manifest_csv = os.path.join(PROJECT_ROOT, "models_manifest.csv")
    git_commit = get_git_commit(PROJECT_ROOT)
    log_model_manifest(manifest_csv, f"source_{args.dataset}_s{args.seed}", weights_path, git_commit)

    metrics, comp_df = evaluate_test_predictions(
        trained_model, test_loader, device, train_stats, in_cols, out_cols,
        exp_dir=out_dir, exp_name=f"source_{args.dataset}_seed{args.seed}"
    )

    print(f"\nFinal Test Metrics for Source {args.dataset}:")
    print(f"  Mean R2   : {metrics['mean_r2']:.4f}")
    print(f"  Overall R2: {metrics['overall_r2']:.4f}")
    print(f"  MAPE      : {metrics['mape']:.2f}%")
    print(f"  RMSE      : {metrics['rmse']:.4f}")


if __name__ == "__main__":
    main()
