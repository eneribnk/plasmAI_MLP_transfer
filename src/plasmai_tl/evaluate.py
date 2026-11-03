import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from .data import unscale_matrix


def evaluate_test_predictions(model, test_loader, device, stats, input_cols, output_cols,
                              exp_dir=None, exp_name=None):
    """
    Evaluate trained model on unseen test set in physical units (unscaled).
    Computes R2, RMSE, MAE, MAPE (overall and per-point).
    """
    model.eval()
    all_inputs, all_targets, all_preds = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            preds = model(inputs)
            all_preds.append(preds.cpu().numpy())

    all_inputs = np.vstack(all_inputs)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)

    # Convert back to physical units (Bug 7 fix: use exact output_cols)
    un_in = unscale_matrix(all_inputs, input_cols, stats)
    un_tar = unscale_matrix(all_targets, output_cols, stats)
    un_pred = unscale_matrix(all_preds, output_cols, stats)

    df_in = pd.DataFrame(un_in, columns=input_cols)
    df_tar = pd.DataFrame(un_tar, columns=output_cols)
    df_pred = pd.DataFrame(un_pred, columns=[f"{c}_pred" for c in output_cols])
    comparison_df = pd.concat([df_in, df_tar, df_pred], axis=1)

    if exp_dir:
        os.makedirs(exp_dir, exist_ok=True)
        comp_path = os.path.join(exp_dir, "comparison.txt")
        comparison_df.to_csv(comp_path, sep="\t", index=False)

    per_point_r2 = {}
    r2_list = []
    for idx, col in enumerate(output_cols):
        r2_val = float(r2_score(un_tar[:, idx], un_pred[:, idx]))
        r2_list.append(r2_val)
        per_point_r2[col] = r2_val

    mean_r2 = float(np.mean(r2_list))
    overall_r2 = float(r2_score(un_tar.flatten(), un_pred.flatten()))

    diff = np.abs(un_tar - un_pred)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # MAPE with protective epsilon
    resid_pct = diff / np.maximum(un_tar, 1e-3) * 100.0
    mape = float(np.mean(resid_pct))

    per_point_mape = {
        col: float(np.mean(resid_pct[:, idx]))
        for idx, col in enumerate(output_cols)
    }

    metrics = {
        "experiment": exp_name or "evaluation",
        "mean_r2": mean_r2,
        "overall_r2": overall_r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "test_size": len(un_tar),
        "per_point_r2": per_point_r2,
        "per_point_mape": per_point_mape
    }

    if exp_dir:
        with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

    return metrics, comparison_df
