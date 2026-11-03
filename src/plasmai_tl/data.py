import os
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter


class TabularDataset(Dataset):
    """PyTorch Dataset for normalized tabular features and targets."""
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset_registry(config_path):
    """Load dataset registry from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_dataset(dataset_cfg, root_dir=None, smoothing_cfg=None):
    """
    Load raw dataframe, clean column names, apply optional Savitzky-Golay target smoothing.
    """
    path = dataset_cfg["data_path"]
    if root_dir and not os.path.isabs(path):
        path = os.path.join(root_dir, path)

    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()

    input_cols = dataset_cfg["inputs"]
    output_cols = dataset_cfg["outputs"]
    required_cols = input_cols + output_cols

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if smoothing_cfg and smoothing_cfg.get("apply", False):
        win = smoothing_cfg.get("window_length", 5)
        poly = smoothing_cfg.get("polyorder", 2)
        if win > len(output_cols):
            win = len(output_cols)
        if win % 2 == 0:
            win -= 1
        if win >= 3:
            y_vals = df[output_cols].values
            df[output_cols] = savgol_filter(y_vals, window_length=win, polyorder=poly, axis=1)

    return df, input_cols, output_cols


def split_fixed_test(n_total, test_ratio=0.15, test_seed=41):
    """
    Carve out a fixed, strictly held-out test split (15%) for unbiased evaluation across all experiments.
    Returns (pool_indices, test_indices).
    """
    rng = np.random.default_rng(test_seed)
    shuffled = rng.permutation(n_total)
    test_size = int(test_ratio * n_total)
    test_idx = shuffled[:test_size].tolist()
    pool_idx = shuffled[test_size:].tolist()
    return pool_idx, test_idx


def get_nested_budget_split(pool_indices, budget, seed, val_ratio=0.20):
    """
    Deterministically creates nested subsets for data scarcity budgets.
    For a given seed, budget N=50 is an exact subset of N=100, which is an exact subset of N=200, etc.
    Validation samples are taken from within the budget.
    """
    rng = np.random.default_rng(seed)
    permuted_pool = [pool_indices[i] for i in rng.permutation(len(pool_indices))]

    if budget == "full" or budget is None:
        budget_samples = permuted_pool
    else:
        budget_n = int(budget)
        budget_samples = permuted_pool[:min(budget_n, len(permuted_pool))]

    n_samples = len(budget_samples)
    val_size = max(4, int(val_ratio * n_samples)) if n_samples > 10 else max(2, int(val_ratio * n_samples))
    train_size = n_samples - val_size

    train_idx = budget_samples[:train_size]
    val_idx = budget_samples[train_size:]
    return train_idx, val_idx


def compute_statistics(df, row_indices, columns):
    """
    Compute mean and std exclusively on specified training rows to prevent data leakage.
    """
    sub_df = df.iloc[row_indices]
    stats = {}
    for col in columns:
        mean_val = float(sub_df[col].mean())
        std_val = float(sub_df[col].std())
        stats[col] = {
            "mean": mean_val,
            "std": std_val if std_val > 1e-7 else 1.0
        }
    return stats


def scale_matrix(data_matrix, columns, stats):
    """Standardize a 2D numpy array using stats dict."""
    arr = np.asarray(data_matrix, dtype=np.float32).copy()
    for col_idx, col_name in enumerate(columns):
        m = stats[col_name]["mean"]
        s = stats[col_name]["std"]
        arr[:, col_idx] = (arr[:, col_idx] - m) / s
    return arr


def unscale_matrix(data_matrix, columns, stats, clip_min=None):
    """Inverse standardize predictions/targets to physical units."""
    arr = np.asarray(data_matrix, dtype=np.float32).copy()
    if arr.ndim == 1:
        arr = arr[None, :]
    for col_idx, col_name in enumerate(columns):
        m = stats[col_name]["mean"]
        s = stats[col_name]["std"]
        arr[:, col_idx] = arr[:, col_idx] * s + m
    if clip_min is not None:
        arr = np.maximum(arr, clip_min)
    return arr


def create_dataloaders(df, train_idx, val_idx, test_idx, input_cols, output_cols, stats, batch_size=32):
    """
    Prepare normalized PyTorch DataLoaders for train, val, and test.
    Batch size for training adapts dynamically to budget size.
    """
    X_raw = df[input_cols].values
    y_raw = df[output_cols].values

    X_norm = scale_matrix(X_raw, input_cols, stats)
    y_norm = scale_matrix(y_raw, output_cols, stats)

    train_ds = TabularDataset(X_norm[train_idx], y_norm[train_idx])
    val_ds = TabularDataset(X_norm[val_idx], y_norm[val_idx])
    test_ds = TabularDataset(X_norm[test_idx], y_norm[test_idx])

    # Dynamic batch size following plan: batch = min(batch_size, max(4, n_train // 8))
    eff_batch = min(batch_size, max(4, len(train_idx) // 4)) if len(train_idx) >= 8 else len(train_idx)

    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=eff_batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    return train_loader, val_loader, test_loader, eff_batch
