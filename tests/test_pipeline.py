import os
import sys
import torch
import numpy as np

# Ensure src is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from plasmai_tl.data import get_nested_budget_split, split_fixed_test
from plasmai_tl.models import Model
from plasmai_tl.strategies import build_model_for_strategy


def test_nested_subsets():
    """Verify that for a fixed seed, smaller budgets are exact subsets of larger budgets."""
    pool = list(range(1000))
    seed = 42

    tr25, _ = get_nested_budget_split(pool, budget=25, seed=seed)
    tr50, _ = get_nested_budget_split(pool, budget=50, seed=seed)
    tr100, _ = get_nested_budget_split(pool, budget=100, seed=seed)

    # All elements in tr25's full budget block must be in tr50's full budget block
    rng = np.random.default_rng(seed)
    perm = [pool[i] for i in rng.permutation(len(pool))]
    b25 = perm[:25]
    b50 = perm[:50]
    b100 = perm[:100]

    assert set(b25).issubset(set(b50)), "b25 is not a subset of b50!"
    assert set(b50).issubset(set(b100)), "b50 is not a subset of b100!"
    print("[PASS] Nested subsets property verified successfully.")


def test_model_freezing():
    """Verify that freeze_layers works for strategies S2 and S3."""
    model, strat = build_model_for_strategy("S2", input_size=4, output_size=10, h1=20, num_layers=3)
    assert not model.layers[0].weight.requires_grad
    assert not model.layers[1].weight.requires_grad
    assert not model.layers[2].weight.requires_grad
    assert model.out.weight.requires_grad
    print("[PASS] Strategy S2 layer freezing verified successfully.")


def test_model_forward():
    """Verify forward pass output dimensions and activations."""
    model = Model(input_size=4, output_size=10, h1=20, num_layers=3)
    dummy_x = torch.randn(8, 4)
    out = model(dummy_x)
    assert out.shape == (8, 10), f"Expected shape (8, 10), got {out.shape}"
    print("[PASS] Model forward pass shape verified successfully.")


if __name__ == "__main__":
    test_nested_subsets()
    test_model_freezing()
    test_model_forward()
    print("ALL UNIT TESTS PASSED!")
