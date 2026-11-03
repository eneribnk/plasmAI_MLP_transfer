# plasmAI_MLP_transfer

Framework for Data-Efficient Transfer Learning on Plasma Surrogate Neural Networks (COBRA Reactor: Ar, O₂, C₄F₈).

---

## 1. Project Overview & Phase 0 Implementation

This repository implements the standardized Transfer Learning experimental protocol described in `TransferLearning_Experiment_Πλανο.docx`.

### Addressed Core Bugs & Enhancements (Section 6.2):
1. **Early Stopping State Dict Deepcopy (`train.py`)**: Replaced shallow reference `best_model_state = model.state_dict()` with `copy.deepcopy(model.state_dict())`, guaranteeing that the checkpoint with the lowest validation loss is restored (even if training runs through all epochs).
2. **Within-Budget Validation Splitting (`data.py`)**: For each budget $N \in \{25, 50, 100, 200, 400, 800, \text{full}\}$, validation samples (20%) are carved out from within the budget. The 15% test set remains strictly held out and unseen.
3. **Nested Budget Subsets (`data.py`)**: For any fixed seed, smaller budgets are guaranteed strict subsets of larger budgets ($N=25 \subset N=50 \subset N=100 \dots$), eliminating sampling noise from learning curves.
4. **Configurable Model & Strategy Architecture (`models.py`, `strategies.py`)**: Configurable input sizes, output heads, and parameterized donor layer indices without hardcoded numbers.
5. **Dynamic Hyperparameters & Logging (`configs/hparams.yaml`)**: Learning rates, fine-tuning scaling factors (`finetune_lr_factor`), unfreeze epoch (`unfreeze_epoch`), and layer gradient norms (`grad_norms.csv`) logged every epoch.
6. **Leak-Free Statistics Normalization (`data.py`)**: Statistics ($z$-score mean & std) computed exclusively on the active training partition of the budget.
7. **Clean Per-Point Metric Evaluation (`evaluate.py`)**: $R^2$, MAPE, and RMSE evaluated across dynamic output point lists in physical units.

---

## 2. Directory Structure

```
plasmAI_MLP_transfer/
├── configs/
│   ├── datasets.yaml          # Registry: paths, inputs, outputs, rate types
│   ├── hparams.yaml           # Architecture & hyperparameter specs
│   └── sweeps/                # Sweeps configurations
├── data/
│   ├── raw/                   # Raw datasets (Ar, O2, C4F8)
│   └── processed/             # Preprocessed splits / stats
├── src/plasmai_tl/            # Core package
│   ├── data.py                # Nested data budgeting, splits, scalers
│   ├── models.py              # Configurable MLP & multi-head architectures
│   ├── loss.py                # Weighted MSE with spatial smoothness & L1
│   ├── strategies.py          # S0 (Scratch), S1 (Fine-tune), S2 (Frozen), S3 (Gradual)
│   ├── train.py               # Training loop, early stopping, gradient logging
│   └── evaluate.py            # Physical evaluation (R2, MAPE, RMSE)
├── scripts/
│   ├── train_source.py        # Pretrain full source models (e.g. Ar, O2)
│   ├── run_transfer.py        # Run single transfer learning / scratch experiment
│   ├── run_sanity_phase0.py   # Automated Phase 0 validation runner
│   ├── analyse.py             # Aggregate all_runs.csv into summary tables
│   └── make_figures.py        # Generate publication figures (F2 learning curves)
├── results/
│   ├── all_runs.csv           # Master experiment ledger
│   ├── summary/               # Aggregated mean +/- std tables
│   └── runs/                  # Per-run models, configs, grad_norms, and metrics
├── figures/                   # Generated high-resolution plots
└── tests/
    └── test_pipeline.py       # Unit test suite
```

---

## 3. Running Phase 0 (Sanity Check & Baseline Reproduction)

To complete **Phase 0**, run the automated sanity script:

```bash
cd C:\v31model\plasmAI_MLP_transfer
python scripts/run_sanity_phase0.py
```

This script:
1. Trains Argon ($Ar$) source models on the full dataset across 3 seeds (`seed=0, 1, 2`).
2. Trains Oxygen ($O_2$) baseline from scratch ($S0$) on full budget across 3 seeds.
3. Performs Transfer Learning ($Ar \rightarrow O_2$, strategy $S1$) on full budget across 3 seeds.
4. Validates that test $R^2 \ge 0.98$ and produces `results/summary/phase0_sanity_summary.csv`.

---

## 4. Running Custom Experiments

To run any single experiment (e.g., $Ar \rightarrow O_2$, Strategy $S1$, Budget $N=100$, Seed 0):
```bash
python scripts/run_transfer.py --source Ar --target O2 --strategy S1 --n_target 100 --seed 0
```

To run a scratch baseline ($S0$ on $C_4F_8$, Budget $N=50$, Seed 1):
```bash
python scripts/run_transfer.py --source None --target C4F8 --strategy S0 --n_target 50 --seed 1
```

To summarize all experiments and produce learning curves:
```bash
python scripts/analyse.py
python scripts/make_figures.py
```
