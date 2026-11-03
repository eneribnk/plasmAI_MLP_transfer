# Datasets Documentation

This directory documents the origin, physical units, and parameters of the datasets used for Transfer Learning experiments.

## 1. Datasets Overview

| Dataset ID | Source File | Samples | Inputs (4) | Outputs (10) | Physical Units |
|---|---|---|---|---|---|
| **Ar** | `v10_Ar_eth_dataset_full.txt` | 1511 | Power, Pressure, Feed, Vbias | Point1_EtchRate .. Point10_EtchRate | nm/min |
| **O2** | `v3_3_1_O2_eth_dataset_full.txt` | 1632 | Power, Pressure, Feed, Vbias | Point1_EtchRate .. Point10_EtchRate | nm/min |
| **C4F8** | `v10_C4F8_eth_dataset_full.txt` | 1500 | Power, Pressure, Feed, Vbias | Point1_DepositionRate .. Point10_DepositionRate | nm/min |

## 2. Shared Inputs Alignment
All 3 datasets share the exact same 4 input parameters in the COBRA reactor:
1. `Power` (W)
2. `Pressure` (Torr)
3. `Feed` (sccm)
4. `Vbias` (V)

## 3. Evaluation Split Protocol
- **Test Set**: Fixed 15% randomly partitioned with fixed seed (41), strictly held out across all runs.
- **Budget Pools**: 85% remaining pool used for nested budget partitions ($N \in \{25, 50, 100, 200, 400, 800, \text{full}\}$).
- **Validation**: 20% carved from within each active budget.
