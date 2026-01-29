# Stress-Testing CNNs (CIFAR-10) — Reproducible Runs

This repository trains a **baseline CNN** and a **single constrained modification** (residual connections) on **CIFAR-10** using **PyTorch**, then exports diagnostics, failure cases, and Grad-CAM visualizations.

## Assignment compliance notes (what we did)

- **Framework**: PyTorch only.
- **Dataset**: CIFAR-10 via `torchvision.datasets.CIFAR10` with the **official train/test split** only (no extra data).
- **No pretrained weights**: all models are trained **from scratch**.
- **Input resolution**: CIFAR-10 \(32 \times 32\).
- **Fixed seed**: `SEED = 2026` (see `code/train.py`) used for determinism (including the train/val split).
- **Epoch budget**: `EPOCHS = 40` (≤ 50).
- **Single constrained modification**: baseline vs modified differs **only** by enabling **residual connections** (`residual=False` vs `residual=True`) in the same `FlexibleCNN` backbone.
- **Train vs validation transforms**: training uses standard augmentation; validation/test use **no augmentation** (normalization only).

## Environment

- PyTorch (GPU recommended)
- torchvision
- matplotlib
- numpy
- scikit-learn

On the provided cluster module, this is handled by:

```bash
module load python/3.10.pytorch
```

## Reproduce all results

From `assignment1/code/`:

```bash
python3 train.py
python3 explain.py
```

Or submit via Slurm:

```bash
sbatch run.sh
```

## What gets generated

All outputs are written to `assignment1/results/`:

- **Training/validation loss curves**: `baseline_dynamics.png`, `modified_dynamics.png`
- **Training/validation accuracy curves**: `baseline_accuracy.png`, `modified_accuracy.png`
- **Confusion matrices**: `baseline_confusion_matrix.png`, `modified_confusion_matrix.png`
- **Per-class accuracy comparison**: `per_class_accuracy.png`
- **Validation accuracy comparison**: `comparison_curves.png`
- **Top high-confidence failures (with test indices)**: `baseline_failures.png`, `modified_failures.png`
- **Machine-readable failure lists**: `baseline_failures.json`, `modified_failures.json`
- **Chosen 3 canonical failure cases** (for the report): `chosen_failure_cases.json`
- **Grad-CAM comparison on the chosen cases**: `gradcam_comparison.png`

## Notes for the report

- Dataset: **CIFAR-10**, using the official train/test split from `torchvision.datasets.CIFAR10`.
- Reproducibility: a fixed seed is used (see `SEED` in `code/train.py`) and the train/val split is generated deterministically.
- Constrained improvement: the only change between baseline and modified models is enabling **residual connections** in the same `FlexibleCNN` backbone.

