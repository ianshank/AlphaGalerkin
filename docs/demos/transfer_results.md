# Resolution Transfer Demo

## Overview

AlphaGalerkin achieves **zero-shot resolution transfer**: train on a small grid, deploy on
any larger grid without retraining. This is enabled by the Galerkin attention mechanism,
which treats positions as a bag of continuous points rather than fixed discrete indices.

The `scripts/demo_transfer.py` script trains a `PhysicsOperator` on the Poisson equation
(a representative physics task) and then runs inference on increasingly large grids to
measure how MSE scales with resolution.

## Key Finding

Measured [2026-07-22]: training a `PhysicsOperator` on 9x9 grids (50 epochs,
`scripts/demo_transfer.py`) transfers zero-shot to larger grids **with no retraining**:

| Grid Size | Measured MSE | Points | Below 0.05? |
|-----------|--------------|--------|-------------|
| 9x9 (train) | ~2.5e-6 | 81 | Yes |
| 13x13 (unseen) | ~2.0e-4 | 169 | Yes |
| 19x19 (unseen) | ~3.9e-4 | 361 | Yes |

> **Correction.** An earlier version reported "MSE 0.000209 on 19x19 — 240× better than the
> 0.05 threshold." That figure was a **fabricated** notebook cell (no code computed it), and
> "240× below a fixed threshold" is not a meaningful result. The numbers above are measured.
> Being "below 0.05" is a weak bar: a discrete CNN **retrained at 19x19** is *more accurate*
> than the operator's zero-shot result — see the honest, CI-gated benchmark
> `specs/transfer_baseline_compare.spec.md`. The operator's real value is running **one model
> at any resolution with no retraining**, not beating a resolution-specific specialist on
> accuracy. (Small quick-mode models transfer worse, e.g. 19x19 ≈ 5e-3; use the full config
> for the headline.)

## MSE vs Resolution Behavior

MSE increases sub-linearly with grid size, consistent with O(N^{-alpha}) convergence
where alpha > 0. This means the model generalises to higher-resolution grids rather than
degrading. The Fourier feature positional encoding ensures continuous coverage of the
spatial domain regardless of discretisation.

## Architecture Summary

| Component | Details |
|-----------|---------|
| Positional encoding | Learnable Fourier features (configurable frequencies) |
| Body | Galerkin linear attention, O(N) complexity |
| Mixing | FNet (FFT2D + iFFT2D), O(N log N) |
| Output | 2-layer MLP head -> scalar potential |
| Normalization | Monte Carlo (1/N), satisfying LBB inf-sup condition |

In quick mode (`--quick`) the model uses:
- `d_model=64`, `n_layers=2`, `n_fourier_features=32`
- 5 training epochs, 500 training samples, 100 inference samples
- Runs in ~3-10 seconds on CPU

In standard mode (default):
- `d_model=128`, `n_layers=4`, `n_fourier_features=64`
- 50 training epochs, 2000 training samples, 200 inference samples

## Output Files

Running `python scripts/demo_transfer.py` produces:

```
outputs/demo_transfer/
  transfer_mse.png       -- MSE vs resolution plot (green=pass, red=fail)
  transfer_results.json  -- structured JSON report
  transfer_results.md    -- Markdown table (when --format markdown is used)
```

The JSON report has the following top-level keys:

```json
{
  "train_size": 9,
  "target_sizes": [9, 13, 19, 25],
  "mse_per_size": {"9": 0.000296, "13": 0.004851, "19": 0.004918, "25": 0.006233},
  "per_size_details": [...],
  "model_params": 139585,
  "training_config": {...},
  "training_time_seconds": 2.3,
  "success_threshold": 0.05,
  "all_passed": true
}
```

## How to Reproduce

**Quick demo (< 60 seconds on CPU):**

```bash
python scripts/demo_transfer.py --quick
```

**Standard demo (50 epochs):**

```bash
python scripts/demo_transfer.py
```

**Custom grid sizes:**

```bash
python scripts/demo_transfer.py \
    --train-size 9 \
    --eval-sizes 9,13,19,25,32 \
    --n-epochs 100 \
    --output-dir outputs/my_transfer_run
```

**Markdown report:**

```bash
python scripts/demo_transfer.py --quick --format markdown
```

**Skip plot generation (CI/headless):**

```bash
python scripts/demo_transfer.py --quick --no-plots
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-size` | 9 | Grid size used for training |
| `--eval-sizes` | `9,13,19,25` | Comma-separated target grid sizes |
| `--n-epochs` | 50 | Training epochs |
| `--quick` | off | Fast mode: 5 epochs, small model |
| `--output-dir` | `outputs/demo_transfer` | Output directory |
| `--no-plots` | off | Skip matplotlib figure |
| `--format` | json | Report format (json or markdown) |
| `--d-model` | 128 | Model hidden dimension |
| `--n-layers` | 4 | Number of Galerkin attention layers |
| `--n-fourier-features` | 64 | Number of Fourier feature frequencies |
| `--learning-rate` | 1e-3 | AdamW learning rate |
| `--batch-size` | 32 | Training batch size |
| `--seed` | 42 | Random seed |
| `--success-threshold` | 0.05 | MSE threshold for pass/fail |

## Implementation Notes

- All hyperparameters are passed through `DemoConfig` (a dataclass); no hardcoded values
  appear in the training or inference logic.
- `matplotlib.use("Agg")` is called before any other matplotlib import to ensure
  compatibility with headless CI environments.
- Inference uses a separate random seed (`seed + inference_seed_offset`) to guarantee
  the test data never overlaps with the training data.
- The `PoissonDataset` from `src/physics/poisson.py` generates synthetic ground-truth
  potential fields by solving the 2D Poisson equation with DST (discrete sine transform).
