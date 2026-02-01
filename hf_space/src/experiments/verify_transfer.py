#!/usr/bin/env python3
"""Zero-Shot Transfer Verification Script.

This script verifies that a model trained on one resolution can
generalize to different resolutions without retraining.

Primary test: Train on 9x9 → Evaluate on 19x19 (MSE < 0.05)

Usage:
    python -m src.experiments.verify_transfer
    python -m src.experiments.verify_transfer --model-path outputs/physics_poc/best_model.pt
    python -m src.experiments.verify_transfer --train-size 9 --eval-sizes 9,13,19
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from src.experiments.physics_model import PhysicsOperator
from src.physics.poisson import PoissonDataset

logger = structlog.get_logger(__name__)

# Module-level constants with documented rationale
# These can be overridden via function parameters where appropriate

# Seed offset ensures eval data doesn't overlap with training data
# Using large offset (50000) provides clear separation
DEFAULT_EVAL_SEED_OFFSET: int = 50000

# Default resolutions for zero-shot transfer testing
# 9x9 is typical training size, 19x19 is standard Go board
DEFAULT_EVAL_SIZES: list[int] = [9, 13, 19]
DEFAULT_RESOLUTION_TEST_SIZES: list[int] = [9, 13, 19, 25]

# Charge position bounds (fraction of grid)
# Keeping charges away from boundaries (0.1-0.9) avoids boundary artifacts
DEFAULT_CHARGE_POSITION_MIN: float = 0.1
DEFAULT_CHARGE_POSITION_MAX: float = 0.9

# Primary evaluation size for zero-shot transfer (standard Go board)
PRIMARY_EVAL_SIZE: int = 19


@dataclass
class TransferResult:
    """Result of zero-shot transfer test."""

    train_size: int
    eval_size: int
    mse: float
    mae: float
    rmse: float
    max_error: float
    n_samples: int
    passed: bool  # MSE < threshold


def load_model(
    model_path: Path,
    device: torch.device,
) -> tuple[PhysicsOperator, dict[str, Any]]:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Tuple of (model, config dict).

    """
    logger.debug("loading_model", path=str(model_path), device=str(device))

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})

    model = PhysicsOperator(
        d_model=config.get("d_model", 128),
        n_heads=config.get("n_heads", 4),
        n_layers=config.get("n_layers", 4),
        n_fourier_features=config.get("n_fourier_features", 64),
        fourier_scale=config.get("fourier_scale", 10.0),
        use_fnet=config.get("use_fnet", True),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model_loaded", path=str(model_path), n_parameters=n_params)

    return model, config


@torch.no_grad()
def evaluate_transfer(
    model: PhysicsOperator,
    train_size: int,
    eval_size: int,
    n_samples: int,
    device: torch.device,
    seed: int = 42,
    threshold: float = 0.05,
    n_charges: int = 5,
    batch_size: int = 32,
) -> TransferResult:
    """Evaluate zero-shot transfer to a different resolution.

    Args:
        model: Trained model.
        train_size: Grid size used for training.
        eval_size: Grid size for evaluation.
        n_samples: Number of samples to evaluate.
        device: Computation device.
        seed: Random seed for reproducibility.
        threshold: MSE threshold for passing.
        n_charges: Number of point charges per sample.
        batch_size: Batch size for evaluation.

    Returns:
        TransferResult with metrics.

    """
    # Use offset seed to ensure eval data differs from training
    eval_seed = seed + DEFAULT_EVAL_SEED_OFFSET
    dataset = PoissonDataset(
        grid_size=eval_size,
        n_samples=n_samples,
        n_charges=n_charges,
        seed=eval_seed,
    )

    logger.debug(
        "evaluate_transfer_start",
        train_size=train_size,
        eval_size=eval_size,
        n_samples=n_samples,
        threshold=threshold,
    )

    all_predictions = []
    all_targets = []

    indices = list(range(len(dataset)))

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        coords = torch.tensor(
            np.stack([s.coords for s in samples]), device=device
        )
        charges = torch.tensor(
            np.stack([s.charges for s in samples]), device=device
        )
        targets = np.stack([s.potential for s in samples])

        predictions = model(coords, charges).cpu().numpy()

        all_predictions.append(predictions)
        all_targets.append(targets)

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    errors = predictions - targets
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    max_error = float(np.max(np.abs(errors)))

    return TransferResult(
        train_size=train_size,
        eval_size=eval_size,
        mse=mse,
        mae=mae,
        rmse=rmse,
        max_error=max_error,
        n_samples=n_samples,
        passed=mse < threshold,
    )


def run_verification(
    model_path: Path | None = None,
    train_size: int = 9,
    eval_sizes: list[int] | None = None,
    n_samples: int = 500,
    threshold: float = 0.05,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full zero-shot transfer verification.

    Args:
        model_path: Path to trained model. If None, trains a new model.
        train_size: Grid size used for training.
        eval_sizes: List of grid sizes to evaluate on.
        n_samples: Number of samples per evaluation.
        threshold: MSE threshold for passing.
        output_dir: Directory to save results.

    Returns:
        Dictionary with verification results.

    """
    if eval_sizes is None:
        eval_sizes = DEFAULT_EVAL_SIZES.copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("verification_starting", device=str(device))

    # Load or train model
    if model_path is None or not model_path.exists():
        logger.info("no_model_found_training", model_path=str(model_path))
        # Train a new model
        from src.experiments.train_physics import TrainingConfig, train

        config = TrainingConfig(
            train_grid_size=train_size,
            eval_grid_size=max(eval_sizes),
            output_dir=str(output_dir) if output_dir else "outputs/physics_poc",
        )
        train(config)
        model_path = Path(config.output_dir) / "best_model.pt"

    model, config = load_model(model_path, device)
    logger.info("model_loaded", path=str(model_path))

    # Run transfer tests
    results: list[TransferResult] = []

    for eval_size in eval_sizes:
        logger.info("evaluating_transfer", eval_size=eval_size)

        result = evaluate_transfer(
            model=model,
            train_size=train_size,
            eval_size=eval_size,
            n_samples=n_samples,
            device=device,
            threshold=threshold,
        )
        results.append(result)

        status = "[PASS]" if result.passed else "[FAIL]"
        logger.info(
            "transfer_result",
            eval_size=eval_size,
            mse=f"{result.mse:.6f}",
            status=status,
        )

    # Compute summary
    all_passed = all(r.passed for r in results)
    # Primary result is the standard Go board size (19x19) if available
    primary_result = next(
        (r for r in results if r.eval_size == PRIMARY_EVAL_SIZE), results[-1]
    )

    summary = {
        "model_path": str(model_path),
        "train_size": train_size,
        "threshold": threshold,
        "results": [
            {
                "eval_size": r.eval_size,
                "mse": r.mse,
                "mae": r.mae,
                "rmse": r.rmse,
                "max_error": r.max_error,
                "passed": r.passed,
            }
            for r in results
        ],
        "primary_transfer": {
            "from": train_size,
            "to": primary_result.eval_size,
            "mse": primary_result.mse,
            "passed": primary_result.passed,
        },
        "all_passed": all_passed,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("ZERO-SHOT TRANSFER VERIFICATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Trained on: {train_size}x{train_size}")
    print(f"Threshold: MSE < {threshold}")
    print()
    print(f"{'Eval Size':<12} {'MSE':<12} {'RMSE':<12} {'Status':<10}")
    print("-" * 46)

    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"{r.eval_size}x{r.eval_size:<7} {r.mse:<12.6f} {r.rmse:<12.6f} {status}")

    print()
    if all_passed:
        print("[PASS] ALL TESTS PASSED - Zero-shot transfer verified!")
    else:
        print("[FAIL] SOME TESTS FAILED - Zero-shot transfer not achieved")
    print("=" * 60)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "transfer_verification.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def verify_resolution_independence(
    model: PhysicsOperator,
    device: torch.device,
    resolutions: list[int] | None = None,
    n_samples: int = 100,
    n_charges: int = 5,
    seed: int = 12345,
) -> dict[str, NDArray[np.float32]]:
    """Verify that model predictions are consistent across resolutions.

    For the same underlying physical problem, predictions at different
    resolutions should converge to the same continuous solution.

    Args:
        model: Trained model.
        device: Computation device.
        resolutions: List of resolutions to test.
        n_samples: Number of test problems.
        n_charges: Number of point charges per sample.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with consistency metrics.

    """
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTION_TEST_SIZES.copy()

    rng = np.random.default_rng(seed)

    logger.info(
        "resolution_independence_start",
        resolutions=resolutions,
        n_samples=n_samples,
    )

    # We'll test at common points that exist in all grids
    # Use the coarsest grid points and interpolate others

    coarsest = min(resolutions)
    finest = max(resolutions)

    # Generate random charge configurations
    # For consistency, we place charges at normalized positions
    all_errors = []
    log_interval = max(1, n_samples // 10)  # Log ~10 times

    for sample_idx in range(n_samples):
        if sample_idx > 0 and sample_idx % log_interval == 0:
            logger.debug(
                "resolution_independence_progress",
                completed=sample_idx,
                total=n_samples,
                mean_error_so_far=float(np.mean(all_errors)) if all_errors else 0.0,
            )
        # Random charge positions (normalized)
        charge_positions = rng.uniform(
            DEFAULT_CHARGE_POSITION_MIN,
            DEFAULT_CHARGE_POSITION_MAX,
            size=(n_charges, 2),
        )
        charge_magnitudes = rng.normal(0, 1, size=n_charges)

        predictions_at_finest: dict[int, NDArray[np.float32]] = {}

        for res in resolutions:
            # Create grid
            x = np.linspace(0, 1, res, dtype=np.float32)
            coords = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)

            # Create charge field by placing charges
            charges = np.zeros(res * res, dtype=np.float32)
            for (px, py), mag in zip(charge_positions, charge_magnitudes, strict=True):
                # Find nearest grid point
                ix = int(px * (res - 1))
                iy = int(py * (res - 1))
                idx = ix * res + iy
                charges[idx] += mag

            # Predict
            coords_t = torch.tensor(coords[None], device=device)
            charges_t = torch.tensor(charges[None], device=device)
            pred = model(coords_t, charges_t).cpu().numpy()[0]

            predictions_at_finest[res] = pred

        # Compare predictions at common points (coarsest grid points)
        # For each coarsest grid point, find corresponding finest grid prediction
        coarse_preds = predictions_at_finest[coarsest]
        fine_preds = predictions_at_finest[finest]

        # Interpolate fine to coarse grid positions
        # Simple: take predictions at coarse grid locations in fine grid
        fine_at_coarse = []
        for i in range(coarsest):
            for j in range(coarsest):
                fine_i = i * (finest - 1) // (coarsest - 1)
                fine_j = j * (finest - 1) // (coarsest - 1)
                fine_idx = fine_i * finest + fine_j
                fine_at_coarse.append(fine_preds[fine_idx])

        fine_at_coarse = np.array(fine_at_coarse)
        error = np.mean((coarse_preds - fine_at_coarse) ** 2)
        all_errors.append(error)

    results = {
        "mean_consistency_error": np.mean(all_errors),
        "std_consistency_error": np.std(all_errors),
        "max_consistency_error": np.max(all_errors),
        "resolutions_tested": resolutions,
    }

    logger.info(
        "resolution_independence_complete",
        mean_error=float(results["mean_consistency_error"]),
        max_error=float(results["max_consistency_error"]),
    )

    return results


def main() -> None:
    """Run the zero-shot transfer verification."""
    parser = argparse.ArgumentParser(
        description="Verify zero-shot transfer of trained model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/physics_poc/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=9,
        help="Grid size the model was trained on",
    )
    parser.add_argument(
        "--eval-sizes",
        type=str,
        default="9,13,19",
        help="Comma-separated list of evaluation grid sizes",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of samples per evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="MSE threshold for passing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/physics_poc",
        help="Directory to save results",
    )

    args = parser.parse_args()

    eval_sizes = [int(x) for x in args.eval_sizes.split(",")]
    model_path = Path(args.model_path)

    results = run_verification(
        model_path=model_path if model_path.exists() else None,
        train_size=args.train_size,
        eval_sizes=eval_sizes,
        n_samples=args.n_samples,
        threshold=args.threshold,
        output_dir=Path(args.output_dir),
    )

    # Exit with appropriate code
    sys.exit(0 if results["all_passed"] else 1)


if __name__ == "__main__":
    main()
