#!/usr/bin/env python3
"""Supervised Training on Synthetic Physics Data.

This script trains the PhysicsOperator on Poisson equation data,
validating the resolution-independent learning capability.

Training: 9x9 grids
Testing: 19x19 grids (zero-shot transfer)

Success criterion: MSE < 0.05 on 19x19 data without retraining.

Usage:
    python -m src.experiments.train_physics
    python -m src.experiments.train_physics --train-size 9 --eval-size 19
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.experiments.physics_model import PhysicsLoss, PhysicsOperator
from src.physics.poisson import PoissonDataset

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for physics training."""

    # Model
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    n_fourier_features: int = 64
    fourier_scale: float = 10.0
    use_fnet: bool = True

    # Data
    train_grid_size: int = 9
    eval_grid_size: int = 19
    n_train_samples: int = 5000
    n_eval_samples: int = 500
    n_charges: int = 5
    batch_size: int = 32

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 100
    log_interval: int = 10
    eval_interval: int = 10

    # Success criteria
    success_threshold: float = 0.05  # MSE threshold for zero-shot transfer success

    # Seed offsets for reproducible data splits
    # These ensure train/eval/transfer datasets don't overlap
    train_eval_seed_offset: int = 10000  # Offset for same-resolution eval data
    transfer_eval_seed_offset: int = 20000  # Offset for zero-shot transfer eval data

    # Output
    output_dir: str = "outputs/physics_poc"
    seed: int = 42


def train_epoch(
    model: PhysicsOperator,
    dataset: PoissonDataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: PhysicsLoss,
    device: torch.device,
    batch_size: int,
    log_interval: int = 50,
) -> float:
    """Train for one epoch.

    Args:
        model: The neural network model.
        dataset: Training dataset.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        device: Computation device.
        batch_size: Batch size for training.
        log_interval: Log progress every N batches.

    Returns:
        Average loss for the epoch.

    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_total_batches = (len(dataset) + batch_size - 1) // batch_size

    logger.debug(
        "epoch_start",
        dataset_size=len(dataset),
        batch_size=batch_size,
        n_batches=n_total_batches,
    )

    # Manual batching (no DataLoader for simplicity)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        # Stack into tensors
        coords = torch.tensor(
            np.stack([s.coords for s in samples]), device=device
        )
        charges = torch.tensor(
            np.stack([s.charges for s in samples]), device=device
        )
        targets = torch.tensor(
            np.stack([s.potential for s in samples]), device=device
        )

        # Forward pass
        optimizer.zero_grad()
        predictions = model(coords, charges)

        # Loss and backward
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Log progress periodically
        if n_batches % log_interval == 0:
            logger.debug(
                "batch_progress",
                batch=n_batches,
                total_batches=n_total_batches,
                running_loss=total_loss / n_batches,
            )

    avg_loss = total_loss / n_batches
    logger.debug("epoch_complete", avg_loss=avg_loss, n_batches=n_batches)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: PhysicsOperator,
    dataset: PoissonDataset,
    loss_fn: PhysicsLoss,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Evaluate model on dataset.

    Args:
        model: The neural network model.
        dataset: Evaluation dataset.
        loss_fn: Loss function (used for consistency, not computed here).
        device: Computation device.
        batch_size: Batch size for evaluation.

    Returns:
        Dictionary with MSE, MAE, and RMSE metrics.

    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    logger.debug(
        "evaluation_start",
        dataset_size=len(dataset),
        grid_size=dataset.grid_size,
    )

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
        targets = torch.tensor(
            np.stack([s.potential for s in samples]), device=device
        )

        predictions = model(coords, charges)

        # Compute metrics
        mse = torch.nn.functional.mse_loss(predictions, targets, reduction="sum")
        mae = torch.nn.functional.l1_loss(predictions, targets, reduction="sum")

        total_mse += mse.item()
        total_mae += mae.item()
        n_samples += predictions.numel()

    metrics = {
        "mse": total_mse / n_samples,
        "mae": total_mae / n_samples,
        "rmse": np.sqrt(total_mse / n_samples),
    }

    logger.debug(
        "evaluation_complete",
        mse=metrics["mse"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
    )

    return metrics


def train(config: TrainingConfig) -> dict[str, Any]:
    """Run full training pipeline.

    Returns:
        Dictionary with training results.

    """
    # Setup
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("training_starting", device=str(device))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    logger.info("creating_datasets", train_size=config.train_grid_size)

    train_dataset = PoissonDataset(
        grid_size=config.train_grid_size,
        n_samples=config.n_train_samples,
        n_charges=config.n_charges,
        seed=config.seed,
    )

    eval_train_dataset = PoissonDataset(
        grid_size=config.train_grid_size,
        n_samples=config.n_eval_samples,
        n_charges=config.n_charges,
        seed=config.seed + config.train_eval_seed_offset,
    )

    # Zero-shot evaluation dataset (different grid size!)
    eval_transfer_dataset = PoissonDataset(
        grid_size=config.eval_grid_size,
        n_samples=config.n_eval_samples,
        n_charges=config.n_charges,
        seed=config.seed + config.transfer_eval_seed_offset,
    )

    logger.info(
        "datasets_created",
        train_samples=len(train_dataset),
        train_stats=train_dataset.get_statistics(),
    )

    # Create model
    model = PhysicsOperator(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_fourier_features=config.n_fourier_features,
        fourier_scale=config.fourier_scale,
        use_fnet=config.use_fnet,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model_created", n_parameters=f"{n_params:,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    loss_fn = PhysicsLoss()

    # Training history
    history = {
        "train_loss": [],
        "eval_mse_same_res": [],
        "eval_mse_transfer": [],
        "learning_rate": [],
    }

    best_transfer_mse = float("inf")
    start_time = time.time()

    # Training loop
    for epoch in range(config.n_epochs):
        # Train
        train_loss = train_epoch(
            model, train_dataset, optimizer, loss_fn, device, config.batch_size
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["learning_rate"].append(scheduler.get_last_lr()[0])

        # Evaluate
        if (epoch + 1) % config.eval_interval == 0:
            # Same resolution
            eval_same = evaluate(
                model, eval_train_dataset, loss_fn, device, config.batch_size
            )
            history["eval_mse_same_res"].append(eval_same["mse"])

            # Zero-shot transfer (different resolution!)
            eval_transfer = evaluate(
                model, eval_transfer_dataset, loss_fn, device, config.batch_size
            )
            history["eval_mse_transfer"].append(eval_transfer["mse"])

            if eval_transfer["mse"] < best_transfer_mse:
                best_transfer_mse = eval_transfer["mse"]
                # Save best model
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": vars(config),
                        "epoch": epoch,
                        "transfer_mse": best_transfer_mse,
                    },
                    output_dir / "best_model.pt",
                )

            logger.info(
                "evaluation",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.6f}",
                eval_mse_same=f"{eval_same['mse']:.6f}",
                eval_mse_transfer=f"{eval_transfer['mse']:.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
        elif (epoch + 1) % config.log_interval == 0:
            logger.info(
                "training_step",
                epoch=epoch + 1,
                train_loss=f"{train_loss:.6f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

    # Final evaluation
    final_same = evaluate(
        model, eval_train_dataset, loss_fn, device, config.batch_size
    )
    final_transfer = evaluate(
        model, eval_transfer_dataset, loss_fn, device, config.batch_size
    )

    elapsed = time.time() - start_time

    # Results
    results = {
        "config": vars(config),
        "history": history,
        "final_metrics": {
            "same_resolution": final_same,
            "zero_shot_transfer": final_transfer,
        },
        "best_transfer_mse": best_transfer_mse,
        "training_time_seconds": elapsed,
        "success": final_transfer["mse"] < config.success_threshold,
        "success_threshold": config.success_threshold,
    }

    # Save results
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training time: {elapsed:.1f} seconds")
    print(f"Train grid size: {config.train_grid_size}x{config.train_grid_size}")
    print(f"Eval grid size: {config.eval_grid_size}x{config.eval_grid_size}")
    print(f"Final MSE (same resolution): {final_same['mse']:.6f}")
    print(f"Final MSE (zero-shot transfer): {final_transfer['mse']:.6f}")
    print(f"Best transfer MSE: {best_transfer_mse:.6f}")
    print(f"Success threshold: {config.success_threshold}")
    print()

    if results["success"]:
        print(f"✓ PASS: Zero-shot transfer MSE < {config.success_threshold}")
    else:
        print(f"✗ FAIL: Zero-shot transfer MSE >= {config.success_threshold}")

    print("=" * 60)

    return results


def main() -> None:
    """Run the physics training script."""
    parser = argparse.ArgumentParser(description="Train PhysicsOperator on Poisson data")
    parser.add_argument("--train-size", type=int, default=9, help="Training grid size")
    parser.add_argument("--eval-size", type=int, default=19, help="Evaluation grid size")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--fourier-scale", type=float, default=10.0, help="Fourier scale")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--success-threshold", type=float, default=0.05,
                        help="MSE threshold for zero-shot transfer success")
    parser.add_argument("--output-dir", type=str, default="outputs/physics_poc")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = TrainingConfig(
        train_grid_size=args.train_size,
        eval_grid_size=args.eval_size,
        n_epochs=args.n_epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        fourier_scale=args.fourier_scale,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        success_threshold=args.success_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    train(config)


if __name__ == "__main__":
    main()
