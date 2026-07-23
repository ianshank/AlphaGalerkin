#!/usr/bin/env python3
"""Resolution Transfer Demo -- AlphaGalerkin Zero-Shot Transfer.

Trains a PhysicsOperator on a small grid and tests it on multiple
larger grids without retraining, demonstrating resolution independence.

Usage:
    python scripts/demo_transfer.py --quick
    python scripts/demo_transfer.py --train-size 9 --eval-sizes 9,13,19,25
    python scripts/demo_transfer.py --n-epochs 50 --output-dir outputs/demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for CI compatibility
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root on path when run as script
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.experiments.physics_model import PhysicsLoss, PhysicsOperator  # noqa: E402
from src.physics.poisson import PoissonDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DemoConfig:
    """All demo parameters -- no hardcoded values in logic below."""

    # Grid sizes
    train_size: int = 9
    target_sizes: list[int] = field(default_factory=lambda: [9, 13, 19, 25])

    # Model architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    n_fourier_features: int = 64
    fourier_scale: float = 10.0
    use_fnet: bool = True
    dropout: float = 0.1

    # Training
    n_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    n_train_samples: int = 2000
    n_inference_samples: int = 200
    n_charges: int = 5
    log_interval: int = 5

    # Reproducibility
    seed: int = 42

    # Thresholds
    success_threshold: float = 0.05
    # Seed offset keeps inference data separate from training data
    inference_seed_offset: int = 50000

    # Output
    output_dir: str = "outputs/demo_transfer"
    report_format: str = "json"  # "json" | "markdown"


@dataclass
class SizeMetrics:
    """Metrics for a single target grid size."""

    grid_size: int
    mse: float
    mae: float
    rmse: float
    n_points: int  # grid_size ** 2
    passed: bool


@dataclass
class DemoResult:
    """Full demo result container."""

    train_size: int
    target_sizes: list[int]
    mse_per_size: dict[str, float]  # str key for JSON serialisation
    per_size_details: list[dict[str, Any]]
    model_params: int
    training_config: dict[str, Any]
    training_time_seconds: float
    success_threshold: float
    all_passed: bool


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _build_model(cfg: DemoConfig, device: torch.device) -> PhysicsOperator:
    """Instantiate and move model to device."""
    return PhysicsOperator(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        n_fourier_features=cfg.n_fourier_features,
        fourier_scale=cfg.fourier_scale,
        use_fnet=cfg.use_fnet,
        dropout=cfg.dropout,
    ).to(device)


def _train_epoch(
    model: PhysicsOperator,
    dataset: PoissonDataset,
    optimizer: torch.optim.Optimizer,
    loss_fn: PhysicsLoss,
    device: torch.device,
    batch_size: int,
) -> float:
    """Train one epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        samples = [dataset[i] for i in batch_idx]

        coords = torch.tensor(
            np.stack([s.coords for s in samples]), dtype=torch.float32, device=device
        )
        charges = torch.tensor(
            np.stack([s.charges for s in samples]), dtype=torch.float32, device=device
        )
        targets = torch.tensor(
            np.stack([s.potential for s in samples]), dtype=torch.float32, device=device
        )

        optimizer.zero_grad()
        predictions = model(coords, charges)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _run_inference(
    model: PhysicsOperator,
    grid_size: int,
    cfg: DemoConfig,
    device: torch.device,
) -> SizeMetrics:
    """Run inference on a target grid size and compute metrics."""
    model.train(False)

    dataset = PoissonDataset(
        grid_size=grid_size,
        n_samples=cfg.n_inference_samples,
        n_charges=cfg.n_charges,
        seed=cfg.seed + cfg.inference_seed_offset,
    )

    sq_errors: list[float] = []
    abs_errors: list[float] = []

    indices = list(range(len(dataset)))
    for start in range(0, len(indices), cfg.batch_size):
        batch_idx = indices[start : start + cfg.batch_size]
        samples = [dataset[i] for i in batch_idx]

        coords = torch.tensor(
            np.stack([s.coords for s in samples]), dtype=torch.float32, device=device
        )
        charges = torch.tensor(
            np.stack([s.charges for s in samples]), dtype=torch.float32, device=device
        )
        targets = np.stack([s.potential for s in samples])

        preds = model(coords, charges).cpu().numpy()
        errors = preds - targets
        sq_errors.extend((errors**2).flatten().tolist())
        abs_errors.extend(np.abs(errors).flatten().tolist())

    mse = float(np.mean(sq_errors))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(mse))

    return SizeMetrics(
        grid_size=grid_size,
        mse=mse,
        mae=mae,
        rmse=rmse,
        n_points=grid_size**2,
        passed=mse < cfg.success_threshold,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_mse_curve(
    cfg: DemoConfig,
    metrics: list[SizeMetrics],
    output_path: Path,
) -> None:
    """Generate MSE vs resolution curve and save to output_path."""
    sizes = [m.grid_size for m in metrics]
    mses = [m.mse for m in metrics]
    colors = ["#2ecc71" if m.passed else "#e74c3c" for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sizes, mses, "k--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.scatter(sizes, mses, c=colors, s=100, zorder=2, edgecolors="black", linewidth=0.5)

    threshold_line = ax.axhline(
        cfg.success_threshold,
        color="#e74c3c",
        linestyle=":",
        linewidth=1.2,
        label=f"Threshold MSE={cfg.success_threshold}",
    )

    if max(mses) / (min(mses) + 1e-12) > 100:
        ax.set_yscale("log")

    ax.set_xlabel("Grid Size (N x N)", fontsize=12)
    ax.set_ylabel("Mean Squared Error", fontsize=12)
    sizes_label = ", ".join(str(s) + "x" + str(s) for s in sizes)
    ax.set_title(
        f"Zero-Shot Resolution Transfer\nTrained on {cfg.train_size}x{cfg.train_size}"
        f" -> Tested on {sizes_label}",
        fontsize=11,
    )
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}x{s}" for s in sizes])

    for m in metrics:
        ax.annotate(
            f"{m.mse:.2e}",
            (m.grid_size, m.mse),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Pass"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Fail"),
        threshold_line,
    ]
    ax.legend(handles=legend_handles, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _build_result(
    cfg: DemoConfig,
    metrics: list[SizeMetrics],
    model: PhysicsOperator,
    training_time: float,
) -> DemoResult:
    n_params = sum(p.numel() for p in model.parameters())
    return DemoResult(
        train_size=cfg.train_size,
        target_sizes=cfg.target_sizes,
        mse_per_size={str(m.grid_size): m.mse for m in metrics},
        per_size_details=[asdict(m) for m in metrics],
        model_params=n_params,
        training_config={
            "train_size": cfg.train_size,
            "n_epochs": cfg.n_epochs,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "n_train_samples": cfg.n_train_samples,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_fourier_features": cfg.n_fourier_features,
            "fourier_scale": cfg.fourier_scale,
            "use_fnet": cfg.use_fnet,
            "seed": cfg.seed,
        },
        training_time_seconds=training_time,
        success_threshold=cfg.success_threshold,
        all_passed=all(m.passed for m in metrics),
    )


def _save_json_report(result: DemoResult, path: Path) -> None:
    with open(path, "w") as fh:
        json.dump(asdict(result), fh, indent=2)


def _save_markdown_report(result: DemoResult, path: Path) -> None:
    header = [
        "# Zero-Shot Resolution Transfer Results",
        "",
        f"**Trained on:** {result.train_size}x{result.train_size}",
        f"**Model parameters:** {result.model_params:,}",
        f"**Training time:** {result.training_time_seconds:.1f} s",
        f"**Success threshold:** MSE < {result.success_threshold}",
        "",
        "## Per-Resolution MSE",
        "",
        "| Grid Size | MSE | RMSE | Points | Pass? |",
        "|-----------|-----|------|--------|-------|",
    ]
    rows = []
    for d in result.per_size_details:
        g = d["grid_size"]
        status = "Yes" if d["passed"] else "No"
        rows.append(f"| {g}x{g} | {d['mse']:.6f} | {d['rmse']:.6f} | {d['n_points']} | {status} |")
    footer = [
        "",
        "**All passed:** " + ("Yes" if result.all_passed else "No"),
    ]
    path.write_text("\n".join(header + rows + footer))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_demo(cfg: DemoConfig, no_plots: bool = False) -> DemoResult:
    """Execute training and multi-resolution inference pipeline."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nBuilding training dataset ({cfg.train_size}x{cfg.train_size}, "
        f"{cfg.n_train_samples} samples)..."
    )
    train_dataset = PoissonDataset(
        grid_size=cfg.train_size,
        n_samples=cfg.n_train_samples,
        n_charges=cfg.n_charges,
        seed=cfg.seed,
    )

    model = _build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters | device: {device}")

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    loss_fn = PhysicsLoss()

    print(f"\nTraining for {cfg.n_epochs} epoch(s)...")
    t0 = time.time()
    for epoch in range(cfg.n_epochs):
        loss_val = _train_epoch(model, train_dataset, optimizer, loss_fn, device, cfg.batch_size)
        scheduler.step()
        if (epoch + 1) % cfg.log_interval == 0 or epoch == 0:
            elapsed = time.time() - t0
            width = len(str(cfg.n_epochs))
            print(
                f"  Epoch {epoch + 1:>{width}}/{cfg.n_epochs} "
                f"| loss={loss_val:.6f} | {elapsed:.1f}s elapsed"
            )
    training_time = time.time() - t0
    print(f"Training complete in {training_time:.1f}s")

    print(f"\nRunning inference on {len(cfg.target_sizes)} grid sizes: {cfg.target_sizes}")
    metrics: list[SizeMetrics] = []
    for size in cfg.target_sizes:
        m = _run_inference(model, size, cfg, device)
        status_str = "PASS" if m.passed else "FAIL"
        print(f"  {size:>2}x{size:<2}  MSE={m.mse:.6f}  RMSE={m.rmse:.6f}  [{status_str}]")
        metrics.append(m)

    if not no_plots:
        plot_path = output_dir / "transfer_mse.png"
        _plot_mse_curve(cfg, metrics, plot_path)
        print(f"\nPlot saved -> {plot_path}")

    result = _build_result(cfg, metrics, model, training_time)

    json_path = output_dir / "transfer_results.json"
    _save_json_report(result, json_path)
    print(f"JSON report saved -> {json_path}")

    if cfg.report_format == "markdown":
        md_path = output_dir / "transfer_results.md"
        _save_markdown_report(result, md_path)
        print(f"Markdown report saved -> {md_path}")

    print("\n" + "=" * 60)
    print("ZERO-SHOT TRANSFER DEMO SUMMARY")
    print("=" * 60)
    print(f"Trained on : {cfg.train_size}x{cfg.train_size}")
    header_line = f"{'Grid':<10} {'MSE':>12}  Status"
    print(header_line)
    print("-" * 32)
    for m in metrics:
        label = "[PASS]" if m.passed else "[FAIL]"
        print(f"{m.grid_size}x{m.grid_size:<6} {m.mse:>12.6f}  {label}")
    print("=" * 60)
    overall = "ALL PASSED" if result.all_passed else "SOME FAILED"
    print(f"Result: {overall}")
    print(f"Training time: {training_time:.1f}s | Model params: {n_params:,}")

    return result


def _parse_args() -> tuple[DemoConfig, bool]:
    parser = argparse.ArgumentParser(description="AlphaGalerkin Zero-Shot Resolution Transfer Demo")
    parser.add_argument(
        "--train-size",
        type=int,
        default=9,
        help="Grid size used for training (default: 9)",
    )
    parser.add_argument(
        "--eval-sizes",
        type=str,
        default="9,13,19,25",
        help="Comma-separated target grid sizes (default: 9,13,19,25)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast mode: 5 epochs, smaller model, 500 training samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/demo_transfer",
        help="Output directory (default: outputs/demo_transfer)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib figure generation",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Report format (default: json)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model hidden dim (default: 128)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Galerkin layers (default: 4)",
    )
    parser.add_argument(
        "--n-fourier-features",
        type=int,
        default=64,
        help="Fourier features (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.05,
        help="MSE threshold for transfer success (default: 0.05)",
    )

    args = parser.parse_args()
    target_sizes = [int(s.strip()) for s in args.eval_sizes.split(",")]

    cfg = DemoConfig(
        train_size=args.train_size,
        target_sizes=target_sizes,
        n_epochs=args.n_epochs,
        output_dir=args.output_dir,
        report_format=args.format,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_fourier_features=args.n_fourier_features,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        success_threshold=args.success_threshold,
    )

    if args.quick:
        # Minimal settings for fast demo (<60 s on CPU)
        cfg.n_epochs = 5
        cfg.d_model = 64
        cfg.n_layers = 2
        cfg.n_fourier_features = 32
        cfg.n_train_samples = 500
        cfg.n_inference_samples = 100
        cfg.log_interval = 1

    return cfg, args.no_plots


def main() -> None:
    """CLI entry point."""
    cfg, no_plots = _parse_args()
    result = run_demo(cfg, no_plots=no_plots)
    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
