#!/usr/bin/env python3
"""PDE Solver Demo -- AlphaGalerkin MCTS-Guided Galerkin Basis Selection.

Runs the PDETrainer for N episodes, tracks error reduction per episode,
and outputs a convergence plot + JSON metrics file.

Usage::

    python scripts/demo_pde_solver.py --pde-type poisson --n-episodes 10
    python scripts/demo_pde_solver.py --pde-type burgers --n-episodes 20 --mcts-sims 20
    python scripts/demo_pde_solver.py --pde-type poisson --n-episodes 5 --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when the script is run directly
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.pde.trainer import PDETrainer, PDETrainingConfig, PDETrainingResult  # noqa: E402

# ---------------------------------------------------------------------------
# Supported PDE types (mirrors SUPPORTED_PDE_TYPES in trainer.py)
# ---------------------------------------------------------------------------

_VALID_PDE_TYPES: tuple[str, ...] = ("poisson", "burgers", "advection_diffusion")

# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------


@dataclass
class DemoConfig:
    """All demo parameters -- no hardcoded values appear in logic below."""

    pde_type: str = "poisson"
    n_episodes: int = 10
    mcts_sims: int = 10
    output_dir: str = "outputs/pde_demo"
    no_plots: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Metrics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EpisodeSummary:
    """Per-episode summary for reporting."""

    episode: int
    initial_error: float
    final_error: float
    n_steps: int
    converged: bool
    reduction_ratio: float  # initial_error / final_error (>1 means improvement)


@dataclass
class DemoMetrics:
    """Full set of metrics produced by the demo."""

    pde_type: str
    n_episodes: int
    mcts_simulations: int
    seed: int
    initial_error: float  # error at episode 0, step 0
    final_error: float  # best final error across all episodes
    improvement_ratio: float  # initial_error / best_final_error
    per_episode_errors: list[float]
    episode_summaries: list[dict[str, float | int | bool]]
    n_converged: int
    total_time_seconds: float


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_convergence(
    cfg: DemoConfig,
    per_episode_errors: list[float],
    initial_error: float,
    output_path: Path,
) -> None:
    """Save a convergence plot: error vs episode number."""
    episodes = list(range(len(per_episode_errors)))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline: constant initial error
    ax.axhline(
        initial_error,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.2,
        label=f"Initial error = {initial_error:.4f}",
    )

    # Per-episode final error
    ax.plot(
        episodes,
        per_episode_errors,
        marker="o",
        color="#2980b9",
        linewidth=1.5,
        markersize=6,
        label="Episode final error",
    )

    # Shade improvement area
    ax.fill_between(
        episodes,
        per_episode_errors,
        [initial_error] * len(episodes),
        where=[e < initial_error for e in per_episode_errors],
        alpha=0.15,
        color="#2ecc71",
        label="Improvement region",
    )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Final Error (L2 residual)", fontsize=12)
    ax.set_title(
        f"MCTS-Guided Galerkin Basis Selection -- {cfg.pde_type.upper()}\n"
        f"{cfg.n_episodes} episodes, {cfg.mcts_sims} MCTS simulations each",
        fontsize=11,
    )
    ax.set_xticks(episodes)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Log scale if range is large
    if max(per_episode_errors) / (min(per_episode_errors) + 1e-12) > 50:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def _extract_metrics(
    cfg: DemoConfig,
    result: PDETrainingResult,
    elapsed: float,
) -> DemoMetrics:
    """Compute the full DemoMetrics from a PDETrainingResult."""
    episodes = result.episodes

    # Initial error: first step of first episode
    initial_error: float
    if episodes and episodes[0].error_history:
        initial_error = episodes[0].error_history[0]
    elif episodes:
        initial_error = episodes[0].initial_error
    else:
        initial_error = float("inf")

    per_episode_errors: list[float] = result.errors  # final error per episode
    best_final = result.best_final_error

    improvement_ratio = initial_error / best_final if best_final > 0 else float("inf")

    summaries: list[EpisodeSummary] = []
    for ep in episodes:
        ep_initial = ep.error_history[0] if ep.error_history else ep.initial_error
        ep_final = ep.final_error
        reduction = ep_initial / ep_final if ep_final > 0 else float("inf")
        summaries.append(
            EpisodeSummary(
                episode=ep.episode_idx,
                initial_error=ep_initial,
                final_error=ep_final,
                n_steps=ep.n_steps,
                converged=ep.converged,
                reduction_ratio=reduction,
            )
        )

    n_converged = sum(1 for ep in episodes if ep.converged)

    return DemoMetrics(
        pde_type=cfg.pde_type,
        n_episodes=cfg.n_episodes,
        mcts_simulations=cfg.mcts_sims,
        seed=cfg.seed,
        initial_error=initial_error,
        final_error=best_final,
        improvement_ratio=improvement_ratio,
        per_episode_errors=per_episode_errors,
        episode_summaries=[asdict(s) for s in summaries],
        n_converged=n_converged,
        total_time_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(metrics: DemoMetrics) -> None:
    """Print a human-readable summary table."""
    col_w = 10

    print()
    print("=" * 62)
    print(f"PDE SOLVER DEMO -- {metrics.pde_type.upper()}")
    print("=" * 62)
    print(
        f"{'Episode':>{col_w}}  {'Final Error':>14}  {'Reduction':>12}  {'Steps':>6}  {'Conv?':>5}"
    )
    print("-" * 62)

    for s in metrics.episode_summaries:
        conv_str = "YES" if s["converged"] else "no"
        ratio_str = f"{s['reduction_ratio']:.3f}x"
        print(
            f"{s['episode']:>{col_w}}  "
            f"{s['final_error']:>14.6f}  "
            f"{ratio_str:>12}  "
            f"{s['n_steps']:>6}  "
            f"{conv_str:>5}"
        )

    print("=" * 62)
    print(f"Initial error      : {metrics.initial_error:.6f}")
    print(f"Best final error   : {metrics.final_error:.6f}")
    print(f"Overall improvement: {metrics.improvement_ratio:.3f}x")
    print(f"Episodes converged : {metrics.n_converged}/{metrics.n_episodes}")
    print(f"Elapsed time       : {metrics.total_time_seconds:.2f}s")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Main demo runner
# ---------------------------------------------------------------------------


def run_demo(cfg: DemoConfig) -> DemoMetrics:
    """Execute the PDE solver demo end-to-end.

    Args:
        cfg: All demo parameters.

    Returns:
        ``DemoMetrics`` with aggregated results.

    """
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\nRunning PDE Solver Demo: pde_type={cfg.pde_type}, "
        f"n_episodes={cfg.n_episodes}, mcts_sims={cfg.mcts_sims}, seed={cfg.seed}"
    )
    print(f"Output directory: {output_dir}")

    training_config = PDETrainingConfig(
        name="pde_demo",
        pde_type=cfg.pde_type,
        n_episodes=cfg.n_episodes,
        mcts_simulations=cfg.mcts_sims,
        seed=cfg.seed,
    )

    trainer = PDETrainer(training_config)
    print(f"\nRunning {cfg.n_episodes} episode(s)...")
    t0 = time.time()
    result: PDETrainingResult = trainer.run()
    elapsed = time.time() - t0

    metrics = _extract_metrics(cfg, result, elapsed)

    # Save JSON metrics
    json_path = output_dir / "pde_results.json"
    with open(json_path, "w") as fh:
        json.dump(asdict(metrics), fh, indent=2)
    print(f"JSON metrics saved -> {json_path}")

    # Save convergence plot
    if not cfg.no_plots:
        plot_path = output_dir / "convergence.png"
        _plot_convergence(cfg, metrics.per_episode_errors, metrics.initial_error, plot_path)
        print(f"Convergence plot saved -> {plot_path}")

    _print_summary(metrics)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(
        description="AlphaGalerkin PDE Solver Demo: MCTS-guided Galerkin basis selection"
    )
    parser.add_argument(
        "--pde-type",
        choices=list(_VALID_PDE_TYPES),
        default="poisson",
        help="PDE equation type (default: poisson)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of training episodes (default: 10)",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=10,
        help="MCTS simulations per search step (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pde_demo",
        help="Directory for outputs (default: outputs/pde_demo)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib convergence plot generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()
    return DemoConfig(
        pde_type=args.pde_type,
        n_episodes=args.n_episodes,
        mcts_sims=args.mcts_sims,
        output_dir=args.output_dir,
        no_plots=args.no_plots,
        seed=args.seed,
    )


def main() -> None:
    """CLI entry point."""
    cfg = _parse_args()
    run_demo(cfg)
    sys.exit(0)


if __name__ == "__main__":
    main()
