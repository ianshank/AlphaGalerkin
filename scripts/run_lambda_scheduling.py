"""Run the λ-window scheduling ablation and write the committed artifacts.

Produces ``results/lambda_scheduling.{png,csv}``: ΔG standard error vs sample
budget for greedy / uniform / MCTS at ``surrogate_bias ∈ {0, 0.25}`` (MCTS median
over seeds with a min/max band).

**This is a negative result.** MCTS does not beat greedy — it over-splits and
fragments the sample budget. See ``specs/lambda_scheduling.spec.md``.

    python -m scripts.run_lambda_scheduling --output-dir results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.thermo.config import LambdaSchedulingConfig
from src.thermo.game import LambdaSchedulingGame, total_stderr
from src.thermo.outer_loop import resolved_seeds
from src.thermo.surrogate import AnalyticSurrogate, MismatchedSurrogate, VarianceSurrogate

if TYPE_CHECKING:
    from src.refinement.state import RefinementState


def _true_stderr(state: RefinementState, truth: AnalyticSurrogate) -> float:
    return total_stderr(state.values.astype(np.float64).reshape(-1, 3), truth)


def _greedy_trajectory(
    game: LambdaSchedulingGame, truth: AnalyticSurrogate
) -> list[tuple[int, float]]:
    state = game.get_initial_state()
    maxw = game.params.max_windows
    traj = [(state.dof, _true_stderr(state, truth))]
    while not game.is_terminal(state):
        valid = [a for a in game.get_valid_actions(state) if a < maxw]
        if not valid:
            break
        best = max(valid, key=lambda a: float(state.indicators[a]))
        state = game.apply_action(state, best)
        traj.append((state.dof, _true_stderr(state, truth)))
    return traj


def _uniform_trajectory(
    game: LambdaSchedulingGame, truth: AnalyticSurrogate
) -> list[tuple[int, float]]:
    state = game.get_initial_state()
    maxw = game.params.max_windows
    traj = [(state.dof, _true_stderr(state, truth))]
    rr = 0
    while not game.is_terminal(state):
        valid = [a for a in game.get_valid_actions(state) if a < maxw]
        if not valid:
            break
        state = game.apply_action(state, valid[rr % len(valid)])
        rr += 1
        traj.append((state.dof, _true_stderr(state, truth)))
    return traj


def _mcts_trajectory(
    game: LambdaSchedulingGame,
    truth: AnalyticSurrogate,
    seed: int,
    n_simulations: int,
    c_puct: float,
) -> list[tuple[int, float]]:
    import torch

    from src.mcts.evaluator import RandomEvaluator
    from src.mcts.search import MCTS
    from src.refinement.adapter import RefinementGameAdapter

    np.random.seed(seed)
    torch.manual_seed(seed)
    adapter = RefinementGameAdapter(game)
    mcts = MCTS(
        evaluator=RandomEvaluator(n_actions=game.action_space_size),
        n_simulations=n_simulations,
        c_puct=c_puct,
        search_mode=adapter.search_mode,
        use_intermediate_rewards=True,
    )
    traj = [(adapter.state.dof, _true_stderr(adapter.state, truth))]
    while not adapter.is_terminal() and adapter.get_legal_actions():
        action = mcts.get_action(adapter, temperature=0.0, add_noise=True)
        adapter.apply_action(action)
        mcts.advance(action)
        traj.append((adapter.state.dof, _true_stderr(adapter.state, truth)))
    return traj


def _final_stderr_by_budget(traj: list[tuple[int, float]], grid: np.ndarray) -> np.ndarray:
    """Step-interpolate a trajectory onto a common sample-budget grid."""
    dofs = np.array([d for d, _ in traj], dtype=np.float64)
    errs = np.array([e for _, e in traj], dtype=np.float64)
    out = np.empty_like(grid, dtype=np.float64)
    for i, b in enumerate(grid):
        mask = dofs <= b
        out[i] = errs[mask][-1] if mask.any() else errs[0]
    return out


def run(config: LambdaSchedulingConfig, output_dir: Path) -> dict[str, float]:
    """Run the ablation and write CSV + PNG. Returns the headline metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    truth = AnalyticSurrogate(config.hardness)
    params = config.to_params()
    seeds = resolved_seeds(config.seed, config.n_seeds)
    grid = np.linspace(params.n_initial_windows * params.batch_samples, params.sample_budget, 20)

    biases = [0.0, config.primary_bias]
    rows: list[dict[str, object]] = []
    headline: dict[str, float] = {}

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(biases), figsize=(11, 4.5), sharey=True)
    if len(biases) == 1:
        axes = [axes]

    for ax, bias in zip(axes, biases, strict=False):
        planner: VarianceSurrogate
        if bias == 0.0:
            planner = truth
        else:
            planner = MismatchedSurrogate(truth, bias=bias)
        game = LambdaSchedulingGame(params, planner)
        greedy = _final_stderr_by_budget(_greedy_trajectory(game, truth), grid)
        uniform = _final_stderr_by_budget(_uniform_trajectory(game, truth), grid)
        mcts_curves = np.array(
            [
                _final_stderr_by_budget(
                    _mcts_trajectory(game, truth, s, config.n_simulations, config.c_puct),
                    grid,
                )
                for s in seeds
            ]
        )
        mcts_med = np.median(mcts_curves, axis=0)
        mcts_lo = np.min(mcts_curves, axis=0)
        mcts_hi = np.max(mcts_curves, axis=0)

        ax.plot(grid, greedy, "-o", label="greedy", color="#2166ac", markersize=3)
        ax.plot(grid, uniform, "-s", label="uniform", color="#4d9221", markersize=3)
        ax.plot(grid, mcts_med, "-^", label="mcts (median)", color="#b2182b", markersize=3)
        ax.fill_between(grid, mcts_lo, mcts_hi, color="#b2182b", alpha=0.15)
        ax.set_title(f"surrogate_bias = {bias:g}")
        ax.set_xlabel("sample budget")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ratio = float(mcts_med[-1] / max(greedy[-1], 1e-12))
        headline[f"final_ratio_mcts_over_greedy_bias_{bias:g}"] = ratio
        for i, b in enumerate(grid):
            rows.append(
                {
                    "bias": bias,
                    "sample_budget": float(b),
                    "greedy_stderr": float(greedy[i]),
                    "uniform_stderr": float(uniform[i]),
                    "mcts_median_stderr": float(mcts_med[i]),
                    "mcts_min_stderr": float(mcts_lo[i]),
                    "mcts_max_stderr": float(mcts_hi[i]),
                }
            )

    axes[0].set_ylabel("ΔG standard error (analytic-scored)")
    fig.suptitle("λ-window scheduling: MCTS vs greedy vs uniform (NEGATIVE result)")
    fig.tight_layout()
    png = output_dir / "lambda_scheduling.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)

    csv_path = output_dir / "lambda_scheduling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return headline


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--sample-budget", type=int, default=None)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--n-simulations", type=int, default=None)
    # 0.05 is the physical kcal/mol floor; with the default profile the achievable
    # stderr stays above it, so the sample budget (not the tolerance) binds.
    parser.add_argument("--error-tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {"error_tolerance": args.error_tolerance}
    if args.sample_budget is not None:
        overrides["sample_budget"] = args.sample_budget
    if args.n_seeds is not None:
        overrides["n_seeds"] = args.n_seeds
    if args.n_simulations is not None:
        overrides["n_simulations"] = args.n_simulations
    config = LambdaSchedulingConfig(name="lambda_scheduling", **overrides)  # type: ignore[arg-type]

    headline = run(config, Path(args.output_dir))
    for k, v in headline.items():
        print(f"{k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
