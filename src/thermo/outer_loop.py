"""Plan-in-surrogate / act-in-world comparison for λ-window scheduling.

Three schedulers allocate the same sample budget across λ-windows:

* **greedy** — myopically add the next batch to the window with the largest
  variance contribution ``c_i**2 / n_i`` (fixed initial windows, no splits).
* **uniform** — round-robin allocation.
* **mcts** — plan splits + allocations with single-agent MCTS.

All three *plan* against the (possibly biased) ``planner`` surrogate; the final
allocation is scored on the ``truth`` (analytic) surrogate. The honest question is
whether MCTS's lookahead survives ``planner != truth``.

Aggregation is the **median** ΔG-stderr ratio over seeds (a single MCTS run is
high-variance), mirroring the ``lshape_amr_compare`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from src.thermo.game import LambdaSchedulingGame, total_stderr

if TYPE_CHECKING:
    from src.refinement.state import RefinementState
    from src.thermo.config import SchedulingParams
    from src.thermo.surrogate import AnalyticSurrogate, VarianceSurrogate

# Seeds derived from a base seed via a large prime stride for reproducibility.
SEED_PRIME_STRIDE = 7919
# Floor on the greedy stderr when forming a ratio, guarding a degenerate zero.
RATIO_FLOOR = 1e-12


def run_greedy(game: LambdaSchedulingGame) -> RefinementState:
    """Allocate each batch to the highest variance-contribution window."""
    state = game.get_initial_state()
    maxw = game.params.max_windows
    while not game.is_terminal(state):
        valid = [a for a in game.get_valid_actions(state) if a < maxw]
        if not valid:
            break
        contrib = state.indicators
        best = max(valid, key=lambda a: float(contrib[a]))
        state = game.apply_action(state, best)
    return state


def run_uniform(game: LambdaSchedulingGame) -> RefinementState:
    """Round-robin allocation across the (fixed) initial windows."""
    state = game.get_initial_state()
    maxw = game.params.max_windows
    rr = 0
    while not game.is_terminal(state):
        valid = [a for a in game.get_valid_actions(state) if a < maxw]
        if not valid:
            break
        state = game.apply_action(state, valid[rr % len(valid)])
        rr += 1
    return state


def run_mcts(
    game: LambdaSchedulingGame,
    seed: int,
    n_simulations: int,
    c_puct: float,
) -> RefinementState:
    """Plan splits + allocations with single-agent MCTS + intermediate rewards."""
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
    while not adapter.is_terminal() and adapter.get_legal_actions():
        action = mcts.get_action(adapter, temperature=0.0, add_noise=True)
        adapter.apply_action(action)
        mcts.advance(action)
    return adapter.state


@dataclass
class ComparisonResult:
    """True (analytic-scored) ΔG stderr per policy plus MCTS/greedy ratios."""

    seed: int
    bias: float
    mcts_stderr: float
    greedy_stderr: float
    uniform_stderr: float

    @property
    def ratio_mcts_over_greedy(self) -> float:
        return self.mcts_stderr / max(self.greedy_stderr, RATIO_FLOOR)

    @property
    def ratio_mcts_over_uniform(self) -> float:
        return self.mcts_stderr / max(self.uniform_stderr, RATIO_FLOOR)


def run_comparison(
    truth: AnalyticSurrogate,
    planner: VarianceSurrogate,
    params: SchedulingParams,
    seed: int,
    bias: float,
    n_simulations: int,
    c_puct: float,
) -> ComparisonResult:
    """Run all three schedulers against ``planner``; score on ``truth``."""
    game = LambdaSchedulingGame(params, planner)

    greedy_state = run_greedy(game)
    uniform_state = run_uniform(game)
    mcts_state = run_mcts(game, seed, n_simulations, c_puct)

    def scored(state: RefinementState) -> float:
        return total_stderr(state.values.astype(np.float64).reshape(-1, 3), truth)

    return ComparisonResult(
        seed=seed,
        bias=bias,
        mcts_stderr=scored(mcts_state),
        greedy_stderr=scored(greedy_state),
        uniform_stderr=scored(uniform_state),
    )


def resolved_seeds(base_seed: int, n_seeds: int) -> list[int]:
    """Deterministic seed list derived from ``base_seed``."""
    return [base_seed + i * SEED_PRIME_STRIDE for i in range(n_seeds)]


@dataclass
class BiasCell:
    """Aggregated result for one surrogate-bias level over ``n_seeds`` seeds."""

    bias: float
    per_seed: list[ComparisonResult]

    @property
    def ratios(self) -> list[float]:
        return [r.ratio_mcts_over_greedy for r in self.per_seed]

    @property
    def median_ratio(self) -> float:
        return float(np.median(self.ratios))

    @property
    def win_fraction(self) -> float:
        return float(np.mean(np.array(self.ratios) < 1.0))

    def metrics(self) -> dict[str, float]:
        ratios = np.array(self.ratios, dtype=np.float64)
        return {
            "median_ratio": float(np.median(ratios)),
            "win_fraction": float(np.mean(ratios < 1.0)),
            "ratio_min": float(np.min(ratios)),
            "ratio_max": float(np.max(ratios)),
            "ratio_std": float(np.std(ratios)),
        }


def run_bias_sweep(
    truth: AnalyticSurrogate,
    make_planner: object,
    params: SchedulingParams,
    biases: list[float],
    base_seed: int,
    n_seeds: int,
    n_simulations: int,
    c_puct: float,
) -> list[BiasCell]:
    """Sweep the bias levels, aggregating the median MCTS/greedy ratio per bias.

    ``make_planner(bias)`` returns the planner surrogate for a bias level (so the
    caller controls how mismatch is injected). At ``bias == 0`` the planner is the
    truth itself.
    """
    seeds = resolved_seeds(base_seed, n_seeds)
    cells: list[BiasCell] = []
    for bias in biases:
        planner = make_planner(bias)  # type: ignore[operator]
        per_seed = [
            run_comparison(truth, planner, params, seed, bias, n_simulations, c_puct)
            for seed in seeds
        ]
        cells.append(BiasCell(bias=bias, per_seed=per_seed))
    return cells


def replace_params(params: SchedulingParams, **changes: object) -> SchedulingParams:
    """Typed wrapper over dataclasses.replace for the frozen params."""
    return replace(params, **changes)  # type: ignore[arg-type]
