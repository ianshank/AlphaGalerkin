"""Plan-in-surrogate / act-in-world comparison for λ-window scheduling.

Three schedulers allocate the same sample budget across λ-windows:

* **greedy** — myopically add the next batch to the window with the largest
  variance contribution ``c_i**2 / n_i`` (fixed initial windows, no splits).
* **uniform** — round-robin allocation.
* **mcts** — plan splits + allocations with single-agent MCTS.

All three *plan* against the (possibly biased) ``planner`` surrogate; the final
allocation is scored on the ``truth`` (analytic) surrogate. The honest question is
whether MCTS's lookahead survives ``planner != truth``.

Each scheduler is exposed both as an ``iterate_*`` generator (yielding every
intermediate state — used by the plot harness to trace error-vs-budget) and as a
``run_*`` wrapper returning the terminal state, so the trajectory logic lives in
exactly one place.

Aggregation is the **median** ΔG-stderr ratio over seeds (a single MCTS run is
high-variance), mirroring the ``lshape_amr_compare`` convention.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.thermo.game import LambdaSchedulingGame, total_stderr

if TYPE_CHECKING:
    from src.refinement.state import RefinementState
    from src.thermo.config import SchedulingParams
    from src.thermo.surrogate import AnalyticSurrogate, VarianceSurrogate

logger = structlog.get_logger(__name__)

# Seeds derived from a base seed via a large prime stride for reproducibility.
SEED_PRIME_STRIDE = 7919
# Floor on the greedy stderr when forming a ratio, guarding a degenerate zero.
RATIO_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# Schedulers (iterate_* yields every state; run_* returns the terminal state)  #
# --------------------------------------------------------------------------- #


def _allocate_actions(game: LambdaSchedulingGame, state: RefinementState) -> list[int]:
    """Legal *allocate* action indices (splits excluded) in ``state``."""
    maxw = game.params.max_windows
    return [a for a in game.get_valid_actions(state) if a < maxw]


def iterate_greedy(game: LambdaSchedulingGame) -> Iterator[RefinementState]:
    """Yield each state as greedy allocates to the highest-variance window."""
    state = game.get_initial_state()
    yield state
    while not game.is_terminal(state):
        valid = _allocate_actions(game, state)
        if not valid:
            break
        best = max(valid, key=lambda a: float(state.indicators[a]))
        state = game.apply_action(state, best)
        logger.debug("greedy_allocate", window=best, error=state.error_estimate, dof=state.dof)
        yield state


def iterate_uniform(game: LambdaSchedulingGame) -> Iterator[RefinementState]:
    """Yield each state under round-robin allocation over the initial windows."""
    state = game.get_initial_state()
    yield state
    rr = 0
    while not game.is_terminal(state):
        valid = _allocate_actions(game, state)
        if not valid:
            break
        window = valid[rr % len(valid)]
        state = game.apply_action(state, window)
        rr += 1
        logger.debug("uniform_allocate", window=window, error=state.error_estimate, dof=state.dof)
        yield state


def iterate_mcts(
    game: LambdaSchedulingGame,
    seed: int,
    n_simulations: int,
    c_puct: float,
) -> Iterator[RefinementState]:
    """Yield each state as single-agent MCTS plans splits + allocations."""
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
        # The ablation's premise is shaped variance-reduction rewards, so this is
        # on by design; the discount is the tunable (SchedulingParams.reward_discount).
        use_intermediate_rewards=True,
        reward_discount=game.params.reward_discount,
    )
    yield adapter.state
    while not adapter.is_terminal() and adapter.get_legal_actions():
        action = mcts.get_action(adapter, temperature=0.0, add_noise=True)
        adapter.apply_action(action)
        mcts.advance(action)
        yield adapter.state


def _last(states: Iterator[RefinementState]) -> RefinementState:
    """Consume an iterator and return its final element (non-empty by design)."""
    final: RefinementState | None = None
    for state in states:
        final = state
    if final is None:  # unreachable: the iterate_* generators always yield ≥1 state
        raise ValueError("scheduler produced no states")
    return final


def run_greedy(game: LambdaSchedulingGame) -> RefinementState:
    """Terminal state of the greedy scheduler."""
    return _last(iterate_greedy(game))


def run_uniform(game: LambdaSchedulingGame) -> RefinementState:
    """Terminal state of the uniform scheduler."""
    return _last(iterate_uniform(game))


def run_mcts(
    game: LambdaSchedulingGame,
    seed: int,
    n_simulations: int,
    c_puct: float,
) -> RefinementState:
    """Terminal state of the single-agent MCTS scheduler."""
    return _last(iterate_mcts(game, seed, n_simulations, c_puct))


# --------------------------------------------------------------------------- #
# Comparison                                                                  #
# --------------------------------------------------------------------------- #


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


def score_true_stderr(state: RefinementState, truth: AnalyticSurrogate) -> float:
    """ΔG standard error of ``state``'s allocation scored on the truth surrogate."""
    return total_stderr(state.values.astype(np.float64).reshape(-1, 3), truth)


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

    result = ComparisonResult(
        seed=seed,
        bias=bias,
        mcts_stderr=score_true_stderr(run_mcts(game, seed, n_simulations, c_puct), truth),
        greedy_stderr=score_true_stderr(run_greedy(game), truth),
        uniform_stderr=score_true_stderr(run_uniform(game), truth),
    )
    logger.debug(
        "lambda_comparison_done",
        seed=seed,
        bias=bias,
        mcts_stderr=result.mcts_stderr,
        greedy_stderr=result.greedy_stderr,
        uniform_stderr=result.uniform_stderr,
        ratio_mcts_over_greedy=result.ratio_mcts_over_greedy,
    )
    return result


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
    make_planner: Callable[[float], VarianceSurrogate],
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
        planner = make_planner(bias)
        per_seed = [
            run_comparison(truth, planner, params, seed, bias, n_simulations, c_puct)
            for seed in seeds
        ]
        cell = BiasCell(bias=bias, per_seed=per_seed)
        logger.info(
            "lambda_bias_cell_done",
            bias=bias,
            n_seeds=n_seeds,
            median_ratio=cell.median_ratio,
            win_fraction=cell.win_fraction,
        )
        cells.append(cell)
    return cells
