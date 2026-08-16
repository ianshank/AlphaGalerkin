"""Performance benchmarks for MCTS."""

from typing import Any

import numpy as np
import pytest

from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS, GameInterface, SearchMode
from tests.benchmarks.conftest import BenchmarkTimerProtocol


class MockGame(GameInterface):
    """A lightweight mock game for MCTS benchmarking."""

    def __init__(self, step: int = 0, max_steps: int = 10) -> None:
        self.step = step
        self.max_steps = max_steps
        self.n_actions = 4

    def get_state(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.zeros((1, 8, 8), dtype=np.float32)

    def get_legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return list(range(self.n_actions))

    def apply_action(self, action: int) -> None:
        self.step += 1

    def is_terminal(self) -> bool:
        return self.step >= self.max_steps

    def get_winner(self) -> int:
        return 0

    def clone(self) -> GameInterface:
        return MockGame(step=self.step, max_steps=self.max_steps)


@pytest.mark.benchmark
def test_mcts_throughput(benchmark_timer: BenchmarkTimerProtocol) -> None:
    """Benchmark MCTS search throughput across simulation budgets."""
    simulation_budgets = [100, 400, 800]
    n_actions = 4
    evaluator = RandomEvaluator(n_actions=n_actions)

    throughputs = {}
    times = {}

    for budget in simulation_budgets:
        mcts = MCTS(evaluator=evaluator, n_simulations=budget, search_mode=SearchMode.SINGLE_AGENT)

        # We benchmark a single full MCTS search for a root state
        def run_search() -> None:
            game = MockGame()
            mcts.search(game)

        stats = benchmark_timer(run_search, warmup_rounds=2, timing_rounds=5)
        times[budget] = stats.mean_time_s
        # Throughput here is simulations per second
        # stats.mean_time_s is time for `budget` simulations
        if stats.mean_time_s > 0:
            throughputs[budget] = budget / stats.mean_time_s
        else:
            throughputs[budget] = 0.0

    # Verify throughput is strictly positive and finite (>50 sims/sec on CPU)
    for budget in simulation_budgets:
        assert throughputs[budget] > 50.0, f"Throughput {throughputs[budget]:.1f} too low for budget {budget}"
        assert times[budget] > 0.0
        assert np.isfinite(throughputs[budget])
