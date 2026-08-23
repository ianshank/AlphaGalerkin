"""Unit tests for reusable agent skills (BenchmarkSkill, SelfPlaySkill)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.agents.skills.benchmark_skill import BenchmarkConfig, BenchmarkSkill
from src.agents.skills.self_play_skill import SelfPlayConfig, SelfPlaySkill
from src.mcts.evaluator import EvaluationResult


class _MockSkillGame:
    """Mock game conforming to GameProtocol."""

    def __init__(self, step: int = 0, max_steps: int = 5) -> None:
        self.step = step
        self.max_steps = max_steps

    def get_state(self) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.zeros((1, 8, 8), dtype=np.float32)

    def get_legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [0, 1, 2]

    def apply_action(self, action: int) -> None:
        self.step += 1

    def is_terminal(self) -> bool:
        return self.step >= self.max_steps

    def get_winner(self) -> float:
        return 1.0

    def clone(self) -> _MockSkillGame:
        return _MockSkillGame(step=self.step, max_steps=self.max_steps)


class _MockSkillEvaluator:
    """Mock evaluator conforming to EvaluatorProtocol."""

    def evaluate(
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],
        legal_actions: list[int],
    ) -> EvaluationResult:
        policy = {a: 1.0 / len(legal_actions) for a in legal_actions}
        return EvaluationResult(policy=policy, value=0.5)

    def evaluate_batch(
        self,
        states: list[np.ndarray[Any, np.dtype[np.float32]]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=False)]


def test_benchmark_skill_execution() -> None:
    """Test BenchmarkSkill warmup, timing, and statistics calculation."""
    config = BenchmarkConfig(name="test_bench", warmup_rounds=2, timing_rounds=3)
    skill = BenchmarkSkill(config)

    def dummy_func(x: int) -> int:
        time.sleep(0.001)
        return x * 2

    result = skill.execute(dummy_func, 5, item_count=10)

    assert result.name == "test_bench"
    assert result.total_rounds == 3
    assert len(result.raw_durations_s) == 3
    assert result.mean_duration_s > 0.0
    assert result.median_duration_s > 0.0
    assert result.min_duration_s <= result.max_duration_s
    assert result.throughput_items_per_s > 0.0


def test_self_play_skill_generation() -> None:
    """Test SelfPlaySkill rollout and batch generation."""
    config = SelfPlayConfig(n_games=2, max_moves_per_game=5)
    skill = SelfPlaySkill(config)

    def game_factory() -> _MockSkillGame:
        return _MockSkillGame(max_steps=3)

    def eval_factory() -> _MockSkillEvaluator:
        return _MockSkillEvaluator()

    result = skill.generate(game_factory, eval_factory)

    assert result.total_games == 2
    assert result.total_positions == 6  # 2 games * 3 steps
    assert len(result.outcomes) == 2
    assert result.outcomes == [1.0, 1.0]
    assert result.generation_duration_s >= 0.0
    assert result.positions_per_second > 0.0
