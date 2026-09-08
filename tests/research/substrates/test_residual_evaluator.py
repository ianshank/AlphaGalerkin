"""Tests for the arena headline leaf evaluator.

``ResidualPriorErrorValueEvaluator`` must read the trailing value slot, never
``state[0]``. ``EncodedValueEvaluator`` and ``RandomEvaluator`` are forbidden as
the published MCTS-vs-classical arm.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.research.substrates.residual_evaluator import ResidualPriorErrorValueEvaluator


def _evaluator(n_actions: int = 4) -> ResidualPriorErrorValueEvaluator:
    return ResidualPriorErrorValueEvaluator(n_actions=n_actions)


class TestConstruction:
    def test_rejects_non_positive_action_count(self) -> None:
        with pytest.raises(ValueError, match="n_actions"):
            ResidualPriorErrorValueEvaluator(n_actions=0)

    def test_rejects_non_positive_temperature(self) -> None:
        with pytest.raises(ValueError, match="prior_temperature"):
            ResidualPriorErrorValueEvaluator(n_actions=2, prior_temperature=0.0)


class TestEvaluate:
    def test_value_is_last_slot(self) -> None:
        tensor = np.array([0.1, 0.9, 0.2, 0.05, 0.33], dtype=np.float32)
        result = _evaluator(n_actions=4).evaluate(tensor, [0, 1, 2, 3])
        assert result.value == pytest.approx(0.33, abs=1e-6)

    def test_prior_concentrates_on_largest_legal_residual(self) -> None:
        tensor = np.array([0.1, 5.0, 0.2, 0.0, 0.0], dtype=np.float32)
        result = _evaluator(n_actions=4).evaluate(tensor, [0, 1, 2, 3])
        assert int(np.argmax(result.policy)) == 1
        assert result.policy[1] > result.policy[0]

    def test_uniform_fallback_when_residuals_are_zero(self) -> None:
        tensor = np.zeros(5, dtype=np.float32)
        result = _evaluator(n_actions=4).evaluate(tensor, [0, 2])
        assert result.policy[0] == pytest.approx(0.5)
        assert result.policy[2] == pytest.approx(0.5)
        assert result.policy[1] == 0.0

    def test_empty_legal_set_returns_zero_policy(self) -> None:
        tensor = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        result = _evaluator(n_actions=2).evaluate(tensor, [])
        assert result.policy.tolist() == [0.0, 0.0]
        assert result.value == pytest.approx(0.5)

    def test_non_finite_value_falls_back_to_zero(self) -> None:
        tensor = np.array([1.0, np.nan], dtype=np.float32)
        result = _evaluator(n_actions=1).evaluate(tensor, [0])
        assert result.value == 0.0

    def test_evaluate_batch_matches_per_item(self) -> None:
        evaluator = _evaluator(n_actions=2)
        states = [
            np.array([1.0, 0.0, 0.2], dtype=np.float32),
            np.array([0.0, 3.0, -0.4], dtype=np.float32),
        ]
        legal = [[0, 1], [1]]
        batched = evaluator.evaluate_batch(states, legal)
        singles = [evaluator.evaluate(s, la) for s, la in zip(states, legal, strict=True)]
        assert len(batched) == 2
        assert batched[0].value == pytest.approx(singles[0].value)
        assert batched[1].policy[1] == pytest.approx(singles[1].policy[1])
