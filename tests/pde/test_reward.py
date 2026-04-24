"""Tests for ``src.pde.reward.log_reward``."""

from __future__ import annotations

import math

import pytest

from src.pde.reward import log_reward


class TestLogReward:
    def test_matches_formula(self) -> None:
        """For plain inputs the helper matches -alpha*log(e) - beta*log(c)."""
        value = log_reward(
            error=0.1,
            cost=10.0,
            alpha=1.0,
            beta=0.5,
            epsilon=1e-12,
        )
        expected = -1.0 * math.log(0.1) - 0.5 * math.log(10.0)
        assert value == pytest.approx(expected, rel=1e-9)

    def test_monotone_in_error(self) -> None:
        """Lower error yields a higher reward with fixed cost."""
        high = log_reward(error=0.5, cost=10.0, alpha=1.0, beta=0.5, epsilon=1e-12)
        low = log_reward(error=0.01, cost=10.0, alpha=1.0, beta=0.5, epsilon=1e-12)
        assert low > high

    def test_monotone_in_cost(self) -> None:
        """Lower cost yields a higher reward with fixed error."""
        cheap = log_reward(error=0.1, cost=5.0, alpha=1.0, beta=0.5, epsilon=1e-12)
        expensive = log_reward(error=0.1, cost=500.0, alpha=1.0, beta=0.5, epsilon=1e-12)
        assert cheap > expensive

    def test_epsilon_floor_prevents_negative_infinity(self) -> None:
        """Zero error/cost is clamped to epsilon so the result stays finite."""
        value = log_reward(
            error=0.0, cost=0.0, alpha=1.0, beta=0.5, epsilon=1e-9
        )
        assert math.isfinite(value)

    def test_invalid_epsilon_rejected(self) -> None:
        """Non-positive epsilon is rejected explicitly."""
        with pytest.raises(ValueError):
            log_reward(error=0.1, cost=1.0, alpha=1.0, beta=0.5, epsilon=0.0)
        with pytest.raises(ValueError):
            log_reward(error=0.1, cost=1.0, alpha=1.0, beta=0.5, epsilon=-1.0)

    def test_zero_coefficients_drop_terms(self) -> None:
        """alpha=0 zeroes out the error term; beta=0 zeroes out the cost term."""
        only_cost = log_reward(error=0.1, cost=5.0, alpha=0.0, beta=1.0, epsilon=1e-12)
        only_error = log_reward(error=0.1, cost=5.0, alpha=1.0, beta=0.0, epsilon=1e-12)
        assert only_cost == pytest.approx(-math.log(5.0), rel=1e-9)
        assert only_error == pytest.approx(-math.log(0.1), rel=1e-9)
