"""Tests for Dirichlet noise injection."""
from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.core.types import ActionType, ElementID
from src.alphagalerkin.env.actions import Action
from src.alphagalerkin.mcts.noise import DirichletNoise


class TestDirichletNoise:
    """Unit tests for DirichletNoise."""

    def _make_priors(self) -> dict[Action, float]:
        """Create test prior distribution."""
        actions = [
            Action(element_id=ElementID(f"e{i}"), action_type=ActionType.NO_OP)
            for i in range(5)
        ]
        return dict.fromkeys(actions, 0.2)

    def test_apply_preserves_sum_to_one(self) -> None:
        noise = DirichletNoise(alpha=0.3, epsilon=0.25)
        rng = np.random.default_rng(42)
        priors = self._make_priors()
        noisy = noise.apply(priors, rng)
        total = sum(noisy.values())
        assert abs(total - 1.0) < 1e-6

    def test_apply_preserves_action_set(self) -> None:
        noise = DirichletNoise(alpha=0.3, epsilon=0.25)
        rng = np.random.default_rng(42)
        priors = self._make_priors()
        noisy = noise.apply(priors, rng)
        assert set(noisy.keys()) == set(priors.keys())

    def test_zero_epsilon_returns_original(self) -> None:
        noise = DirichletNoise(alpha=0.3, epsilon=0.0)
        rng = np.random.default_rng(42)
        priors = self._make_priors()
        noisy = noise.apply(priors, rng)
        for a in priors:
            assert abs(noisy[a] - priors[a]) < 1e-6

    def test_full_epsilon_is_pure_noise(self) -> None:
        noise = DirichletNoise(alpha=0.3, epsilon=1.0)
        rng = np.random.default_rng(42)
        priors = self._make_priors()
        noisy = noise.apply(priors, rng)
        # With epsilon=1.0, priors are completely replaced
        # Should still sum to 1
        assert abs(sum(noisy.values()) - 1.0) < 1e-6

    def test_all_values_positive(self) -> None:
        noise = DirichletNoise(alpha=0.3, epsilon=0.25)
        rng = np.random.default_rng(42)
        priors = self._make_priors()
        noisy = noise.apply(priors, rng)
        for v in noisy.values():
            assert v > 0.0

    def test_empty_priors_returned(self) -> None:
        """Empty priors dict is returned unchanged."""
        noise = DirichletNoise(alpha=0.3, epsilon=0.25)
        result = noise.apply({})
        assert result == {}

    def test_invalid_alpha_raises(self) -> None:
        """Non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            DirichletNoise(alpha=0.0)

    def test_invalid_epsilon_raises(self) -> None:
        """Epsilon outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="epsilon must be in"):
            DirichletNoise(alpha=0.3, epsilon=1.5)

    def test_properties(self) -> None:
        """Properties expose noise parameters."""
        noise = DirichletNoise(alpha=0.5, epsilon=0.3)
        assert noise.alpha == 0.5
        assert noise.epsilon == 0.3

    def test_reproducible_with_rng(self) -> None:
        """Same RNG seed produces same noise."""
        noise = DirichletNoise(alpha=0.3, epsilon=0.25)
        priors = {"a": 0.5, "b": 0.3, "c": 0.2}
        result1 = noise.apply(priors, rng=np.random.default_rng(99))
        result2 = noise.apply(priors, rng=np.random.default_rng(99))
        for key in priors:
            assert result1[key] == pytest.approx(result2[key])
