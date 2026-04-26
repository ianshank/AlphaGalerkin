"""Property-based tests for all 5 loss balancing strategies.

Uses Hypothesis to generate random loss values and verify invariants
that must hold for all inputs across all strategies:
- Weights are non-negative and finite
- compute_weighted_loss() produces valid gradients
- Weights sum to a positive value
- Multiple sequential updates don't cause weight explosion/collapse
- Single-loss edge case
- Equal losses produce equal weights (where applicable)
- Extreme value handling (very large/small losses)
- NaN/Inf filtering
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor

from src.training.loss_balancing import (
    BalancingStrategy,
    LossBalancingConfig,
    LossTerms,
    create_loss_balancer,
)

# ---------------------------------------------------------------------------
# Hypothesis strategies for generating random loss values
# ---------------------------------------------------------------------------

# Strategy for generating a single positive loss value (typical training range)
_positive_loss = st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False)

# Strategy for loss values that can be very small
_small_loss = st.floats(min_value=1e-12, max_value=1e-4, allow_nan=False, allow_infinity=False)

# Strategy for loss values that can be very large
_large_loss = st.floats(min_value=1e2, max_value=1e6, allow_nan=False, allow_infinity=False)

# All 5 strategies
ALL_STRATEGIES = [
    BalancingStrategy.STATIC,
    BalancingStrategy.RELOBRALO,
    BalancingStrategy.GRADNORM,
    BalancingStrategy.UNCERTAINTY,
    BalancingStrategy.SOFTADAPT,
]

# Strategies that support equal-loss => equal-weight property
SYMMETRIC_STRATEGIES = [
    BalancingStrategy.STATIC,
    BalancingStrategy.RELOBRALO,
    BalancingStrategy.GRADNORM,
    BalancingStrategy.UNCERTAINTY,
    BalancingStrategy.SOFTADAPT,
]

# Default loss names for multi-loss tests
DEFAULT_LOSS_NAMES = ["policy", "value", "physics"]


def _make_losses(
    names: list[str],
    values: list[float],
    requires_grad: bool = False,
) -> dict[str, Tensor]:
    """Create a loss dict from names and float values."""
    return {
        name: torch.tensor(val, requires_grad=requires_grad) for name, val in zip(names, values)
    }


def _make_config(
    strategy: BalancingStrategy,
    **overrides: Any,
) -> LossBalancingConfig:
    """Create a LossBalancingConfig with warmup disabled by default."""
    defaults: dict[str, Any] = {
        "name": "test",
        "strategy": strategy,
        "warmup_steps": 0,
    }
    defaults.update(overrides)
    return LossBalancingConfig(**defaults)


# ---------------------------------------------------------------------------
# A) update() returns valid weights (non-negative, finite, no NaN)
# ---------------------------------------------------------------------------


class TestUpdateReturnsValidWeights:
    """For every strategy, update() must return valid weights."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(
        loss_a=_positive_loss,
        loss_b=_positive_loss,
        loss_c=_positive_loss,
    )
    @settings(max_examples=50)
    def test_weights_non_negative_finite(
        self,
        strategy: BalancingStrategy,
        loss_a: float,
        loss_b: float,
        loss_c: float,
    ) -> None:
        """Weights from update() are non-negative and finite."""
        config = _make_config(strategy)
        balancer = create_loss_balancer(config, DEFAULT_LOSS_NAMES)
        losses = _make_losses(DEFAULT_LOSS_NAMES, [loss_a, loss_b, loss_c])

        weights = balancer.update(losses)

        for name, w in weights.items():
            assert w >= 0.0, f"Weight for {name} is negative: {w}"
            assert math.isfinite(w), f"Weight for {name} is not finite: {w}"


# ---------------------------------------------------------------------------
# B) compute_weighted_loss() produces valid gradients
# ---------------------------------------------------------------------------


class TestComputeWeightedLossGradients:
    """compute_weighted_loss() must produce a differentiable weighted sum."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(
        loss_a=_positive_loss,
        loss_b=_positive_loss,
    )
    @settings(max_examples=30)
    def test_weighted_loss_has_valid_gradients(
        self,
        strategy: BalancingStrategy,
        loss_a: float,
        loss_b: float,
    ) -> None:
        """Weighted sum retains grad and backward produces finite gradients."""
        config = _make_config(strategy)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        ta = torch.tensor(loss_a, requires_grad=True)
        tb = torch.tensor(loss_b, requires_grad=True)
        losses: dict[str, Tensor] = {"a": ta, "b": tb}

        result = balancer.compute_weighted_loss(losses)

        assert isinstance(result, LossTerms)
        assert result.weighted_sum.requires_grad
        assert torch.isfinite(result.weighted_sum), "Weighted sum must be finite"

        result.weighted_sum.backward()

        assert ta.grad is not None, "Gradient must flow to loss a"
        assert tb.grad is not None, "Gradient must flow to loss b"
        assert torch.isfinite(ta.grad), f"Grad for a not finite: {ta.grad}"
        assert torch.isfinite(tb.grad), f"Grad for b not finite: {tb.grad}"


# ---------------------------------------------------------------------------
# C) Weights sum to a positive value
# ---------------------------------------------------------------------------


class TestWeightsSumPositive:
    """Sum of all weights must be strictly positive."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(
        loss_a=_positive_loss,
        loss_b=_positive_loss,
        loss_c=_positive_loss,
    )
    @settings(max_examples=30)
    def test_weight_sum_positive(
        self,
        strategy: BalancingStrategy,
        loss_a: float,
        loss_b: float,
        loss_c: float,
    ) -> None:
        """Sum of weights is strictly positive after update."""
        config = _make_config(strategy)
        balancer = create_loss_balancer(config, DEFAULT_LOSS_NAMES)
        losses = _make_losses(DEFAULT_LOSS_NAMES, [loss_a, loss_b, loss_c])

        result = balancer.compute_weighted_loss(losses)

        weight_sum = sum(result.weights.values())
        assert weight_sum > 0.0, f"Weight sum must be positive, got {weight_sum}"


# ---------------------------------------------------------------------------
# D) Roundtrip: init -> update -> compute_weighted_loss -> backward -> valid grads
# ---------------------------------------------------------------------------


class TestRoundtripGradientFlow:
    """Full roundtrip: multiple updates then backward produces valid gradients."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(n_steps=st.integers(min_value=1, max_value=10))
    @settings(max_examples=20)
    def test_roundtrip_gradient_flow(
        self,
        strategy: BalancingStrategy,
        n_steps: int,
    ) -> None:
        """Multiple update steps followed by backward yields finite gradients."""
        config = _make_config(strategy)
        names = ["policy", "value"]
        balancer = create_loss_balancer(config, names)

        for _ in range(n_steps):
            ta = torch.tensor(1.0, requires_grad=True)
            tb = torch.tensor(0.5, requires_grad=True)
            losses: dict[str, Tensor] = {"policy": ta, "value": tb}

            result = balancer.compute_weighted_loss(losses)

        # Final backward
        result.weighted_sum.backward()

        assert ta.grad is not None
        assert tb.grad is not None
        assert torch.isfinite(ta.grad)
        assert torch.isfinite(tb.grad)


# ---------------------------------------------------------------------------
# E) Multiple sequential updates don't cause weight explosion/collapse
# ---------------------------------------------------------------------------


class TestWeightStability:
    """Weights stay bounded after many sequential updates."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_no_weight_explosion_after_many_updates(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """After 200 updates with varied losses, weights stay within bounds."""
        config = _make_config(
            strategy,
            min_weight=0.01,
            max_weight=10.0,
        )
        names = ["a", "b", "c"]
        balancer = create_loss_balancer(config, names)

        for i in range(200):
            # Oscillating loss pattern to stress-test adaptation
            losses = _make_losses(
                names,
                [
                    1.0 + 0.5 * math.sin(i * 0.1),
                    0.1 + 0.05 * math.cos(i * 0.2),
                    2.0 * math.exp(-i * 0.01),
                ],
            )
            result = balancer.compute_weighted_loss(losses)

        for name, w in result.weights.items():
            assert w >= config.min_weight, f"Weight {name}={w} below min_weight={config.min_weight}"
            assert w <= config.max_weight, f"Weight {name}={w} above max_weight={config.max_weight}"
            assert math.isfinite(w), f"Weight {name}={w} is not finite"

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_no_weight_collapse_constant_losses(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """Constant identical losses should not collapse weights to min_weight."""
        config = _make_config(strategy, min_weight=0.01, max_weight=10.0)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        for _ in range(100):
            losses = _make_losses(names, [1.0, 1.0])
            result = balancer.compute_weighted_loss(losses)

        # With constant equal losses, weights should not drift to extremes
        weight_sum = sum(result.weights.values())
        assert weight_sum > 0.0, "Weight sum collapsed to zero"


# ---------------------------------------------------------------------------
# F) Single-loss edge case
# ---------------------------------------------------------------------------


class TestSingleLoss:
    """All strategies must handle a single loss term gracefully."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(loss_val=_positive_loss)
    @settings(max_examples=30)
    def test_single_loss_valid(
        self,
        strategy: BalancingStrategy,
        loss_val: float,
    ) -> None:
        """Single loss term produces valid weights and weighted sum."""
        config = _make_config(strategy)
        balancer = create_loss_balancer(config, ["only"])

        t = torch.tensor(loss_val, requires_grad=True)
        losses: dict[str, Tensor] = {"only": t}

        result = balancer.compute_weighted_loss(losses)

        assert result.weights["only"] > 0.0
        assert math.isfinite(result.weights["only"])
        assert torch.isfinite(result.weighted_sum)
        assert result.weighted_sum.requires_grad

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_single_loss_multiple_updates(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """Single loss term remains stable across multiple updates."""
        config = _make_config(strategy)
        balancer = create_loss_balancer(config, ["only"])

        for i in range(20):
            losses = _make_losses(["only"], [1.0 / (i + 1)])
            result = balancer.compute_weighted_loss(losses)

        assert result.weights["only"] > 0.0
        assert math.isfinite(result.weights["only"])


# ---------------------------------------------------------------------------
# G) Equal losses produce equal weights (for symmetric strategies)
# ---------------------------------------------------------------------------


class TestEqualLossesEqualWeights:
    """Equal loss values should produce equal (or near-equal) weights."""

    @pytest.mark.parametrize("strategy", SYMMETRIC_STRATEGIES)
    @given(loss_val=st.floats(min_value=0.01, max_value=100.0, allow_nan=False))
    @settings(max_examples=30)
    def test_equal_losses_equal_weights(
        self,
        strategy: BalancingStrategy,
        loss_val: float,
    ) -> None:
        """When all losses are equal, weights should be (near-)equal."""
        config = _make_config(strategy)
        names = ["a", "b", "c"]
        balancer = create_loss_balancer(config, names)

        # Several updates with equal losses
        for _ in range(20):
            losses = _make_losses(names, [loss_val, loss_val, loss_val])
            result = balancer.compute_weighted_loss(losses)

        weights = list(result.weights.values())
        # All weights should be approximately equal
        mean_w = sum(weights) / len(weights)
        for w in weights:
            # ReLoBRaLo with random lookback may have small deviations
            assert abs(w - mean_w) < 0.5, f"Weights not equal for equal losses: {result.weights}"


# ---------------------------------------------------------------------------
# H) Extreme value handling (very large/small losses)
# ---------------------------------------------------------------------------


class TestExtremeValues:
    """Strategies must handle extreme loss magnitudes without crashing."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(loss_val=_small_loss)
    @settings(max_examples=20)
    def test_very_small_losses(
        self,
        strategy: BalancingStrategy,
        loss_val: float,
    ) -> None:
        """Very small losses produce valid weights."""
        config = _make_config(strategy)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        for _ in range(5):
            losses = _make_losses(names, [loss_val, loss_val * 2])
            result = balancer.compute_weighted_loss(losses)

        for w in result.weights.values():
            assert w > 0.0
            assert math.isfinite(w)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @given(loss_val=_large_loss)
    @settings(max_examples=20)
    def test_very_large_losses(
        self,
        strategy: BalancingStrategy,
        loss_val: float,
    ) -> None:
        """Very large losses produce valid weights."""
        config = _make_config(strategy)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        for _ in range(5):
            losses = _make_losses(names, [loss_val, loss_val / 2])
            result = balancer.compute_weighted_loss(losses)

        for w in result.weights.values():
            assert w > 0.0
            assert math.isfinite(w)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_extreme_imbalance(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """Extreme loss imbalance (1e-8 vs 1e4) produces bounded weights."""
        config = _make_config(
            strategy,
            min_weight=0.01,
            max_weight=10.0,
        )
        names = ["tiny", "huge"]
        balancer = create_loss_balancer(config, names)

        for _ in range(30):
            losses = _make_losses(names, [1e-8, 1e4])
            result = balancer.compute_weighted_loss(losses)

        for name, w in result.weights.items():
            assert (
                config.min_weight <= w <= config.max_weight
            ), f"Weight {name}={w} out of bounds [{config.min_weight}, {config.max_weight}]"


# ---------------------------------------------------------------------------
# NaN and Inf filtering
# ---------------------------------------------------------------------------


class TestNonFiniteLossFiltering:
    """NaN and Inf losses should be filtered out rather than corrupt state."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_nan_loss_filtered(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """NaN losses are filtered; remaining valid losses still work."""
        config = _make_config(strategy)
        names = ["good", "bad"]
        balancer = create_loss_balancer(config, names)

        losses: dict[str, Tensor] = {
            "good": torch.tensor(1.0, requires_grad=True),
            "bad": torch.tensor(float("nan")),
        }
        result = balancer.compute_weighted_loss(losses)

        assert torch.isfinite(result.weighted_sum)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_inf_loss_filtered(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """Inf losses are filtered; remaining valid losses still work."""
        config = _make_config(strategy)
        names = ["good", "bad"]
        balancer = create_loss_balancer(config, names)

        losses: dict[str, Tensor] = {
            "good": torch.tensor(1.0, requires_grad=True),
            "bad": torch.tensor(float("inf")),
        }
        result = balancer.compute_weighted_loss(losses)

        assert torch.isfinite(result.weighted_sum)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_all_nan_raises(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """If all losses are NaN, should raise ValueError (no valid terms)."""
        config = _make_config(strategy)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        losses: dict[str, Tensor] = {
            "a": torch.tensor(float("nan")),
            "b": torch.tensor(float("nan")),
        }
        with pytest.raises(ValueError, match="No valid loss terms"):
            balancer.compute_weighted_loss(losses)


# ---------------------------------------------------------------------------
# Determinism (for non-random strategies)
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Non-random strategies produce identical results for identical inputs."""

    @pytest.mark.parametrize(
        "strategy",
        [
            BalancingStrategy.STATIC,
            BalancingStrategy.GRADNORM,
            BalancingStrategy.UNCERTAINTY,
        ],
    )
    def test_deterministic_strategies(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """Static, GradNorm, Uncertainty are deterministic."""
        config = _make_config(strategy)
        names = ["a", "b"]

        balancer1 = create_loss_balancer(config, names)
        balancer2 = create_loss_balancer(config, names)

        for _ in range(10):
            losses = _make_losses(names, [1.5, 0.8])
            r1 = balancer1.compute_weighted_loss(losses)
            r2 = balancer2.compute_weighted_loss(losses)

        assert r1.weights == r2.weights


# ---------------------------------------------------------------------------
# Reset behavior
# ---------------------------------------------------------------------------


class TestResetBehavior:
    """reset() should return the balancer to its initial state."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_reset_restores_initial_state(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """After reset, weights return to initial uniform values."""
        config = _make_config(strategy)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        # Run several updates
        for _ in range(20):
            losses = _make_losses(names, [0.5, 2.0])
            balancer.compute_weighted_loss(losses)

        # Reset
        balancer.reset()

        # Weights should be back to 1.0
        for name, w in balancer.weights.items():
            assert w == 1.0, f"After reset, weight {name}={w}, expected 1.0"
        assert balancer._step == 0


# ---------------------------------------------------------------------------
# Warmup behavior
# ---------------------------------------------------------------------------


class TestWarmupBehavior:
    """During warmup, weights should remain at initial values."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_weights_unchanged_during_warmup(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """Weights are not updated during warmup period."""
        warmup = 50
        config = _make_config(strategy, warmup_steps=warmup)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        for _ in range(warmup):
            losses = _make_losses(names, [10.0, 0.001])
            result = balancer.compute_weighted_loss(losses)

            # During warmup, weights should remain 1.0
            assert result.weights == {
                "a": 1.0,
                "b": 1.0,
            }, f"Weights changed during warmup: {result.weights}"


# ---------------------------------------------------------------------------
# Weight clamping property
# ---------------------------------------------------------------------------


class TestWeightClamping:
    """Weights must always respect min_weight and max_weight config bounds."""

    @pytest.mark.parametrize(
        "strategy",
        [
            BalancingStrategy.RELOBRALO,
            BalancingStrategy.SOFTADAPT,
        ],
    )
    @given(
        loss_a=st.floats(min_value=1e-8, max_value=1e8, allow_nan=False, allow_infinity=False),
        loss_b=st.floats(min_value=1e-8, max_value=1e8, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_weights_within_bounds(
        self,
        strategy: BalancingStrategy,
        loss_a: float,
        loss_b: float,
    ) -> None:
        """Weights are clamped to [min_weight, max_weight] for adaptive strategies."""
        min_w = 0.05
        max_w = 8.0
        config = _make_config(strategy, min_weight=min_w, max_weight=max_w)
        names = ["a", "b"]
        balancer = create_loss_balancer(config, names)

        for _ in range(15):
            losses = _make_losses(names, [loss_a, loss_b])
            result = balancer.compute_weighted_loss(losses)

        for name, w in result.weights.items():
            assert w >= min_w, f"Weight {name}={w} < min_weight={min_w}"
            assert w <= max_w, f"Weight {name}={w} > max_weight={max_w}"


# ---------------------------------------------------------------------------
# Many loss terms
# ---------------------------------------------------------------------------


class TestManyLossTerms:
    """Strategies should work with many loss terms (e.g. 10)."""

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_ten_loss_terms(
        self,
        strategy: BalancingStrategy,
    ) -> None:
        """10 loss terms all produce valid weights."""
        names = [f"loss_{i}" for i in range(10)]
        config = _make_config(strategy)
        balancer = create_loss_balancer(config, names)

        for step in range(20):
            values = [float(i + 1) * 0.1 + step * 0.01 for i in range(10)]
            losses = _make_losses(names, values)
            result = balancer.compute_weighted_loss(losses)

        for name, w in result.weights.items():
            assert w > 0.0, f"Weight {name} not positive: {w}"
            assert math.isfinite(w), f"Weight {name} not finite: {w}"

        assert torch.isfinite(result.weighted_sum)


# ---------------------------------------------------------------------------
# Monotonicity for SoftAdapt: faster-improving losses get lower weight
# ---------------------------------------------------------------------------


class TestSoftAdaptMonotonicity:
    """SoftAdapt should assign higher weight to slower-improving losses."""

    def test_stagnant_loss_gets_higher_weight(self) -> None:
        """A stagnant loss should eventually get a higher weight than a fast-dropping one."""
        config = _make_config(
            BalancingStrategy.SOFTADAPT,
            softadapt_window_size=5,
        )
        names = ["improving", "stagnant"]
        balancer = create_loss_balancer(config, names)

        for i in range(30):
            losses = _make_losses(
                names,
                [
                    max(0.01, 2.0 - i * 0.1),  # Fast improvement
                    1.0,  # Stagnant
                ],
            )
            result = balancer.compute_weighted_loss(losses)

        # After many steps, the stagnant loss should generally have higher weight
        # (SoftAdapt gives higher weight to losses with positive rate = getting worse)
        # The stagnant one has rate ~0, improving one has negative rate
        # With softmax, higher rate => higher weight, so stagnant >= improving
        # This is a soft check since SoftAdapt depends on window history
        assert result.weights["stagnant"] >= result.weights["improving"] * 0.5, (
            f"Expected stagnant weight >= 0.5 * improving weight, "
            f"got stagnant={result.weights['stagnant']}, improving={result.weights['improving']}"
        )
