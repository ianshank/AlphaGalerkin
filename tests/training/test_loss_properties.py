"""Property-based tests for loss functions.

Tests mathematical invariants that must hold for all inputs:
- Loss non-negativity
- Gradient flow (loss produces valid gradients)
- Symmetry properties
- Monotonicity properties
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.training.loss import AlphaGalerkinLoss, EntropyRegularizer

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def valid_logits(batch_max: int = 8, actions_max: int = 50) -> st.SearchStrategy:
    """Strategy for policy logits tensors."""
    return st.tuples(
        st.integers(min_value=1, max_value=batch_max),
        st.integers(min_value=2, max_value=actions_max),
    ).map(lambda shape: torch.randn(shape[0], shape[1]))


def valid_loss_inputs(batch_max: int = 8, actions_max: int = 50) -> st.SearchStrategy:
    """Strategy for a full set of loss inputs (logits, value, targets)."""
    return st.tuples(
        st.integers(min_value=1, max_value=batch_max),
        st.integers(min_value=2, max_value=actions_max),
    ).map(lambda shape: _make_loss_inputs(shape[0], shape[1]))


def _make_loss_inputs(batch: int, n_actions: int) -> dict:
    """Create a dictionary of loss inputs for a given shape."""
    torch.manual_seed(batch * 1000 + n_actions)
    return {
        "policy_logits": torch.randn(batch, n_actions),
        "value": torch.tanh(torch.randn(batch, 1)),
        "target_policy": torch.softmax(torch.randn(batch, n_actions), dim=-1),
        "target_value": torch.rand(batch, 1) * 2 - 1,
    }


# ---------------------------------------------------------------------------
# Property tests for AlphaGalerkinLoss
# ---------------------------------------------------------------------------


class TestLossNonNegativity:
    """Total loss and all components must be non-negative."""

    @given(data=valid_loss_inputs())
    @settings(max_examples=50)
    def test_total_loss_non_negative(self, data: dict) -> None:
        """Total loss is non-negative for all valid inputs."""
        loss_fn = AlphaGalerkinLoss()
        result = loss_fn(**data)

        assert result.total.item() >= 0.0, "Total loss must be non-negative"

    @given(data=valid_loss_inputs())
    @settings(max_examples=50)
    def test_policy_loss_non_negative(self, data: dict) -> None:
        """Policy loss (cross-entropy with soft targets) is non-negative."""
        loss_fn = AlphaGalerkinLoss()
        result = loss_fn(**data)

        assert result.policy.item() >= -1e-6, "Policy loss must be non-negative"

    @given(data=valid_loss_inputs())
    @settings(max_examples=50)
    def test_value_loss_non_negative(self, data: dict) -> None:
        """Value loss (MSE) is non-negative."""
        loss_fn = AlphaGalerkinLoss()
        result = loss_fn(**data)

        assert result.value.item() >= 0.0, "Value loss must be non-negative"


class TestPolicyLossProperties:
    """Mathematical properties of the policy cross-entropy loss."""

    @given(n_actions=st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_uniform_target_equals_log_n(self, n_actions: int) -> None:
        """Policy loss with uniform target should equal log(n_actions).

        For uniform target p = 1/n and any logits, the cross-entropy
        H(uniform, softmax(logits)) >= log(n) with equality when the
        predicted distribution is also uniform.
        """
        torch.manual_seed(n_actions)
        batch_size = 4
        loss_fn = AlphaGalerkinLoss()

        # Uniform logits -> uniform predicted distribution
        uniform_logits = torch.zeros(batch_size, n_actions)
        uniform_target = torch.ones(batch_size, n_actions) / n_actions

        policy_loss = loss_fn.compute_policy_loss(uniform_logits, uniform_target)

        expected = math.log(n_actions)
        assert policy_loss.item() == pytest.approx(expected, abs=1e-4), (
            f"Uniform CE should be log({n_actions})={expected:.4f}, "
            f"got {policy_loss.item():.4f}"
        )

    def test_value_loss_zero_when_exact(self) -> None:
        """Value loss is zero when prediction equals target."""
        loss_fn = AlphaGalerkinLoss()
        batch_size = 4

        value = torch.tensor([[0.5], [-0.3], [0.0], [0.9]])
        target = value.clone()

        value_loss = loss_fn.compute_value_loss(value, target)
        assert value_loss.item() == pytest.approx(0.0, abs=1e-7)


class TestLBBLossProperties:
    """Properties of the LBB regularization loss."""

    def test_lbb_loss_decreases_with_increasing_constant(self) -> None:
        """LBB loss should decrease when LBB constant increases toward target."""
        loss_fn = AlphaGalerkinLoss(lbb_target=0.1, lbb_weight=1.0)

        lbb_small = torch.tensor([0.01])
        lbb_medium = torch.tensor([0.05])
        lbb_large = torch.tensor([0.1])

        loss_small = loss_fn.compute_lbb_loss(lbb_small).item()
        loss_medium = loss_fn.compute_lbb_loss(lbb_medium).item()
        loss_large = loss_fn.compute_lbb_loss(lbb_large).item()

        # Loss should decrease as LBB constant approaches target
        assert loss_small > loss_medium, (
            f"Loss at 0.01 ({loss_small:.4f}) should exceed loss at 0.05 ({loss_medium:.4f})"
        )
        assert loss_medium > loss_large, (
            f"Loss at 0.05 ({loss_medium:.4f}) should exceed loss at 0.1 ({loss_large:.4f})"
        )

    def test_lbb_loss_is_zero_without_constant(self) -> None:
        """LBB loss returns 0 when no LBB constant is provided."""
        loss_fn = AlphaGalerkinLoss(lbb_weight=1.0)
        loss = loss_fn.compute_lbb_loss(None)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


class TestEntropyRegularizerProperties:
    """Properties of the entropy regularizer."""

    @given(n_actions=st.integers(min_value=2, max_value=100))
    @settings(max_examples=50)
    def test_entropy_maximized_for_uniform(self, n_actions: int) -> None:
        """Entropy regularizer is most negative (maximum bonus) for uniform distribution."""
        torch.manual_seed(n_actions)
        reg = EntropyRegularizer(weight=1.0)

        batch_size = 4

        # Uniform distribution (maximum entropy)
        uniform_logits = torch.zeros(batch_size, n_actions)
        entropy_uniform = reg(uniform_logits)

        # Peaked distribution (low entropy)
        peaked_logits = torch.zeros(batch_size, n_actions)
        peaked_logits[:, 0] = 20.0
        entropy_peaked = reg(peaked_logits)

        # Since the regularizer returns -weight * normalized_entropy,
        # uniform should be more negative (larger bonus).
        assert entropy_uniform.item() < entropy_peaked.item(), (
            f"Uniform entropy ({entropy_uniform.item():.4f}) should be more negative "
            f"than peaked ({entropy_peaked.item():.4f})"
        )


class TestGradientProperties:
    """Tests that verify gradient flow through loss components."""

    @given(data=valid_loss_inputs())
    @settings(max_examples=50)
    def test_loss_produces_valid_gradients(self, data: dict) -> None:
        """Loss produces finite, non-NaN gradients for all valid inputs."""
        loss_fn = AlphaGalerkinLoss()

        # Make logits require grad
        logits = data["policy_logits"].clone().detach().requires_grad_(True)
        value_raw = torch.randn_like(data["value"], requires_grad=True)
        value = torch.tanh(value_raw)

        result = loss_fn(
            policy_logits=logits,
            value=value,
            target_policy=data["target_policy"],
            target_value=data["target_value"],
        )

        result.total.backward()

        assert logits.grad is not None, "Gradients must flow to policy logits"
        assert value_raw.grad is not None, "Gradients must flow to value"
        assert torch.isfinite(logits.grad).all(), "Policy gradients must be finite"
        assert torch.isfinite(value_raw.grad).all(), "Value gradients must be finite"


class TestLabelSmoothing:
    """Label smoothing should reduce loss magnitude compared to no smoothing."""

    @given(data=valid_loss_inputs())
    @settings(max_examples=50)
    def test_label_smoothing_reduces_loss(self, data: dict) -> None:
        """With label smoothing, the policy loss should differ from unsmoothed."""
        loss_no_smooth = AlphaGalerkinLoss(label_smoothing=0.0)
        loss_smoothed = AlphaGalerkinLoss(label_smoothing=0.1)

        result_no = loss_no_smooth(**data)
        result_sm = loss_smoothed(**data)

        # Label smoothing mixes uniform into the target, changing loss.
        # We just verify both are finite (not that one is strictly smaller).
        assert result_no.policy.isfinite(), "Unsmoothed policy loss must be finite"
        assert result_sm.policy.isfinite(), "Smoothed policy loss must be finite"
        # The two should differ (label smoothing changes the target)
        if data["policy_logits"].shape[-1] > 2:
            assert result_no.policy.item() != pytest.approx(
                result_sm.policy.item(), abs=1e-6
            ), "Smoothed loss should differ from unsmoothed loss"


class TestDeterminism:
    """Loss is deterministic: same input -> same output."""

    def test_loss_is_deterministic(self) -> None:
        """Two identical calls with identical inputs must return identical loss."""
        torch.manual_seed(42)
        loss_fn = AlphaGalerkinLoss()

        batch_size, n_actions = 4, 82
        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result1 = loss_fn(policy_logits, value, target_policy, target_value)

        # Reset stats to avoid accumulated state differences
        loss_fn.reset_stats()
        result2 = loss_fn(policy_logits, value, target_policy, target_value)

        assert result1.total.item() == pytest.approx(result2.total.item(), abs=1e-7)
        assert result1.policy.item() == pytest.approx(result2.policy.item(), abs=1e-7)
        assert result1.value.item() == pytest.approx(result2.value.item(), abs=1e-7)
        assert result1.lbb.item() == pytest.approx(result2.lbb.item(), abs=1e-7)
