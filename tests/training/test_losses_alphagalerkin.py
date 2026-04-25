"""Tests for AlphaGalerkin composite loss (src.training.losses.alphagalerkin)."""

from __future__ import annotations

import pytest
import torch

from src.training.losses.alphagalerkin import AlphaGalerkinLoss, EntropyRegularizer
from src.training.losses.base import LossOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
BOARD_SIZE = 5
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE  # 25


def _make_inputs(
    batch_size: int = BATCH_SIZE,
    action_size: int = ACTION_SIZE,
    requires_grad: bool = False,
):
    """Create standard test inputs."""
    policy_logits = torch.randn(batch_size, action_size, requires_grad=requires_grad)
    value = torch.tanh(torch.randn(batch_size, 1))
    if requires_grad:
        value = torch.randn(batch_size, 1, requires_grad=True)
        value_out = torch.tanh(value)
        value_out.retain_grad()
        return policy_logits, value_out, value  # value is raw for grad check
    target_policy = torch.softmax(torch.randn(batch_size, action_size), dim=-1)
    target_value = torch.rand(batch_size, 1) * 2 - 1
    return policy_logits, value, target_policy, target_value


# ---------------------------------------------------------------------------
# AlphaGalerkinLoss tests
# ---------------------------------------------------------------------------


class TestAlphaGalerkinLossDefault:
    """Test AlphaGalerkinLoss with default parameters."""

    def test_basic_computation(self) -> None:
        """Loss computation returns LossOutput with scalar tensors."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits, value, target_policy, target_value = _make_inputs()

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert isinstance(result, LossOutput)
        assert result.total.ndim == 0
        assert result.policy.ndim == 0
        assert result.value.ndim == 0
        assert result.lbb.ndim == 0

    def test_total_equals_weighted_sum(self) -> None:
        """Total loss equals weighted sum of components (no LBB constant)."""
        loss_fn = AlphaGalerkinLoss(policy_weight=1.0, value_weight=1.0, lbb_weight=0.01)
        policy_logits, value, target_policy, target_value = _make_inputs()

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        expected = 1.0 * result.policy + 1.0 * result.value + 0.01 * result.lbb
        torch.testing.assert_close(result.total, expected)

    def test_to_dict(self) -> None:
        """LossOutput.to_dict returns float values."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits, value, target_policy, target_value = _make_inputs()
        result = loss_fn(policy_logits, value, target_policy, target_value)
        d = result.to_dict()

        for key in ("total", "policy", "value", "lbb"):
            assert key in d
            assert isinstance(d[key], float)


class TestAlphaGalerkinLossCustomWeights:
    """Test AlphaGalerkinLoss with custom LBB weights."""

    def test_custom_lbb_weight(self) -> None:
        """Custom LBB weight changes total loss."""
        policy_logits, value, target_policy, target_value = _make_inputs()
        lbb_constant = torch.rand(BATCH_SIZE) * 0.05  # Small -> large LBB loss

        loss_lo = AlphaGalerkinLoss(lbb_weight=0.001)
        loss_hi = AlphaGalerkinLoss(lbb_weight=1.0)

        r_lo = loss_lo(policy_logits, value, target_policy, target_value, lbb_constant)
        r_hi = loss_hi(policy_logits, value, target_policy, target_value, lbb_constant)

        # Same component losses, different total due to weighting
        torch.testing.assert_close(r_lo.policy, r_hi.policy)
        torch.testing.assert_close(r_lo.value, r_hi.value)
        torch.testing.assert_close(r_lo.lbb, r_hi.lbb)
        assert r_hi.total > r_lo.total

    def test_zero_lbb_weight_removes_lbb_contribution(self) -> None:
        """LBB weight 0 means LBB loss does not affect total."""
        loss_fn = AlphaGalerkinLoss(lbb_weight=0.0)
        policy_logits, value, target_policy, target_value = _make_inputs()
        lbb_constant = torch.rand(BATCH_SIZE) * 0.05

        result = loss_fn(policy_logits, value, target_policy, target_value, lbb_constant)
        expected = result.policy + result.value
        torch.testing.assert_close(result.total, expected)


class TestLossNonNegativity:
    """Verify loss components are non-negative."""

    def test_all_components_non_negative(self) -> None:
        """All loss components should be >= 0 over many random inputs."""
        loss_fn = AlphaGalerkinLoss()

        for _ in range(20):
            policy_logits, value, target_policy, target_value = _make_inputs()
            result = loss_fn(policy_logits, value, target_policy, target_value)

            assert result.total >= 0, "Total loss must be non-negative"
            assert result.policy >= 0, "Policy loss must be non-negative"
            assert result.value >= 0, "Value loss must be non-negative"
            assert result.lbb >= 0, "LBB loss must be non-negative"

    def test_non_negative_with_lbb_constant(self) -> None:
        """Non-negativity holds when LBB constant is provided."""
        loss_fn = AlphaGalerkinLoss()

        for _ in range(10):
            policy_logits, value, target_policy, target_value = _make_inputs()
            lbb_constant = torch.rand(BATCH_SIZE) * 2.0
            result = loss_fn(policy_logits, value, target_policy, target_value, lbb_constant)
            assert result.total.isfinite()
            assert result.policy >= 0
            assert result.value >= 0


class TestIndividualComponents:
    """Test individual loss components in isolation."""

    def test_policy_loss_with_perfect_prediction(self) -> None:
        """Policy CE is minimal when logits match target distribution."""
        loss_fn = AlphaGalerkinLoss()

        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        # Convert target back to logits (log of softmax)
        logits = torch.log(target_policy + 1e-10)

        loss = loss_fn.compute_policy_loss(logits, target_policy)
        assert loss >= 0
        # Should be close to the entropy of the target distribution
        entropy = -(target_policy * torch.log(target_policy + 1e-10)).sum(dim=-1).mean()
        assert abs(loss.item() - entropy.item()) < 0.5

    def test_value_loss_zero_when_perfect(self) -> None:
        """Value MSE is zero when prediction matches target."""
        loss_fn = AlphaGalerkinLoss()
        target = torch.randn(BATCH_SIZE, 1)
        loss = loss_fn.compute_value_loss(target, target)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=1e-7)

    def test_value_loss_positive_when_different(self) -> None:
        """Value MSE is positive when prediction differs from target."""
        loss_fn = AlphaGalerkinLoss()
        pred = torch.zeros(BATCH_SIZE, 1)
        target = torch.ones(BATCH_SIZE, 1)
        loss = loss_fn.compute_value_loss(pred, target)
        assert loss > 0

    def test_lbb_loss_none_returns_zero(self) -> None:
        """LBB loss returns 0 when constant is None."""
        loss_fn = AlphaGalerkinLoss()
        loss = loss_fn.compute_lbb_loss(None)
        assert loss.item() == 0.0

    def test_lbb_loss_positive_for_small_constant(self) -> None:
        """LBB loss is positive when constant is below target."""
        loss_fn = AlphaGalerkinLoss(lbb_target=0.1, lbb_eps=1e-8)
        lbb_constant = torch.tensor([0.01, 0.02, 0.03, 0.05])
        loss = loss_fn.compute_lbb_loss(lbb_constant)
        assert loss > 0

    def test_lbb_loss_decreases_with_larger_constant(self) -> None:
        """LBB loss decreases as the constant approaches/exceeds target."""
        loss_fn = AlphaGalerkinLoss(lbb_target=0.1)
        small_lbb = torch.tensor([0.01])
        large_lbb = torch.tensor([0.5])
        loss_small = loss_fn.compute_lbb_loss(small_lbb)
        loss_large = loss_fn.compute_lbb_loss(large_lbb)
        assert loss_small > loss_large


class TestLabelSmoothing:
    """Test label smoothing effect on policy loss."""

    def test_label_smoothing_changes_loss(self) -> None:
        """Label smoothing should change the loss value."""
        loss_no_smooth = AlphaGalerkinLoss(label_smoothing=0.0)
        loss_smooth = AlphaGalerkinLoss(label_smoothing=0.1)

        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        # One-hot-ish target
        target_policy = torch.zeros(BATCH_SIZE, ACTION_SIZE)
        target_policy[:, 0] = 1.0

        l_no = loss_no_smooth.compute_policy_loss(policy_logits, target_policy)
        l_sm = loss_smooth.compute_policy_loss(policy_logits, target_policy)

        assert not torch.allclose(l_no, l_sm), "Smoothing should change loss"

    def test_label_smoothing_zero_is_noop(self) -> None:
        """Zero label smoothing should have no effect."""
        loss_fn = AlphaGalerkinLoss(label_smoothing=0.0)
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        loss = loss_fn.compute_policy_loss(policy_logits, target_policy)
        assert loss.isfinite()


class TestLogBarrierWeight:
    """Test log barrier weight in LBB loss."""

    def test_log_barrier_weight_effect(self) -> None:
        """Higher log barrier weight increases LBB loss."""
        loss_lo = AlphaGalerkinLoss(log_barrier_weight=0.001)
        loss_hi = AlphaGalerkinLoss(log_barrier_weight=1.0)

        lbb_constant = torch.tensor([0.5, 0.6, 0.7, 0.8])

        l_lo = loss_lo.compute_lbb_loss(lbb_constant)
        l_hi = loss_hi.compute_lbb_loss(lbb_constant)

        # Higher barrier weight -> higher LBB loss
        assert l_hi > l_lo

    def test_log_barrier_zero_weight(self) -> None:
        """Zero barrier weight removes the log penalty term."""
        loss_fn = AlphaGalerkinLoss(log_barrier_weight=0.0, lbb_target=0.1)
        # LBB constant above target -> threshold penalty is 0
        lbb_constant = torch.tensor([0.5, 0.6])
        loss = loss_fn.compute_lbb_loss(lbb_constant)
        # Only threshold penalty (which is 0 here) + 0 * log_penalty
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestEntropyRegularizerComputation:
    """Test EntropyRegularizer."""

    def test_uniform_has_max_entropy(self) -> None:
        """Uniform logits produce maximum entropy (most negative output)."""
        reg = EntropyRegularizer(weight=1.0)
        uniform_logits = torch.zeros(BATCH_SIZE, ACTION_SIZE)
        peaked_logits = torch.zeros(BATCH_SIZE, ACTION_SIZE)
        peaked_logits[:, 0] = 100.0

        e_uniform = reg(uniform_logits)
        e_peaked = reg(peaked_logits)

        # Return is -weight * entropy, so uniform is more negative
        assert e_uniform < e_peaked

    def test_entropy_is_scalar(self) -> None:
        """EntropyRegularizer returns a scalar."""
        reg = EntropyRegularizer()
        logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        result = reg(logits)
        assert result.ndim == 0

    def test_entropy_with_mask(self) -> None:
        """Entropy works with action mask."""
        reg = EntropyRegularizer(weight=1.0)
        logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        mask = torch.ones(BATCH_SIZE, ACTION_SIZE)
        mask[:, ACTION_SIZE // 2 :] = 0.0

        result = reg(logits, mask=mask)
        assert result.isfinite()

    def test_entropy_all_masked(self) -> None:
        """All actions masked should not produce NaN."""
        reg = EntropyRegularizer(weight=1.0)
        logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        mask = torch.zeros(BATCH_SIZE, ACTION_SIZE)

        result = reg(logits, mask=mask)
        assert result.isfinite()

    def test_entropy_single_valid_action(self) -> None:
        """Single valid action has zero entropy."""
        reg = EntropyRegularizer(weight=1.0)
        logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        mask = torch.zeros(BATCH_SIZE, ACTION_SIZE)
        mask[:, 0] = 1.0

        result = reg(logits, mask=mask)
        assert result.isfinite()

    def test_weight_scales_output(self) -> None:
        """Changing weight scales the output proportionally."""
        logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        reg1 = EntropyRegularizer(weight=1.0)
        reg2 = EntropyRegularizer(weight=2.0)

        e1 = reg1(logits)
        e2 = reg2(logits)
        torch.testing.assert_close(e2, 2.0 * e1, atol=1e-6, rtol=1e-5)


class TestGradientFlow:
    """Test gradient propagation through loss."""

    def test_gradients_flow_to_policy_logits(self) -> None:
        """Gradients flow back to policy_logits."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE, requires_grad=True)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        result.total.backward()

        assert policy_logits.grad is not None
        assert policy_logits.grad.shape == policy_logits.shape

    def test_gradients_flow_to_value(self) -> None:
        """Gradients flow back through value head."""
        loss_fn = AlphaGalerkinLoss()
        raw_value = torch.randn(BATCH_SIZE, 1, requires_grad=True)
        value = torch.tanh(raw_value)
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        result.total.backward()

        assert raw_value.grad is not None

    def test_gradients_flow_through_lbb(self) -> None:
        """Gradients flow through the LBB constant."""
        loss_fn = AlphaGalerkinLoss(lbb_weight=0.1)
        lbb_constant = torch.rand(BATCH_SIZE, requires_grad=True)
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value, lbb_constant)
        result.total.backward()

        assert lbb_constant.grad is not None

    def test_entropy_regularizer_gradient(self) -> None:
        """Gradients flow through entropy regularizer."""
        reg = EntropyRegularizer(weight=0.01)
        logits = torch.randn(BATCH_SIZE, ACTION_SIZE, requires_grad=True)
        loss = reg(logits)
        loss.backward()
        assert logits.grad is not None


class TestEdgeCases:
    """Edge cases for AlphaGalerkinLoss."""

    def test_uniform_policy_target(self) -> None:
        """Uniform target policy should produce finite loss."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.ones(BATCH_SIZE, ACTION_SIZE) / ACTION_SIZE
        target_value = torch.randn(BATCH_SIZE, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        assert result.total.isfinite()

    def test_zero_value_target(self) -> None:
        """Zero value target should produce finite loss."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        target_value = torch.zeros(BATCH_SIZE, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        assert result.total.isfinite()

    def test_action_mask_all_valid(self) -> None:
        """All-ones mask should behave like no mask."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits = torch.randn(BATCH_SIZE, ACTION_SIZE)
        value = torch.randn(BATCH_SIZE, 1)
        target_policy = torch.softmax(torch.randn(BATCH_SIZE, ACTION_SIZE), dim=-1)
        target_value = torch.randn(BATCH_SIZE, 1)

        mask = torch.ones(BATCH_SIZE, ACTION_SIZE)
        r_mask = loss_fn.compute_policy_loss(policy_logits, target_policy, mask)
        r_none = loss_fn.compute_policy_loss(policy_logits, target_policy, None)

        torch.testing.assert_close(r_mask, r_none, atol=1e-5, rtol=1e-5)

    def test_running_stats_reset(self) -> None:
        """Running stats reset correctly."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits, value, target_policy, target_value = _make_inputs()
        loss_fn(policy_logits, value, target_policy, target_value)

        stats = loss_fn.get_running_stats()
        assert stats["policy"] > 0 or stats["value"] > 0

        loss_fn.reset_stats()
        stats = loss_fn.get_running_stats()
        assert stats["policy"] == 0.0
        assert stats["value"] == 0.0
        assert stats["lbb"] == 0.0

    def test_batch_size_one(self) -> None:
        """Works with batch size 1."""
        loss_fn = AlphaGalerkinLoss()
        policy_logits = torch.randn(1, ACTION_SIZE)
        value = torch.randn(1, 1)
        target_policy = torch.softmax(torch.randn(1, ACTION_SIZE), dim=-1)
        target_value = torch.randn(1, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)
        assert result.total.isfinite()
