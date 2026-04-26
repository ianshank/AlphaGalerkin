"""Numerical stability tests for training components.

Tests behavior under extreme conditions:
- Very large/small inputs
- Near-zero denominators
- Ill-conditioned matrices
- NaN/Inf propagation
"""

from __future__ import annotations

import pytest
import torch

from src.training.losses import AlphaGalerkinLoss, EntropyRegularizer


class TestLossNumericalStability:
    """Test loss functions under extreme numerical conditions."""

    def test_policy_loss_with_extreme_logits(self) -> None:
        """Large logits should not cause overflow."""
        loss_fn = AlphaGalerkinLoss()
        batch_size, n_actions = 4, 82

        # Logits of magnitude 1000 -- log_softmax must handle this
        policy_logits = torch.randn(batch_size, n_actions) * 1000.0
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total), "Total loss must be finite with extreme logits"
        assert torch.isfinite(result.policy), "Policy loss must be finite with extreme logits"

    def test_policy_loss_with_near_zero_targets(self) -> None:
        """Near-zero probability targets should not cause NaN."""
        loss_fn = AlphaGalerkinLoss()
        batch_size, n_actions = 4, 82

        policy_logits = torch.randn(batch_size, n_actions)
        # Target that is almost one-hot (many near-zero entries)
        target_policy = torch.full((batch_size, n_actions), 1e-10)
        target_policy[:, 0] = 1.0 - (n_actions - 1) * 1e-10

        value = torch.tanh(torch.randn(batch_size, 1))
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total), "Loss must be finite with near-zero targets"
        assert torch.isfinite(result.policy), "Policy loss must be finite with near-zero targets"

    def test_value_loss_with_extreme_values(self) -> None:
        """Extreme value predictions should produce finite loss."""
        loss_fn = AlphaGalerkinLoss()
        batch_size = 4

        # Values at the boundaries of tanh range
        value = torch.tensor([[1.0], [-1.0], [0.9999], [-0.9999]])
        target_value = torch.tensor([[-1.0], [1.0], [-0.9999], [0.9999]])

        value_loss = loss_fn.compute_value_loss(value, target_value)

        assert torch.isfinite(value_loss), "Value loss must be finite at tanh boundaries"
        assert value_loss.item() > 0.0, "Value loss must be positive for different pred/target"

    def test_lbb_loss_with_near_zero_constant(self) -> None:
        """Near-zero LBB constant should produce large but finite loss."""
        loss_fn = AlphaGalerkinLoss(lbb_target=0.1, lbb_weight=1.0)

        lbb_constant = torch.tensor([1e-12])
        loss = loss_fn.compute_lbb_loss(lbb_constant)

        assert torch.isfinite(loss), "LBB loss must be finite with near-zero constant"
        assert loss.item() > 0.0, "LBB loss must be positive for near-zero constant"

    def test_lbb_loss_with_very_large_constant(self) -> None:
        """Very large LBB constant should produce near-zero threshold penalty."""
        loss_fn = AlphaGalerkinLoss(lbb_target=0.1, lbb_weight=1.0, log_barrier_weight=0.0)

        lbb_constant = torch.tensor([100.0])
        loss = loss_fn.compute_lbb_loss(lbb_constant)

        assert torch.isfinite(loss), "LBB loss must be finite with very large constant"
        # Threshold penalty should be 0 (100 >> 0.1 target)
        # With log_barrier_weight=0, loss should be ~0
        assert loss.item() == pytest.approx(0.0, abs=1e-4), (
            "LBB threshold loss should be ~0 when constant greatly exceeds target"
        )

    def test_lbb_loss_with_negative_constant_clamped(self) -> None:
        """Negative LBB constant should be clamped and produce finite loss."""
        loss_fn = AlphaGalerkinLoss(lbb_target=0.1, lbb_weight=1.0)

        lbb_constant = torch.tensor([-0.5])
        loss = loss_fn.compute_lbb_loss(lbb_constant)

        assert torch.isfinite(loss), "LBB loss must be finite with negative constant (clamped)"

    def test_entropy_with_all_masked_actions(self) -> None:
        """All actions masked should handle gracefully without NaN."""
        reg = EntropyRegularizer(weight=1.0)
        batch_size, n_actions = 4, 10

        logits = torch.randn(batch_size, n_actions)
        mask = torch.zeros(batch_size, n_actions)  # All masked

        entropy = reg(logits, mask=mask)

        assert torch.isfinite(entropy), "Entropy must be finite when all actions are masked"

    def test_entropy_with_single_valid_action(self) -> None:
        """Single valid action should give zero or near-zero entropy."""
        reg = EntropyRegularizer(weight=1.0)
        batch_size, n_actions = 4, 10

        logits = torch.randn(batch_size, n_actions)
        mask = torch.zeros(batch_size, n_actions)
        for i in range(batch_size):
            mask[i, i % n_actions] = 1.0

        entropy = reg(logits, mask=mask)

        assert torch.isfinite(entropy), "Entropy must be finite with single valid action"
        # With a single valid action, the distribution is deterministic -> entropy = 0
        # The regularizer returns -weight * normalized_entropy, so result should be ~0
        assert entropy.item() == pytest.approx(0.0, abs=1e-5), (
            "Entropy should be ~0 with single valid action per sample"
        )

    def test_loss_gradient_flow_through_all_components(self) -> None:
        """Verify gradients flow through all loss components including LBB."""
        loss_fn = AlphaGalerkinLoss(lbb_weight=0.1)
        batch_size, n_actions = 4, 82

        policy_logits = torch.randn(batch_size, n_actions, requires_grad=True)
        value_raw = torch.randn(batch_size, 1, requires_grad=True)
        value = torch.tanh(value_raw)
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1
        # Create lbb_constant as a leaf tensor so .grad is populated
        lbb_constant = torch.tensor([0.4, 0.13, 0.13, 0.04], requires_grad=True)

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            lbb_constant=lbb_constant,
        )

        result.total.backward()

        assert policy_logits.grad is not None, "Gradients must flow to policy logits"
        assert value_raw.grad is not None, "Gradients must flow through tanh to value"
        assert lbb_constant.grad is not None, "Gradients must flow to LBB constant"

        assert torch.isfinite(policy_logits.grad).all(), "Policy grads must be finite"
        assert torch.isfinite(value_raw.grad).all(), "Value grads must be finite"
        assert torch.isfinite(lbb_constant.grad).all(), "LBB grads must be finite"

    def test_mixed_precision_compatibility(self) -> None:
        """Loss should work with float16 inputs (AMP training)."""
        loss_fn = AlphaGalerkinLoss()
        batch_size, n_actions = 4, 82

        policy_logits = torch.randn(batch_size, n_actions, dtype=torch.float16)
        value = torch.tanh(torch.randn(batch_size, 1, dtype=torch.float16))
        target_policy = torch.softmax(
            torch.randn(batch_size, n_actions, dtype=torch.float16), dim=-1
        )
        target_value = torch.rand(batch_size, 1, dtype=torch.float16) * 2 - 1

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total), "Loss must be finite with float16 inputs"
        assert torch.isfinite(result.policy), "Policy loss must be finite with float16"
        assert torch.isfinite(result.value), "Value loss must be finite with float16"

    @pytest.mark.parametrize("batch_size", [1, 2, 16])
    @pytest.mark.parametrize("n_actions", [2, 10, 362])
    def test_various_shapes_produce_finite_loss(self, batch_size: int, n_actions: int) -> None:
        """Loss must be finite across a wide range of tensor shapes."""
        torch.manual_seed(batch_size * 1000 + n_actions)
        loss_fn = AlphaGalerkinLoss()

        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total), (
            f"Loss not finite for shape ({batch_size}, {n_actions})"
        )

    def test_policy_loss_with_identical_logits(self) -> None:
        """All-identical logits should produce a valid uniform-like prediction."""
        loss_fn = AlphaGalerkinLoss()
        batch_size, n_actions = 4, 82

        # All logits the same -> softmax gives uniform
        policy_logits = torch.ones(batch_size, n_actions) * 5.0
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        value = torch.zeros(batch_size, 1)
        target_value = torch.zeros(batch_size, 1)

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert torch.isfinite(result.total), "Loss must be finite with constant logits"

    def test_entropy_with_very_large_logits(self) -> None:
        """Entropy regularizer should be finite with extreme logits."""
        reg = EntropyRegularizer(weight=1.0)
        batch_size, n_actions = 4, 10

        logits = torch.zeros(batch_size, n_actions)
        logits[:, 0] = 1e6  # Extremely peaked

        entropy = reg(logits)
        assert torch.isfinite(entropy), "Entropy must be finite with extreme logits"

    def test_entropy_with_very_negative_logits(self) -> None:
        """Entropy regularizer should be finite when all logits are very negative."""
        reg = EntropyRegularizer(weight=1.0)
        batch_size, n_actions = 4, 10

        logits = torch.full((batch_size, n_actions), -1e6)

        entropy = reg(logits)
        assert torch.isfinite(entropy), "Entropy must be finite with very negative logits"
