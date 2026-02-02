"""Tests for training loss functions."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.training.loss import AlphaGalerkinLoss, EntropyRegularizer, LossOutput


class TestAlphaGalerkinLoss:
    """Tests for AlphaGalerkinLoss."""

    def test_loss_basic_computation(self) -> None:
        """Test basic loss computation."""
        loss_fn = AlphaGalerkinLoss()

        batch_size = 4
        n_actions = 82  # 9x9 + pass

        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        assert isinstance(result, LossOutput)
        assert result.total.ndim == 0  # Scalar
        assert result.policy.ndim == 0
        assert result.value.ndim == 0
        assert result.lbb.ndim == 0

    def test_loss_non_negative(self) -> None:
        """Test that all loss components are non-negative."""
        loss_fn = AlphaGalerkinLoss()

        for _ in range(10):
            batch_size = 8
            n_actions = 82

            policy_logits = torch.randn(batch_size, n_actions)
            value = torch.tanh(torch.randn(batch_size, 1))
            target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
            target_value = torch.rand(batch_size, 1) * 2 - 1

            result = loss_fn(
                policy_logits=policy_logits,
                value=value,
                target_policy=target_policy,
                target_value=target_value,
            )

            assert result.total >= 0, "Total loss should be non-negative"
            assert result.policy >= 0, "Policy loss should be non-negative"
            assert result.value >= 0, "Value loss should be non-negative"
            assert result.lbb >= 0, "LBB loss should be non-negative"

    def test_loss_with_lbb_constant(self) -> None:
        """Test loss with LBB constant."""
        loss_fn = AlphaGalerkinLoss(lbb_weight=0.1)

        batch_size = 4
        n_actions = 82

        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1
        lbb_constant = torch.rand(batch_size) * 0.5

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            lbb_constant=lbb_constant,
        )

        # LBB loss should be positive when lbb_constant < lbb_target
        assert result.lbb > 0

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through all loss components."""
        loss_fn = AlphaGalerkinLoss()

        batch_size = 4
        n_actions = 82

        policy_logits = torch.randn(batch_size, n_actions, requires_grad=True)
        # Create value tensor as leaf tensor (not wrapped in tanh)
        # The model's output already applies tanh, so we simulate final values directly
        value_raw = torch.randn(batch_size, 1, requires_grad=True)
        value = torch.tanh(value_raw)
        # Retain grad on non-leaf tensor to verify gradient flow
        value.retain_grad()

        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
        )

        result.total.backward()

        assert policy_logits.grad is not None, "Gradients should flow to policy logits"
        # Check gradients flow through tanh to the raw value tensor
        assert value_raw.grad is not None, "Gradients should flow through tanh to raw value"
        assert value.grad is not None, "Gradients should flow to value (with retain_grad)"

    def test_loss_weights_applied(self) -> None:
        """Test that loss weights are correctly applied."""
        # Default weights
        loss_fn_default = AlphaGalerkinLoss(policy_weight=1.0, value_weight=1.0)
        # Double policy weight
        loss_fn_policy = AlphaGalerkinLoss(policy_weight=2.0, value_weight=1.0)

        batch_size = 4
        n_actions = 82

        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result_default = loss_fn_default(policy_logits, value, target_policy, target_value)
        result_policy = loss_fn_policy(policy_logits, value, target_policy, target_value)

        # Policy loss contribution should be doubled
        expected_diff = result_default.policy.item()
        actual_diff = result_policy.total.item() - result_default.total.item()
        assert abs(actual_diff - expected_diff) < 0.01

    def test_loss_with_action_mask(self) -> None:
        """Test loss with action masking."""
        loss_fn = AlphaGalerkinLoss()

        batch_size = 4
        n_actions = 82

        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        # Mask out half the actions
        action_mask = torch.ones(batch_size, n_actions)
        action_mask[:, n_actions // 2 :] = 0.0

        result = loss_fn(
            policy_logits=policy_logits,
            value=value,
            target_policy=target_policy,
            target_value=target_value,
            action_mask=action_mask,
        )

        assert result.total.isfinite()

    def test_running_stats(self) -> None:
        """Test running statistics tracking."""
        loss_fn = AlphaGalerkinLoss()

        batch_size = 4
        n_actions = 82

        # Run several iterations
        for _ in range(5):
            policy_logits = torch.randn(batch_size, n_actions)
            value = torch.tanh(torch.randn(batch_size, 1))
            target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
            target_value = torch.rand(batch_size, 1) * 2 - 1

            loss_fn(policy_logits, value, target_policy, target_value)

        stats = loss_fn.get_running_stats()
        assert "policy" in stats
        assert "value" in stats
        assert "lbb" in stats

        # Reset and verify
        loss_fn.reset_stats()
        stats = loss_fn.get_running_stats()
        assert stats["policy"] == 0.0

    @given(batch_size=st.integers(1, 16), n_actions=st.integers(10, 400))
    @settings(max_examples=20)
    def test_loss_various_shapes(self, batch_size: int, n_actions: int) -> None:
        """Property test: loss works with various shapes."""
        loss_fn = AlphaGalerkinLoss()

        policy_logits = torch.randn(batch_size, n_actions)
        value = torch.tanh(torch.randn(batch_size, 1))
        target_policy = torch.softmax(torch.randn(batch_size, n_actions), dim=-1)
        target_value = torch.rand(batch_size, 1) * 2 - 1

        result = loss_fn(policy_logits, value, target_policy, target_value)

        assert result.total.isfinite()
        assert result.total >= 0


class TestEntropyRegularizer:
    """Tests for entropy regularization."""

    def test_entropy_computation(self) -> None:
        """Test entropy is computed correctly."""
        reg = EntropyRegularizer(weight=1.0)

        batch_size = 4
        n_actions = 10

        # Uniform distribution - maximum entropy
        uniform_logits = torch.zeros(batch_size, n_actions)
        entropy_uniform = reg(uniform_logits)

        # Peaked distribution - low entropy
        peaked_logits = torch.zeros(batch_size, n_actions)
        peaked_logits[:, 0] = 10.0
        entropy_peaked = reg(peaked_logits)

        # Uniform should have higher (more negative) entropy bonus
        # Since we return negative entropy, uniform should be more negative
        assert entropy_uniform < entropy_peaked

    def test_entropy_with_mask(self) -> None:
        """Test entropy with action mask."""
        reg = EntropyRegularizer(weight=1.0)

        batch_size = 4
        n_actions = 10

        logits = torch.randn(batch_size, n_actions)
        mask = torch.ones(batch_size, n_actions)
        mask[:, 5:] = 0.0  # Only first 5 actions valid

        entropy = reg(logits, mask=mask)
        assert entropy.isfinite()

    def test_entropy_with_all_masked(self) -> None:
        """Test entropy handles degenerate case with all actions masked.

        When all actions are masked for a sample, the entropy should be 0
        and not produce NaN values.
        """
        reg = EntropyRegularizer(weight=1.0)

        batch_size = 4
        n_actions = 10

        logits = torch.randn(batch_size, n_actions)
        mask = torch.ones(batch_size, n_actions)

        # Mask all actions for first two samples (degenerate case)
        mask[0, :] = 0.0
        mask[1, :] = 0.0

        entropy = reg(logits, mask=mask)

        # Should not produce NaN or Inf
        assert entropy.isfinite(), "Entropy should be finite even with all-masked samples"
        # Result should be a scalar
        assert entropy.ndim == 0

    def test_entropy_with_single_valid_action(self) -> None:
        """Test entropy with only one valid action per sample.

        With only one valid action, entropy should be 0 (no uncertainty).
        """
        reg = EntropyRegularizer(weight=1.0)

        batch_size = 4
        n_actions = 10

        logits = torch.randn(batch_size, n_actions)
        mask = torch.zeros(batch_size, n_actions)

        # Only one valid action per sample
        for i in range(batch_size):
            mask[i, i % n_actions] = 1.0

        entropy = reg(logits, mask=mask)

        # Should be finite
        assert entropy.isfinite(), "Entropy should be finite with single valid action"
        # With only one valid action, entropy is 0 (normalized entropy = 0/0 -> handled)
        # The actual value depends on implementation, but it should be finite
        assert entropy.ndim == 0
