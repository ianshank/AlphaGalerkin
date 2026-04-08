"""Tests for stability monitoring (LBB condition) in Galerkin attention.

Covers:
    - StabilityGuard initialization and state tracking
    - LBB constant computation (single-head and multi-head)
    - Stability checking with threshold logic
    - Regularization loss computation
    - Forward pass (combined check + regularization)
    - Summary statistics
    - StableGalerkinInitializer initialization and adjustment
    - Edge cases: near-zero keys, identity-like keys, RuntimeError fallback
"""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.stability import StabilityGuard, StableGalerkinInitializer

# ---------------------------------------------------------------------------
# StabilityGuard -- basic construction
# ---------------------------------------------------------------------------


class TestStabilityGuardInit:
    """Tests for StabilityGuard initialization."""

    def test_default_parameters(self) -> None:
        guard = StabilityGuard()
        assert guard.beta_threshold == 1e-6
        assert guard.regularization_strength == 0.01
        assert guard.log_interval == 100
        assert guard.step_counter.item() == 0
        assert guard.min_beta_seen.item() == float("inf")
        assert guard.max_beta_seen.item() == 0.0
        assert guard._beta_history == []

    def test_custom_parameters(self) -> None:
        guard = StabilityGuard(
            beta_threshold=0.01,
            regularization_strength=0.1,
            log_interval=50,
        )
        assert guard.beta_threshold == 0.01
        assert guard.regularization_strength == 0.1
        assert guard.log_interval == 50

    def test_is_nn_module(self) -> None:
        guard = StabilityGuard()
        assert isinstance(guard, nn.Module)

    def test_buffers_registered(self) -> None:
        guard = StabilityGuard()
        buffer_names = [n for n, _ in guard.named_buffers()]
        assert "step_counter" in buffer_names
        assert "min_beta_seen" in buffer_names
        assert "max_beta_seen" in buffer_names


# ---------------------------------------------------------------------------
# compute_lbb_constant
# ---------------------------------------------------------------------------


class TestComputeLBBConstant:
    """Tests for LBB constant computation."""

    def test_output_shape_single_batch(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(1, 10, 8)
        beta = guard.compute_lbb_constant(keys)
        assert beta.shape == (1,)

    def test_output_shape_multi_batch(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(4, 10, 8)
        beta = guard.compute_lbb_constant(keys)
        assert beta.shape == (4,)

    def test_positive_beta_for_random_keys(self) -> None:
        """Random keys should produce positive singular values."""
        guard = StabilityGuard()
        keys = torch.randn(2, 16, 8)
        beta = guard.compute_lbb_constant(keys)
        assert (beta > 0).all()

    def test_near_zero_keys_give_small_beta(self) -> None:
        """Keys close to zero should yield a very small beta."""
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 8) * 1e-10
        beta = guard.compute_lbb_constant(keys)
        assert (beta < 1e-10).all()

    def test_orthogonal_keys_give_large_beta(self) -> None:
        """Orthogonal keys should produce a well-conditioned Gram matrix."""
        guard = StabilityGuard()
        # Create a batch of orthogonal key matrices
        q, _ = torch.linalg.qr(torch.randn(2, 8, 8))
        keys = q  # (2, 8, 8) -- orthonormal columns
        beta = guard.compute_lbb_constant(keys)
        # beta should be close to 1/n since K^T K / n ~ I/n
        assert (beta > 0.01).all()

    def test_deterministic_with_seed(self) -> None:
        """Same input should produce the same beta."""
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 6)
        b1 = guard.compute_lbb_constant(keys)
        b2 = guard.compute_lbb_constant(keys)
        assert torch.allclose(b1, b2)

    def test_no_nan_output(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(3, 12, 4)
        beta = guard.compute_lbb_constant(keys)
        assert not torch.isnan(beta).any()

    def test_zero_keys_fallback_or_zero(self) -> None:
        """Exactly-zero keys should not crash; beta should be ~0."""
        guard = StabilityGuard()
        keys = torch.zeros(1, 10, 4)
        beta = guard.compute_lbb_constant(keys)
        assert beta.shape == (1,)
        assert beta.item() < 1e-12


# ---------------------------------------------------------------------------
# compute_multihead_lbb
# ---------------------------------------------------------------------------


class TestComputeMultiheadLBB:
    """Tests for multi-head LBB computation."""

    def test_output_shapes(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 4, 10, 8)  # batch=2, heads=4, n=10, d=8
        beta_min, beta_per_head = guard.compute_multihead_lbb(keys)
        assert beta_min.shape == (2,)
        assert beta_per_head.shape == (2, 4)

    def test_min_across_heads(self) -> None:
        """beta_min should equal the per-head minimum."""
        guard = StabilityGuard()
        keys = torch.randn(3, 2, 8, 6)
        beta_min, beta_per_head = guard.compute_multihead_lbb(keys)
        expected_min = beta_per_head.min(dim=-1).values
        assert torch.allclose(beta_min, expected_min)

    def test_positive_values(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 4, 12, 8)
        beta_min, beta_per_head = guard.compute_multihead_lbb(keys)
        assert (beta_min > 0).all()
        assert (beta_per_head > 0).all()

    def test_single_head_matches_compute_lbb(self) -> None:
        """With 1 head, multihead should match single-head result."""
        guard = StabilityGuard()
        keys_3d = torch.randn(2, 10, 8)
        keys_4d = keys_3d.unsqueeze(1)  # add head dim
        beta_single = guard.compute_lbb_constant(keys_3d)
        beta_min, beta_per_head = guard.compute_multihead_lbb(keys_4d)
        assert torch.allclose(beta_single, beta_min, atol=1e-6)
        assert torch.allclose(beta_single, beta_per_head.squeeze(1), atol=1e-6)


# ---------------------------------------------------------------------------
# check_stability
# ---------------------------------------------------------------------------


class TestCheckStability:
    """Tests for stability checking."""

    def test_returns_tuple(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 8)
        result = guard.check_stability(keys)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_stable_keys_detected(self) -> None:
        """Well-conditioned keys should be stable."""
        guard = StabilityGuard(beta_threshold=1e-8)
        keys = torch.randn(2, 10, 8)
        is_stable, beta = guard.check_stability(keys)
        assert is_stable is True
        assert (beta > 1e-8).all()

    def test_degenerate_keys_detected_unstable(self) -> None:
        """Near-zero keys should be detected as unstable."""
        guard = StabilityGuard(beta_threshold=1.0)
        keys = torch.randn(2, 10, 8) * 1e-8
        is_stable, beta = guard.check_stability(keys)
        assert is_stable is False

    def test_step_counter_increments(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(1, 8, 4)
        assert guard.step_counter.item() == 0
        guard.check_stability(keys)
        assert guard.step_counter.item() == 1
        guard.check_stability(keys)
        assert guard.step_counter.item() == 2

    def test_min_beta_tracking(self) -> None:
        guard = StabilityGuard()
        keys_large = torch.randn(1, 10, 8) * 10.0
        guard.check_stability(keys_large)
        first_min = guard.min_beta_seen.item()

        keys_tiny = torch.randn(1, 10, 8) * 1e-6
        guard.check_stability(keys_tiny)
        assert guard.min_beta_seen.item() <= first_min

    def test_max_beta_tracking(self) -> None:
        guard = StabilityGuard()
        keys_tiny = torch.randn(1, 10, 8) * 1e-6
        guard.check_stability(keys_tiny)
        first_max = guard.max_beta_seen.item()

        keys_large = torch.randn(1, 10, 8) * 10.0
        guard.check_stability(keys_large)
        assert guard.max_beta_seen.item() >= first_max

    def test_multihead_mode(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 4, 10, 8)
        is_stable, beta = guard.check_stability(keys, multihead=True)
        assert isinstance(is_stable, bool)
        assert beta.shape == (2,)

    def test_logging_triggered_at_interval(self) -> None:
        """After log_interval steps the logger should fire (no crash)."""
        guard = StabilityGuard(log_interval=3)
        keys = torch.randn(1, 8, 4)
        for _ in range(3):
            guard.check_stability(keys)
        # beta_history should be populated after the logging call
        assert len(guard._beta_history) == 1

    def test_logging_not_triggered_before_interval(self) -> None:
        guard = StabilityGuard(log_interval=5)
        keys = torch.randn(1, 8, 4)
        for _ in range(4):
            guard.check_stability(keys)
        assert len(guard._beta_history) == 0


# ---------------------------------------------------------------------------
# _log_stability (exercised indirectly but also directly)
# ---------------------------------------------------------------------------


class TestLogStability:
    """Test the internal logging helper."""

    def test_appends_to_history(self) -> None:
        guard = StabilityGuard()
        beta = torch.tensor([0.5, 0.3])
        guard._log_stability(beta, is_stable=True)
        assert len(guard._beta_history) == 1
        # Mean of [0.5, 0.3] = 0.4
        assert abs(guard._beta_history[0] - 0.4) < 1e-6

    def test_multiple_calls_accumulate(self) -> None:
        guard = StabilityGuard()
        for _ in range(5):
            guard._log_stability(torch.tensor([1.0]), is_stable=True)
        assert len(guard._beta_history) == 5


# ---------------------------------------------------------------------------
# regularization_loss
# ---------------------------------------------------------------------------


class TestRegularizationLoss:
    """Tests for the regularization loss."""

    def test_scalar_output(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 8)
        loss = guard.regularization_loss(keys)
        assert loss.dim() == 0  # scalar

    def test_non_negative(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 8)
        loss = guard.regularization_loss(keys)
        assert loss.item() >= 0.0

    def test_higher_for_degenerate_keys(self) -> None:
        """Loss should be larger when keys are near-zero (low singular values)."""
        guard = StabilityGuard(regularization_strength=0.1, beta_threshold=1e-3)
        good_keys = torch.randn(2, 10, 8)
        bad_keys = torch.randn(2, 10, 8) * 1e-8
        loss_good = guard.regularization_loss(good_keys)
        loss_bad = guard.regularization_loss(bad_keys)
        assert loss_bad.item() >= loss_good.item()

    def test_zero_strength_gives_zero_loss(self) -> None:
        guard = StabilityGuard(regularization_strength=0.0)
        keys = torch.randn(2, 10, 8) * 1e-8  # degenerate
        loss = guard.regularization_loss(keys)
        assert loss.item() == 0.0

    def test_multihead_mode(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 4, 10, 8)
        loss = guard.regularization_loss(keys, multihead=True)
        assert loss.dim() == 0

    def test_gradient_flows(self) -> None:
        """Regularization loss should support backprop."""
        guard = StabilityGuard(regularization_strength=0.1, beta_threshold=0.1)
        keys = torch.randn(2, 10, 8, requires_grad=True)
        loss = guard.regularization_loss(keys)
        loss.backward()
        assert keys.grad is not None


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    """Tests for the combined forward pass."""

    def test_returns_three_values(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 8)
        result = guard(keys)
        assert len(result) == 3

    def test_types(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 10, 8)
        is_stable, beta, reg_loss = guard(keys)
        assert isinstance(is_stable, bool)
        assert isinstance(beta, torch.Tensor)
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.dim() == 0

    def test_multihead_forward(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(2, 4, 10, 8)
        is_stable, beta, reg_loss = guard(keys, multihead=True)
        assert beta.shape == (2,)
        assert reg_loss.dim() == 0

    def test_step_counter_increments_in_forward(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(1, 8, 4)
        guard(keys)
        assert guard.step_counter.item() == 1

    def test_forward_no_nan(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(3, 12, 6)
        _, beta, reg_loss = guard(keys)
        assert not torch.isnan(beta).any()
        assert not torch.isnan(reg_loss).any()


# ---------------------------------------------------------------------------
# get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    """Tests for summary statistics."""

    def test_initial_summary(self) -> None:
        guard = StabilityGuard(beta_threshold=0.001)
        summary = guard.get_summary()
        assert summary["total_steps"] == 0
        assert summary["min_beta_seen"] == float("inf")
        assert summary["max_beta_seen"] == 0.0
        assert summary["beta_threshold"] == 0.001
        assert summary["recent_beta_mean"] == 0.0

    def test_summary_after_steps(self) -> None:
        guard = StabilityGuard(log_interval=1)
        keys = torch.randn(1, 8, 4)
        for _ in range(5):
            guard.check_stability(keys)
        summary = guard.get_summary()
        assert summary["total_steps"] == 5
        assert summary["min_beta_seen"] < float("inf")
        assert summary["max_beta_seen"] > 0.0
        assert summary["recent_beta_mean"] > 0.0

    def test_summary_keys(self) -> None:
        guard = StabilityGuard()
        expected_keys = {
            "total_steps",
            "min_beta_seen",
            "max_beta_seen",
            "beta_threshold",
            "recent_beta_mean",
        }
        assert set(guard.get_summary().keys()) == expected_keys


# ---------------------------------------------------------------------------
# StableGalerkinInitializer
# ---------------------------------------------------------------------------


class TestStableGalerkinInitializer:
    """Tests for the stable initialization helper."""

    def test_default_parameters(self) -> None:
        init = StableGalerkinInitializer()
        assert init.beta_target == 0.1
        assert init.max_iterations == 10
        assert isinstance(init.stability_guard, StabilityGuard)

    def test_custom_parameters(self) -> None:
        init = StableGalerkinInitializer(beta_target=0.5, max_iterations=5)
        assert init.beta_target == 0.5
        assert init.max_iterations == 5
        # stability_guard threshold is beta_target / 10
        assert init.stability_guard.beta_threshold == 0.05

    def test_initialize_projection(self) -> None:
        """Weight should be modified in-place."""
        init = StableGalerkinInitializer()
        weight = torch.empty(8, 16)
        init.initialize_projection(weight, d_key=8)
        # After initialization, weight should have reasonable scale
        assert weight.std().item() > 0
        assert not torch.isnan(weight).any()

    def test_initialize_projection_scale(self) -> None:
        """Verify the scaling factor 1/sqrt(d_key)."""
        init = StableGalerkinInitializer()
        weight = torch.empty(16, 16)
        init.initialize_projection(weight, d_key=16)
        # Orthogonal init gives unit-scale, then scaled by 1/sqrt(16) = 0.25
        # So std should be roughly 0.25
        assert weight.std().item() < 1.0

    def test_verify_and_adjust_with_to_k(self) -> None:
        """Module with to_k attribute should be adjusted."""
        init = StableGalerkinInitializer(beta_target=1e-8, max_iterations=3)
        module = nn.Module()
        module.to_k = nn.Linear(8, 8, bias=False)
        sample_input = torch.randn(2, 10, 8)
        result = init.verify_and_adjust(module, sample_input)
        assert isinstance(result, bool)

    def test_verify_and_adjust_with_to_key(self) -> None:
        """Module with to_key attribute should be adjusted."""
        init = StableGalerkinInitializer(beta_target=1e-8, max_iterations=3)
        module = nn.Module()
        module.to_key = nn.Linear(8, 8, bias=False)
        sample_input = torch.randn(2, 10, 8)
        result = init.verify_and_adjust(module, sample_input)
        assert isinstance(result, bool)

    def test_verify_raises_for_missing_attribute(self) -> None:
        """Module without to_k or to_key should raise ValueError."""
        init = StableGalerkinInitializer()
        module = nn.Module()
        sample_input = torch.randn(2, 10, 8)
        import pytest

        with pytest.raises(ValueError, match="to_k"):
            init.verify_and_adjust(module, sample_input)

    def test_verify_returns_true_for_easy_target(self) -> None:
        """With a very low target, verification should succeed quickly."""
        init = StableGalerkinInitializer(beta_target=1e-12, max_iterations=5)
        module = nn.Module()
        module.to_k = nn.Linear(8, 8, bias=False)
        nn.init.orthogonal_(module.to_k.weight)
        sample_input = torch.randn(2, 10, 8)
        result = init.verify_and_adjust(module, sample_input)
        assert result is True

    def test_verify_returns_false_for_impossible_target(self) -> None:
        """With an impossibly high target, verification should fail."""
        init = StableGalerkinInitializer(beta_target=1e10, max_iterations=3)
        module = nn.Module()
        module.to_k = nn.Linear(4, 4, bias=False)
        sample_input = torch.randn(1, 6, 4)
        result = init.verify_and_adjust(module, sample_input)
        assert result is False

    def test_adjust_for_stability_scales_weights(self) -> None:
        """_adjust_for_stability should scale weights up."""
        init = StableGalerkinInitializer(beta_target=1.0)
        module = nn.Module()
        module.to_k = nn.Linear(4, 4, bias=False)
        # Set small weights
        module.to_k.weight.data.fill_(0.01)
        original_norm = module.to_k.weight.data.norm().item()

        current_beta = torch.tensor([0.001])
        init._adjust_for_stability(module, current_beta)
        new_norm = module.to_k.weight.data.norm().item()
        # Weights should have been scaled up
        assert new_norm > original_norm

    def test_adjust_for_stability_with_to_key(self) -> None:
        init = StableGalerkinInitializer(beta_target=1.0)
        module = nn.Module()
        module.to_key = nn.Linear(4, 4, bias=False)
        module.to_key.weight.data.fill_(0.01)
        original_norm = module.to_key.weight.data.norm().item()
        init._adjust_for_stability(module, torch.tensor([0.001]))
        assert module.to_key.weight.data.norm().item() > original_norm

    def test_adjust_for_stability_no_attribute(self) -> None:
        """Should be a no-op when neither attribute exists."""
        init = StableGalerkinInitializer()
        module = nn.Module()
        # Should not raise
        init._adjust_for_stability(module, torch.tensor([0.001]))

    def test_adjust_clamps_scale_factor(self) -> None:
        """Scale factor should be clamped to [1.0, 2.0]."""
        init = StableGalerkinInitializer(beta_target=100.0)
        module = nn.Module()
        module.to_k = nn.Linear(4, 4, bias=False)
        module.to_k.weight.data.fill_(1.0)
        original_norm = module.to_k.weight.data.norm().item()

        # With a very low current_beta and high target, scale_factor would
        # be huge, but it gets clamped to 2.0
        init._adjust_for_stability(module, torch.tensor([1e-10]))
        new_norm = module.to_k.weight.data.norm().item()
        # Should be at most 2x the original
        assert new_norm <= original_norm * 2.0 + 1e-6


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_svd_fallback_on_singular_matrix(self) -> None:
        """Verify behavior when SVD might struggle with perfectly singular data."""
        guard = StabilityGuard()
        # All-same rows should yield rank-1 Gram matrix
        keys = torch.ones(1, 10, 4)
        beta = guard.compute_lbb_constant(keys)
        assert beta.shape == (1,)
        # Beta should be very small (min singular value of rank-1 matrix)
        # The min SV of KTK/n will be 0 for the degenerate directions
        assert beta.item() < 1e-4

    def test_large_batch(self) -> None:
        guard = StabilityGuard()
        keys = torch.randn(32, 8, 4)
        beta = guard.compute_lbb_constant(keys)
        assert beta.shape == (32,)
        assert (beta > 0).all()

    def test_state_dict_save_load(self) -> None:
        """Buffers should survive state_dict round-trip."""
        guard = StabilityGuard()
        keys = torch.randn(1, 8, 4)
        guard.check_stability(keys)

        state = guard.state_dict()
        guard2 = StabilityGuard()
        guard2.load_state_dict(state)
        assert guard2.step_counter.item() == 1
        assert guard2.min_beta_seen.item() == guard.min_beta_seen.item()

    def test_sequential_stability_calls(self) -> None:
        """Multiple check_stability calls should consistently track state."""
        guard = StabilityGuard(log_interval=2)
        keys = torch.randn(1, 8, 4)
        for i in range(6):
            guard.check_stability(keys)
        assert guard.step_counter.item() == 6
        # Logging should have fired at steps 2, 4, 6
        assert len(guard._beta_history) == 3
