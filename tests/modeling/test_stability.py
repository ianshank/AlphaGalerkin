"""Coverage tests for LBB stability monitoring.

Tests cover:
- StabilityGuard: LBB constant computation, stability checking, regularization
- StableGalerkinInitializer: Weight initialization and verification
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.modeling.stability import StabilityGuard, StableGalerkinInitializer

SEED = 42
BATCH_SIZE = 2
SEQ_LEN = 16
D_KEY = 8
N_HEADS = 2


@pytest.fixture
def stability_guard() -> StabilityGuard:
    return StabilityGuard(
        beta_threshold=1e-6,
        regularization_strength=0.01,
        log_interval=5,
    )


@pytest.fixture
def keys() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_KEY)


@pytest.fixture
def multihead_keys() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, D_KEY)


class TestStabilityGuard:
    """Tests for StabilityGuard module."""

    def test_initialization(self, stability_guard: StabilityGuard) -> None:
        assert stability_guard.beta_threshold == 1e-6
        assert stability_guard.regularization_strength == 0.01
        assert stability_guard.log_interval == 5
        assert stability_guard.step_counter.item() == 0
        assert stability_guard.min_beta_seen.item() == float("inf")
        assert stability_guard.max_beta_seen.item() == 0.0

    def test_compute_lbb_constant(
        self, stability_guard: StabilityGuard, keys: torch.Tensor
    ) -> None:
        beta = stability_guard.compute_lbb_constant(keys)
        assert beta.shape == (BATCH_SIZE,)
        assert (beta >= 0).all()

    def test_compute_lbb_constant_orthogonal(self, stability_guard: StabilityGuard) -> None:
        """Orthogonal keys should have high stability."""
        torch.manual_seed(SEED)
        # Create orthogonal-like keys
        keys = torch.eye(D_KEY).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
        # Pad to have more points than dimensions
        keys = torch.cat([keys, keys], dim=1)
        beta = stability_guard.compute_lbb_constant(keys)
        assert (beta > 0).all()

    def test_compute_multihead_lbb(
        self, stability_guard: StabilityGuard, multihead_keys: torch.Tensor
    ) -> None:
        beta_min, beta_per_head = stability_guard.compute_multihead_lbb(multihead_keys)
        assert beta_min.shape == (BATCH_SIZE,)
        assert beta_per_head.shape == (BATCH_SIZE, N_HEADS)
        # Min across heads should match
        expected_min = beta_per_head.min(dim=-1).values
        torch.testing.assert_close(beta_min, expected_min)

    def test_check_stability(
        self, stability_guard: StabilityGuard, keys: torch.Tensor
    ) -> None:
        is_stable, beta = stability_guard.check_stability(keys)
        assert isinstance(is_stable, bool)
        assert beta.shape == (BATCH_SIZE,)
        # Step counter should increment
        assert stability_guard.step_counter.item() == 1

    def test_check_stability_multihead(
        self, stability_guard: StabilityGuard, multihead_keys: torch.Tensor
    ) -> None:
        is_stable, beta = stability_guard.check_stability(multihead_keys, multihead=True)
        assert isinstance(is_stable, bool)
        assert beta.shape == (BATCH_SIZE,)

    def test_check_stability_updates_tracking(
        self, stability_guard: StabilityGuard, keys: torch.Tensor
    ) -> None:
        stability_guard.check_stability(keys)
        assert stability_guard.min_beta_seen.item() < float("inf")
        assert stability_guard.max_beta_seen.item() > 0.0

    def test_check_stability_logging_interval(
        self, stability_guard: StabilityGuard, keys: torch.Tensor
    ) -> None:
        """Test that logging happens at the right interval."""
        for _ in range(stability_guard.log_interval):
            stability_guard.check_stability(keys)
        assert stability_guard.step_counter.item() == stability_guard.log_interval

    def test_regularization_loss(
        self, stability_guard: StabilityGuard, keys: torch.Tensor
    ) -> None:
        loss = stability_guard.regularization_loss(keys)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_regularization_loss_multihead(
        self, stability_guard: StabilityGuard, multihead_keys: torch.Tensor
    ) -> None:
        loss = stability_guard.regularization_loss(multihead_keys, multihead=True)
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_forward(
        self, stability_guard: StabilityGuard, keys: torch.Tensor
    ) -> None:
        is_stable, beta, reg_loss = stability_guard(keys)
        assert isinstance(is_stable, bool)
        assert beta.shape == (BATCH_SIZE,)
        assert reg_loss.shape == ()

    def test_forward_multihead(
        self, stability_guard: StabilityGuard, multihead_keys: torch.Tensor
    ) -> None:
        is_stable, beta, reg_loss = stability_guard(multihead_keys, multihead=True)
        assert isinstance(is_stable, bool)
        assert beta.shape == (BATCH_SIZE,)
        assert reg_loss.shape == ()

    def test_get_summary(self, stability_guard: StabilityGuard, keys: torch.Tensor) -> None:
        stability_guard.check_stability(keys)
        summary = stability_guard.get_summary()
        assert "total_steps" in summary
        assert "min_beta_seen" in summary
        assert "max_beta_seen" in summary
        assert "beta_threshold" in summary
        assert "recent_beta_mean" in summary
        assert summary["total_steps"] == 1

    def test_get_summary_empty_history(self, stability_guard: StabilityGuard) -> None:
        summary = stability_guard.get_summary()
        assert summary["recent_beta_mean"] == 0.0

    def test_gradient_flow(self, stability_guard: StabilityGuard) -> None:
        """Test that gradients flow through regularization loss."""
        torch.manual_seed(SEED)
        keys = torch.randn(BATCH_SIZE, SEQ_LEN, D_KEY, requires_grad=True)
        loss = stability_guard.regularization_loss(keys)
        loss.backward()
        # Gradient should exist (may be zero if no violation)
        assert keys.grad is not None


class TestStableGalerkinInitializer:
    """Tests for StableGalerkinInitializer."""

    def test_initialization(self) -> None:
        init = StableGalerkinInitializer(beta_target=0.1, max_iterations=5)
        assert init.beta_target == 0.1
        assert init.max_iterations == 5

    def test_initialize_projection(self) -> None:
        init = StableGalerkinInitializer()
        weight = torch.empty(D_KEY, D_KEY)
        init.initialize_projection(weight, d_key=D_KEY)
        # Weight should be modified
        assert weight.shape == (D_KEY, D_KEY)
        # Check it's not all zeros
        assert weight.abs().sum() > 0

    def test_verify_and_adjust_with_to_key(self) -> None:
        torch.manual_seed(SEED)
        init = StableGalerkinInitializer(beta_target=0.001, max_iterations=5)

        module = nn.Module()
        module.to_key = nn.Linear(D_KEY, D_KEY)
        nn.init.orthogonal_(module.to_key.weight)

        sample_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_KEY)
        result = init.verify_and_adjust(module, sample_input)
        assert isinstance(result, bool)

    def test_verify_and_adjust_with_to_k(self) -> None:
        torch.manual_seed(SEED)
        init = StableGalerkinInitializer(beta_target=0.001, max_iterations=5)

        module = nn.Module()
        module.to_k = nn.Linear(D_KEY, D_KEY)
        nn.init.orthogonal_(module.to_k.weight)

        sample_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_KEY)
        result = init.verify_and_adjust(module, sample_input)
        assert isinstance(result, bool)

    def test_verify_and_adjust_no_key_raises(self) -> None:
        torch.manual_seed(SEED)
        init = StableGalerkinInitializer(beta_target=0.001, max_iterations=5)
        module = nn.Module()  # No to_key or to_k
        sample_input = torch.randn(BATCH_SIZE, SEQ_LEN, D_KEY)
        with pytest.raises(ValueError, match="to_k"):
            init.verify_and_adjust(module, sample_input)
