"""Tests for StabilityGuard."""

from __future__ import annotations

import pytest
import torch

from src.alphagalerkin.core.constants import MIN_SINGULAR_VALUE
from src.alphagalerkin.nn.stability_guard import StabilityGuard


class TestStabilityGuard:
    """Unit tests for LBB stability monitor."""

    def test_well_conditioned_matrix_zero_loss(self) -> None:
        """Well-conditioned matrix produces near-zero loss."""
        guard = StabilityGuard(beta=1e-6)
        # Identity-like matrix has sigma_min = 1.0
        ktv = torch.eye(16).unsqueeze(0)
        loss = guard(ktv)
        assert float(loss.item()) == 0.0

    def test_ill_conditioned_matrix_positive_loss(self) -> None:
        """Ill-conditioned matrix triggers penalty."""
        guard = StabilityGuard(beta=0.5)
        # Near-singular matrix
        ktv = torch.zeros(8, 8)
        ktv[0, 0] = 1.0  # Only one non-zero singular value
        loss = guard(ktv)
        assert float(loss.item()) > 0.0

    def test_diagnostics_populated(self) -> None:
        """After forward, diagnostics dict is populated."""
        guard = StabilityGuard()
        ktv = torch.eye(4)
        guard(ktv)
        diag = guard.diagnostics
        assert "sigma_min" in diag
        assert "sigma_max" in diag
        assert "condition_number" in diag
        assert "num_singular_values" in diag

    def test_is_stable_with_identity(self) -> None:
        """Identity matrix is stable."""
        guard = StabilityGuard(beta=1e-6)
        guard(torch.eye(4))
        assert guard.is_stable()

    def test_penalty_weight_scales_loss(self) -> None:
        """Higher penalty weight increases loss proportionally."""
        ktv = torch.zeros(4, 4)
        ktv[0, 0] = 1.0  # singular

        guard1 = StabilityGuard(beta=0.5, penalty_weight=1.0)
        guard2 = StabilityGuard(beta=0.5, penalty_weight=2.0)
        loss1 = float(guard1(ktv).item())
        loss2 = float(guard2(ktv).item())
        assert loss2 == pytest.approx(2.0 * loss1, rel=1e-5)

    def test_batched_input(self) -> None:
        """Handles batched KTV matrices."""
        guard = StabilityGuard()
        ktv = torch.randn(4, 2, 8, 8)  # batch of KTV matrices
        loss = guard(ktv)
        assert loss.shape == ()  # scalar

    def test_gradient_flows(self) -> None:
        """Gradients propagate through the guard."""
        guard = StabilityGuard(beta=1.0)
        ktv = torch.randn(4, 4) * 0.01
        ktv = ktv.detach().requires_grad_(True)
        loss = guard(ktv)
        loss.backward()
        assert ktv.grad is not None

    def test_default_beta_from_constants(self) -> None:
        """Default beta uses MIN_SINGULAR_VALUE from constants."""
        guard = StabilityGuard()
        assert guard.beta == MIN_SINGULAR_VALUE

    def test_identity_batched_zero_loss(self) -> None:
        """Batched identity matrices produce zero loss."""
        guard = StabilityGuard(beta=0.01)
        ktv = torch.eye(8).unsqueeze(0).unsqueeze(0).expand(2, 4, 8, 8)
        loss = guard(ktv)
        assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)
