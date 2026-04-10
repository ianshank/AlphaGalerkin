"""Tests for operator learning loss functions (src.training.losses.operator)."""

from __future__ import annotations

import pytest
import torch

from src.training.losses import get_loss
from src.training.losses.operator import H1Loss, L2RelativeLoss, MSELoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
BOARD_SIZE = 5


# ---------------------------------------------------------------------------
# L2RelativeLoss tests
# ---------------------------------------------------------------------------


class TestL2RelativeLoss:
    """Test L2RelativeLoss computation and properties."""

    def test_zero_error(self) -> None:
        """Identical pred and target produce zero loss."""
        loss_fn = L2RelativeLoss()
        x = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_error(self) -> None:
        """Different pred and target produce positive loss."""
        loss_fn = L2RelativeLoss()
        pred = torch.zeros(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        target = torch.ones(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss > 0

    def test_scale_invariance(self) -> None:
        """Relative L2 is approximately scale-invariant."""
        loss_fn = L2RelativeLoss()
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE) + 2

        loss_orig = loss_fn(pred, target)
        loss_scaled = loss_fn(pred * 10, target * 10)

        torch.testing.assert_close(loss_orig, loss_scaled, atol=1e-4, rtol=1e-4)

    def test_reduction_mean(self) -> None:
        """Mean reduction returns scalar."""
        loss_fn = L2RelativeLoss(reduction="mean")
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0

    def test_reduction_sum(self) -> None:
        """Sum reduction returns scalar."""
        loss_fn = L2RelativeLoss(reduction="sum")
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0

    def test_reduction_none(self) -> None:
        """None reduction returns per-sample values."""
        loss_fn = L2RelativeLoss(reduction="none")
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.shape == (BATCH_SIZE,)

    def test_sum_ge_mean(self) -> None:
        """Sum reduction >= mean reduction for batch > 1."""
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)

        loss_mean = L2RelativeLoss(reduction="mean")(pred, target)
        loss_sum = L2RelativeLoss(reduction="sum")(pred, target)

        assert loss_sum >= loss_mean

    def test_non_negative(self) -> None:
        """L2 relative loss is always non-negative."""
        loss_fn = L2RelativeLoss()
        for _ in range(20):
            pred = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
            target = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
            loss = loss_fn(pred, target)
            assert loss >= 0

    def test_gradient_flow(self) -> None:
        """Gradients flow through L2RelativeLoss."""
        loss_fn = L2RelativeLoss()
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE, requires_grad=True)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_eps_prevents_division_by_zero(self) -> None:
        """Epsilon prevents NaN when target is zero."""
        loss_fn = L2RelativeLoss(eps=1e-8)
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE)
        target = torch.zeros(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.isfinite()

    def test_custom_eps(self) -> None:
        """Custom epsilon is stored correctly."""
        loss_fn = L2RelativeLoss(eps=1e-4)
        assert loss_fn.eps == 1e-4


# ---------------------------------------------------------------------------
# H1Loss tests
# ---------------------------------------------------------------------------


class TestH1Loss:
    """Test H1 Sobolev loss computation."""

    def test_zero_error(self) -> None:
        """Identical pred and target produce zero loss."""
        loss_fn = H1Loss()
        x = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_error(self) -> None:
        """Different pred and target produce positive loss."""
        loss_fn = H1Loss()
        pred = torch.zeros(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        target = torch.ones(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss > 0

    def test_gradient_term_effect(self) -> None:
        """Non-zero lambda_grad adds gradient penalty."""
        pred = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)

        loss_no_grad = H1Loss(lambda_grad=0.0)(pred, target)
        loss_with_grad = H1Loss(lambda_grad=1.0)(pred, target)

        assert loss_with_grad >= loss_no_grad - 1e-6

    def test_lambda_grad_zero_equals_l2(self) -> None:
        """H1 with lambda_grad=0 is equivalent to L2 relative loss."""
        pred = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)

        h1_loss = H1Loss(lambda_grad=0.0)(pred, target)
        l2_loss = L2RelativeLoss()(pred, target)

        torch.testing.assert_close(h1_loss, l2_loss, atol=1e-6, rtol=1e-6)

    def test_gradient_flow(self) -> None:
        """Gradients flow through H1Loss."""
        loss_fn = H1Loss(lambda_grad=0.1)
        pred = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
        target = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_non_negative(self) -> None:
        """H1 loss is always non-negative."""
        loss_fn = H1Loss(lambda_grad=0.5)
        for _ in range(10):
            pred = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
            target = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
            loss = loss_fn(pred, target)
            assert loss >= 0

    def test_finite_gradient(self) -> None:
        """H1Loss gradient computation produces finite values."""
        loss_fn = H1Loss(lambda_grad=0.1)
        x = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        dx, dy = loss_fn._compute_gradient(x)

        assert dx.isfinite().all()
        assert dy.isfinite().all()
        assert dx.shape == x.shape
        assert dy.shape == x.shape

    def test_gradient_of_constant_is_zero(self) -> None:
        """Finite-difference gradient of constant field is zero."""
        loss_fn = H1Loss()
        x = torch.ones(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE) * 3.14
        dx, dy = loss_fn._compute_gradient(x)

        # All differences should be zero except padding
        assert torch.allclose(dx[:, :, :-1, :], torch.zeros_like(dx[:, :, :-1, :]))
        assert torch.allclose(dy[:, :, :, :-1], torch.zeros_like(dy[:, :, :, :-1]))

    def test_higher_lambda_grad_larger_loss(self) -> None:
        """Higher lambda_grad should produce >= loss for non-identical gradients."""
        pred = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)

        loss_low = H1Loss(lambda_grad=0.1)(pred, target)
        loss_high = H1Loss(lambda_grad=10.0)(pred, target)

        assert loss_high >= loss_low - 1e-6


# ---------------------------------------------------------------------------
# MSELoss tests
# ---------------------------------------------------------------------------


class TestMSELoss:
    """Test MSELoss wrapper."""

    def test_zero_error(self) -> None:
        """Identical inputs produce zero loss."""
        loss_fn = MSELoss()
        x = torch.randn(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_error(self) -> None:
        """Different inputs produce positive loss."""
        loss_fn = MSELoss()
        pred = torch.zeros(BATCH_SIZE, BOARD_SIZE)
        target = torch.ones(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)

    def test_gradient_flow(self) -> None:
        """Gradients flow through MSELoss."""
        loss_fn = MSELoss()
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE, requires_grad=True)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_symmetric(self) -> None:
        """MSE is symmetric: MSE(a, b) == MSE(b, a)."""
        loss_fn = MSELoss()
        a = torch.randn(BATCH_SIZE, BOARD_SIZE)
        b = torch.randn(BATCH_SIZE, BOARD_SIZE)
        torch.testing.assert_close(loss_fn(a, b), loss_fn(b, a))

    def test_non_negative(self) -> None:
        """MSE is always non-negative."""
        loss_fn = MSELoss()
        for _ in range(10):
            pred = torch.randn(BATCH_SIZE, BOARD_SIZE)
            target = torch.randn(BATCH_SIZE, BOARD_SIZE)
            loss = loss_fn(pred, target)
            assert loss >= 0

    def test_scalar_output(self) -> None:
        """MSE returns a scalar tensor."""
        loss_fn = MSELoss()
        pred = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.ndim == 0

    def test_known_value(self) -> None:
        """MSE of [0,0] vs [1,1] is 1.0."""
        loss_fn = MSELoss()
        pred = torch.zeros(1, 2)
        target = torch.ones(1, 2)
        loss = loss_fn(pred, target)
        assert loss.item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Loss registry (get_loss) tests
# ---------------------------------------------------------------------------


class TestLossRegistry:
    """Test loss registry and get_loss factory."""

    def test_get_l2_relative(self) -> None:
        """get_loss('l2_relative') returns L2RelativeLoss."""
        loss = get_loss("l2_relative")
        assert isinstance(loss, L2RelativeLoss)

    def test_get_h1(self) -> None:
        """get_loss('h1') returns H1Loss."""
        loss = get_loss("h1")
        assert isinstance(loss, H1Loss)

    def test_get_mse(self) -> None:
        """get_loss('mse') returns MSELoss."""
        loss = get_loss("mse")
        assert isinstance(loss, MSELoss)

    def test_get_alphagalerkin(self) -> None:
        """get_loss('alphagalerkin') returns AlphaGalerkinLoss."""
        from src.training.losses.alphagalerkin import AlphaGalerkinLoss

        loss = get_loss("alphagalerkin")
        assert isinstance(loss, AlphaGalerkinLoss)

    def test_get_unknown_raises(self) -> None:
        """get_loss with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("nonexistent_loss_xyz")

    def test_get_with_kwargs(self) -> None:
        """get_loss passes kwargs to constructor."""
        loss = get_loss("l2_relative", eps=1e-4)
        assert loss.eps == 1e-4

    def test_alias_l2(self) -> None:
        """'l2' alias resolves to L2RelativeLoss."""
        loss = get_loss("l2")
        assert isinstance(loss, L2RelativeLoss)

    def test_alias_sobolev(self) -> None:
        """'sobolev' alias resolves to H1Loss."""
        loss = get_loss("sobolev")
        assert isinstance(loss, H1Loss)

    def test_registry_returns_callable(self) -> None:
        """All registered losses are callable."""
        for name in ["l2_relative", "h1", "mse"]:
            loss = get_loss(name)
            assert callable(loss)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for operator losses."""

    def test_l2_zero_inputs(self) -> None:
        """L2 loss with all-zero inputs is finite."""
        loss_fn = L2RelativeLoss()
        z = torch.zeros(BATCH_SIZE, BOARD_SIZE)
        loss = loss_fn(z, z)
        assert loss.isfinite()
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_h1_single_pixel(self) -> None:
        """H1 loss works with 1x1 spatial dims (trivial gradient)."""
        loss_fn = H1Loss(lambda_grad=0.1)
        pred = torch.randn(BATCH_SIZE, 1, 1, 1)
        target = torch.randn(BATCH_SIZE, 1, 1, 1)
        loss = loss_fn(pred, target)
        assert loss.isfinite()

    def test_l2_batch_size_one(self) -> None:
        """L2 loss works with batch size 1."""
        loss_fn = L2RelativeLoss()
        pred = torch.randn(1, BOARD_SIZE)
        target = torch.randn(1, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.isfinite()

    def test_mse_large_values(self) -> None:
        """MSE handles large values without overflow in float32."""
        loss_fn = MSELoss()
        pred = torch.ones(BATCH_SIZE, BOARD_SIZE) * 1e4
        target = torch.ones(BATCH_SIZE, BOARD_SIZE) * 1e4 + 1
        loss = loss_fn(pred, target)
        assert loss.isfinite()
        assert loss.item() == pytest.approx(1.0, abs=1e-3)

    def test_l2_high_dimensional(self) -> None:
        """L2 loss handles high-dimensional inputs by flattening."""
        loss_fn = L2RelativeLoss()
        pred = torch.randn(BATCH_SIZE, 2, 3, BOARD_SIZE, BOARD_SIZE)
        target = torch.randn(BATCH_SIZE, 2, 3, BOARD_SIZE, BOARD_SIZE)
        loss = loss_fn(pred, target)
        assert loss.isfinite()
        assert loss.ndim == 0

    def test_all_losses_scalar_output(self) -> None:
        """All losses with default reduction produce scalar output."""
        pred_2d = torch.randn(BATCH_SIZE, BOARD_SIZE)
        target_2d = torch.randn(BATCH_SIZE, BOARD_SIZE)
        pred_4d = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)
        target_4d = torch.randn(BATCH_SIZE, 1, BOARD_SIZE, BOARD_SIZE)

        assert L2RelativeLoss()(pred_2d, target_2d).ndim == 0
        assert H1Loss()(pred_4d, target_4d).ndim == 0
        assert MSELoss()(pred_2d, target_2d).ndim == 0


class TestConstantBackedDefaults:
    """Verify that constant-backed defaults match the constant values."""

    def test_l2_eps_matches_numeric_epsilon(self) -> None:
        from src.constants import NUMERIC_EPSILON

        fn = L2RelativeLoss()
        assert fn.eps == NUMERIC_EPSILON

    def test_h1_lambda_grad_matches_default_constant(self) -> None:
        from src.constants import DEFAULT_H1_GRADIENT_WEIGHT

        fn = H1Loss()
        assert fn.lambda_grad == DEFAULT_H1_GRADIENT_WEIGHT

    def test_override_eps_works(self) -> None:
        fn = L2RelativeLoss(eps=1e-4)
        assert fn.eps == pytest.approx(1e-4)

    def test_override_lambda_grad_works(self) -> None:
        fn = H1Loss(lambda_grad=0.5)
        assert fn.lambda_grad == pytest.approx(0.5)
