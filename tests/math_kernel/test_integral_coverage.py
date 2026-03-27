"""Additional coverage tests for integral approximation.

Tests cover uncovered paths in src/math_kernel/integral.py:
- MonteCarloIntegral: Weighted quadrature, forward alias
- GalerkinProjection: Projection, LBB constant computation
- PetrovGalerkinProjection: LBB validation, projection, dimension handling
- Factory functions: create_monte_carlo_integral, create_galerkin_projection,
  create_petrov_galerkin_projection
"""

from __future__ import annotations

import pytest
import torch

from src.math_kernel.integral import (
    GalerkinProjection,
    MonteCarloIntegral,
    PetrovGalerkinProjection,
    create_galerkin_projection,
    create_monte_carlo_integral,
    create_petrov_galerkin_projection,
)

SEED = 42
BATCH_SIZE = 2
SEQ_LEN = 16
D_MODEL = 16
D_KEY = 8
D_VALUE = 8
D_TRIAL = 12
D_TEST = 8


class TestMonteCarloIntegralExtended:
    """Extended tests for MonteCarloIntegral."""

    def test_uniform_weights(self) -> None:
        torch.manual_seed(SEED)
        mc = MonteCarloIntegral()
        values = torch.ones(BATCH_SIZE, SEQ_LEN, D_MODEL)
        result = mc.integrate(values)
        # Mean of ones should be ones
        torch.testing.assert_close(result, torch.ones(BATCH_SIZE, D_MODEL))

    def test_weighted_integration(self) -> None:
        torch.manual_seed(SEED)
        mc = MonteCarloIntegral()
        values = torch.ones(BATCH_SIZE, SEQ_LEN, D_MODEL)
        weights = torch.ones(BATCH_SIZE, SEQ_LEN)
        result = mc.integrate(values, weights)
        torch.testing.assert_close(result, torch.ones(BATCH_SIZE, D_MODEL))

    def test_weighted_non_uniform(self) -> None:
        torch.manual_seed(SEED)
        mc = MonteCarloIntegral()
        values = torch.zeros(1, 4, 1)
        values[0, 0, 0] = 1.0  # Only first point has value
        weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        result = mc.integrate(values, weights)
        assert result.item() == pytest.approx(1.0)

    def test_forward_equals_integrate(self) -> None:
        torch.manual_seed(SEED)
        mc = MonteCarloIntegral()
        values = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        result_int = mc.integrate(values)
        result_fwd = mc.forward(values)
        torch.testing.assert_close(result_int, result_fwd)

    def test_1d_values(self) -> None:
        mc = MonteCarloIntegral()
        values = torch.randn(BATCH_SIZE, SEQ_LEN)
        result = mc.integrate(values)
        assert result.shape == (BATCH_SIZE,)


class TestGalerkinProjectionExtended:
    """Extended tests for GalerkinProjection."""

    @pytest.fixture
    def projection(self) -> GalerkinProjection:
        torch.manual_seed(SEED)
        return GalerkinProjection(d_model=D_MODEL, d_key=D_KEY, d_value=D_VALUE)

    def test_initialization(self, projection: GalerkinProjection) -> None:
        assert projection.d_model == D_MODEL
        assert projection.d_key == D_KEY
        assert projection.d_value == D_VALUE

    def test_project_shape(self, projection: GalerkinProjection) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output = projection.project(x)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_forward_equals_project(self, projection: GalerkinProjection) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        result_proj = projection.project(x)
        result_fwd = projection.forward(x)
        torch.testing.assert_close(result_proj, result_fwd)

    def test_compute_lbb_constant(self, projection: GalerkinProjection) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        beta = projection.compute_lbb_constant(x)
        assert beta.shape == (BATCH_SIZE,)
        assert (beta >= 0).all()

    def test_gradient_flow(self, projection: GalerkinProjection) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, requires_grad=True)
        output = projection(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_no_nan_output(self, projection: GalerkinProjection) -> None:
        torch.manual_seed(SEED)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output = projection(x)
        assert not torch.isnan(output).any()


class TestPetrovGalerkinProjectionExtended:
    """Extended tests for PetrovGalerkinProjection."""

    def test_valid_initialization(self) -> None:
        torch.manual_seed(SEED)
        proj = PetrovGalerkinProjection(
            d_model=D_MODEL, d_trial=D_TRIAL, d_test=D_TEST, d_value=D_VALUE
        )
        assert proj.d_trial == D_TRIAL
        assert proj.d_test == D_TEST

    def test_lbb_violation_raises(self) -> None:
        with pytest.raises(ValueError, match="LBB violation"):
            PetrovGalerkinProjection(
                d_model=D_MODEL, d_trial=4, d_test=8, d_value=D_VALUE
            )

    def test_equal_trial_test(self) -> None:
        torch.manual_seed(SEED)
        proj = PetrovGalerkinProjection(
            d_model=D_MODEL, d_trial=D_KEY, d_test=D_KEY, d_value=D_VALUE
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output = proj(x)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_trial_greater_than_test(self) -> None:
        torch.manual_seed(SEED)
        proj = PetrovGalerkinProjection(
            d_model=D_MODEL, d_trial=D_TRIAL, d_test=D_TEST, d_value=D_VALUE
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output = proj.project(x)
        assert output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)

    def test_forward_equals_project(self) -> None:
        torch.manual_seed(SEED)
        proj = PetrovGalerkinProjection(
            d_model=D_MODEL, d_trial=D_TRIAL, d_test=D_TEST, d_value=D_VALUE
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        torch.testing.assert_close(proj.forward(x), proj.project(x))


class TestIntegralFactoryFunctions:
    """Tests for factory functions."""

    def test_create_monte_carlo_integral_torch(self) -> None:
        mc = create_monte_carlo_integral(backend="torch")
        assert isinstance(mc, MonteCarloIntegral)

    def test_create_monte_carlo_integral_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_monte_carlo_integral(backend="invalid")

    def test_create_galerkin_projection_torch(self) -> None:
        proj = create_galerkin_projection(
            d_model=D_MODEL, d_key=D_KEY, d_value=D_VALUE, backend="torch"
        )
        assert isinstance(proj, GalerkinProjection)

    def test_create_galerkin_projection_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_galerkin_projection(
                d_model=D_MODEL, d_key=D_KEY, d_value=D_VALUE, backend="invalid"
            )

    def test_create_petrov_galerkin_projection_torch(self) -> None:
        proj = create_petrov_galerkin_projection(
            d_model=D_MODEL, d_trial=D_TRIAL, d_test=D_TEST, d_value=D_VALUE, backend="torch"
        )
        assert isinstance(proj, PetrovGalerkinProjection)

    def test_create_petrov_galerkin_projection_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_petrov_galerkin_projection(
                d_model=D_MODEL, d_trial=D_TRIAL, d_test=D_TEST, d_value=D_VALUE, backend="invalid"
            )
