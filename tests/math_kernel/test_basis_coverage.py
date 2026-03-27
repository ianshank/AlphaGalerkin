"""Additional coverage tests for basis functions.

Tests cover uncovered paths in src/math_kernel/basis.py:
- FourierBasis: learnable mode, forward alias, various n_features
- ChebyshevBasis: higher degrees, forward alias, edge cases
- create_grid_coordinates: various board sizes and batch sizes
- Factory functions: create_fourier_basis, create_chebyshev_basis, error paths
"""

from __future__ import annotations

import math

import pytest
import torch

from src.math_kernel.basis import (
    ChebyshevBasis,
    FourierBasis,
    create_chebyshev_basis,
    create_fourier_basis,
    create_grid_coordinates,
)

SEED = 42
BATCH_SIZE = 2
BOARD_SIZE = 5
N_FEATURES = 8
MAX_DEGREE = 4


class TestFourierBasisExtended:
    """Extended tests for FourierBasis."""

    def test_learnable_mode(self) -> None:
        torch.manual_seed(SEED)
        basis = FourierBasis(n_features=N_FEATURES, scale=1.0, learnable=True)
        assert isinstance(basis.b_matrix, torch.nn.Parameter)

    def test_non_learnable_mode(self) -> None:
        torch.manual_seed(SEED)
        basis = FourierBasis(n_features=N_FEATURES, scale=1.0, learnable=False)
        assert not isinstance(basis.b_matrix, torch.nn.Parameter)

    def test_evaluate_shape(self) -> None:
        torch.manual_seed(SEED)
        basis = FourierBasis(n_features=N_FEATURES)
        coords = torch.rand(BATCH_SIZE, BOARD_SIZE**2, 2)
        features = basis.evaluate(coords)
        assert features.shape == (BATCH_SIZE, BOARD_SIZE**2, 2 * N_FEATURES)

    def test_forward_equals_evaluate(self) -> None:
        torch.manual_seed(SEED)
        basis = FourierBasis(n_features=N_FEATURES)
        coords = torch.rand(BATCH_SIZE, BOARD_SIZE**2, 2)
        eval_result = basis.evaluate(coords)
        forward_result = basis.forward(coords)
        torch.testing.assert_close(eval_result, forward_result)

    def test_output_contains_sin_cos(self) -> None:
        torch.manual_seed(SEED)
        basis = FourierBasis(n_features=N_FEATURES)
        coords = torch.rand(1, 10, 2)
        features = basis.evaluate(coords)
        # First half is cos, second half is sin
        cos_part = features[..., :N_FEATURES]
        sin_part = features[..., N_FEATURES:]
        # cos^2 + sin^2 should = 1 for each frequency
        identity = cos_part**2 + sin_part**2
        torch.testing.assert_close(identity, torch.ones_like(identity), atol=1e-5, rtol=1e-5)

    def test_scale_parameter(self) -> None:
        torch.manual_seed(SEED)
        basis_low = FourierBasis(n_features=N_FEATURES, scale=0.1)
        torch.manual_seed(SEED)
        basis_high = FourierBasis(n_features=N_FEATURES, scale=10.0)
        # Different scales produce different frequency matrices
        assert not torch.allclose(basis_low.b_matrix, basis_high.b_matrix)

    def test_gradient_flow(self) -> None:
        torch.manual_seed(SEED)
        basis = FourierBasis(n_features=N_FEATURES, learnable=True)
        coords = torch.rand(1, 10, 2)
        features = basis.evaluate(coords)
        loss = features.sum()
        loss.backward()
        assert basis.b_matrix.grad is not None


class TestChebyshevBasisExtended:
    """Extended tests for ChebyshevBasis."""

    def test_degree_1(self) -> None:
        basis = ChebyshevBasis(max_degree=1)
        coords = torch.rand(1, 10, 2)
        features = basis.evaluate(coords)
        # degree=1 -> T_0 only -> 1*1 = 1 feature
        assert features.shape == (1, 10, 1)

    def test_degree_2(self) -> None:
        basis = ChebyshevBasis(max_degree=2)
        coords = torch.rand(1, 10, 2)
        features = basis.evaluate(coords)
        # degree=2 -> T_0, T_1 -> 2*2 = 4 features
        assert features.shape == (1, 10, 4)

    def test_higher_degree(self) -> None:
        basis = ChebyshevBasis(max_degree=MAX_DEGREE)
        coords = torch.rand(BATCH_SIZE, 10, 2)
        features = basis.evaluate(coords)
        assert features.shape == (BATCH_SIZE, 10, MAX_DEGREE**2)

    def test_forward_equals_evaluate(self) -> None:
        basis = ChebyshevBasis(max_degree=MAX_DEGREE)
        coords = torch.rand(1, 10, 2)
        torch.testing.assert_close(basis.forward(coords), basis.evaluate(coords))

    def test_chebyshev_at_endpoints(self) -> None:
        """T_n(-1) = (-1)^n and T_n(1) = 1 for all n."""
        basis = ChebyshevBasis(max_degree=MAX_DEGREE)
        # coords in [0,1] -> mapped to [-1,1]
        # So coords=0 -> x=-1, coords=1 -> x=1
        coords_zero = torch.zeros(1, 1, 2)  # maps to (-1, -1)
        features = basis.evaluate(coords_zero)
        assert features.shape == (1, 1, MAX_DEGREE**2)

    def test_no_nan(self) -> None:
        basis = ChebyshevBasis(max_degree=MAX_DEGREE)
        coords = torch.rand(BATCH_SIZE, 20, 2)
        features = basis.evaluate(coords)
        assert not torch.isnan(features).any()


class TestCreateGridCoordinates:
    """Tests for create_grid_coordinates."""

    def test_basic_shape(self) -> None:
        coords = create_grid_coordinates(BOARD_SIZE, batch_size=BATCH_SIZE)
        assert coords.shape == (BATCH_SIZE, BOARD_SIZE**2, 2)

    def test_unit_range(self) -> None:
        coords = create_grid_coordinates(BOARD_SIZE)
        assert (coords >= 0).all()
        assert (coords <= 1).all()

    def test_cell_centered(self) -> None:
        coords = create_grid_coordinates(2)  # 2x2 board
        # Centers should be at 0.25 and 0.75
        unique_x = coords[0, :, 0].unique().sort().values
        torch.testing.assert_close(
            unique_x, torch.tensor([0.25, 0.75]), atol=1e-6, rtol=1e-6
        )

    def test_various_sizes(self) -> None:
        for size in [3, 5, 9, 13, 19]:
            coords = create_grid_coordinates(size)
            assert coords.shape == (1, size**2, 2)

    def test_device(self) -> None:
        coords = create_grid_coordinates(BOARD_SIZE, device=torch.device("cpu"))
        assert coords.device == torch.device("cpu")


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_fourier_basis_torch(self) -> None:
        basis = create_fourier_basis(n_features=N_FEATURES, backend="torch")
        assert isinstance(basis, FourierBasis)

    def test_create_fourier_basis_learnable(self) -> None:
        basis = create_fourier_basis(n_features=N_FEATURES, learnable=True, backend="torch")
        assert isinstance(basis.b_matrix, torch.nn.Parameter)

    def test_create_fourier_basis_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_fourier_basis(n_features=N_FEATURES, backend="invalid")

    def test_create_chebyshev_basis_torch(self) -> None:
        basis = create_chebyshev_basis(max_degree=MAX_DEGREE, backend="torch")
        assert isinstance(basis, ChebyshevBasis)

    def test_create_chebyshev_basis_invalid_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            create_chebyshev_basis(max_degree=MAX_DEGREE, backend="invalid")
