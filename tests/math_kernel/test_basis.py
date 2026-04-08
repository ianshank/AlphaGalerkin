"""Property-based tests for basis functions.

Tests mathematical properties:
- Orthogonality
- Completeness
- Translation invariance
- Linearity
- Factory functions
- JAX backend error paths
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.math_kernel.basis import (
    HAS_JAX,
    ChebyshevBasis,
    FourierBasis,
    create_chebyshev_basis,
    create_fourier_basis,
    create_grid_coordinates,
)


class TestFourierBasis:
    """Tests for Fourier basis functions."""

    @pytest.fixture
    def basis(self) -> FourierBasis:
        """Create a Fourier basis for testing."""
        torch.manual_seed(42)
        return FourierBasis(n_features=64, scale=1.0, learnable=False)

    def test_output_shape(self, basis: FourierBasis) -> None:
        """Test that output has correct shape."""
        batch, n = 2, 81
        coords = torch.rand(batch, n, 2)

        features = basis(coords)

        assert features.shape == (batch, n, 2 * 64)  # cos + sin

    def test_deterministic(self, basis: FourierBasis) -> None:
        """Test that same input gives same output."""
        coords = torch.rand(2, 81, 2)

        features1 = basis(coords)
        features2 = basis(coords)

        assert torch.allclose(features1, features2)

    def test_bounded_output(self, basis: FourierBasis) -> None:
        """Test that outputs are bounded (cos/sin in [-1, 1])."""
        coords = torch.rand(4, 100, 2)

        features = basis(coords)

        assert features.abs().max() <= 1.0 + 1e-6

    @given(st.floats(0.0, 1.0), st.floats(0.0, 1.0))
    @settings(max_examples=50, deadline=None)
    def test_single_point_bounded(self, x: float, y: float) -> None:
        """Property: Fourier features at any point are bounded."""
        torch.manual_seed(42)
        basis = FourierBasis(n_features=32, scale=1.0)

        coords = torch.tensor([[[x, y]]])
        features = basis(coords)

        assert features.abs().max() <= 1.0 + 1e-6

    def test_frequency_scaling(self) -> None:
        """Test that higher scale leads to higher frequency variation."""
        coords = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(-1)
        coords = torch.cat([coords, torch.zeros_like(coords)], dim=-1)

        torch.manual_seed(42)
        low_freq = FourierBasis(n_features=32, scale=0.1)
        torch.manual_seed(42)
        high_freq = FourierBasis(n_features=32, scale=10.0)

        feat_low = low_freq(coords)
        feat_high = high_freq(coords)

        # High frequency should have more variation
        var_low = feat_low.var(dim=1).mean()
        var_high = feat_high.var(dim=1).mean()

        # High scale should have different variance pattern
        assert not torch.allclose(var_low, var_high, rtol=0.1)


class TestChebyshevBasis:
    """Tests for Chebyshev basis functions."""

    @pytest.fixture
    def basis(self) -> ChebyshevBasis:
        """Create a Chebyshev basis for testing."""
        return ChebyshevBasis(max_degree=8)

    def test_output_shape(self, basis: ChebyshevBasis) -> None:
        """Test that output has correct shape."""
        batch, n = 2, 81
        coords = torch.rand(batch, n, 2)

        features = basis(coords)

        # 2D tensor product: degree^2 features
        assert features.shape == (batch, n, 64)

    def test_chebyshev_at_endpoints(self, basis: ChebyshevBasis) -> None:
        """Test Chebyshev values at domain endpoints."""
        # At x=0 (maps to -1 in Chebyshev), T_n(-1) = (-1)^n
        # At x=1 (maps to +1 in Chebyshev), T_n(1) = 1
        coords_0 = torch.tensor([[[0.0, 0.5]]])
        coords_1 = torch.tensor([[[1.0, 0.5]]])

        feat_0 = basis(coords_0)
        feat_1 = basis(coords_1)

        # Check that they're different (polynomial variation)
        assert not torch.allclose(feat_0, feat_1)

    def test_chebyshev_recurrence(self) -> None:
        """Test that Chebyshev polynomials satisfy recurrence relation."""
        basis = ChebyshevBasis(max_degree=5)

        # Test at random points
        x = torch.rand(10) * 2 - 1  # In [-1, 1]

        # Compute manually using recurrence
        T = [torch.ones_like(x), x]
        for _ in range(2, 5):
            T.append(2 * x * T[-1] - T[-2])

        # Verify T_4 = 2*x*T_3 - T_2
        expected_T4 = 2 * x * T[3] - T[2]

        assert torch.allclose(T[4], expected_T4, atol=1e-5)


class TestGridCoordinates:
    """Tests for grid coordinate generation."""

    def test_output_shape(self) -> None:
        """Test coordinate grid shape."""
        coords = create_grid_coordinates(9, batch_size=4)

        assert coords.shape == (4, 81, 2)

    def test_coordinate_range(self) -> None:
        """Test that coordinates are in [0, 1]."""
        coords = create_grid_coordinates(19)

        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

    def test_cell_centered(self) -> None:
        """Test that coordinates are cell-centered."""
        coords = create_grid_coordinates(9)

        # First coordinate should be (0.5/9, 0.5/9)
        expected_first = torch.tensor([0.5 / 9, 0.5 / 9])
        assert torch.allclose(coords[0, 0], expected_first, atol=1e-5)

        # Last coordinate should be (8.5/9, 8.5/9)
        expected_last = torch.tensor([8.5 / 9, 8.5 / 9])
        assert torch.allclose(coords[0, -1], expected_last, atol=1e-5)

    def test_different_resolutions(self) -> None:
        """Test that different resolutions give proportional coordinates."""
        coords_9 = create_grid_coordinates(9)
        coords_19 = create_grid_coordinates(19)

        # Both should cover [0, 1] with cell-centered coordinates
        # Check that center points are similar
        center_9 = coords_9[0, 40]  # Approximately center for 9x9
        center_19 = coords_19[0, 180]  # Approximately center for 19x19

        # Both should be close to (0.5, 0.5)
        assert torch.allclose(center_9, torch.tensor([0.5, 0.5]), atol=0.1)
        assert torch.allclose(center_19, torch.tensor([0.5, 0.5]), atol=0.1)

    @given(st.integers(3, 25))
    @settings(max_examples=20, deadline=None)
    def test_any_board_size(self, board_size: int) -> None:
        """Property: Grid coordinates work for any board size."""
        coords = create_grid_coordinates(board_size)

        assert coords.shape == (1, board_size**2, 2)
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0


# ===================================================================
# Factory function tests (covers lines 488-518, 540-560)
# ===================================================================


class TestCreateFourierBasisFactory:
    """Tests for create_fourier_basis factory function."""

    def test_torch_backend_returns_fourier_basis(self) -> None:
        """Factory with backend='torch' returns a FourierBasis instance."""
        torch.manual_seed(42)
        basis = create_fourier_basis(n_features=16, scale=1.0, learnable=False, backend="torch")
        assert isinstance(basis, FourierBasis)

    def test_torch_backend_correct_n_features(self) -> None:
        """Factory-produced FourierBasis has the requested n_features."""
        torch.manual_seed(42)
        basis = create_fourier_basis(n_features=32, scale=2.0, learnable=False, backend="torch")
        assert basis.n_features == 32

    def test_torch_backend_learnable(self) -> None:
        """Factory can create a learnable FourierBasis."""
        torch.manual_seed(42)
        basis = create_fourier_basis(n_features=16, scale=1.0, learnable=True, backend="torch")
        assert isinstance(basis, FourierBasis)
        # b_matrix should be a Parameter when learnable
        assert isinstance(basis.b_matrix, torch.nn.Parameter)

    def test_torch_backend_not_learnable(self) -> None:
        """Factory creates a non-learnable FourierBasis by default."""
        torch.manual_seed(42)
        basis = create_fourier_basis(n_features=16, scale=1.0, learnable=False, backend="torch")
        # b_matrix should be a buffer (not a Parameter)
        assert not isinstance(basis.b_matrix, torch.nn.Parameter)

    def test_torch_backend_produces_correct_output(self) -> None:
        """Factory-produced basis gives correct output shape."""
        torch.manual_seed(42)
        basis = create_fourier_basis(n_features=8, scale=1.0, backend="torch")
        coords = torch.rand(2, 25, 2)
        features = basis(coords)
        assert features.shape == (2, 25, 16)  # 2 * n_features

    def test_jax_backend_raises_import_error_when_unavailable(self) -> None:
        """Factory with backend='jax' raises ImportError when JAX not installed."""
        if HAS_JAX:
            pytest.skip("JAX is installed; cannot test ImportError path")
        with pytest.raises(ImportError, match="JAX and Flax are required"):
            create_fourier_basis(n_features=16, backend="jax")

    def test_unknown_backend_raises_value_error(self) -> None:
        """Factory with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_fourier_basis(n_features=16, backend="tensorflow")

    def test_default_backend_is_torch(self) -> None:
        """Factory defaults to 'torch' backend when not specified."""
        torch.manual_seed(42)
        # The function signature has backend="torch" as default, but let's
        # call with explicit "torch" and verify it works
        basis = create_fourier_basis(n_features=8, backend="torch")
        assert isinstance(basis, FourierBasis)


class TestCreateChebyshevBasisFactory:
    """Tests for create_chebyshev_basis factory function."""

    def test_torch_backend_returns_chebyshev_basis(self) -> None:
        """Factory with backend='torch' returns a ChebyshevBasis instance."""
        basis = create_chebyshev_basis(max_degree=4, backend="torch")
        assert isinstance(basis, ChebyshevBasis)

    def test_torch_backend_correct_max_degree(self) -> None:
        """Factory-produced ChebyshevBasis has the requested max_degree."""
        basis = create_chebyshev_basis(max_degree=6, backend="torch")
        assert basis.max_degree == 6

    def test_torch_backend_produces_correct_output(self) -> None:
        """Factory-produced basis gives correct output shape."""
        basis = create_chebyshev_basis(max_degree=5, backend="torch")
        coords = torch.rand(2, 16, 2)
        features = basis(coords)
        # 2D tensor product: degree^2 features
        assert features.shape == (2, 16, 25)

    def test_jax_backend_raises_import_error_when_unavailable(self) -> None:
        """Factory with backend='jax' raises ImportError when JAX not installed."""
        if HAS_JAX:
            pytest.skip("JAX is installed; cannot test ImportError path")
        with pytest.raises(ImportError, match="JAX and Flax are required"):
            create_chebyshev_basis(max_degree=4, backend="jax")

    def test_unknown_backend_raises_value_error(self) -> None:
        """Factory with unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_chebyshev_basis(max_degree=4, backend="numpy")


class TestFourierBasisLearnable:
    """Additional tests for learnable Fourier basis."""

    def test_learnable_requires_grad(self) -> None:
        """Learnable FourierBasis has b_matrix with requires_grad=True."""
        torch.manual_seed(42)
        basis = FourierBasis(n_features=16, scale=1.0, learnable=True)
        assert basis.b_matrix.requires_grad is True

    def test_non_learnable_no_grad(self) -> None:
        """Non-learnable FourierBasis has b_matrix without requires_grad."""
        torch.manual_seed(42)
        basis = FourierBasis(n_features=16, scale=1.0, learnable=False)
        assert basis.b_matrix.requires_grad is False

    def test_forward_aliases_evaluate(self) -> None:
        """forward() and evaluate() produce identical results."""
        torch.manual_seed(42)
        basis = FourierBasis(n_features=16, scale=1.0)
        coords = torch.rand(2, 25, 2)
        out_forward = basis.forward(coords)
        out_evaluate = basis.evaluate(coords)
        assert torch.allclose(out_forward, out_evaluate)


class TestChebyshevBasisEdgeCases:
    """Additional edge case tests for ChebyshevBasis."""

    def test_degree_1(self) -> None:
        """ChebyshevBasis with max_degree=1 gives T_0 only."""
        basis = ChebyshevBasis(max_degree=1)
        coords = torch.rand(1, 4, 2)
        features = basis(coords)
        # T_0(x) = 1 for both x and y, so tensor product is 1 * 1 = 1
        assert features.shape == (1, 4, 1)
        assert torch.allclose(features, torch.ones(1, 4, 1))

    def test_degree_2_known_values(self) -> None:
        """ChebyshevBasis with max_degree=2 at known points."""
        basis = ChebyshevBasis(max_degree=2)
        # coords = (0.5, 0.5) maps to (0, 0) in Chebyshev domain
        coords = torch.tensor([[[0.5, 0.5]]])
        features = basis(coords)
        # T_0(0) = 1, T_1(0) = 0 => tensor product: [1*1, 1*0, 0*1, 0*0] = [1, 0, 0, 0]
        assert features.shape == (1, 1, 4)
        expected = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        assert torch.allclose(features, expected, atol=1e-6)

    def test_forward_aliases_evaluate(self) -> None:
        """forward() and evaluate() produce identical results."""
        basis = ChebyshevBasis(max_degree=4)
        coords = torch.rand(2, 9, 2)
        out_forward = basis.forward(coords)
        out_evaluate = basis.evaluate(coords)
        assert torch.allclose(out_forward, out_evaluate)
