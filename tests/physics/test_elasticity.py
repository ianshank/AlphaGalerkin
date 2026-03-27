"""Tests for Linear Elasticity solver.

Tests cover:
- ElasticitySample: Elasticity-specific sample dataclass
- ElasticitySolver: Spectral (FFT) solver with Navier-Cauchy equations
- Physical properties: zero force, finiteness, symmetry, linearity
- Determinism and reproducibility
- Different grid sizes
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")

from src.physics.elasticity import (
    ElasticitySample,
    ElasticitySolver,
)
from src.physics.solver import generate_random_field

# --- Fixtures ---

GRID_SIZES = [8, 16, 32]
DEFAULT_GRID = 16


@pytest.fixture
def solver() -> ElasticitySolver:
    """Create default Elasticity solver."""
    return ElasticitySolver(resolution=DEFAULT_GRID)


@pytest.fixture
def solver_custom() -> ElasticitySolver:
    """Create Elasticity solver with custom material parameters."""
    return ElasticitySolver(young_modulus=2.0, poisson_ratio=0.25, resolution=DEFAULT_GRID)


@pytest.fixture
def force_field() -> numpy.ndarray:
    """Create a sample body force field with zero mean (equilibrium)."""
    Fx = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)
    Fy = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=43)
    Fx -= numpy.mean(Fx)
    Fy -= numpy.mean(Fy)
    F = numpy.stack([Fx, Fy], axis=-1).reshape(-1, 2).astype(numpy.float32)
    return F


@pytest.fixture
def zero_force() -> numpy.ndarray:
    """Create a zero body force field."""
    return numpy.zeros((DEFAULT_GRID * DEFAULT_GRID, 2), dtype=numpy.float32)


# --- ElasticitySample Tests ---


class TestElasticitySample:
    """Tests for ElasticitySample dataclass."""

    def test_sample_creation(self):
        """Test ElasticitySample can be created with required fields."""
        n_points = 16
        sample = ElasticitySample(
            input_field=numpy.random.randn(n_points, 2).astype(numpy.float32),
            output_field=numpy.random.randn(n_points, 2).astype(numpy.float32),
            coords=numpy.random.rand(n_points, 2).astype(numpy.float32),
            grid_size=4,
        )

        assert sample.input_field.shape == (n_points, 2)
        assert sample.output_field.shape == (n_points, 2)

    def test_sample_metadata(self):
        """Test ElasticitySample stores metadata."""
        sample = ElasticitySample(
            input_field=numpy.zeros((4, 2), dtype=numpy.float32),
            output_field=numpy.zeros((4, 2), dtype=numpy.float32),
            coords=numpy.zeros((4, 2), dtype=numpy.float32),
            grid_size=2,
            metadata={"E": 1.0, "nu": 0.3},
        )

        assert sample.metadata is not None
        assert sample.metadata["E"] == 1.0
        assert sample.metadata["nu"] == 0.3


# --- ElasticitySolver Initialization Tests ---


class TestElasticitySolverInit:
    """Tests for ElasticitySolver initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        solver = ElasticitySolver()

        assert solver.E == 1.0
        assert solver.nu == 0.3
        assert solver.resolution == 32

    def test_custom_values(self):
        """Test custom initialization."""
        solver = ElasticitySolver(
            young_modulus=210.0,
            poisson_ratio=0.25,
            resolution=16,
        )

        assert solver.E == 210.0
        assert solver.nu == 0.25
        assert solver.resolution == 16

    def test_lame_parameters_computed(self):
        """Test Lame parameters are computed correctly from E and nu."""
        E = 2.0
        nu = 0.3
        solver = ElasticitySolver(young_modulus=E, poisson_ratio=nu)

        expected_mu = E / (2 * (1 + nu))
        expected_lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))

        assert numpy.isclose(solver.mu, expected_mu)
        assert numpy.isclose(solver.lam, expected_lam)


# --- ElasticitySolver Solve Tests ---


class TestElasticitySolverSolve:
    """Tests for ElasticitySolver.solve method."""

    def test_solve_output_shape(self, solver: ElasticitySolver, force_field):
        """Test solve produces displacement with correct shape."""
        displacement = solver.solve(force_field)

        assert displacement.shape == force_field.shape
        assert displacement.dtype == numpy.float32

    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_solve_different_grid_sizes(self, grid_size: int):
        """Test solve works on different grid sizes."""
        solver = ElasticitySolver(resolution=grid_size)
        Fx = generate_random_field(grid_size=grid_size, smooth=True, seed=42)
        Fy = generate_random_field(grid_size=grid_size, smooth=True, seed=43)
        Fx -= numpy.mean(Fx)
        Fy -= numpy.mean(Fy)
        F = numpy.stack([Fx, Fy], axis=-1).reshape(-1, 2).astype(numpy.float32)

        displacement = solver.solve(F)

        assert displacement.shape == (grid_size * grid_size, 2)
        assert displacement.dtype == numpy.float32

    def test_solve_deterministic(self, solver: ElasticitySolver, force_field):
        """Test solver is deterministic for same input."""
        u1 = solver.solve(force_field)
        u2 = solver.solve(force_field)

        assert numpy.allclose(u1, u2)

    def test_solve_no_nan_inf(self, solver: ElasticitySolver, force_field):
        """Test solution contains no NaN or Inf values."""
        displacement = solver.solve(force_field)

        assert numpy.all(numpy.isfinite(displacement))

    def test_zero_force_gives_zero_displacement(self, solver: ElasticitySolver, zero_force):
        """Test zero body force produces zero displacement."""
        displacement = solver.solve(zero_force)

        assert numpy.allclose(displacement, 0.0, atol=1e-7)


# --- Physical Properties Tests ---


class TestElasticityPhysicalProperties:
    """Tests for physical correctness of Elasticity solver."""

    def test_linearity_scaling(self, solver: ElasticitySolver, force_field):
        """Test solver is linear: scaling force scales displacement."""
        scale = 3.0
        u1 = solver.solve(force_field)
        u2 = solver.solve(scale * force_field)

        assert numpy.allclose(u2, scale * u1, rtol=1e-4)

    def test_superposition(self, solver: ElasticitySolver):
        """Test superposition principle: u(F1 + F2) = u(F1) + u(F2)."""
        n = DEFAULT_GRID

        Fx1 = generate_random_field(grid_size=n, smooth=True, seed=10)
        Fy1 = generate_random_field(grid_size=n, smooth=True, seed=11)
        Fx1 -= numpy.mean(Fx1)
        Fy1 -= numpy.mean(Fy1)
        F1 = numpy.stack([Fx1, Fy1], axis=-1).reshape(-1, 2).astype(numpy.float32)

        Fx2 = generate_random_field(grid_size=n, smooth=True, seed=20)
        Fy2 = generate_random_field(grid_size=n, smooth=True, seed=21)
        Fx2 -= numpy.mean(Fx2)
        Fy2 -= numpy.mean(Fy2)
        F2 = numpy.stack([Fx2, Fy2], axis=-1).reshape(-1, 2).astype(numpy.float32)

        u1 = solver.solve(F1)
        u2 = solver.solve(F2)
        u_combined = solver.solve(F1 + F2)

        assert numpy.allclose(u_combined, u1 + u2, rtol=1e-4)

    def test_solution_smoothness(self, solver: ElasticitySolver, force_field):
        """Test displacement field is smooth (small gradient variance)."""
        displacement = solver.solve(force_field)
        n = DEFAULT_GRID

        # Reshape to 2D grid for gradient computation
        ux = displacement[:, 0].reshape(n, n)

        grad_x = numpy.diff(ux, axis=0)
        max_val = numpy.abs(ux).max()

        if max_val > 1e-10:
            # Relative gradient should be bounded
            assert grad_x.var() < max_val**2 * 100

    def test_stiffer_material_smaller_displacement(self):
        """Test that higher Young's modulus produces smaller displacement."""
        n = DEFAULT_GRID
        Fx = generate_random_field(grid_size=n, smooth=True, seed=42)
        Fy = generate_random_field(grid_size=n, smooth=True, seed=43)
        Fx -= numpy.mean(Fx)
        Fy -= numpy.mean(Fy)
        F = numpy.stack([Fx, Fy], axis=-1).reshape(-1, 2).astype(numpy.float32)

        solver_soft = ElasticitySolver(young_modulus=1.0, resolution=n)
        solver_stiff = ElasticitySolver(young_modulus=10.0, resolution=n)

        u_soft = solver_soft.solve(F)
        u_stiff = solver_stiff.solve(F)

        # Stiffer material -> smaller displacement magnitude
        assert numpy.linalg.norm(u_stiff) < numpy.linalg.norm(u_soft)

    def test_displacement_finiteness(self, solver: ElasticitySolver, force_field):
        """Test all displacement values are finite."""
        displacement = solver.solve(force_field)

        assert numpy.all(numpy.isfinite(displacement))

    def test_displacement_vector_components(self, solver: ElasticitySolver, force_field):
        """Test displacement has two components (ux, uy) per point."""
        displacement = solver.solve(force_field)

        assert displacement.shape[1] == 2


# --- ElasticitySolver Generate Sample Tests ---


class TestElasticitySolverGenerateSample:
    """Tests for ElasticitySolver.generate_sample method."""

    def test_generate_sample_returns_elasticity_sample(self, solver: ElasticitySolver):
        """Test generate_sample returns ElasticitySample."""
        sample = solver.generate_sample(seed=42)
        assert isinstance(sample, ElasticitySample)

    def test_generate_sample_shapes(self, solver: ElasticitySolver):
        """Test generated sample has correct shapes."""
        sample = solver.generate_sample(seed=42)
        n_points = solver.resolution**2

        assert sample.input_field.shape == (n_points, 2)
        assert sample.output_field.shape == (n_points, 2)
        assert sample.coords.shape == (n_points, 2)
        assert sample.grid_size == solver.resolution

    def test_generate_sample_deterministic(self, solver: ElasticitySolver):
        """Test sample generation is deterministic with seed."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        assert numpy.allclose(sample1.input_field, sample2.input_field)
        assert numpy.allclose(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: ElasticitySolver):
        """Test different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=43)

        assert not numpy.allclose(sample1.input_field, sample2.input_field)

    def test_generate_sample_coords_normalized(self, solver: ElasticitySolver):
        """Test coordinates are in [0, 1]."""
        sample = solver.generate_sample(seed=42)

        assert sample.coords.min() >= 0.0
        assert sample.coords.max() <= 1.0

    def test_generate_sample_metadata(self, solver: ElasticitySolver):
        """Test sample metadata contains material parameters."""
        sample = solver.generate_sample(seed=42)

        assert sample.metadata is not None
        assert "E" in sample.metadata
        assert "nu" in sample.metadata
        assert sample.metadata["E"] == solver.E
        assert sample.metadata["nu"] == solver.nu

    def test_generate_sample_zero_mean_force(self, solver: ElasticitySolver):
        """Test generated sample has zero-mean force (equilibrium condition)."""
        sample = solver.generate_sample(seed=42)
        F = sample.input_field.reshape(solver.resolution, solver.resolution, 2)

        # Force should have approximately zero mean (enforced in generate_sample)
        assert numpy.allclose(numpy.mean(F[..., 0]), 0.0, atol=1e-6)
        assert numpy.allclose(numpy.mean(F[..., 1]), 0.0, atol=1e-6)


# --- Edge Cases ---


class TestElasticityEdgeCases:
    """Edge case tests for Elasticity solver."""

    def test_small_grid(self):
        """Test with small grid size."""
        solver = ElasticitySolver(resolution=4)
        sample = solver.generate_sample(seed=42)

        assert sample.grid_size == 4
        assert sample.input_field.shape == (16, 2)
        assert sample.output_field.shape == (16, 2)

    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_various_grid_sizes_generate_sample(self, grid_size: int):
        """Test generate_sample works on different grid sizes."""
        solver = ElasticitySolver(resolution=grid_size)
        sample = solver.generate_sample(seed=42)

        n_points = grid_size**2
        assert sample.input_field.shape == (n_points, 2)
        assert sample.output_field.shape == (n_points, 2)
        assert numpy.all(numpy.isfinite(sample.output_field))

    def test_incompressible_limit(self):
        """Test near-incompressible material (nu close to 0.5).

        Poisson ratio near 0.5 approaches incompressibility; lambda diverges
        but the solver should still produce finite results.
        """
        solver = ElasticitySolver(young_modulus=1.0, poisson_ratio=0.49, resolution=8)
        sample = solver.generate_sample(seed=42)

        assert numpy.all(numpy.isfinite(sample.output_field))

    def test_constant_force_field(self):
        """Test with spatially constant force (should give zero displacement for periodic BCs).

        A constant force has zero Fourier modes except k=0, which is set to zero
        displacement. So the result should be zero.
        """
        n = DEFAULT_GRID
        F = numpy.ones((n * n, 2), dtype=numpy.float32)

        solver = ElasticitySolver(resolution=n)
        displacement = solver.solve(F)

        # Constant force -> zero displacement in periodic spectral solver
        # (k=0 mode is zeroed out)
        assert numpy.allclose(displacement, 0.0, atol=1e-6)
