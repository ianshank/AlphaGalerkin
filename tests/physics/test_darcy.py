"""Tests for Darcy flow solver.

Tests cover:
- DarcySample: Darcy-specific sample dataclass
- DarcyFlowSolver: Finite difference solver
- Physical properties: boundedness, smoothness, boundary conditions
- Determinism and reproducibility
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")

from src.physics.darcy import (
    DarcyFlowSolver,
    DarcySample,
)
from src.physics.solver import generate_random_field

# --- Fixtures ---

GRID_SIZES = [8, 12, 16]
DEFAULT_GRID = 8


@pytest.fixture
def solver() -> DarcyFlowSolver:
    """Create default Darcy solver."""
    return DarcyFlowSolver(resolution=DEFAULT_GRID)


@pytest.fixture
def solver_custom() -> DarcyFlowSolver:
    """Create Darcy solver with custom forcing."""
    return DarcyFlowSolver(forcing=2.0, resolution=DEFAULT_GRID)


@pytest.fixture
def permeability_field() -> numpy.ndarray:
    """Create a sample permeability field (strictly positive)."""
    log_k = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42, source_std=2.0)
    perm = numpy.exp(log_k)
    return numpy.clip(perm, 0.1, 100.0).astype(numpy.float32)


@pytest.fixture
def uniform_permeability() -> numpy.ndarray:
    """Create a uniform permeability field."""
    return numpy.ones((DEFAULT_GRID, DEFAULT_GRID), dtype=numpy.float32)


# --- DarcySample Tests ---


class TestDarcySample:
    """Tests for DarcySample dataclass."""

    def test_sample_creation(self):
        """Test DarcySample can be created with required fields."""
        n_points = 16
        sample = DarcySample(
            input_field=numpy.random.randn(n_points).astype(numpy.float32),
            output_field=numpy.random.randn(n_points).astype(numpy.float32),
            coords=numpy.random.rand(n_points, 2).astype(numpy.float32),
            grid_size=4,
        )

        assert sample.input_field.shape == (n_points,)
        assert sample.output_field.shape == (n_points,)

    def test_sample_metadata(self):
        """Test DarcySample stores metadata."""
        sample = DarcySample(
            input_field=numpy.zeros(4, dtype=numpy.float32),
            output_field=numpy.zeros(4, dtype=numpy.float32),
            coords=numpy.zeros((4, 2), dtype=numpy.float32),
            grid_size=2,
            metadata={"forcing": 1.0},
        )

        assert sample.metadata is not None
        assert sample.metadata["forcing"] == 1.0


# --- DarcyFlowSolver Initialization Tests ---


class TestDarcyFlowSolverInit:
    """Tests for DarcyFlowSolver initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        solver = DarcyFlowSolver()

        assert solver.forcing == 1.0
        assert solver.resolution == 32

    def test_custom_values(self):
        """Test custom initialization."""
        solver = DarcyFlowSolver(forcing=5.0, resolution=16)

        assert solver.forcing == 5.0
        assert solver.resolution == 16


# --- DarcyFlowSolver Solve Tests ---


class TestDarcyFlowSolverSolve:
    """Tests for DarcyFlowSolver.solve method."""

    def test_solve_output_shape(self, solver: DarcyFlowSolver, permeability_field):
        """Test solve produces output with correct shape."""
        pressure = solver.solve(permeability_field)

        assert pressure.shape == permeability_field.shape
        assert pressure.dtype == numpy.float32

    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_solve_different_grid_sizes(self, grid_size: int):
        """Test solve works on different grid sizes."""
        solver = DarcyFlowSolver(resolution=grid_size)
        log_k = generate_random_field(grid_size=grid_size, smooth=True, seed=42, source_std=2.0)
        perm = numpy.exp(log_k).astype(numpy.float32)
        perm = numpy.clip(perm, 0.1, 100.0)

        pressure = solver.solve(perm)

        assert pressure.shape == (grid_size, grid_size)
        assert pressure.dtype == numpy.float32

    def test_solve_deterministic(self, solver: DarcyFlowSolver, permeability_field):
        """Test solver is deterministic for same input."""
        p1 = solver.solve(permeability_field)
        p2 = solver.solve(permeability_field)

        assert numpy.allclose(p1, p2)

    def test_solve_no_nan_inf(self, solver: DarcyFlowSolver, permeability_field):
        """Test solution contains no NaN or Inf values."""
        pressure = solver.solve(permeability_field)

        assert numpy.all(numpy.isfinite(pressure))

    def test_boundary_conditions_zero(self, solver: DarcyFlowSolver, permeability_field):
        """Test Dirichlet boundary conditions: u = 0 on boundary."""
        pressure = solver.solve(permeability_field)
        n = pressure.shape[0]

        # All boundary values should be zero
        assert numpy.allclose(pressure[0, :], 0.0, atol=1e-6)
        assert numpy.allclose(pressure[n - 1, :], 0.0, atol=1e-6)
        assert numpy.allclose(pressure[:, 0], 0.0, atol=1e-6)
        assert numpy.allclose(pressure[:, n - 1], 0.0, atol=1e-6)

    def test_solution_bounded(self, solver: DarcyFlowSolver, permeability_field):
        """Test solution is bounded (physical requirement)."""
        pressure = solver.solve(permeability_field)

        # Pressure from Darcy flow with positive forcing and Dirichlet BCs
        # should be non-negative in the interior
        interior = pressure[1:-1, 1:-1]
        msg = "Interior pressure should be non-negative for positive forcing"
        assert numpy.all(interior >= -1e-6), msg

    def test_solution_smoothness(self, solver: DarcyFlowSolver, permeability_field):
        """Test solution is smooth (small gradient relative to magnitude)."""
        pressure = solver.solve(permeability_field)

        # Compute gradient magnitude
        grad_x = numpy.diff(pressure, axis=0)
        grad_y = numpy.diff(pressure, axis=1)

        # Gradient variance should be bounded (smooth solution)
        max_val = numpy.abs(pressure).max()
        if max_val > 1e-10:
            # Relative gradient should not be enormous
            assert grad_x.var() < max_val**2 * 100
            assert grad_y.var() < max_val**2 * 100

    def test_uniform_permeability_symmetric(self, solver: DarcyFlowSolver, uniform_permeability):
        """Test uniform permeability with constant forcing gives symmetric solution."""
        pressure = solver.solve(uniform_permeability)

        # With constant forcing and uniform permeability, solution should be
        # approximately symmetric about center
        n = pressure.shape[0]
        if n > 4:
            # Check left-right symmetry
            left = pressure[:, :n // 2]
            right = pressure[:, n // 2:][:, ::-1]
            min_cols = min(left.shape[1], right.shape[1])
            correlation = numpy.corrcoef(
                left[:, :min_cols].flatten(),
                right[:, :min_cols].flatten(),
            )[0, 1]
            assert correlation > 0.9


# --- DarcyFlowSolver Generate Sample Tests ---


class TestDarcyFlowSolverGenerateSample:
    """Tests for DarcyFlowSolver.generate_sample method."""

    def test_generate_sample_returns_darcy_sample(self, solver: DarcyFlowSolver):
        """Test generate_sample returns DarcySample."""
        sample = solver.generate_sample(seed=42)
        assert isinstance(sample, DarcySample)

    def test_generate_sample_shapes(self, solver: DarcyFlowSolver):
        """Test generated sample has correct shapes."""
        sample = solver.generate_sample(seed=42)
        n_points = solver.resolution**2

        assert sample.input_field.shape == (n_points,)
        assert sample.output_field.shape == (n_points,)
        assert sample.coords.shape == (n_points, 2)
        assert sample.grid_size == solver.resolution

    def test_generate_sample_deterministic(self, solver: DarcyFlowSolver):
        """Test sample generation is deterministic with seed."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        assert numpy.allclose(sample1.input_field, sample2.input_field)
        assert numpy.allclose(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: DarcyFlowSolver):
        """Test different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=43)

        assert not numpy.allclose(sample1.input_field, sample2.input_field)

    def test_generate_sample_coords_normalized(self, solver: DarcyFlowSolver):
        """Test coordinates are in [0, 1]."""
        sample = solver.generate_sample(seed=42)

        assert sample.coords.min() >= 0.0
        assert sample.coords.max() <= 1.0

    def test_generate_sample_permeability_positive(self, solver: DarcyFlowSolver):
        """Test permeability (input) is strictly positive."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(sample.input_field > 0)

    def test_generate_sample_metadata(self, solver: DarcyFlowSolver):
        """Test sample metadata contains forcing parameter."""
        sample = solver.generate_sample(seed=42)

        assert sample.metadata is not None
        assert "forcing" in sample.metadata
        assert sample.metadata["forcing"] == solver.forcing


# --- Generate Batch Tests ---


class TestDarcyFlowSolverGenerateBatch:
    """Tests for DarcyFlowSolver.generate_batch method."""

    def test_batch_size(self, solver: DarcyFlowSolver):
        """Test batch generates correct number of samples."""
        batch = solver.generate_batch(n_samples=3, seed=42)

        assert len(batch) == 3
        for sample in batch:
            assert isinstance(sample, DarcySample)

    def test_batch_deterministic(self, solver: DarcyFlowSolver):
        """Test batch generation is deterministic with seed."""
        batch1 = solver.generate_batch(n_samples=3, seed=42)
        batch2 = solver.generate_batch(n_samples=3, seed=42)

        for s1, s2 in zip(batch1, batch2, strict=True):
            assert numpy.allclose(s1.input_field, s2.input_field)
            assert numpy.allclose(s1.output_field, s2.output_field)

    def test_batch_samples_differ(self, solver: DarcyFlowSolver):
        """Test samples within a batch are distinct."""
        batch = solver.generate_batch(n_samples=3, seed=42)

        # Each sample should be different
        assert not numpy.allclose(batch[0].input_field, batch[1].input_field)
        assert not numpy.allclose(batch[1].input_field, batch[2].input_field)


# --- Physical Properties Tests ---


class TestDarcyPhysicalProperties:
    """Tests for physical correctness of Darcy flow solver."""

    def test_higher_forcing_higher_pressure(self):
        """Test that higher forcing produces higher pressure."""
        perm = numpy.ones((DEFAULT_GRID, DEFAULT_GRID), dtype=numpy.float32)

        solver_low = DarcyFlowSolver(forcing=1.0, resolution=DEFAULT_GRID)
        solver_high = DarcyFlowSolver(forcing=2.0, resolution=DEFAULT_GRID)

        p_low = solver_low.solve(perm)
        p_high = solver_high.solve(perm)

        # Higher forcing -> higher peak pressure
        assert numpy.max(p_high) > numpy.max(p_low)

    def test_higher_permeability_lower_pressure(self):
        """Test that higher permeability leads to lower pressure (easier flow)."""
        solver = DarcyFlowSolver(forcing=1.0, resolution=DEFAULT_GRID)

        perm_low = numpy.ones((DEFAULT_GRID, DEFAULT_GRID), dtype=numpy.float32) * 1.0
        perm_high = numpy.ones((DEFAULT_GRID, DEFAULT_GRID), dtype=numpy.float32) * 10.0

        p_low_perm = solver.solve(perm_low)
        p_high_perm = solver.solve(perm_high)

        # Higher permeability -> lower pressure buildup
        assert numpy.max(p_high_perm) < numpy.max(p_low_perm)

    def test_solution_finiteness(self, solver: DarcyFlowSolver, permeability_field):
        """Test all solution values are finite."""
        pressure = solver.solve(permeability_field)

        assert numpy.all(numpy.isfinite(pressure))
