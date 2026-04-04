"""Tests for Darcy flow equation solver.

Tests cover:
- DarcySample: Darcy-specific sample dataclass
- DarcyFlowSolver: Solver initialization, solve, sample generation, batch generation
- Boundary conditions and physical properties
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")

from src.physics.darcy import DarcyFlowSolver, DarcySample
from src.physics.solver import generate_random_field

# --- Fixtures ---


@pytest.fixture
def solver() -> DarcyFlowSolver:
    """Create default Darcy solver with small grid."""
    return DarcyFlowSolver(resolution=8)


@pytest.fixture
def solver_16() -> DarcyFlowSolver:
    """Create Darcy solver with 16x16 grid."""
    return DarcyFlowSolver(resolution=16)


@pytest.fixture
def uniform_permeability() -> numpy.ndarray:
    """Create uniform permeability field."""
    return numpy.ones((8, 8), dtype=numpy.float32)


# --- DarcySample Tests ---


class TestDarcySample:
    """Tests for DarcySample dataclass."""

    def test_sample_fields(self) -> None:
        """Test DarcySample has expected fields."""
        sample = DarcySample(
            input_field=numpy.ones(64, dtype=numpy.float32),
            output_field=numpy.zeros(64, dtype=numpy.float32),
            coords=numpy.random.rand(64, 2).astype(numpy.float32),
            grid_size=8,
            metadata={"forcing": 1.0},
        )

        assert sample.input_field.shape == (64,)
        assert sample.output_field.shape == (64,)
        assert sample.grid_size == 8
        assert sample.metadata["forcing"] == 1.0

    def test_sample_with_none_metadata(self) -> None:
        """Test DarcySample with no metadata."""
        sample = DarcySample(
            input_field=numpy.ones(64, dtype=numpy.float32),
            output_field=numpy.zeros(64, dtype=numpy.float32),
            coords=numpy.random.rand(64, 2).astype(numpy.float32),
            grid_size=8,
        )

        assert sample.metadata is None


# --- DarcyFlowSolver Initialization Tests ---


class TestDarcyFlowSolverInit:
    """Tests for DarcyFlowSolver initialization."""

    def test_default_init(self) -> None:
        """Test default solver parameters."""
        solver = DarcyFlowSolver()

        assert solver.forcing == 1.0
        assert solver.resolution == 32

    def test_custom_init(self) -> None:
        """Test custom solver parameters."""
        solver = DarcyFlowSolver(forcing=2.5, resolution=16)

        assert solver.forcing == 2.5
        assert solver.resolution == 16

    def test_zero_forcing(self) -> None:
        """Test solver with zero forcing."""
        solver = DarcyFlowSolver(forcing=0.0, resolution=8)
        assert solver.forcing == 0.0

    def test_negative_forcing(self) -> None:
        """Test solver accepts negative forcing."""
        solver = DarcyFlowSolver(forcing=-1.0, resolution=8)
        assert solver.forcing == -1.0


# --- Solve Tests ---


class TestDarcyFlowSolverSolve:
    """Tests for DarcyFlowSolver.solve."""

    def test_solve_shape(
        self, solver: DarcyFlowSolver, uniform_permeability: numpy.ndarray
    ) -> None:
        """Test output shape matches input."""
        result = solver.solve(uniform_permeability)

        assert result.shape == (8, 8)
        assert result.dtype == numpy.float32

    def test_boundary_conditions(
        self, solver: DarcyFlowSolver, uniform_permeability: numpy.ndarray
    ) -> None:
        """Test Dirichlet boundary conditions (u=0 on boundary)."""
        result = solver.solve(uniform_permeability)

        # Top and bottom rows
        numpy.testing.assert_allclose(result[0, :], 0.0, atol=1e-10)
        numpy.testing.assert_allclose(result[-1, :], 0.0, atol=1e-10)
        # Left and right columns
        numpy.testing.assert_allclose(result[:, 0], 0.0, atol=1e-10)
        numpy.testing.assert_allclose(result[:, -1], 0.0, atol=1e-10)

    def test_interior_positive_with_positive_forcing(
        self, solver: DarcyFlowSolver, uniform_permeability: numpy.ndarray
    ) -> None:
        """Test that positive forcing gives positive interior values."""
        result = solver.solve(uniform_permeability)

        interior = result[1:-1, 1:-1]
        assert numpy.all(interior > 0), "Interior should be positive with positive forcing"

    def test_symmetry_with_uniform_permeability(
        self, solver: DarcyFlowSolver, uniform_permeability: numpy.ndarray
    ) -> None:
        """Test solution symmetry for uniform permeability."""
        result = solver.solve(uniform_permeability)

        numpy.testing.assert_allclose(result, result.T, atol=1e-6)

    def test_higher_permeability_lower_pressure(self, solver: DarcyFlowSolver) -> None:
        """Test that higher permeability leads to lower pressure for same forcing."""
        perm_low = numpy.ones((8, 8), dtype=numpy.float32) * 1.0
        perm_high = numpy.ones((8, 8), dtype=numpy.float32) * 10.0

        result_low = solver.solve(perm_low)
        result_high = solver.solve(perm_high)

        assert numpy.max(result_high) < numpy.max(result_low)

    def test_different_forcing(self) -> None:
        """Test that different forcing values scale the solution."""
        solver_f1 = DarcyFlowSolver(forcing=1.0, resolution=8)
        solver_f2 = DarcyFlowSolver(forcing=2.0, resolution=8)

        perm = numpy.ones((8, 8), dtype=numpy.float32)

        result_f1 = solver_f1.solve(perm)
        result_f2 = solver_f2.solve(perm)

        # Solution should scale linearly with forcing
        numpy.testing.assert_allclose(result_f2, 2.0 * result_f1, atol=1e-5)

    def test_solve_with_random_permeability(self, solver: DarcyFlowSolver) -> None:
        """Test solve with random permeability produces finite values."""
        log_k = generate_random_field(grid_size=8, smooth=True, seed=42, source_std=2.0)
        permeability = numpy.exp(log_k)
        permeability = numpy.clip(permeability, 0.1, 100.0)

        result = solver.solve(permeability)

        assert numpy.all(numpy.isfinite(result))
        assert result.shape == (8, 8)

    def test_zero_forcing_gives_zero_solution(self) -> None:
        """Test that zero forcing produces zero solution."""
        solver = DarcyFlowSolver(forcing=0.0, resolution=8)
        perm = numpy.ones((8, 8), dtype=numpy.float32)

        result = solver.solve(perm)

        numpy.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_negative_forcing_negative_interior(self) -> None:
        """Test that negative forcing gives negative interior values."""
        solver = DarcyFlowSolver(forcing=-1.0, resolution=8)
        perm = numpy.ones((8, 8), dtype=numpy.float32)

        result = solver.solve(perm)

        interior = result[1:-1, 1:-1]
        assert numpy.all(interior < 0), "Interior should be negative with negative forcing"


# --- Generate Sample Tests ---


class TestDarcyFlowSolverGenerateSample:
    """Tests for DarcyFlowSolver.generate_sample."""

    def test_generate_sample_structure(self, solver: DarcyFlowSolver) -> None:
        """Test generated sample has correct structure."""
        sample = solver.generate_sample(seed=42)

        assert isinstance(sample, DarcySample)
        assert sample.input_field.shape == (64,)  # 8*8
        assert sample.output_field.shape == (64,)
        assert sample.coords.shape == (64, 2)
        assert sample.grid_size == 8

    def test_generate_sample_metadata(self, solver: DarcyFlowSolver) -> None:
        """Test generated sample metadata."""
        sample = solver.generate_sample(seed=42)

        assert sample.metadata is not None
        assert "forcing" in sample.metadata
        assert sample.metadata["forcing"] == 1.0

    def test_generate_sample_reproducibility(self, solver: DarcyFlowSolver) -> None:
        """Test that same seed produces same sample."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        numpy.testing.assert_array_equal(sample1.input_field, sample2.input_field)
        numpy.testing.assert_array_equal(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: DarcyFlowSolver) -> None:
        """Test that different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=123)

        assert not numpy.array_equal(sample1.input_field, sample2.input_field)

    def test_generate_sample_permeability_positive(self, solver: DarcyFlowSolver) -> None:
        """Test that permeability field is strictly positive."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(sample.input_field > 0)

    def test_generate_sample_finite_output(self, solver: DarcyFlowSolver) -> None:
        """Test that output field is finite."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(numpy.isfinite(sample.output_field))

    def test_generate_sample_coords_range(self, solver: DarcyFlowSolver) -> None:
        """Test that coordinates are in [0, 1] range."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(sample.coords >= 0.0)
        assert numpy.all(sample.coords <= 1.0)


# --- Batch Generation Tests ---


class TestDarcyFlowSolverBatch:
    """Tests for DarcyFlowSolver.generate_batch."""

    def test_batch_size(self, solver: DarcyFlowSolver) -> None:
        """Test batch generates correct number of samples."""
        batch = solver.generate_batch(n_samples=3, seed=42)

        assert len(batch) == 3
        assert all(isinstance(s, DarcySample) for s in batch)

    def test_batch_reproducibility(self, solver: DarcyFlowSolver) -> None:
        """Test batch reproducibility with same seed."""
        batch1 = solver.generate_batch(n_samples=2, seed=42)
        batch2 = solver.generate_batch(n_samples=2, seed=42)

        numpy.testing.assert_array_equal(batch1[0].input_field, batch2[0].input_field)
        numpy.testing.assert_array_equal(batch1[1].input_field, batch2[1].input_field)

    def test_batch_samples_different(self, solver: DarcyFlowSolver) -> None:
        """Test that batch samples are different from each other."""
        batch = solver.generate_batch(n_samples=3, seed=42)

        assert not numpy.array_equal(batch[0].input_field, batch[1].input_field)

    def test_batch_none_seed(self, solver: DarcyFlowSolver) -> None:
        """Test batch generation with no seed."""
        batch = solver.generate_batch(n_samples=2, seed=None)

        assert len(batch) == 2
        assert all(numpy.all(numpy.isfinite(s.output_field)) for s in batch)

    def test_batch_different_resolution(self, solver_16: DarcyFlowSolver) -> None:
        """Test batch with different resolution."""
        batch = solver_16.generate_batch(n_samples=2, seed=42)

        assert all(s.grid_size == 16 for s in batch)
        assert all(s.input_field.shape == (256,) for s in batch)


# --- Parameter Validation Tests ---


class TestDarcyParameterValidation:
    """Tests for parameter validation and edge cases."""

    def test_small_resolution(self) -> None:
        """Test solver works with small resolution."""
        solver = DarcyFlowSolver(resolution=4)
        perm = numpy.ones((4, 4), dtype=numpy.float32)
        result = solver.solve(perm)
        assert result.shape == (4, 4)
        assert numpy.all(numpy.isfinite(result))

    def test_high_permeability_contrast(self) -> None:
        """Test solver handles high permeability contrast."""
        solver = DarcyFlowSolver(resolution=8)
        perm = numpy.ones((8, 8), dtype=numpy.float32)
        perm[:4, :] = 0.1
        perm[4:, :] = 100.0

        result = solver.solve(perm)
        assert numpy.all(numpy.isfinite(result))

    def test_forcing_stored_in_metadata(self) -> None:
        """Test that custom forcing is stored in sample metadata."""
        solver = DarcyFlowSolver(forcing=3.14, resolution=8)
        sample = solver.generate_sample(seed=42)
        assert sample.metadata["forcing"] == pytest.approx(3.14)
