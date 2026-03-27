"""Tests for Heat equation solver.

Tests cover:
- HeatSample: Heat-specific sample dataclass
- HeatSolver: Spectral (FFT) solver
- Time evolution and conservation properties
- Steady state convergence
- Determinism and reproducibility
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")
scipy = pytest.importorskip("scipy")

from src.physics.heat import (
    HeatSample,
    HeatSolver,
)
from src.physics.solver import generate_random_field

# --- Fixtures ---

GRID_SIZES = [8, 16, 32]
DEFAULT_GRID = 16


@pytest.fixture
def solver() -> HeatSolver:
    """Create default Heat solver."""
    return HeatSolver(resolution=DEFAULT_GRID)


@pytest.fixture
def solver_fast_diffusion() -> HeatSolver:
    """Create Heat solver with fast diffusion."""
    return HeatSolver(alpha=0.1, total_time=1.0, resolution=DEFAULT_GRID)


@pytest.fixture
def initial_condition() -> numpy.ndarray:
    """Create a sample initial temperature distribution."""
    return generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)


# --- HeatSample Tests ---


class TestHeatSample:
    """Tests for HeatSample dataclass."""

    def test_sample_creation(self):
        """Test HeatSample can be created with required fields."""
        n_points = 16
        sample = HeatSample(
            input_field=numpy.random.randn(n_points).astype(numpy.float32),
            output_field=numpy.random.randn(n_points).astype(numpy.float32),
            coords=numpy.random.rand(n_points, 2).astype(numpy.float32),
            grid_size=4,
        )

        assert sample.input_field.shape == (n_points,)
        assert sample.output_field.shape == (n_points,)

    def test_sample_metadata(self):
        """Test HeatSample stores metadata."""
        sample = HeatSample(
            input_field=numpy.zeros(4, dtype=numpy.float32),
            output_field=numpy.zeros(4, dtype=numpy.float32),
            coords=numpy.zeros((4, 2), dtype=numpy.float32),
            grid_size=2,
            metadata={"alpha": 0.01, "total_time": 1.0},
        )

        assert sample.metadata is not None
        assert sample.metadata["alpha"] == 0.01
        assert sample.metadata["total_time"] == 1.0


# --- HeatSolver Initialization Tests ---


class TestHeatSolverInit:
    """Tests for HeatSolver initialization."""

    def test_default_values(self):
        """Test default initialization values."""
        solver = HeatSolver()

        assert solver.alpha == 0.01
        assert solver.time_step == 0.001
        assert solver.total_time == 1.0
        assert solver.resolution == 32

    def test_custom_values(self):
        """Test custom initialization."""
        solver = HeatSolver(
            alpha=0.05,
            time_step=0.01,
            total_time=2.0,
            resolution=16,
        )

        assert solver.alpha == 0.05
        assert solver.time_step == 0.01
        assert solver.total_time == 2.0
        assert solver.resolution == 16


# --- HeatSolver Solve Tests ---


class TestHeatSolverSolve:
    """Tests for HeatSolver.solve method."""

    def test_solve_output_shape(self, solver: HeatSolver, initial_condition):
        """Test solve produces output with correct shape."""
        result = solver.solve(initial_condition)

        assert result.shape == initial_condition.shape
        assert result.dtype == numpy.float32

    @pytest.mark.parametrize("grid_size", GRID_SIZES)
    def test_solve_different_grid_sizes(self, grid_size: int):
        """Test solve works on different grid sizes."""
        solver = HeatSolver(resolution=grid_size)
        u0 = generate_random_field(grid_size=grid_size, smooth=True, seed=42)

        result = solver.solve(u0)

        assert result.shape == (grid_size, grid_size)
        assert result.dtype == numpy.float32

    def test_solve_deterministic(self, solver: HeatSolver, initial_condition):
        """Test solver is deterministic for same input."""
        r1 = solver.solve(initial_condition)
        r2 = solver.solve(initial_condition)

        assert numpy.allclose(r1, r2)

    def test_solve_no_nan_inf(self, solver: HeatSolver, initial_condition):
        """Test solution contains no NaN or Inf values."""
        result = solver.solve(initial_condition)

        assert numpy.all(numpy.isfinite(result))

    def test_zero_initial_stays_zero(self, solver: HeatSolver):
        """Test zero initial condition stays zero."""
        u0 = numpy.zeros((DEFAULT_GRID, DEFAULT_GRID), dtype=numpy.float32)
        result = solver.solve(u0)

        assert numpy.allclose(result, 0.0, atol=1e-7)


# --- Time Evolution Tests ---


class TestHeatTimeEvolution:
    """Tests for heat equation time evolution properties."""

    def test_solution_changes_over_time(self):
        """Test that solution evolves (is not identical to initial condition)."""
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)

        solver = HeatSolver(alpha=0.01, total_time=0.5, resolution=DEFAULT_GRID)
        result = solver.solve(u0)

        # Solution should differ from initial condition (diffusion occurred)
        assert not numpy.allclose(result, u0, atol=1e-4)

    def test_diffusion_smooths_field(self):
        """Test that diffusion makes the field smoother over time."""
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=False, seed=42)

        solver = HeatSolver(alpha=0.05, total_time=1.0, resolution=DEFAULT_GRID)
        result = solver.solve(u0)

        # Compute variance of gradients as smoothness measure
        grad_x_initial = numpy.diff(u0, axis=0)
        grad_x_final = numpy.diff(result, axis=0)

        assert grad_x_final.var() < grad_x_initial.var()

    def test_more_time_more_diffusion(self):
        """Test that longer time produces more diffusion (smoother result)."""
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)

        solver_short = HeatSolver(alpha=0.01, total_time=0.1, resolution=DEFAULT_GRID)
        solver_long = HeatSolver(alpha=0.01, total_time=2.0, resolution=DEFAULT_GRID)

        result_short = solver_short.solve(u0)
        result_long = solver_long.solve(u0)

        # Longer time -> closer to uniform (smaller variance)
        assert result_long.var() <= result_short.var() + 1e-10

    def test_higher_alpha_faster_diffusion(self):
        """Test that higher thermal diffusivity causes faster diffusion."""
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)

        solver_slow = HeatSolver(alpha=0.001, total_time=1.0, resolution=DEFAULT_GRID)
        solver_fast = HeatSolver(alpha=0.1, total_time=1.0, resolution=DEFAULT_GRID)

        result_slow = solver_slow.solve(u0)
        result_fast = solver_fast.solve(u0)

        # Higher alpha -> smoother result (lower variance)
        assert result_fast.var() <= result_slow.var() + 1e-10


# --- Conservation Properties Tests ---


class TestHeatConservation:
    """Tests for conservation properties of the heat equation."""

    def test_mean_temperature_conserved(self, solver: HeatSolver):
        """Test mean temperature is conserved (periodic BCs, no sources).

        For the heat equation with periodic BCs and no source term,
        the mean temperature is conserved: mean(u(t)) = mean(u(0)).
        """
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)
        result = solver.solve(u0)

        # Mean should be conserved (periodic BCs preserve integral)
        assert numpy.allclose(numpy.mean(result), numpy.mean(u0), rtol=1e-4)

    def test_energy_decreases(self, solver: HeatSolver):
        """Test that L2 norm (energy) decreases over time.

        The heat equation is dissipative: ||u(t)||_2 <= ||u(0)||_2.
        """
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)
        # Remove mean to avoid trivial constant solution
        u0_centered = u0 - numpy.mean(u0)

        solver_dissipative = HeatSolver(alpha=0.01, total_time=1.0, resolution=DEFAULT_GRID)
        result = solver_dissipative.solve(u0_centered)

        energy_initial = numpy.sum(u0_centered**2)
        energy_final = numpy.sum(result**2)

        assert energy_final <= energy_initial + 1e-6


# --- Steady State Tests ---


class TestHeatSteadyState:
    """Tests for steady state convergence of the heat equation."""

    def test_steady_state_convergence(self):
        """Test that long-time solution converges to steady state (uniform)."""
        u0 = generate_random_field(grid_size=DEFAULT_GRID, smooth=True, seed=42)

        # Very long time -> should converge to mean
        solver = HeatSolver(alpha=0.1, total_time=100.0, resolution=DEFAULT_GRID)
        result = solver.solve(u0)

        # Should be approximately uniform (equal to mean of initial)
        mean_val = numpy.mean(u0)
        assert numpy.allclose(result, mean_val, atol=1e-3)

    def test_uniform_initial_stays_uniform(self):
        """Test that a uniform initial condition remains uniform."""
        value = 5.0
        u0 = numpy.full((DEFAULT_GRID, DEFAULT_GRID), value, dtype=numpy.float32)

        solver = HeatSolver(alpha=0.01, total_time=1.0, resolution=DEFAULT_GRID)
        result = solver.solve(u0)

        assert numpy.allclose(result, value, atol=1e-5)


# --- HeatSolver Generate Sample Tests ---


class TestHeatSolverGenerateSample:
    """Tests for HeatSolver.generate_sample method."""

    def test_generate_sample_returns_heat_sample(self, solver: HeatSolver):
        """Test generate_sample returns HeatSample."""
        sample = solver.generate_sample(seed=42)
        assert isinstance(sample, HeatSample)

    def test_generate_sample_shapes(self, solver: HeatSolver):
        """Test generated sample has correct shapes."""
        sample = solver.generate_sample(seed=42)
        n_points = solver.resolution**2

        assert sample.input_field.shape == (n_points,)
        assert sample.output_field.shape == (n_points,)
        assert sample.coords.shape == (n_points, 2)
        assert sample.grid_size == solver.resolution

    def test_generate_sample_deterministic(self, solver: HeatSolver):
        """Test sample generation is deterministic with seed."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        assert numpy.allclose(sample1.input_field, sample2.input_field)
        assert numpy.allclose(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: HeatSolver):
        """Test different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=43)

        assert not numpy.allclose(sample1.input_field, sample2.input_field)

    def test_generate_sample_coords_normalized(self, solver: HeatSolver):
        """Test coordinates are in [0, 1]."""
        sample = solver.generate_sample(seed=42)

        assert sample.coords.min() >= 0.0
        assert sample.coords.max() <= 1.0

    def test_generate_sample_metadata(self, solver: HeatSolver):
        """Test sample metadata contains solver parameters."""
        sample = solver.generate_sample(seed=42)

        assert sample.metadata is not None
        assert "alpha" in sample.metadata
        assert "total_time" in sample.metadata
        assert sample.metadata["alpha"] == solver.alpha
        assert sample.metadata["total_time"] == solver.total_time
