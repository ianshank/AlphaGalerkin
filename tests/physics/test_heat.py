"""Tests for Heat equation solver.

Tests cover:
- HeatSample: Heat-specific sample dataclass
- HeatSolver: Initialization, spectral solve, sample generation
- Physical properties: decay, energy conservation, time stepping
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from src.physics.heat import HeatSample, HeatSolver

# --- Fixtures ---


@pytest.fixture
def solver() -> HeatSolver:
    """Create default heat solver with small grid."""
    return HeatSolver(resolution=16)


@pytest.fixture
def solver_fast() -> HeatSolver:
    """Create heat solver with fast diffusion."""
    return HeatSolver(alpha=0.1, total_time=1.0, resolution=16)


@pytest.fixture
def solver_slow() -> HeatSolver:
    """Create heat solver with slow diffusion."""
    return HeatSolver(alpha=0.001, total_time=0.1, resolution=16)


# --- HeatSample Tests ---


class TestHeatSample:
    """Tests for HeatSample dataclass."""

    def test_sample_fields(self) -> None:
        """Test HeatSample has expected fields."""
        n = 16
        sample = HeatSample(
            input_field=numpy.ones(n * n, dtype=numpy.float32),
            output_field=numpy.zeros(n * n, dtype=numpy.float32),
            coords=numpy.random.rand(n * n, 2).astype(numpy.float32),
            grid_size=n,
            metadata={"alpha": 0.01, "total_time": 1.0},
        )

        assert sample.input_field.shape == (n * n,)
        assert sample.output_field.shape == (n * n,)
        assert sample.grid_size == n
        assert sample.metadata["alpha"] == 0.01

    def test_sample_with_none_metadata(self) -> None:
        """Test HeatSample with no metadata."""
        n = 8
        sample = HeatSample(
            input_field=numpy.ones(n * n, dtype=numpy.float32),
            output_field=numpy.zeros(n * n, dtype=numpy.float32),
            coords=numpy.random.rand(n * n, 2).astype(numpy.float32),
            grid_size=n,
        )
        assert sample.metadata is None


# --- HeatSolver Initialization Tests ---


class TestHeatSolverInit:
    """Tests for HeatSolver initialization."""

    def test_default_init(self) -> None:
        """Test default solver parameters."""
        solver = HeatSolver()

        assert solver.alpha == 0.01
        assert solver.time_step == 0.001
        assert solver.total_time == 1.0
        assert solver.resolution == 32

    def test_custom_init(self) -> None:
        """Test custom solver parameters."""
        solver = HeatSolver(alpha=0.05, time_step=0.01, total_time=2.0, resolution=64)

        assert solver.alpha == 0.05
        assert solver.time_step == 0.01
        assert solver.total_time == 2.0
        assert solver.resolution == 64

    def test_zero_alpha(self) -> None:
        """Test solver with zero diffusivity preserves initial condition."""
        solver = HeatSolver(alpha=0.0, resolution=8)
        u0 = numpy.random.randn(8, 8).astype(numpy.float32)
        result = solver.solve(u0)
        numpy.testing.assert_allclose(result, u0, atol=1e-6)

    def test_zero_time(self) -> None:
        """Test solver with zero total time preserves initial condition."""
        solver = HeatSolver(total_time=0.0, resolution=8)
        u0 = numpy.random.randn(8, 8).astype(numpy.float32)
        result = solver.solve(u0)
        numpy.testing.assert_allclose(result, u0, atol=1e-6)


# --- Solve Tests ---


class TestHeatSolverSolve:
    """Tests for HeatSolver.solve."""

    def test_solve_shape(self, solver: HeatSolver) -> None:
        """Test output shape matches input."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)
        result = solver.solve(u0)

        assert result.shape == (16, 16)
        assert result.dtype == numpy.float32

    def test_solve_finite(self, solver: HeatSolver) -> None:
        """Test output is finite."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)
        result = solver.solve(u0)

        assert numpy.all(numpy.isfinite(result))

    def test_diffusion_smoothing(self, solver: HeatSolver) -> None:
        """Test that diffusion smooths the initial condition."""
        u0 = numpy.zeros((16, 16), dtype=numpy.float32)
        u0[8, 8] = 10.0

        result = solver.solve(u0)

        assert numpy.max(result) < numpy.max(u0)

    def test_constant_field_unchanged(self, solver: HeatSolver) -> None:
        """Test that a constant field remains constant under diffusion."""
        u0 = numpy.ones((16, 16), dtype=numpy.float32) * 5.0

        result = solver.solve(u0)

        numpy.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_energy_decay(self, solver: HeatSolver) -> None:
        """Test that total energy (L2 norm) decreases or stays same."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)

        result = solver.solve(u0)

        energy_initial = numpy.sum(u0**2)
        energy_final = numpy.sum(result**2)

        assert energy_final <= energy_initial + 1e-6

    def test_zero_initial_condition(self, solver: HeatSolver) -> None:
        """Test that zero initial condition remains zero."""
        u0 = numpy.zeros((16, 16), dtype=numpy.float32)

        result = solver.solve(u0)

        numpy.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_higher_alpha_faster_decay(self) -> None:
        """Test that higher thermal diffusivity gives faster decay."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)

        solver_low = HeatSolver(alpha=0.001, total_time=1.0, resolution=16)
        solver_high = HeatSolver(alpha=0.1, total_time=1.0, resolution=16)

        result_low = solver_low.solve(u0.copy())
        result_high = solver_high.solve(u0.copy())

        energy_low = numpy.sum(result_low**2)
        energy_high = numpy.sum(result_high**2)

        assert energy_high < energy_low

    def test_longer_time_more_decay(self) -> None:
        """Test that longer time gives more diffusion."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)

        solver_short = HeatSolver(alpha=0.01, total_time=0.1, resolution=16)
        solver_long = HeatSolver(alpha=0.01, total_time=10.0, resolution=16)

        result_short = solver_short.solve(u0.copy())
        result_long = solver_long.solve(u0.copy())

        energy_short = numpy.sum(result_short**2)
        energy_long = numpy.sum(result_long**2)

        assert energy_long < energy_short

    def test_mean_preservation(self, solver: HeatSolver) -> None:
        """Test that mean is preserved (periodic BCs conserve integral)."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)

        result = solver.solve(u0)

        numpy.testing.assert_allclose(
            numpy.mean(result), numpy.mean(u0), atol=1e-5,
            err_msg="Mean should be preserved under periodic heat equation"
        )

    def test_small_resolution(self) -> None:
        """Test solve works with small resolution."""
        solver = HeatSolver(resolution=4)
        u0 = numpy.random.randn(4, 4).astype(numpy.float32)
        result = solver.solve(u0)
        assert result.shape == (4, 4)
        assert numpy.all(numpy.isfinite(result))


# --- Time Stepping Tests ---


class TestHeatSolverTimeStepping:
    """Tests for time-stepping related properties."""

    def test_monotonic_energy_decay(self) -> None:
        """Test energy decays monotonically across time steps."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)
        energies = []

        for t in [0.0, 0.1, 0.5, 1.0, 5.0]:
            solver = HeatSolver(alpha=0.01, total_time=t, resolution=16)
            result = solver.solve(u0.copy())
            energies.append(numpy.sum(result**2))

        # Energy should be non-increasing
        for i in range(len(energies) - 1):
            assert energies[i + 1] <= energies[i] + 1e-6

    def test_long_time_approaches_mean(self) -> None:
        """Test that very long diffusion approaches uniform (mean) field."""
        u0 = numpy.random.randn(16, 16).astype(numpy.float32)
        mean_val = numpy.mean(u0)

        solver = HeatSolver(alpha=0.1, total_time=100.0, resolution=16)
        result = solver.solve(u0)

        numpy.testing.assert_allclose(result, mean_val, atol=1e-3)


# --- Generate Sample Tests ---


class TestHeatSolverGenerateSample:
    """Tests for HeatSolver.generate_sample."""

    def test_generate_sample_structure(self, solver: HeatSolver) -> None:
        """Test generated sample has correct structure."""
        sample = solver.generate_sample(seed=42)

        assert isinstance(sample, HeatSample)
        assert sample.input_field.shape == (256,)  # 16*16
        assert sample.output_field.shape == (256,)
        assert sample.coords.shape == (256, 2)
        assert sample.grid_size == 16

    def test_generate_sample_metadata(self, solver: HeatSolver) -> None:
        """Test generated sample metadata."""
        sample = solver.generate_sample(seed=42)

        assert sample.metadata is not None
        assert "alpha" in sample.metadata
        assert "total_time" in sample.metadata
        assert sample.metadata["alpha"] == 0.01
        assert sample.metadata["total_time"] == 1.0

    def test_generate_sample_reproducibility(self, solver: HeatSolver) -> None:
        """Test same seed produces same sample."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        numpy.testing.assert_array_equal(sample1.input_field, sample2.input_field)
        numpy.testing.assert_array_equal(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: HeatSolver) -> None:
        """Test different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=123)

        assert not numpy.array_equal(sample1.input_field, sample2.input_field)

    def test_generate_sample_finite(self, solver: HeatSolver) -> None:
        """Test all generated fields are finite."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(numpy.isfinite(sample.input_field))
        assert numpy.all(numpy.isfinite(sample.output_field))

    def test_generate_sample_coords_range(self, solver: HeatSolver) -> None:
        """Test coordinates are in [0, 1]."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(sample.coords >= 0.0)
        assert numpy.all(sample.coords <= 1.0)

    def test_generate_sample_output_smoother(self, solver_fast: HeatSolver) -> None:
        """Test that output is smoother than input (fast diffusion)."""
        sample = solver_fast.generate_sample(seed=42)

        input_2d = sample.input_field.reshape(16, 16)
        output_2d = sample.output_field.reshape(16, 16)

        # Compute gradient magnitude as roughness proxy
        grad_input = numpy.sum(numpy.diff(input_2d, axis=0) ** 2) + numpy.sum(
            numpy.diff(input_2d, axis=1) ** 2
        )
        grad_output = numpy.sum(numpy.diff(output_2d, axis=0) ** 2) + numpy.sum(
            numpy.diff(output_2d, axis=1) ** 2
        )

        assert grad_output < grad_input, "Output should be smoother than input"
