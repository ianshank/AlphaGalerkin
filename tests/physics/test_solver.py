"""Tests for physics solver base classes.

Tests cover:
- PhysicsSample: Generic physics sample dataclass
- DiffEqSolver: Abstract base solver
- generate_random_field: Random field generation
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from src.physics.solver import DiffEqSolver, PhysicsSample, generate_random_field

# --- PhysicsSample Tests ---


class TestPhysicsSample:
    """Tests for PhysicsSample dataclass."""

    @pytest.fixture
    def sample_data(self) -> dict:
        """Create sample data for testing."""
        grid_size = 4
        n_points = grid_size**2

        return {
            "input_field": numpy.random.randn(n_points).astype(numpy.float32),
            "output_field": numpy.random.randn(n_points).astype(numpy.float32),
            "coords": numpy.random.rand(n_points, 2).astype(numpy.float32),
            "grid_size": grid_size,
        }

    def test_init_with_required_fields(self, sample_data: dict):
        """Test initialization with required fields."""
        sample = PhysicsSample(**sample_data)

        assert sample.grid_size == sample_data["grid_size"]
        assert numpy.array_equal(sample.input_field, sample_data["input_field"])
        assert numpy.array_equal(sample.output_field, sample_data["output_field"])
        assert numpy.array_equal(sample.coords, sample_data["coords"])

    def test_init_with_metadata(self, sample_data: dict):
        """Test initialization with metadata."""
        metadata = {"viscosity": 0.1, "time": 1.0}
        sample = PhysicsSample(**sample_data, metadata=metadata)

        assert sample.metadata == metadata
        assert sample.metadata["viscosity"] == 0.1

    def test_default_metadata_is_none(self, sample_data: dict):
        """Test default metadata is None."""
        sample = PhysicsSample(**sample_data)
        assert sample.metadata is None

    def test_generic_types(self, sample_data: dict):
        """Test sample works with different field types."""
        # Vector field (e.g., for elasticity)
        vector_input = numpy.random.randn(16, 2).astype(numpy.float32)
        vector_output = numpy.random.randn(16, 2).astype(numpy.float32)

        sample = PhysicsSample(
            input_field=vector_input,
            output_field=vector_output,
            coords=sample_data["coords"],
            grid_size=4,
        )

        assert sample.input_field.shape == (16, 2)
        assert sample.output_field.shape == (16, 2)


# --- DiffEqSolver Tests ---


class TestDiffEqSolver:
    """Tests for DiffEqSolver abstract base class."""

    def test_is_abstract(self):
        """Test DiffEqSolver cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            DiffEqSolver()

    def test_default_resolution(self):
        """Test default resolution value via concrete implementation."""

        class ConcreteSolver(DiffEqSolver):
            def solve(self, input_field):
                return input_field

            def generate_sample(self, seed=None):
                return PhysicsSample(
                    input_field=numpy.zeros(1),
                    output_field=numpy.zeros(1),
                    coords=numpy.zeros((1, 2)),
                    grid_size=1,
                )

        solver = ConcreteSolver()
        assert solver.resolution == 32

    def test_custom_resolution(self):
        """Test custom resolution initialization."""

        class ConcreteSolver(DiffEqSolver):
            def solve(self, input_field):
                return input_field

            def generate_sample(self, seed=None):
                return PhysicsSample(
                    input_field=numpy.zeros(1),
                    output_field=numpy.zeros(1),
                    coords=numpy.zeros((1, 2)),
                    grid_size=1,
                )

        solver = ConcreteSolver(resolution=64)
        assert solver.resolution == 64

    def test_get_grid_coords(self):
        """Test grid coordinate generation."""

        class ConcreteSolver(DiffEqSolver):
            def solve(self, input_field):
                return input_field

            def generate_sample(self, seed=None):
                return PhysicsSample(
                    input_field=numpy.zeros(1),
                    output_field=numpy.zeros(1),
                    coords=numpy.zeros((1, 2)),
                    grid_size=1,
                )

        solver = ConcreteSolver()
        coords = solver._get_grid_coords(4)

        # Should have N^2 coordinates
        assert coords.shape == (16, 2)

        # Values should be in [0, 1]
        assert coords.min() >= 0.0
        assert coords.max() <= 1.0

        # Should include corner points
        assert numpy.any(numpy.all(coords == [0.0, 0.0], axis=1))

    def test_get_grid_coords_dtype(self):
        """Test grid coordinates are float32."""

        class ConcreteSolver(DiffEqSolver):
            def solve(self, input_field):
                return input_field

            def generate_sample(self, seed=None):
                return PhysicsSample(
                    input_field=numpy.zeros(1),
                    output_field=numpy.zeros(1),
                    coords=numpy.zeros((1, 2)),
                    grid_size=1,
                )

        solver = ConcreteSolver()
        coords = solver._get_grid_coords(5)

        assert coords.dtype == numpy.float32


# --- generate_random_field Tests ---


class TestGenerateRandomField:
    """Tests for generate_random_field function."""

    def test_output_shape(self):
        """Test output has correct shape."""
        field = generate_random_field(grid_size=8)
        assert field.shape == (8, 8)

    def test_output_dtype(self):
        """Test output is float32."""
        field = generate_random_field(grid_size=8)
        assert field.dtype == numpy.float32

    def test_deterministic_with_seed(self):
        """Test same seed produces same field."""
        field1 = generate_random_field(grid_size=8, seed=42)
        field2 = generate_random_field(grid_size=8, seed=42)

        assert numpy.allclose(field1, field2)

    def test_different_seeds_produce_different_fields(self):
        """Test different seeds produce different fields."""
        field1 = generate_random_field(grid_size=8, seed=42)
        field2 = generate_random_field(grid_size=8, seed=43)

        assert not numpy.allclose(field1, field2)

    def test_continuous_field_default(self):
        """Test default produces continuous field."""
        field = generate_random_field(grid_size=8, seed=42)

        # Continuous field should have many non-zero values
        non_zero_count = numpy.count_nonzero(field)
        assert non_zero_count > 32  # At least half

    def test_point_sources(self):
        """Test sparse point source generation."""
        field = generate_random_field(grid_size=16, n_sources=3, smooth=False, seed=42)

        # With 3 sources and no smoothing, should have few non-zero values
        non_zero_count = numpy.count_nonzero(field)
        assert non_zero_count <= 3

    def test_source_std_affects_magnitude(self):
        """Test source_std affects field magnitude."""
        field_low = generate_random_field(grid_size=8, source_std=0.1, seed=42)
        field_high = generate_random_field(grid_size=8, source_std=10.0, seed=42)

        # Higher std should produce larger magnitudes
        assert numpy.abs(field_high).max() > numpy.abs(field_low).max()

    def test_smoothing_applied(self):
        """Test Gaussian smoothing is applied."""
        field_smooth = generate_random_field(grid_size=16, smooth=True, seed=42)
        field_raw = generate_random_field(grid_size=16, smooth=False, seed=42)

        # Smoothed field should have smaller variance locally
        # Check variance of gradients
        grad_smooth = numpy.diff(field_smooth, axis=0)
        grad_raw = numpy.diff(field_raw, axis=0)

        assert grad_smooth.var() < grad_raw.var()

    def test_no_smoothing(self):
        """Test smoothing can be disabled."""
        field = generate_random_field(grid_size=8, smooth=False, seed=42)
        # Should still produce valid output
        assert field.shape == (8, 8)

    def test_large_grid(self):
        """Test with larger grid size."""
        field = generate_random_field(grid_size=64, seed=42)
        assert field.shape == (64, 64)

    def test_small_grid(self):
        """Test with small grid size."""
        field = generate_random_field(grid_size=3, seed=42)
        assert field.shape == (3, 3)


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests for physics solver module."""

    def test_single_point_grid(self):
        """Test field generation with 1x1 grid."""
        field = generate_random_field(grid_size=1, seed=42)
        assert field.shape == (1, 1)

    def test_many_sources_small_grid(self):
        """Test more sources than grid cells."""
        # Multiple sources can land on same cell
        field = generate_random_field(
            grid_size=3, n_sources=20, smooth=False, seed=42
        )
        assert field.shape == (3, 3)

    def test_zero_sources(self):
        """Test with zero sources specified."""
        field = generate_random_field(grid_size=8, n_sources=0, smooth=False, seed=42)
        # Zero sources should produce all-zero field
        assert numpy.allclose(field, 0.0)

    def test_physics_sample_with_empty_metadata(self):
        """Test PhysicsSample with empty metadata dict."""
        sample = PhysicsSample(
            input_field=numpy.zeros(4),
            output_field=numpy.zeros(4),
            coords=numpy.zeros((4, 2)),
            grid_size=2,
            metadata={},
        )
        assert sample.metadata == {}
