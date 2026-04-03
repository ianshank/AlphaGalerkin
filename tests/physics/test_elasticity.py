"""Tests for Linear Elasticity equation solver.

Tests cover:
- ElasticitySample: Elasticity-specific sample dataclass
- ElasticitySolver: Initialization, Lame parameters, spectral solve
- Physical properties: symmetry, zero force, material parameter effects
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from src.physics.elasticity import ElasticitySample, ElasticitySolver
from src.physics.solver import generate_random_field


# --- Fixtures ---


@pytest.fixture
def solver() -> ElasticitySolver:
    """Create default elasticity solver with small grid."""
    return ElasticitySolver(resolution=16)


@pytest.fixture
def solver_stiff() -> ElasticitySolver:
    """Create solver with high stiffness."""
    return ElasticitySolver(young_modulus=10.0, poisson_ratio=0.3, resolution=16)


@pytest.fixture
def solver_soft() -> ElasticitySolver:
    """Create solver with low stiffness."""
    return ElasticitySolver(young_modulus=0.1, poisson_ratio=0.3, resolution=16)


@pytest.fixture
def zero_mean_force() -> numpy.ndarray:
    """Create a zero-mean force field for periodic stability."""
    n = 16
    rng = numpy.random.default_rng(42)
    F = rng.normal(0, 1, (n * n, 2)).astype(numpy.float32)
    # Enforce zero mean for periodic equilibrium
    F -= F.mean(axis=0)
    return F


# --- ElasticitySample Tests ---


class TestElasticitySample:
    """Tests for ElasticitySample dataclass."""

    def test_sample_fields(self) -> None:
        """Test ElasticitySample has expected fields."""
        n = 16
        sample = ElasticitySample(
            input_field=numpy.ones((n * n, 2), dtype=numpy.float32),
            output_field=numpy.zeros((n * n, 2), dtype=numpy.float32),
            coords=numpy.random.rand(n * n, 2).astype(numpy.float32),
            grid_size=n,
            metadata={"E": 1.0, "nu": 0.3},
        )

        assert sample.input_field.shape == (n * n, 2)
        assert sample.output_field.shape == (n * n, 2)
        assert sample.grid_size == n
        assert sample.metadata["E"] == 1.0
        assert sample.metadata["nu"] == 0.3


# --- ElasticitySolver Initialization Tests ---


class TestElasticitySolverInit:
    """Tests for ElasticitySolver initialization."""

    def test_default_init(self) -> None:
        """Test default solver parameters."""
        solver = ElasticitySolver()

        assert solver.E == 1.0
        assert solver.nu == 0.3
        assert solver.resolution == 32

    def test_custom_init(self) -> None:
        """Test custom solver parameters."""
        solver = ElasticitySolver(young_modulus=200.0, poisson_ratio=0.25, resolution=64)

        assert solver.E == 200.0
        assert solver.nu == 0.25
        assert solver.resolution == 64

    def test_lame_parameters(self) -> None:
        """Test Lame parameter computation."""
        E = 1.0
        nu = 0.3
        solver = ElasticitySolver(young_modulus=E, poisson_ratio=nu)

        expected_mu = E / (2 * (1 + nu))
        expected_lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))

        numpy.testing.assert_allclose(solver.mu, expected_mu, rtol=1e-10)
        numpy.testing.assert_allclose(solver.lam, expected_lam, rtol=1e-10)

    def test_lame_parameters_steel(self) -> None:
        """Test Lame parameters for steel-like material."""
        solver = ElasticitySolver(young_modulus=200e9, poisson_ratio=0.3)

        # Mu (shear modulus) should be positive
        assert solver.mu > 0
        # Lambda should be positive for nu < 0.5
        assert solver.lam > 0

    def test_lame_parameters_rubber(self) -> None:
        """Test Lame parameters for nearly-incompressible material."""
        solver = ElasticitySolver(young_modulus=1.0, poisson_ratio=0.499)

        # Lambda should be very large for nearly-incompressible materials
        assert solver.lam > solver.mu * 10


# --- Solve Tests ---


class TestElasticitySolverSolve:
    """Tests for ElasticitySolver.solve."""

    def test_solve_shape(self, solver: ElasticitySolver, zero_mean_force: numpy.ndarray) -> None:
        """Test output shape matches expected displacement shape."""
        result = solver.solve(zero_mean_force)

        assert result.shape == (16 * 16, 2)
        assert result.dtype == numpy.float32

    def test_solve_finite(self, solver: ElasticitySolver, zero_mean_force: numpy.ndarray) -> None:
        """Test output is finite."""
        result = solver.solve(zero_mean_force)

        assert numpy.all(numpy.isfinite(result))

    def test_zero_force_zero_displacement(self, solver: ElasticitySolver) -> None:
        """Test that zero force gives zero displacement."""
        n = 16
        F = numpy.zeros((n * n, 2), dtype=numpy.float32)

        result = solver.solve(F)

        numpy.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_displacement_proportional_to_force(
        self, solver: ElasticitySolver, zero_mean_force: numpy.ndarray
    ) -> None:
        """Test linear scaling: 2*F -> 2*u."""
        u1 = solver.solve(zero_mean_force)
        u2 = solver.solve(2.0 * zero_mean_force)

        numpy.testing.assert_allclose(u2, 2.0 * u1, atol=1e-5)

    def test_stiffer_material_less_displacement(
        self,
        solver_stiff: ElasticitySolver,
        solver_soft: ElasticitySolver,
        zero_mean_force: numpy.ndarray,
    ) -> None:
        """Test that stiffer material gives less displacement."""
        u_stiff = solver_stiff.solve(zero_mean_force)
        u_soft = solver_soft.solve(zero_mean_force)

        norm_stiff = numpy.linalg.norm(u_stiff)
        norm_soft = numpy.linalg.norm(u_soft)

        assert norm_stiff < norm_soft, "Stiffer material should yield smaller displacement"

    def test_x_only_force(self, solver: ElasticitySolver) -> None:
        """Test force only in x-direction."""
        n = 16
        Fx = generate_random_field(n, smooth=True, seed=42)
        Fx -= numpy.mean(Fx)
        Fy = numpy.zeros((n, n), dtype=numpy.float32)
        F = numpy.stack([Fx, Fy], axis=-1).reshape(-1, 2).astype(numpy.float32)

        result = solver.solve(F)

        # X displacement should be nonzero
        ux = result[:, 0]
        assert numpy.max(numpy.abs(ux)) > 1e-10

    def test_solve_different_resolutions(self) -> None:
        """Test solver works at different resolutions."""
        for res in [8, 16, 32]:
            solver = ElasticitySolver(resolution=res)
            n = res
            rng = numpy.random.default_rng(42)
            F = rng.normal(0, 1, (n * n, 2)).astype(numpy.float32)
            F -= F.mean(axis=0)

            result = solver.solve(F)

            assert result.shape == (n * n, 2)
            assert numpy.all(numpy.isfinite(result))


# --- Generate Sample Tests ---


class TestElasticitySolverGenerateSample:
    """Tests for ElasticitySolver.generate_sample."""

    def test_generate_sample_structure(self, solver: ElasticitySolver) -> None:
        """Test generated sample has correct structure."""
        sample = solver.generate_sample(seed=42)

        assert isinstance(sample, ElasticitySample)
        assert sample.input_field.shape == (16 * 16, 2)
        assert sample.output_field.shape == (16 * 16, 2)
        assert sample.coords.shape == (16 * 16, 2)
        assert sample.grid_size == 16

    def test_generate_sample_metadata(self, solver: ElasticitySolver) -> None:
        """Test generated sample metadata."""
        sample = solver.generate_sample(seed=42)

        assert sample.metadata is not None
        assert "E" in sample.metadata
        assert "nu" in sample.metadata
        assert sample.metadata["E"] == 1.0
        assert sample.metadata["nu"] == 0.3

    def test_generate_sample_reproducibility(self, solver: ElasticitySolver) -> None:
        """Test same seed produces same sample."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=42)

        numpy.testing.assert_array_equal(sample1.input_field, sample2.input_field)
        numpy.testing.assert_array_equal(sample1.output_field, sample2.output_field)

    def test_generate_sample_different_seeds(self, solver: ElasticitySolver) -> None:
        """Test different seeds produce different samples."""
        sample1 = solver.generate_sample(seed=42)
        sample2 = solver.generate_sample(seed=123)

        assert not numpy.array_equal(sample1.input_field, sample2.input_field)

    def test_generate_sample_finite(self, solver: ElasticitySolver) -> None:
        """Test all generated fields are finite."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(numpy.isfinite(sample.input_field))
        assert numpy.all(numpy.isfinite(sample.output_field))

    def test_generate_sample_zero_mean_force(self, solver: ElasticitySolver) -> None:
        """Test that generated force has zero mean (equilibrium condition)."""
        sample = solver.generate_sample(seed=42)

        F = sample.input_field  # (N*N, 2)
        mean_force = numpy.mean(F, axis=0)

        numpy.testing.assert_allclose(
            mean_force, 0.0, atol=1e-5,
            err_msg="Force field should have zero mean for periodic equilibrium"
        )

    def test_generate_sample_coords_range(self, solver: ElasticitySolver) -> None:
        """Test coordinates are in [0, 1]."""
        sample = solver.generate_sample(seed=42)

        assert numpy.all(sample.coords >= 0.0)
        assert numpy.all(sample.coords <= 1.0)

    def test_generate_sample_no_seed(self, solver: ElasticitySolver) -> None:
        """Test sample generation without seed."""
        sample = solver.generate_sample(seed=None)

        assert isinstance(sample, ElasticitySample)
        assert numpy.all(numpy.isfinite(sample.output_field))


# --- Material Parameter Effect Tests ---


class TestMaterialParameters:
    """Tests for material parameter effects on solutions."""

    def test_poisson_ratio_zero(self) -> None:
        """Test with zero Poisson ratio (no lateral coupling)."""
        solver = ElasticitySolver(young_modulus=1.0, poisson_ratio=0.0, resolution=16)

        # Lambda should be zero
        numpy.testing.assert_allclose(solver.lam, 0.0, atol=1e-10)

        sample = solver.generate_sample(seed=42)
        assert numpy.all(numpy.isfinite(sample.output_field))

    def test_different_young_moduli(self) -> None:
        """Test that displacement scales inversely with Young's modulus."""
        n = 16
        rng = numpy.random.default_rng(42)
        F = rng.normal(0, 1, (n * n, 2)).astype(numpy.float32)
        F -= F.mean(axis=0)

        solver_E1 = ElasticitySolver(young_modulus=1.0, resolution=n)
        solver_E2 = ElasticitySolver(young_modulus=2.0, resolution=n)

        u_E1 = solver_E1.solve(F)
        u_E2 = solver_E2.solve(F)

        # Double E -> roughly half the displacement
        # (Not exact due to Poisson coupling, but should be in ballpark)
        ratio = numpy.linalg.norm(u_E1) / numpy.linalg.norm(u_E2)
        assert 1.5 < ratio < 2.5, f"Expected ~2x displacement ratio, got {ratio}"
