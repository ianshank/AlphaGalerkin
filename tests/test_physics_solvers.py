"""Comprehensive test suite for Physics Solvers."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.physics.darcy import DarcyFlowSolver, DarcySample
from src.physics.elasticity import ElasticitySample, ElasticitySolver
from src.physics.heat import HeatSample, HeatSolver
from src.physics.poisson import PoissonSample, PoissonSolver
from src.physics.solver import generate_random_field

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(params=[16, 32])
def resolution(request):
    """Parameterized resolution fixture."""
    return request.param


@pytest.fixture
def heat_solver(resolution):
    """Heat solver fixture with configurable resolution."""
    return HeatSolver(resolution=resolution, alpha=0.01, total_time=0.5)


@pytest.fixture
def darcy_solver(resolution):
    """Darcy solver fixture with configurable resolution."""
    return DarcyFlowSolver(resolution=resolution, forcing=1.0)


@pytest.fixture
def elasticity_solver(resolution):
    """Elasticity solver fixture with configurable resolution."""
    return ElasticitySolver(resolution=resolution, young_modulus=1.0, poisson_ratio=0.3)


@pytest.fixture
def poisson_solver(resolution):
    """Poisson solver fixture with configurable resolution."""
    return PoissonSolver(resolution=resolution)


# =============================================================================
# Unit Tests: generate_random_field
# =============================================================================


class TestGenerateRandomField:
    """Tests for the random field generator utility."""

    def test_output_shape(self, resolution):
        """Verify output has correct shape."""
        field = generate_random_field(resolution, seed=42)
        assert field.shape == (resolution, resolution)

    def test_dtype(self, resolution):
        """Verify output has correct dtype."""
        field = generate_random_field(resolution, seed=42)
        assert field.dtype == np.float32

    def test_reproducibility(self, resolution):
        """Verify same seed produces same output."""
        f1 = generate_random_field(resolution, seed=123)
        f2 = generate_random_field(resolution, seed=123)
        assert_allclose(f1, f2)

    def test_different_seeds_produce_different_fields(self, resolution):
        """Verify different seeds produce different outputs."""
        f1 = generate_random_field(resolution, seed=1)
        f2 = generate_random_field(resolution, seed=2)
        assert not np.allclose(f1, f2)

    def test_sparse_sources(self, resolution):
        """Verify sparse source generation."""
        n_sources = 5
        field = generate_random_field(resolution, n_sources=n_sources, smooth=False, seed=42)
        # Should have at most n_sources non-zero values (some may overlap)
        assert np.count_nonzero(field) <= n_sources


# =============================================================================
# Unit Tests: Heat Solver
# =============================================================================


class TestHeatSolver:
    """Tests for the Heat equation solver."""

    def test_solve_output_shape(self, heat_solver, resolution):
        """Verify solve returns correct shape."""
        u0 = np.random.randn(resolution, resolution).astype(np.float32)
        u_final = heat_solver.solve(u0)
        assert u_final.shape == u0.shape

    def test_conservation_of_mean(self, heat_solver, resolution):
        """For periodic BCs, mean should be conserved."""
        u0 = np.random.randn(resolution, resolution).astype(np.float32)
        u_final = heat_solver.solve(u0)
        assert_allclose(np.mean(u0), np.mean(u_final), rtol=1e-5)

    def test_diffusion_reduces_variance(self, heat_solver, resolution):
        """Variance should decrease after diffusion."""
        u0 = np.random.randn(resolution, resolution).astype(np.float32)
        u_final = heat_solver.solve(u0)
        assert np.var(u_final) < np.var(u0)

    def test_generate_sample_returns_heat_sample(self, heat_solver):
        """Verify generate_sample returns HeatSample."""
        sample = heat_solver.generate_sample(seed=42)
        assert isinstance(sample, HeatSample)

    def test_sample_has_correct_metadata(self, heat_solver):
        """Verify sample metadata contains expected keys."""
        sample = heat_solver.generate_sample(seed=42)
        assert "alpha" in sample.metadata
        assert "total_time" in sample.metadata


# =============================================================================
# Unit Tests: Darcy Flow Solver
# =============================================================================


class TestDarcyFlowSolver:
    """Tests for the Darcy flow solver."""

    def test_solve_output_shape(self, darcy_solver, resolution):
        """Verify solve returns correct shape."""
        permeability = np.ones((resolution, resolution), dtype=np.float32)
        u = darcy_solver.solve(permeability)
        assert u.shape == permeability.shape

    def test_constant_permeability_laplacian(self, resolution):
        """With constant permeability, should reduce to Laplacian."""
        # For constant permeability -a*Laplacian(u) = f
        # This is equivalent to Poisson.
        solver = DarcyFlowSolver(resolution=resolution, forcing=1.0)
        k = np.ones((resolution, resolution), dtype=np.float32)
        u = solver.solve(k)
        # Should be concave, max in middle
        center = resolution // 2
        assert u[center, center] > u[0, 0]

    def test_boundary_conditions(self, darcy_solver, resolution):
        """Verify Dirichlet boundary conditions (u=0 on boundary)."""
        k = np.ones((resolution, resolution), dtype=np.float32)
        u = darcy_solver.solve(k)
        assert_allclose(u[0, :], 0, atol=1e-7)
        assert_allclose(u[-1, :], 0, atol=1e-7)
        assert_allclose(u[:, 0], 0, atol=1e-7)
        assert_allclose(u[:, -1], 0, atol=1e-7)

    def test_generate_sample_returns_darcy_sample(self, darcy_solver):
        """Verify generate_sample returns DarcySample."""
        sample = darcy_solver.generate_sample(seed=42)
        assert isinstance(sample, DarcySample)

    def test_permeability_strictly_positive(self, darcy_solver):
        """Verify generated permeability is strictly positive."""
        sample = darcy_solver.generate_sample(seed=42)
        permeability = sample.input_field.reshape(darcy_solver.resolution, -1)
        assert np.all(permeability > 0)


# =============================================================================
# Unit Tests: Elasticity Solver
# =============================================================================


class TestElasticitySolver:
    """Tests for the Linear Elasticity solver."""

    def test_solve_output_shape(self, elasticity_solver, resolution):
        """Verify solve returns correct shape."""
        F = np.random.randn(resolution * resolution, 2).astype(np.float32)
        # Zero mean force for equilibrium
        F -= F.mean(axis=0)
        u = elasticity_solver.solve(F)
        assert u.shape == F.shape

    def test_zero_force_zero_displacement(self, elasticity_solver, resolution):
        """Zero force should give zero displacement."""
        F = np.zeros((resolution * resolution, 2), dtype=np.float32)
        u = elasticity_solver.solve(F)
        assert_allclose(u, 0, atol=1e-7)

    def test_generate_sample_returns_elasticity_sample(self, elasticity_solver):
        """Verify generate_sample returns ElasticitySample."""
        sample = elasticity_solver.generate_sample(seed=42)
        assert isinstance(sample, ElasticitySample)

    def test_sample_has_correct_metadata(self, elasticity_solver):
        """Verify sample metadata contains material properties."""
        sample = elasticity_solver.generate_sample(seed=42)
        assert "E" in sample.metadata
        assert "nu" in sample.metadata


# =============================================================================
# Unit Tests: Poisson Solver
# =============================================================================


class TestPoissonSolver:
    """Tests for the Poisson equation solver."""

    def test_solve_output_shape(self, poisson_solver, resolution):
        """Verify solve returns correct shape."""
        charges = np.random.randn(resolution, resolution).astype(np.float32)
        potential = poisson_solver.solve(charges)
        assert potential.shape == charges.shape

    def test_generate_sample_returns_poisson_sample(self, poisson_solver):
        """Verify generate_sample returns PoissonSample."""
        sample = poisson_solver.generate_sample(seed=42)
        assert isinstance(sample, PoissonSample)


# =============================================================================
# Integration Tests
# =============================================================================


class TestSolverIntegration:
    """Integration tests across all solvers."""

    @pytest.mark.parametrize(
        "SolverClass,kwargs",
        [
            (HeatSolver, {"alpha": 0.01, "total_time": 0.5}),
            (DarcyFlowSolver, {"forcing": 1.0}),
            (ElasticitySolver, {"young_modulus": 1.0, "poisson_ratio": 0.3}),
            (PoissonSolver, {}),
        ],
    )
    def test_sample_coordinates_normalized(self, SolverClass, kwargs, resolution):
        """All samples should have coordinates in [0, 1]."""
        solver = SolverClass(resolution=resolution, **kwargs)
        sample = solver.generate_sample(seed=42)
        assert np.all(sample.coords >= 0)
        assert np.all(sample.coords <= 1)

    @pytest.mark.parametrize(
        "SolverClass,kwargs",
        [
            (HeatSolver, {"alpha": 0.01, "total_time": 0.5}),
            (DarcyFlowSolver, {"forcing": 1.0}),
            (ElasticitySolver, {"young_modulus": 1.0, "poisson_ratio": 0.3}),
            (PoissonSolver, {}),
        ],
    )
    def test_sample_grid_size_correct(self, SolverClass, kwargs, resolution):
        """Sample grid_size should match solver resolution."""
        solver = SolverClass(resolution=resolution, **kwargs)
        sample = solver.generate_sample(seed=42)
        assert sample.grid_size == resolution

    @pytest.mark.parametrize(
        "SolverClass,kwargs",
        [
            (HeatSolver, {"alpha": 0.01, "total_time": 0.5}),
            (DarcyFlowSolver, {"forcing": 1.0}),
            (ElasticitySolver, {"young_modulus": 1.0, "poisson_ratio": 0.3}),
            (PoissonSolver, {}),
        ],
    )
    def test_reproducibility(self, SolverClass, kwargs, resolution):
        """Same seed should produce identical samples."""
        solver = SolverClass(resolution=resolution, **kwargs)
        s1 = solver.generate_sample(seed=99)
        s2 = solver.generate_sample(seed=99)
        assert_allclose(s1.input_field, s2.input_field)
        assert_allclose(s1.output_field, s2.output_field)
