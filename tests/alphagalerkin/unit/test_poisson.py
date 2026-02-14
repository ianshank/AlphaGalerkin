"""Tests for the Poisson physics module (src/alphagalerkin/physics/poisson.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.alphagalerkin.physics.poisson import PoissonModule


@pytest.fixture
def poisson() -> PoissonModule:
    """Create a PoissonModule instance for testing."""
    return PoissonModule()


# -------------------------------------------------------------------
# weak_form
# -------------------------------------------------------------------


class TestWeakForm:
    """Tests for the weak_form placeholder."""

    def test_returns_none(self, poisson: PoissonModule) -> None:
        result = poisson.weak_form(trial=None, test=None, mesh=None)
        assert result is None


# -------------------------------------------------------------------
# boundary_conditions
# -------------------------------------------------------------------


class TestBoundaryConditions:
    """Tests for boundary_conditions method."""

    def test_returns_list(self, poisson: PoissonModule) -> None:
        bcs = poisson.boundary_conditions()
        assert isinstance(bcs, list)

    def test_single_dirichlet_bc(self, poisson: PoissonModule) -> None:
        bcs = poisson.boundary_conditions()
        assert len(bcs) == 1

    def test_bc_type_is_dirichlet(self, poisson: PoissonModule) -> None:
        bcs = poisson.boundary_conditions()
        assert bcs[0].bc_type == "dirichlet"

    def test_bc_value_is_zero(self, poisson: PoissonModule) -> None:
        bcs = poisson.boundary_conditions()
        assert bcs[0].value == 0.0

    def test_bc_region_is_all(self, poisson: PoissonModule) -> None:
        bcs = poisson.boundary_conditions()
        assert bcs[0].region == "all"


# -------------------------------------------------------------------
# manufactured_solution
# -------------------------------------------------------------------


class TestManufacturedSolution:
    """Tests for the manufactured solution u = sin(pi*x)*sin(pi*y)."""

    def test_returns_manufactured_solution(self, poisson: PoissonModule) -> None:
        mms = poisson.manufactured_solution()
        assert mms.name == "poisson_sinsin"
        assert mms.expected_convergence_order == 2.0

    def test_exact_solution_at_origin(self, poisson: PoissonModule) -> None:
        """u(0, 0) = sin(0)*sin(0) = 0."""
        mms = poisson.manufactured_solution()
        points = np.array([[0.0, 0.0]])
        result = mms.exact_solution(points)
        assert result[0] == pytest.approx(0.0, abs=1e-15)

    def test_exact_solution_at_center(self, poisson: PoissonModule) -> None:
        """u(0.5, 0.5) = sin(pi/2)*sin(pi/2) = 1."""
        mms = poisson.manufactured_solution()
        points = np.array([[0.5, 0.5]])
        result = mms.exact_solution(points)
        assert result[0] == pytest.approx(1.0, abs=1e-14)

    def test_exact_solution_at_boundary(self, poisson: PoissonModule) -> None:
        """u(1, y) = sin(pi)*sin(pi*y) = 0 for any y."""
        mms = poisson.manufactured_solution()
        points = np.array([[1.0, 0.3], [0.0, 0.7]])
        result = mms.exact_solution(points)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_forcing_at_center(self, poisson: PoissonModule) -> None:
        """f(0.5, 0.5) = 2*pi^2*sin(pi/2)*sin(pi/2) = 2*pi^2."""
        mms = poisson.manufactured_solution()
        points = np.array([[0.5, 0.5]])
        result = mms.forcing(points)
        expected = 2.0 * np.pi**2
        assert result[0] == pytest.approx(expected, rel=1e-12)

    def test_forcing_at_origin(self, poisson: PoissonModule) -> None:
        """f(0, 0) = 0."""
        mms = poisson.manufactured_solution()
        points = np.array([[0.0, 0.0]])
        result = mms.forcing(points)
        assert result[0] == pytest.approx(0.0, abs=1e-14)

    def test_boundary_data_returns_zeros(self, poisson: PoissonModule) -> None:
        """Boundary data should be all zeros."""
        mms = poisson.manufactured_solution()
        points = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
        result = mms.boundary_data(points)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_exact_and_forcing_batch(self, poisson: PoissonModule) -> None:
        """Batch of points should have correct shapes."""
        mms = poisson.manufactured_solution()
        points = np.random.rand(20, 2)
        exact = mms.exact_solution(points)
        forcing = mms.forcing(points)
        assert exact.shape == (20,)
        assert forcing.shape == (20,)


# -------------------------------------------------------------------
# reward_function
# -------------------------------------------------------------------


class TestRewardFunction:
    """Tests for the reward_function placeholder."""

    def test_returns_zero(self, poisson: PoissonModule) -> None:
        result = poisson.reward_function(state=None, action=None, next_state=None)
        assert result == 0.0

    def test_returns_float(self, poisson: PoissonModule) -> None:
        result = poisson.reward_function(state="s", action="a", next_state="s2")
        assert isinstance(result, float)


# -------------------------------------------------------------------
# state_features
# -------------------------------------------------------------------


class TestStateFeatures:
    """Tests for the state_features placeholder."""

    def test_returns_none(self, poisson: PoissonModule) -> None:
        result = poisson.state_features(None)
        assert result is None


# -------------------------------------------------------------------
# action_validators
# -------------------------------------------------------------------


class TestActionValidators:
    """Tests for the action_validators placeholder."""

    def test_returns_empty_list(self, poisson: PoissonModule) -> None:
        result = poisson.action_validators()
        assert result == []
        assert isinstance(result, list)


# -------------------------------------------------------------------
# default_config
# -------------------------------------------------------------------


class TestDefaultConfig:
    """Tests for the default_config method."""

    def test_returns_dict(self, poisson: PoissonModule) -> None:
        config = poisson.default_config()
        assert isinstance(config, dict)

    def test_has_domain_key(self, poisson: PoissonModule) -> None:
        config = poisson.default_config()
        assert "domain" in config

    def test_has_diffusivity_key(self, poisson: PoissonModule) -> None:
        config = poisson.default_config()
        assert "diffusivity" in config
        assert config["diffusivity"] == 1.0

    def test_domain_is_rectangle(self, poisson: PoissonModule) -> None:
        config = poisson.default_config()
        assert config["domain"]["type"] == "rectangle"

    def test_domain_bounds_are_unit_square(self, poisson: PoissonModule) -> None:
        config = poisson.default_config()
        assert config["domain"]["bounds"] == [
            [0.0, 1.0],
            [0.0, 1.0],
        ]


# -------------------------------------------------------------------
# solve_on_grid
# -------------------------------------------------------------------


class TestSolveOnGrid:
    """Tests for the solve_on_grid finite-difference solver."""

    def test_solve_small_grid(self, poisson: PoissonModule) -> None:
        """Solve on a 5x5 grid and check the result object."""
        result = poisson.solve_on_grid(n=5)
        assert result.converged is True
        assert result.solution.shape == (25,)
        assert result.solve_time_ms > 0.0

    def test_solve_residual_small(self, poisson: PoissonModule) -> None:
        """Residual norm should be near zero (direct solve)."""
        result = poisson.solve_on_grid(n=10)
        assert result.residual_norm < 1e-8

    def test_solve_shape_matches_dof(self, poisson: PoissonModule) -> None:
        """Solution length should be n^2."""
        for n in [5, 8, 10]:
            result = poisson.solve_on_grid(n=n)
            assert len(result.solution) == n * n

    def test_convergence_finer_grid_smaller_error(self, poisson: PoissonModule) -> None:
        """Finer grid should give a smaller error vs exact solution."""
        errors = []
        for n in [5, 10, 20]:
            result = poisson.solve_on_grid(n=n)
            # Compute error vs exact solution
            h = 1.0 / (n + 1)
            x_grid = np.linspace(h, 1.0 - h, n)
            y_grid = np.linspace(h, 1.0 - h, n)
            grid_x, grid_y = np.meshgrid(x_grid, y_grid)
            points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            mms = poisson.manufactured_solution()
            exact = mms.exact_solution(points)
            error = float(np.sqrt(np.mean((result.solution - exact) ** 2)))
            errors.append(error)

        # Each refinement should reduce the error
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]

    def test_residual_norm_is_float(self, poisson: PoissonModule) -> None:
        result = poisson.solve_on_grid(n=5)
        assert isinstance(result.residual_norm, float)

    def test_condition_number_is_one(self, poisson: PoissonModule) -> None:
        """Condition number is set to 1.0 (skipped computation)."""
        result = poisson.solve_on_grid(n=5)
        assert result.condition_number == 1.0
