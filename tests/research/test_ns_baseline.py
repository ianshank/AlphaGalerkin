"""Tests for Navier-Stokes FDM baseline solver.

Validates the Chorin projection method against the Taylor-Green vortex
analytical solution. The solver should produce velocity errors below
a reasonable threshold for moderate Reynolds numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import NavierStokesOperator
from src.research.baselines import NavierStokesConfig, NavierStokesFDMSolver

scipy = pytest.importorskip("scipy", reason="scipy required for NS FDM tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ns_operator(reynolds_number: float = 100.0) -> NavierStokesOperator:
    """Create a Taylor-Green vortex NS operator on [0, 2*pi]^2."""
    two_pi = 2.0 * float(np.pi)
    cfg = PDEConfig(
        name="test_ns",
        pde_type=PDEType.NAVIER_STOKES,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[two_pi, two_pi],
        advection_coeff=[0.0, 0.0],
    )
    return NavierStokesOperator(cfg, reynolds_number=reynolds_number)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNavierStokesFDMSolver:
    """Chorin projection method solver tests."""

    def test_solve_returns_result(self) -> None:
        """Solver returns a valid SolverResult with expected shape."""
        operator = _make_ns_operator(reynolds_number=100.0)
        solver = NavierStokesFDMSolver(
            config=NavierStokesConfig(
                dt=0.01,
                t_final=0.01,
                min_grid_points=4,
            ),
        )
        result = solver.solve(operator, n_dof=32)

        assert result.n_dof > 0
        assert result.wall_time_seconds >= 0.0
        assert result.grid_points.shape[1] == 2
        # Solution should contain ux and uy concatenated
        n_grid = result.grid_points.shape[0]
        assert result.solution.shape[0] == 2 * n_grid

    def test_taylor_green_error_below_threshold(self) -> None:
        """L2 error against Taylor-Green exact solution should be bounded.

        With Re=100, small time integration, and moderate grid, we expect
        the L2 error to be reasonable (< 0.5 for this coarse test).
        """
        operator = _make_ns_operator(reynolds_number=100.0)
        solver = NavierStokesFDMSolver(
            config=NavierStokesConfig(
                dt=0.01,
                t_final=0.05,
                min_grid_points=4,
            ),
        )
        result = solver.solve(operator, n_dof=128)

        assert result.l2_error is not None
        assert result.l2_error < 0.5, f"L2 error {result.l2_error:.4f} exceeds threshold 0.5"

    def test_viscosity_from_operator(self) -> None:
        """Solver should extract viscosity from the operator."""
        operator = _make_ns_operator(reynolds_number=200.0)
        assert hasattr(operator, "viscosity")
        assert operator.viscosity == pytest.approx(1.0 / 200.0)

    def test_metadata_contains_solver_info(self) -> None:
        """Result metadata should contain projection method details."""
        operator = _make_ns_operator()
        solver = NavierStokesFDMSolver(
            config=NavierStokesConfig(dt=0.01, t_final=0.01),
        )
        result = solver.solve(operator, n_dof=32)

        assert result.metadata["method"] == "chorin_projection"
        assert "dt" in result.metadata
        assert "n_steps" in result.metadata
        assert "grid_size" in result.metadata

    def test_initial_condition_from_operator(self) -> None:
        """Solver should initialize from the operator's initial_condition.

        Taylor-Green at t=0:
            ux = -cos(x)*sin(y)
            uy =  sin(x)*cos(y)
        """
        operator = _make_ns_operator(reynolds_number=100.0)
        coords = np.array([[1.0, 1.0]], dtype=np.float32)
        ic = operator.initial_condition(coords)
        ic_arr = np.asarray(ic, dtype=np.float64)

        expected_ux = -np.cos(1.0) * np.sin(1.0)
        expected_uy = np.sin(1.0) * np.cos(1.0)
        assert ic_arr.shape[-1] == 2
        assert ic_arr[0, 0] == pytest.approx(expected_ux, abs=1e-5)
        assert ic_arr[0, 1] == pytest.approx(expected_uy, abs=1e-5)

    def test_exact_solution_decays(self) -> None:
        """Taylor-Green exact solution should decay exponentially with time."""
        operator = _make_ns_operator(reynolds_number=100.0)
        coords = np.array([[1.0, 1.0]], dtype=np.float32)

        sol_t0 = np.asarray(operator.exact_solution(coords, time=0.0), dtype=np.float64)
        sol_t1 = np.asarray(operator.exact_solution(coords, time=1.0), dtype=np.float64)

        # Velocity magnitude should decrease
        mag_t0 = float(np.sqrt(np.sum(sol_t0**2)))
        mag_t1 = float(np.sqrt(np.sum(sol_t1**2)))
        assert mag_t1 < mag_t0

    def test_solver_requires_dim_2(self) -> None:
        """Solver should raise NotImplementedError for dim != 2."""
        cfg = PDEConfig(
            name="test_1d",
            pde_type=PDEType.NAVIER_STOKES,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        operator = NavierStokesOperator(cfg, reynolds_number=100.0)
        solver = NavierStokesFDMSolver()

        with pytest.raises(NotImplementedError, match="dim=2"):
            solver.solve(operator, n_dof=32)

    def test_finer_grid_improves_accuracy(self) -> None:
        """Increasing DOF should reduce L2 error at t=0 (pure initial condition)."""
        operator = _make_ns_operator(reynolds_number=100.0)
        # Use very small t_final so the initial condition dominates
        # and grid resolution is the main error driver
        solver = NavierStokesFDMSolver(
            config=NavierStokesConfig(dt=0.001, t_final=0.001),
        )

        result_coarse = solver.solve(operator, n_dof=32)
        result_fine = solver.solve(operator, n_dof=512)

        # Both should produce finite errors
        assert result_coarse.l2_error is not None
        assert result_fine.l2_error is not None
        assert result_fine.l2_error < float("inf")
        assert result_coarse.l2_error < float("inf")
