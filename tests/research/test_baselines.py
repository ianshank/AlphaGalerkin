"""Tests for src/research/baselines.py.

Covers SolverResult, SolverConfig, UniformFDMSolver, DorflerAMRSolver,
SimplePINNSolver (smoke tests only due to training cost), get_solver,
and list_solvers.

scipy is required for FDM/AMR tests; tests are skipped if not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import (
    BaseSolver,
    DorflerAMRSolver,
    SimplePINNSolver,
    SolverConfig,
    SolverResult,
    UniformFDMSolver,
    get_solver,
    list_solvers,
)

scipy = pytest.importorskip("scipy", reason="scipy required for FDM/AMR tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_poisson_1d(n_grid: int = 50) -> PoissonOperator:
    """1D Poisson operator on [0, 1]."""
    cfg = PDEConfig(
        name="test_poisson_1d",
        pde_type=PDEType.POISSON,
        domain_dim=1,
        domain_min=[0.0],
        domain_max=[1.0],
        advection_coeff=[0.0],
    )
    return PoissonOperator(cfg)


def _make_poisson_2d() -> PoissonOperator:
    """2D Poisson operator on [0,1]^2."""
    cfg = PDEConfig(
        name="test_poisson_2d",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


# ---------------------------------------------------------------------------
# SolverResult
# ---------------------------------------------------------------------------

class TestSolverResult:
    def test_to_dict_basic(self):
        result = SolverResult(
            solution=np.array([1.0, 2.0]),
            grid_points=np.array([[0.5]]),
            n_dof=2,
            wall_time_seconds=0.01,
            l2_error=0.001,
            h1_error=0.005,
        )
        d = result.to_dict()
        assert d["n_dof"] == 2
        assert d["wall_time_seconds"] == pytest.approx(0.01)
        assert d["l2_error"] == pytest.approx(0.001)
        assert d["h1_error"] == pytest.approx(0.005)
        assert isinstance(d["metadata"], dict)

    def test_to_dict_optional_none(self):
        result = SolverResult(
            solution=np.zeros(10),
            grid_points=np.zeros((10, 1)),
            n_dof=10,
            wall_time_seconds=0.0,
        )
        d = result.to_dict()
        assert d["l2_error"] is None
        assert d["h1_error"] is None

    def test_metadata_defaults_to_empty_dict(self):
        result = SolverResult(
            solution=np.zeros(5),
            grid_points=np.zeros((5, 1)),
            n_dof=5,
            wall_time_seconds=0.0,
        )
        assert result.metadata == {}

    def test_metadata_preserved(self):
        meta = {"method": "central_differences", "order": 2}
        result = SolverResult(
            solution=np.zeros(5),
            grid_points=np.zeros((5, 1)),
            n_dof=5,
            wall_time_seconds=0.0,
            metadata=meta,
        )
        assert result.to_dict()["metadata"] == meta


# ---------------------------------------------------------------------------
# SolverConfig
# ---------------------------------------------------------------------------

class TestSolverConfig:
    def test_defaults(self):
        cfg = SolverConfig()
        assert cfg.seed >= 0
        assert cfg.max_iterations >= 1
        assert cfg.tolerance > 0

    def test_extra_fields_forbidden(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SolverConfig(unknown_field=42)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# UniformFDMSolver
# ---------------------------------------------------------------------------

class TestUniformFDMSolver:
    def test_solve_1d_returns_result(self):
        solver = UniformFDMSolver()
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=20)
        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.wall_time_seconds >= 0.0
        assert len(result.solution) == result.n_dof
        assert result.grid_points.shape[-1] == 1

    def test_solve_1d_l2_error_finite(self):
        solver = UniformFDMSolver()
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=50)
        # L2 error may be None if no exact solution, or finite if there is one
        if result.l2_error is not None:
            assert math.isfinite(result.l2_error)
            assert result.l2_error >= 0.0

    def test_solve_2d_returns_result(self):
        solver = UniformFDMSolver()
        op = _make_poisson_2d()
        result = solver.solve(op, n_dof=25)
        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.grid_points.shape[-1] == 2

    def test_solve_more_dof_less_error(self):
        """Refining from 10 to 100 DOF should reduce or maintain L2 error."""
        solver = UniformFDMSolver()
        op = _make_poisson_1d()
        r_coarse = solver.solve(op, n_dof=10)
        r_fine = solver.solve(op, n_dof=100)
        if r_coarse.l2_error is not None and r_fine.l2_error is not None:
            assert r_fine.l2_error <= r_coarse.l2_error + 1e-10

    def test_metadata_has_method(self):
        solver = UniformFDMSolver()
        result = solver.solve(_make_poisson_1d(), n_dof=20)
        assert "method" in result.metadata

    def test_dim3_raises_not_implemented(self):
        from src.pde.config import PDEConfig, PDEType
        from src.pde.operators import HeatOperator
        cfg3d = PDEConfig(
            name="heat_3d",
            pde_type=PDEType.HEAT,
            domain_dim=3,
            domain_min=[0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0],
            advection_coeff=[0.0, 0.0, 0.0],
        )
        op = HeatOperator(cfg3d)
        solver = UniformFDMSolver()
        with pytest.raises(NotImplementedError):
            solver.solve(op, n_dof=10)

    @pytest.mark.parametrize("n_dof", [5, 20, 50, 100])
    def test_solve_1d_various_dof(self, n_dof: int):
        solver = UniformFDMSolver()
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=n_dof)
        assert result.n_dof >= 1


# ---------------------------------------------------------------------------
# DorflerAMRSolver
# ---------------------------------------------------------------------------

class TestDorflerAMRSolver:
    def test_default_construction(self):
        solver = DorflerAMRSolver()
        assert 0.0 < solver.marking_fraction < 1.0

    def test_invalid_marking_fraction(self):
        with pytest.raises(ValueError):
            DorflerAMRSolver(marking_fraction=0.0)
        with pytest.raises(ValueError):
            DorflerAMRSolver(marking_fraction=1.0)
        with pytest.raises(ValueError):
            DorflerAMRSolver(marking_fraction=-0.5)

    def test_solve_1d_returns_result(self):
        solver = DorflerAMRSolver(marking_fraction=0.3, max_refinements=5)
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=30)
        assert isinstance(result, SolverResult)
        assert result.n_dof >= 1
        assert result.wall_time_seconds >= 0.0

    def test_solve_2d_raises_not_implemented(self):
        solver = DorflerAMRSolver()
        op = _make_poisson_2d()
        with pytest.raises(NotImplementedError):
            solver.solve(op, n_dof=50)

    def test_metadata_has_marking_fraction(self):
        mf = 0.4
        solver = DorflerAMRSolver(marking_fraction=mf)
        result = solver.solve(_make_poisson_1d(), n_dof=20)
        assert result.metadata["marking_fraction"] == pytest.approx(mf)

    @pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.7])
    def test_various_marking_fractions(self, fraction: float):
        solver = DorflerAMRSolver(marking_fraction=fraction, max_refinements=3)
        result = solver.solve(_make_poisson_1d(), n_dof=20)
        assert result.n_dof >= 1

    def test_dorfler_mark_classmethod(self):
        """_dorfler_mark returns boolean array of same length as indicators."""
        solver = DorflerAMRSolver(marking_fraction=0.5)
        indicators = np.array([1.0, 0.5, 2.0, 0.1, 1.5])
        marked = solver._dorfler_mark(indicators)
        assert marked.dtype == bool
        assert len(marked) == len(indicators)
        assert marked.any()

    def test_refine_grid_inserts_midpoints(self):
        """_refine_grid adds midpoints for marked elements."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        marked = np.array([True, False, True, False])
        x_new = DorflerAMRSolver._refine_grid(x, marked)
        assert len(x_new) > len(x)
        # New points should be sorted
        assert np.all(np.diff(x_new) > 0)


# ---------------------------------------------------------------------------
# SimplePINNSolver
# ---------------------------------------------------------------------------

class TestSimplePINNSolver:
    """Smoke tests only - full training is too slow for a test suite."""

    def test_construction(self):
        solver = SimplePINNSolver(
            hidden_dim=16, n_layers=2, n_epochs=1, learning_rate=1e-3
        )
        assert solver.hidden_dim == 16
        assert solver.n_layers == 2
        assert solver.n_epochs == 1

    def test_solve_smoke(self):
        """Single epoch to verify it runs end-to-end without crashing."""
        solver = SimplePINNSolver(
            hidden_dim=8, n_layers=2, n_epochs=1,
            n_collocation=20, learning_rate=1e-3
        )
        op = _make_poisson_1d()
        result = solver.solve(op, n_dof=20)
        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.wall_time_seconds >= 0.0

    def test_build_network_architecture(self):
        """Network should have correct layers."""
        import torch.nn
        solver = SimplePINNSolver(hidden_dim=32, n_layers=3)
        net = solver._build_network(input_dim=1)
        assert isinstance(net, torch.nn.Sequential)

    def test_name_and_description(self):
        solver = SimplePINNSolver()
        assert isinstance(solver.name, str)
        assert len(solver.name) > 0
        assert isinstance(solver.description, str)


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_solvers_returns_list(self):
        solvers = list_solvers()
        assert isinstance(solvers, list)
        assert len(solvers) > 0
        assert all(isinstance(s, str) for s in solvers)

    def test_list_solvers_includes_fdm(self):
        assert "uniform_fdm" in list_solvers()

    def test_list_solvers_includes_dorfler(self):
        assert "dorfler_amr" in list_solvers()

    def test_list_solvers_is_sorted(self):
        solvers = list_solvers()
        assert solvers == sorted(solvers)

    def test_get_solver_uniform_fdm(self):
        solver = get_solver("uniform_fdm")
        assert isinstance(solver, UniformFDMSolver)

    def test_get_solver_dorfler_amr(self):
        solver = get_solver("dorfler_amr")
        assert isinstance(solver, DorflerAMRSolver)

    def test_get_solver_dorfler_with_kwargs(self):
        solver = get_solver("dorfler_amr", marking_fraction=0.5)
        assert isinstance(solver, DorflerAMRSolver)
        assert solver.marking_fraction == pytest.approx(0.5)

    def test_get_solver_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown solver"):
            get_solver("nonexistent_solver")

    def test_get_solver_pinn(self):
        solver = get_solver("pinn")
        assert isinstance(solver, SimplePINNSolver)

    def test_all_solvers_are_base_solver(self):
        for name in list_solvers():
            solver = get_solver(name)
            assert isinstance(solver, BaseSolver)
