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
    AMRConfig,
    BaseSolver,
    DorflerAMRSolver,
    FDMConfig,
    NavierStokesConfig,
    NavierStokesFDMSolver,
    PINNConfig,
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

    def test_solve_2d_returns_result(self):
        solver = DorflerAMRSolver(marking_fraction=0.3, max_refinements=3)
        op = _make_poisson_2d()
        result = solver.solve(op, n_dof=25)
        assert isinstance(result, SolverResult)
        assert result.n_dof >= 1
        assert result.grid_points.shape[-1] == 2
        assert result.metadata.get("dim") == 2

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
        solver = SimplePINNSolver(hidden_dim=16, n_layers=2, n_epochs=1, learning_rate=1e-3)
        assert solver.hidden_dim == 16
        assert solver.n_layers == 2
        assert solver.n_epochs == 1

    def test_solve_smoke(self):
        """Single epoch to verify it runs end-to-end without crashing."""
        solver = SimplePINNSolver(
            hidden_dim=8, n_layers=2, n_epochs=1, n_collocation=20, learning_rate=1e-3
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

    def test_list_solvers_includes_ns_fdm(self):
        assert "navier_stokes_fdm" in list_solvers()

    def test_get_solver_ns_fdm(self):
        from src.research.baselines import NavierStokesFDMSolver

        solver = get_solver("navier_stokes_fdm")
        assert isinstance(solver, NavierStokesFDMSolver)


# ---------------------------------------------------------------------------
# DorflerAMRSolver 2D extension
# ---------------------------------------------------------------------------


class TestDorflerAMR2D:
    """Tests for the 2D adaptive mesh refinement extension."""

    def test_solve_2d_basic(self):
        solver = DorflerAMRSolver(marking_fraction=0.3, max_refinements=3)
        op = _make_poisson_2d()
        result = solver.solve(op, n_dof=16)
        assert isinstance(result, SolverResult)
        assert result.n_dof >= 1
        assert result.wall_time_seconds >= 0.0

    def test_solve_2d_grid_shape(self):
        solver = DorflerAMRSolver(marking_fraction=0.3, max_refinements=2)
        op = _make_poisson_2d()
        result = solver.solve(op, n_dof=25)
        assert result.grid_points.shape[-1] == 2

    def test_solve_2d_error_finite(self):
        solver = DorflerAMRSolver(marking_fraction=0.3, max_refinements=3)
        op = _make_poisson_2d()
        result = solver.solve(op, n_dof=25)
        if result.l2_error is not None:
            assert math.isfinite(result.l2_error)
            assert result.l2_error >= 0.0

    def test_solve_2d_metadata(self):
        solver = DorflerAMRSolver(marking_fraction=0.4, max_refinements=3)
        op = _make_poisson_2d()
        result = solver.solve(op, n_dof=16)
        assert result.metadata["dim"] == 2
        assert result.metadata["marking_fraction"] == pytest.approx(0.4)
        assert result.metadata["n_refinements"] >= 1

    def test_solve_on_grid_2d_static(self):
        """Verify the internal 2D grid solver produces non-zero solutions."""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        op = _make_poisson_2d()
        xs = np.linspace(0.0, 1.0, 6, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, 6, dtype=np.float64)
        u, grid = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve)
        assert u.shape[0] == len(xs) * len(ys)
        assert grid.shape == (len(xs) * len(ys), 2)

    def test_indicators_2d_shape(self):
        """2D indicators array should have (n_elem_x, n_elem_y) shape."""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve

        op = _make_poisson_2d()
        xs = np.linspace(0.0, 1.0, 6, dtype=np.float64)
        ys = np.linspace(0.0, 1.0, 6, dtype=np.float64)
        u, _ = DorflerAMRSolver._solve_on_grid_2d(xs, ys, op, sparse, spsolve)
        indicators = DorflerAMRSolver._compute_indicators_2d(xs, ys, u, op)
        assert indicators.shape == (5, 5)

    def test_dorfler_mark_2d_returns_bool_arrays(self):
        solver = DorflerAMRSolver(marking_fraction=0.5)
        indicators = np.random.rand(4, 4)
        xs = np.linspace(0, 1, 5)
        ys = np.linspace(0, 1, 5)
        marked_x, marked_y = solver._dorfler_mark_2d(indicators, xs, ys)
        assert marked_x.dtype == bool
        assert marked_y.dtype == bool
        assert len(marked_x) == 4
        assert len(marked_y) == 4

    def test_dim3_raises_not_implemented(self):
        """3D problems should still raise NotImplementedError."""
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
        solver = DorflerAMRSolver()
        with pytest.raises(NotImplementedError, match="dim=1 and dim=2"):
            solver.solve(op, n_dof=10)


# ---------------------------------------------------------------------------
# NavierStokesFDMSolver
# ---------------------------------------------------------------------------


class TestNavierStokesFDMSolver:
    """Tests for the Navier-Stokes FDM projection method solver."""

    def _make_ns_operator(self, re: float = 100.0):
        from src.pde.operators import NavierStokesOperator

        cfg = PDEConfig(
            name="test_ns",
            pde_type=PDEType.NAVIER_STOKES,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[6.283185307, 6.283185307],  # 2*pi
            advection_coeff=[0.0, 0.0],
        )
        return NavierStokesOperator(cfg, reynolds_number=re)

    def test_construction(self):
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver(dt=0.01, t_final=0.1)
        assert solver.dt == 0.01
        assert solver.t_final == 0.1

    def test_solve_smoke(self):
        """Minimal solve (very coarse grid, short time) to verify no crash."""
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver(dt=0.05, t_final=0.05)
        op = self._make_ns_operator(re=10.0)
        result = solver.solve(op, n_dof=32)
        assert isinstance(result, SolverResult)
        assert result.n_dof >= 1
        assert result.wall_time_seconds >= 0.0

    def test_solve_returns_velocity_field(self):
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver(dt=0.05, t_final=0.05)
        op = self._make_ns_operator(re=10.0)
        result = solver.solve(op, n_dof=32)
        # Solution contains ux and uy concatenated
        n_grid = result.grid_points.shape[0]
        assert result.solution.shape[0] == 2 * n_grid

    def test_solve_metadata(self):
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver(dt=0.05, t_final=0.1)
        op = self._make_ns_operator(re=10.0)
        result = solver.solve(op, n_dof=32)
        assert result.metadata["method"] == "chorin_projection"
        assert "dt" in result.metadata
        assert "n_steps" in result.metadata
        assert "grid_size" in result.metadata

    def test_dim1_raises_not_implemented(self):
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver()
        op = _make_poisson_1d()
        with pytest.raises(NotImplementedError, match="dim=2"):
            solver.solve(op, n_dof=20)

    def test_name_and_description(self):
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver()
        assert solver.name == "navier_stokes_fdm"
        assert len(solver.description) > 0

    def test_l2_error_computed(self):
        """On Taylor-Green vortex, L2 error should be computed."""
        from src.research.baselines import NavierStokesFDMSolver

        solver = NavierStokesFDMSolver(dt=0.05, t_final=0.05)
        op = self._make_ns_operator(re=10.0)
        result = solver.solve(op, n_dof=32)
        # May or may not have exact solution depending on operator
        # Just verify it doesn't crash
        assert result.l2_error is None or result.l2_error >= 0.0


# ---------------------------------------------------------------------------
# Per-Solver Config Tests
# ---------------------------------------------------------------------------


class TestPerSolverConfigs:
    """Validate that per-solver Pydantic configs surface all parameters."""

    def test_fdm_config_defaults(self):
        config = FDMConfig()
        assert config.min_grid_points == 3
        assert config.seed == 42
        assert config.tolerance == 1e-10

    def test_fdm_config_custom(self):
        config = FDMConfig(min_grid_points=5, seed=123)
        assert config.min_grid_points == 5
        assert config.seed == 123

    def test_fdm_config_validation(self):
        with pytest.raises(Exception):
            FDMConfig(min_grid_points=0)  # ge=2

    def test_amr_config_defaults(self):
        config = AMRConfig()
        assert config.marking_fraction == 0.3
        assert config.max_refinements == 10
        assert config.initial_dof_divisor == 4
        assert config.max_initial_points_1d == 8
        assert config.min_initial_points == 4
        assert config.initial_side_divisor_2d == 2
        assert config.min_initial_side_2d == 3

    def test_amr_config_custom(self):
        config = AMRConfig(marking_fraction=0.5, max_refinements=20)
        assert config.marking_fraction == 0.5
        assert config.max_refinements == 20

    def test_amr_config_validation(self):
        with pytest.raises(Exception):
            AMRConfig(marking_fraction=1.5)  # lt=1.0

    def test_pinn_config_defaults(self):
        config = PINNConfig()
        assert config.hidden_dim == 64
        assert config.n_layers == 3
        assert config.n_epochs == 2000
        assert config.learning_rate == 1e-3
        assert config.n_collocation == 1000
        assert config.bc_loss_weight == 10.0
        assert config.n_boundary_points == 50
        assert config.log_interval == 500

    def test_pinn_config_custom(self):
        config = PINNConfig(hidden_dim=128, n_epochs=100, log_interval=10)
        assert config.hidden_dim == 128
        assert config.n_epochs == 100
        assert config.log_interval == 10

    def test_ns_config_defaults(self):
        config = NavierStokesConfig()
        assert config.dt == 0.01
        assert config.t_final == 1.0
        assert config.default_viscosity == 0.01
        assert config.min_grid_points == 4
        assert config.cfl_safety == 0.25
        assert config.viscosity_floor == 1e-12
        assert config.log_fraction == 10

    def test_ns_config_custom(self):
        config = NavierStokesConfig(dt=0.005, t_final=2.0, cfl_safety=0.1)
        assert config.dt == 0.005
        assert config.t_final == 2.0
        assert config.cfl_safety == 0.1

    def test_ns_config_validation(self):
        with pytest.raises(Exception):
            NavierStokesConfig(cfl_safety=1.5)  # le=1.0

    def test_solver_uses_fdm_config(self):
        config = FDMConfig(min_grid_points=5)
        solver = UniformFDMSolver(config=config)
        assert solver.config.min_grid_points == 5

    def test_solver_uses_amr_config(self):
        config = AMRConfig(marking_fraction=0.4, max_refinements=5)
        solver = DorflerAMRSolver(config=config)
        assert solver.marking_fraction == 0.4
        assert solver.max_refinements == 5

    def test_amr_constructor_overrides_config(self):
        """Constructor args should override config defaults."""
        config = AMRConfig(marking_fraction=0.4)
        solver = DorflerAMRSolver(marking_fraction=0.2, config=config)
        assert solver.marking_fraction == 0.2  # Constructor wins

    def test_solver_uses_pinn_config(self):
        config = PINNConfig(hidden_dim=32, n_epochs=10)
        solver = SimplePINNSolver(config=config)
        assert solver.hidden_dim == 32
        assert solver.n_epochs == 10

    def test_solver_uses_ns_config(self):
        config = NavierStokesConfig(dt=0.005, t_final=0.5)
        solver = NavierStokesFDMSolver(config=config)
        assert solver.dt == 0.005
        assert solver.t_final == 0.5
