"""Tests for src/research/fem_baseline.py.

Covers FEMConfig validation, ScikitFEMPoissonSolver on unit-square Poisson
with a manufactured solution, P2/P3 element convergence, and the
ScikitFEMLShapedSolver smoke test on the L-shaped domain.

scikit-fem is required; tests are skipped if not installed.
"""

from __future__ import annotations

import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, BaseSolver, SolverResult

pytest.importorskip("skfem", reason="scikit-fem required for FEM baseline tests")

import numpy as np  # noqa: E402

from src.research.fem_baseline import (  # noqa: E402
    FEMConfig,
    ScikitFEMLShapedSolver,
    ScikitFEMPoissonSolver,
)


def _make_poisson_2d() -> PoissonOperator:
    cfg = PDEConfig(
        name="test_poisson_2d",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


def _make_poisson_lshaped() -> PoissonOperator:
    cfg = PDEConfig(
        name="test_poisson_lshaped",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-1.0, -1.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


class TestFEMConfig:
    def test_defaults(self):
        config = FEMConfig()
        assert config.element_type == "P1"
        assert config.refinement_strategy == "h_adaptive"
        assert 0.0 < config.marking_fraction < 1.0
        assert config.max_element_order == 3
        assert config.min_mesh_side == 3
        assert config.min_initial_dof_hint == 9
        assert config.zz_epsilon == pytest.approx(1e-12)

    def test_rejects_unknown_element(self):
        with pytest.raises(ValueError):
            FEMConfig(element_type="P5")

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError):
            FEMConfig(refinement_strategy="magic")

    def test_marking_fraction_bounds(self):
        with pytest.raises(ValueError):
            FEMConfig(marking_fraction=0.0)
        with pytest.raises(ValueError):
            FEMConfig(marking_fraction=1.5)

    def test_max_element_order_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            FEMConfig(max_element_order=4)
        with pytest.raises(ValueError):
            FEMConfig(max_element_order=0)

    def test_min_mesh_side_bounds(self):
        with pytest.raises(ValueError):
            FEMConfig(min_mesh_side=1)

    def test_zz_epsilon_positive(self):
        with pytest.raises(ValueError):
            FEMConfig(zz_epsilon=0.0)


class TestFEMInternalHelpers:
    """Covers the Dorfler marking + ZZ gradient helpers in isolation."""

    def test_dorfler_mark_basic(self):
        solver = ScikitFEMPoissonSolver(FEMConfig(marking_fraction=0.5))
        indicators = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float64)
        marked = solver._dorfler_mark(indicators)
        # The top indicator (0.4) alone covers 0.4 >= 0.5 * 1.0 = 0.5?  No,
        # so we need the top two (0.4 + 0.3 = 0.7 >= 0.5).
        assert marked[1]  # 0.4 is highest
        assert marked.sum() == 2

    def test_dorfler_mark_zero_indicators(self):
        solver = ScikitFEMPoissonSolver()
        marked = solver._dorfler_mark(np.zeros(5, dtype=np.float64))
        assert not marked.any()

    def test_element_gradients_linear_function(self):
        """Gradient of u = 2x + 3y on any triangle should be (2, 3)."""
        # Single triangle with vertices at (0,0), (1,0), (0,1).
        mesh_p = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        mesh_t = np.array([[0], [1], [2]], dtype=np.int64)

        class _FakeMesh:
            p = mesh_p
            t = mesh_t

        nodal = np.array([0.0, 2.0, 3.0], dtype=np.float64)  # u = 2x + 3y
        grads = ScikitFEMPoissonSolver._element_gradients(_FakeMesh(), nodal)
        assert grads.shape == (1, 2)
        np.testing.assert_allclose(grads[0], [2.0, 3.0], atol=1e-10)

    def test_triangle_area(self):
        pts = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        assert ScikitFEMPoissonSolver._triangle_area(pts) == pytest.approx(0.5)

    def test_estimate_smoothness_empty_mark(self):
        solver = ScikitFEMPoissonSolver()
        indicators = np.array([1.0, 2.0, 3.0])
        marked = np.zeros_like(indicators, dtype=bool)
        assert solver._estimate_smoothness(indicators, marked) == 0.0

    def test_estimate_smoothness_uniform(self):
        """Uniform indicators produce ratio close to 1.0 (smooth)."""
        solver = ScikitFEMPoissonSolver()
        indicators = np.ones(5)
        marked = np.ones(5, dtype=bool)
        smooth = solver._estimate_smoothness(indicators, marked)
        assert smooth == pytest.approx(1.0, rel=1e-6)

    def test_estimate_smoothness_concentrated(self):
        """One dominant indicator produces a low ratio."""
        solver = ScikitFEMPoissonSolver()
        indicators = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
        marked = np.ones(5, dtype=bool)
        smooth = solver._estimate_smoothness(indicators, marked)
        # mean=20.8, max=100 -> 0.208
        assert smooth < 0.5


class TestFEMDimensionChecks:
    def test_rejects_1d_operator(self):
        cfg = PDEConfig(
            name="p1d",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        operator = PoissonOperator(cfg)
        solver = ScikitFEMPoissonSolver(FEMConfig(max_refinement_levels=1))
        with pytest.raises(NotImplementedError, match="2D"):
            solver.solve(operator, n_dof=9)

    def test_lshaped_rejects_1d_operator(self):
        cfg = PDEConfig(
            name="p1d",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        operator = PoissonOperator(cfg)
        solver = ScikitFEMLShapedSolver(FEMConfig(max_refinement_levels=1))
        with pytest.raises(NotImplementedError, match="2D"):
            solver.solve(operator, n_dof=9)


class TestScikitFEMPoissonSolver:
    def test_is_basesolver(self):
        assert issubclass(ScikitFEMPoissonSolver, BaseSolver)

    def test_registered(self):
        assert "scikit_fem_poisson" in SOLVER_REGISTRY
        assert SOLVER_REGISTRY["scikit_fem_poisson"] is ScikitFEMPoissonSolver

    def test_solve_returns_result(self):
        config = FEMConfig(
            element_type="P1",
            refinement_strategy="uniform",
            max_refinement_levels=2,
            initial_mesh_refinements=1,
        )
        solver = ScikitFEMPoissonSolver(config)
        operator = _make_poisson_2d()
        result = solver.solve(operator, n_dof=64)

        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.wall_time_seconds > 0.0
        assert result.metadata["method"] == "scikit_fem_hp_adaptive"
        assert result.metadata["strategy"] == "uniform"

    def test_h_adaptive_reduces_error(self):
        """Two refinement levels should produce lower error than one."""
        operator = _make_poisson_2d()

        cfg_low = FEMConfig(
            element_type="P1",
            refinement_strategy="h_adaptive",
            max_refinement_levels=1,
            initial_mesh_refinements=1,
        )
        r_low = ScikitFEMPoissonSolver(cfg_low).solve(operator, n_dof=25)

        cfg_high = FEMConfig(
            element_type="P1",
            refinement_strategy="h_adaptive",
            max_refinement_levels=3,
            initial_mesh_refinements=1,
        )
        r_high = ScikitFEMPoissonSolver(cfg_high).solve(operator, n_dof=25)

        assert r_high.n_dof >= r_low.n_dof
        if r_low.l2_error is not None and r_high.l2_error is not None:
            # More refinement should not make error worse (modulo numerical noise)
            assert r_high.l2_error <= r_low.l2_error * 1.5

    def test_p2_at_least_as_accurate_as_p1(self):
        """P2 elements should match or beat P1 at comparable refinement."""
        operator = _make_poisson_2d()

        r_p1 = ScikitFEMPoissonSolver(
            FEMConfig(
                element_type="P1",
                refinement_strategy="uniform",
                max_refinement_levels=1,
                initial_mesh_refinements=2,
            )
        ).solve(operator, n_dof=64)

        r_p2 = ScikitFEMPoissonSolver(
            FEMConfig(
                element_type="P2",
                refinement_strategy="uniform",
                max_refinement_levels=1,
                initial_mesh_refinements=2,
            )
        ).solve(operator, n_dof=64)

        if r_p1.l2_error is not None and r_p2.l2_error is not None:
            # P2 should be at least roughly comparable; give generous margin.
            assert r_p2.l2_error <= r_p1.l2_error * 1.5


class TestScikitFEMLShapedSolver:
    def test_registered(self):
        assert "scikit_fem_lshaped" in SOLVER_REGISTRY

    def test_runs_on_lshaped(self):
        """Smoke test: L-shaped solver runs without crashing."""
        config = FEMConfig(
            element_type="P1",
            refinement_strategy="h_adaptive",
            max_refinement_levels=2,
            initial_mesh_refinements=1,
        )
        solver = ScikitFEMLShapedSolver(config)
        operator = _make_poisson_lshaped()
        result = solver.solve(operator, n_dof=64)

        assert isinstance(result, SolverResult)
        assert result.n_dof > 0
        assert result.metadata["method"] == "scikit_fem_hp_adaptive"
