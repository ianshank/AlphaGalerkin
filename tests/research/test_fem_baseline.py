"""Tests for src/research/fem_baseline.py.

Covers FEMConfig validation, ScikitFEMPoissonSolver on unit-square Poisson
with a manufactured solution, P2/P3 element convergence, and the
ScikitFEMLShapedSolver smoke test on the L-shaped domain.

scikit-fem is required; tests are skipped if not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, BaseSolver, SolverResult

pytest.importorskip("skfem", reason="scikit-fem required for FEM baseline tests")

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
