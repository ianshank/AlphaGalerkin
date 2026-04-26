"""Tests for the multigrid + direct Poisson solvers."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import src.research.extra_solvers  # noqa: F401 — populate registry
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, get_solver
from src.research.extra_solvers.multigrid import (
    DirectPoissonConfig,
    DirectPoissonSolver,
    MultigridPoissonConfig,
)


def _make_poisson_op() -> PoissonOperator:
    cfg = PDEConfig(
        name="poisson_test",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
    )
    return PoissonOperator(cfg)


# ---------------------------------------------------------------------------
# Direct
# ---------------------------------------------------------------------------


class TestDirectPoissonConfig:
    def test_defaults(self) -> None:
        cfg = DirectPoissonConfig()
        assert cfg.min_grid_points >= 2

    def test_negative_min_grid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DirectPoissonConfig(min_grid_points=1)


class TestDirectPoissonSolve:
    def test_registered(self) -> None:
        assert "direct_solver" in SOLVER_REGISTRY

    def test_solves_unit_square(self) -> None:
        op = _make_poisson_op()
        solver = DirectPoissonSolver()
        result = solver.solve(op, n_dof=64)  # 8x8
        assert result.n_dof == 64
        assert np.all(np.isfinite(result.solution))
        assert result.solution.size == 100  # (8+2)^2 with BC frame

    def test_grid_size_floor(self) -> None:
        op = _make_poisson_op()
        solver = DirectPoissonSolver(DirectPoissonConfig(min_grid_points=8))
        result = solver.solve(op, n_dof=4)  # request smaller than floor
        assert result.metadata["n_per_side"] >= 8

    def test_higher_dim_rejected(self) -> None:
        cfg = PDEConfig(
            name="p1d",
            pde_type=PDEType.POISSON,
            domain_dim=1,
            domain_min=[0.0],
            domain_max=[1.0],
            advection_coeff=[0.0],
        )
        op = PoissonOperator(cfg)
        with pytest.raises(NotImplementedError, match="2D"):
            DirectPoissonSolver().solve(op, n_dof=16)


# ---------------------------------------------------------------------------
# Multigrid (real or stub depending on environment)
# ---------------------------------------------------------------------------


class TestMultigridPoissonConfig:
    def test_defaults(self) -> None:
        cfg = MultigridPoissonConfig()
        assert cfg.cycle in ("V", "W", "F")
        assert cfg.max_levels >= 2

    def test_extra_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MultigridPoissonConfig(unknown=1)  # type: ignore[call-arg]


class TestMultigridSolverBehavior:
    def test_registered(self) -> None:
        assert "multigrid" in SOLVER_REGISTRY

    def test_stub_or_real(self) -> None:
        """Multigrid is either real (pyamg installed) or stub (raises ImportError).

        Both branches are valid; we just exercise the appropriate one
        to ensure registry plumbing is sound regardless of environment.
        """
        op = _make_poisson_op()
        try:
            import pyamg  # noqa: F401
            real_available = True
        except ImportError:
            real_available = False

        solver = get_solver("multigrid")
        if real_available:
            result = solver.solve(op, n_dof=64)
            assert np.all(np.isfinite(result.solution))
        else:
            with pytest.raises(ImportError, match="pyamg"):
                solver.solve(op, n_dof=64)
