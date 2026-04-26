"""Tests for :class:`SUPGFEMSolver`.

Coverage targets:

* Pydantic config validation (defaults + bounds + extra-forbidden).
* Registry registration and dispatch via ``get_solver``.
* SUPG -> Galerkin reduction below ``pe_low_threshold`` (τ == 0).
* Stable, finite solutions in the high-Peclet regime where central
  differences would oscillate.
* Higher-dim rejection — solver clearly errors on operators it does
  not support.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import src.research.extra_solvers  # noqa: F401 — populate registry
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import AdvectionDiffusionOperator, PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, get_solver
from src.research.extra_solvers.supg_fem import SUPGFEMConfig, SUPGFEMSolver


def _make_advdiff_op(diffusion: float = 0.01, advection: float = 1.0) -> AdvectionDiffusionOperator:
    cfg = PDEConfig(
        name="advdiff_test",
        pde_type=PDEType.ADVECTION_DIFFUSION,
        domain_dim=1,
        domain_min=[0.0],
        domain_max=[1.0],
        advection_coeff=[advection],
        diffusion_coeff=diffusion,
    )
    return AdvectionDiffusionOperator(cfg)


class TestSUPGFEMConfig:
    def test_defaults_valid(self) -> None:
        cfg = SUPGFEMConfig()
        assert cfg.min_grid_points >= 4
        assert cfg.velocity_floor > 0
        assert cfg.diffusion_floor > 0
        assert cfg.pe_low_threshold > 0

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SUPGFEMConfig(unknown=1)  # type: ignore[call-arg]

    def test_negative_velocity_floor_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SUPGFEMConfig(velocity_floor=0.0)


class TestSUPGFEMRegistration:
    def test_registered(self) -> None:
        assert "supg_fem" in SOLVER_REGISTRY

    def test_get_solver_returns_instance(self) -> None:
        solver = get_solver("supg_fem")
        assert isinstance(solver, SUPGFEMSolver)


class TestSUPGFEMSolve:
    def test_low_peclet_reduces_to_galerkin(self) -> None:
        """When ν >> a, Pe << 1 and τ should be exactly zero."""
        op = _make_advdiff_op(diffusion=10.0, advection=1.0)
        solver = SUPGFEMSolver()
        result = solver.solve(op, n_dof=32)
        assert result.metadata["tau"] == pytest.approx(0.0)

    def test_high_peclet_produces_finite_solution(self) -> None:
        """In the convection-dominated regime the SUPG correction must
        keep the solution bounded (no spurious oscillations)."""
        op = _make_advdiff_op(diffusion=1e-4, advection=1.0)
        solver = SUPGFEMSolver()
        result = solver.solve(op, n_dof=64)
        assert np.all(np.isfinite(result.solution))
        # Non-trivial stabilisation
        assert result.metadata["tau"] > 0
        # Solution magnitude is bounded (no blow-up)
        assert np.max(np.abs(result.solution)) < 1e3

    def test_grid_size_floor_respected(self) -> None:
        op = _make_advdiff_op()
        solver = SUPGFEMSolver(SUPGFEMConfig(min_grid_points=16))
        result = solver.solve(op, n_dof=4)
        assert result.n_dof >= 16

    def test_higher_dim_raises(self) -> None:
        # 2D Poisson rejected — SUPG is 1D only by design.
        cfg = PDEConfig(
            name="p2d",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[0.0, 0.0],
            domain_max=[1.0, 1.0],
            advection_coeff=[0.0, 0.0],
        )
        solver = SUPGFEMSolver()
        with pytest.raises(NotImplementedError, match="1D"):
            solver.solve(PoissonOperator(cfg), n_dof=16)

    def test_metadata_completeness(self) -> None:
        op = _make_advdiff_op()
        solver = SUPGFEMSolver()
        result = solver.solve(op, n_dof=32)
        for key in ("method", "peclet", "tau", "h", "diffusion", "advection_velocity"):
            assert key in result.metadata
