"""Tests for the FNO and DeepONet baseline solvers.

Strategy:

* Unit-test the Pydantic configs (no torch dependency).
* Smoke-test ``solve`` end-to-end with very small training budgets so
  the test suite stays fast (n_train_steps=5, grid 8x8).
* Verify registry registration and ``get_solver`` dispatch.

Both solvers train a fresh model per call — we only need shape and
finiteness assertions, not convergence guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import src.research.extra_solvers  # noqa: F401 — populate registry
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import PoissonOperator
from src.research.baselines import SOLVER_REGISTRY, get_solver

torch = pytest.importorskip("torch")


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
# Configs
# ---------------------------------------------------------------------------


class TestNeuralOpConfigs:
    def test_fno_defaults(self) -> None:
        from src.research.extra_solvers.neural_op import FNOSolverConfig

        cfg = FNOSolverConfig()
        assert cfg.modes >= 1
        assert cfg.width >= 4
        assert cfg.n_layers >= 1

    def test_fno_extra_rejected(self) -> None:
        from src.research.extra_solvers.neural_op import FNOSolverConfig

        with pytest.raises(ValidationError):
            FNOSolverConfig(unknown=1)  # type: ignore[call-arg]

    def test_deeponet_defaults(self) -> None:
        from src.research.extra_solvers.neural_op import DeepONetSolverConfig

        cfg = DeepONetSolverConfig()
        assert cfg.branch_width >= 8
        assert cfg.trunk_width >= 8
        assert cfg.latent_dim >= 4


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestNeuralOpRegistration:
    def test_fno_registered(self) -> None:
        assert "fno" in SOLVER_REGISTRY

    def test_deeponet_registered(self) -> None:
        assert "deeponet" in SOLVER_REGISTRY


# ---------------------------------------------------------------------------
# Smoke tests (small budgets)
# ---------------------------------------------------------------------------


class TestFNOSmoke:
    def test_solve_runs(self) -> None:
        from src.research.extra_solvers.neural_op import (
            FNOBaselineSolver,
            FNOSolverConfig,
        )

        op = _make_poisson_op()
        cfg = FNOSolverConfig(
            n_train_steps=3,
            modes=2,
            width=4,
            n_layers=1,
            grid_points_floor=8,
            log_interval=10,
        )
        solver = FNOBaselineSolver(cfg)
        result = solver.solve(op, n_dof=64)
        assert result.n_dof == 64  # 8x8
        assert np.all(np.isfinite(result.solution))
        assert result.metadata["method"] == "fno_2d"


class TestDeepONetSmoke:
    def test_solve_runs(self) -> None:
        from src.research.extra_solvers.neural_op import (
            DeepONetBaselineSolver,
            DeepONetSolverConfig,
        )

        op = _make_poisson_op()
        cfg = DeepONetSolverConfig(
            n_train_steps=3,
            branch_width=16,
            branch_layers=1,
            trunk_width=16,
            trunk_layers=1,
            latent_dim=8,
            grid_points_floor=8,
            log_interval=10,
        )
        solver = DeepONetBaselineSolver(cfg)
        result = solver.solve(op, n_dof=64)
        assert result.n_dof == 64
        assert np.all(np.isfinite(result.solution))
        assert result.metadata["method"] == "deeponet"
