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


class _ManufacturedPoissonND(PoissonOperator):
    """N-D Poisson on the unit cube with a separable manufactured solution.

    ``u(x) = prod_d sin(pi x_d)``, so ``-Delta u = dim * pi^2 * u`` and ``u = 0``
    on the boundary — the exact reference for convergence checks in any dim.
    """

    def source_term(self, coords):  # type: ignore[no-untyped-def]
        c = np.asarray(coords, dtype=np.float64)
        prod = np.ones(c.shape[0], dtype=np.float64)
        for d in range(self.dim):
            prod *= np.sin(np.pi * c[:, d])
        return self.dim * (np.pi**2) * prod

    def boundary_value(self, coords):  # type: ignore[no-untyped-def]
        return np.zeros(np.asarray(coords).shape[0], dtype=np.float64)

    def exact_solution(self, coords):  # type: ignore[no-untyped-def]
        c = np.asarray(coords, dtype=np.float64)
        prod = np.ones(c.shape[0], dtype=np.float64)
        for d in range(self.dim):
            prod *= np.sin(np.pi * c[:, d])
        return prod


def _make_manufactured_op(dim: int) -> _ManufacturedPoissonND:
    cfg = PDEConfig(
        name=f"manufactured_{dim}d",
        pde_type=PDEType.POISSON,
        domain_dim=dim,
        domain_min=[0.0] * dim,
        domain_max=[1.0] * dim,
        advection_coeff=[0.0] * dim,
    )
    return _ManufacturedPoissonND(cfg)


class TestDirectPoissonND:
    """N-D generalization: the scipy-direct solver converges in 1D/2D/3D."""

    def test_1d_solution_shape_and_accuracy(self) -> None:
        op = _make_manufactured_op(1)
        result = DirectPoissonSolver().solve(op, n_dof=64)
        assert result.metadata["dim"] == 1
        n = result.metadata["n_per_side"]
        assert result.solution.size == n + 2  # 1D grid incl. 2 boundaries
        assert result.l2_error is not None and result.l2_error < 1e-2

    def test_3d_solution_shape_and_accuracy(self) -> None:
        op = _make_manufactured_op(3)
        result = DirectPoissonSolver().solve(op, n_dof=8**3)  # ~8 per side
        assert result.metadata["dim"] == 3
        n = result.metadata["n_per_side"]
        assert result.n_dof == n**3
        assert result.solution.size == (n + 2) ** 3
        assert result.l2_error is not None and result.l2_error < 5e-2

    def test_3d_refinement_reduces_error(self) -> None:
        op = _make_manufactured_op(3)
        coarse = DirectPoissonSolver().solve(op, n_dof=6**3)
        fine = DirectPoissonSolver().solve(op, n_dof=12**3)
        assert fine.l2_error is not None and coarse.l2_error is not None
        assert fine.l2_error < coarse.l2_error


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

    def test_unsupported_dim_rejected(self) -> None:
        cfg = PDEConfig(
            name="p4d",
            pde_type=PDEType.POISSON,
            domain_dim=4,
            domain_min=[0.0, 0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0, 1.0],
            advection_coeff=[0.0, 0.0, 0.0, 0.0],
        )
        op = PoissonOperator(cfg)
        with pytest.raises(NotImplementedError, match="1D/2D/3D"):
            DirectPoissonSolver().solve(op, n_dof=16)

    def test_homogeneous_bc_boundary_is_zero(self) -> None:
        """Boundary values for zero-BC Poisson must remain zero in solution."""
        op = _make_poisson_op()
        result = DirectPoissonSolver().solve(op, n_dof=64)
        n = result.metadata["n_per_side"]
        grid = result.solution.reshape(n + 2, n + 2)
        np.testing.assert_array_equal(grid[0, :], 0.0)
        np.testing.assert_array_equal(grid[-1, :], 0.0)
        np.testing.assert_array_equal(grid[:, 0], 0.0)
        np.testing.assert_array_equal(grid[:, -1], 0.0)


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

    def test_invalid_cycle_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MultigridPoissonConfig(cycle="X")  # type: ignore[arg-type]

    def test_all_valid_cycles_accepted(self) -> None:
        for cycle in ("V", "W", "F"):
            cfg = MultigridPoissonConfig(cycle=cycle)  # type: ignore[arg-type]
            assert cfg.cycle == cycle


class TestMultigridSolverBehavior:
    def test_registered(self) -> None:
        assert "multigrid" in SOLVER_REGISTRY

    def test_construction_always_succeeds(self) -> None:
        """Construction must never raise — the dep is checked in solve()."""
        from src.research.extra_solvers.multigrid import MultigridPoissonSolver

        solver = MultigridPoissonSolver()
        # Both attributes exist regardless of pyamg presence.
        assert solver.name == "multigrid"
        assert solver.config.cycle in ("V", "W", "F")

    def test_stub_or_real(self) -> None:
        """Multigrid is either real (pyamg installed) or raises ImportError.

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

    def test_missing_pyamg_raises_importerror(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forcing pyamg out of sys.modules must trigger the install hint."""
        import builtins

        from src.research.extra_solvers.multigrid import MultigridPoissonSolver

        original_import = builtins.__import__

        def _no_pyamg(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "pyamg" or name.startswith("pyamg."):
                raise ImportError("forced missing pyamg for test")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_pyamg)

        solver = MultigridPoissonSolver()
        with pytest.raises(ImportError, match="pip install pyamg"):
            solver.solve(_make_poisson_op(), n_dof=64)

    def test_unsupported_dim_rejected(self) -> None:
        """Dim is validated (before the optional pyamg import) — 4D rejected."""
        from src.research.extra_solvers.multigrid import MultigridPoissonSolver

        cfg = PDEConfig(
            name="p4d",
            pde_type=PDEType.POISSON,
            domain_dim=4,
            domain_min=[0.0, 0.0, 0.0, 0.0],
            domain_max=[1.0, 1.0, 1.0, 1.0],
            advection_coeff=[0.0, 0.0, 0.0, 0.0],
        )
        with pytest.raises(NotImplementedError, match="1D/2D/3D"):
            MultigridPoissonSolver().solve(PoissonOperator(cfg), n_dof=16)
