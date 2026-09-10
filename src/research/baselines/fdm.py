"""Uniform-grid finite-difference baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from src.pde.operators import PDEOperator
from src.research.baselines.base import BaseSolver
from src.research.baselines.schemas import FDMConfig, SolverResult

logger = structlog.get_logger(__name__)


class UniformFDMSolver(BaseSolver):
    """Finite difference method on a uniform grid.

    Uses second-order central differences and scipy.sparse for
    the linear system.  Supports 1D and 2D Poisson-type problems.
    """

    name = "uniform_fdm"
    description = "Uniform-grid finite difference method"

    def __init__(self, config: FDMConfig | None = None) -> None:
        self.config = config or FDMConfig()

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve using finite differences on a uniform grid."""
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise ImportError(
                "UniformFDMSolver requires scipy. Install with: pip install scipy"
            ) from exc

        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("fdm_solve_start")
        t0 = time.perf_counter()

        if operator.dim == 1:
            solution, grid = self._solve_1d(operator, n_dof, sparse, spsolve)
        elif operator.dim == 2:
            solution, grid = self._solve_2d(operator, n_dof, sparse, spsolve)
        else:
            raise NotImplementedError(f"UniformFDMSolver does not support dim={operator.dim}")

        wall_time = time.perf_counter() - t0
        l2_err = self._compute_l2_error(solution, grid, operator)

        log.info("fdm_solve_done", wall_time=wall_time, l2_error=l2_err)
        return SolverResult(
            solution=solution,
            grid_points=grid,
            n_dof=len(solution),
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={"method": "central_differences", "order": 2},
        )

    # ------------------------------------------------------------------
    # 1D solver
    # ------------------------------------------------------------------
    def _solve_1d(
        self,
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = max(n_dof, self.config.min_grid_points)
        x = np.linspace(
            float(operator.domain_min[0]),
            float(operator.domain_max[0]),
            n + 2,
            dtype=np.float64,
        )
        h = x[1] - x[0]
        interior = x[1:-1]

        # Build tridiagonal: -u_{i-1} + 2u_i - u_{i+1} = h^2 * f_i
        diags = [
            np.full(n - 1, -1.0),
            np.full(n, 2.0),
            np.full(n - 1, -1.0),
        ]
        A = sparse.diags(diags, offsets=[-1, 0, 1], format="csc")

        coords = interior.reshape(-1, 1).astype(np.float32)
        f = np.asarray(operator.source_term(coords), dtype=np.float64).flatten()
        rhs = (h**2) * f

        # Boundary conditions
        bc_left = operator.boundary_value(np.array([[operator.domain_min[0]]], dtype=np.float32))
        bc_right = operator.boundary_value(np.array([[operator.domain_max[0]]], dtype=np.float32))
        bc_left_val = float(np.asarray(bc_left).flat[0])
        bc_right_val = float(np.asarray(bc_right).flat[0])
        rhs[0] += bc_left_val
        rhs[-1] += bc_right_val

        u_inner = spsolve(A, rhs)
        u_full = np.concatenate([[bc_left_val], u_inner, [bc_right_val]])
        grid = x.reshape(-1, 1)
        return u_full.astype(np.float64), grid.astype(np.float64)

    # ------------------------------------------------------------------
    # 2D solver
    # ------------------------------------------------------------------
    def _solve_2d(
        self,
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = max(int(np.sqrt(n_dof)), self.config.min_grid_points)
        xs = np.linspace(
            float(operator.domain_min[0]),
            float(operator.domain_max[0]),
            n + 2,
            dtype=np.float64,
        )
        ys = np.linspace(
            float(operator.domain_min[1]),
            float(operator.domain_max[1]),
            n + 2,
            dtype=np.float64,
        )
        h = xs[1] - xs[0]
        xi, yi = xs[1:-1], ys[1:-1]
        XX, YY = np.meshgrid(xi, yi, indexing="ij")
        coords_flat = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)

        # 5-point Laplacian stencil (n*n interior unknowns)
        I_n = sparse.eye(n, format="csc")
        T = sparse.diags(
            [np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)],
            [-1, 0, 1],
            format="csc",
        )
        A = sparse.kron(I_n, T) + sparse.kron(
            sparse.diags(
                [np.full(n - 1, -1.0), np.full(n - 1, -1.0)],
                [-1, 1],
                format="csc",
            ),
            I_n,
        )

        f = np.asarray(operator.source_term(coords_flat), dtype=np.float64).flatten()
        rhs = (h**2) * f

        u_inner = spsolve(A, rhs)

        # Build full grid including boundary for error computation
        XX_full, YY_full = np.meshgrid(xs, ys, indexing="ij")
        grid_full = np.stack([XX_full.ravel(), YY_full.ravel()], axis=-1)

        # Place interior solution into full grid
        u_full = np.zeros((n + 2, n + 2), dtype=np.float64)
        u_full[1:-1, 1:-1] = u_inner.reshape(n, n)

        # Fill boundary values
        for i in range(n + 2):
            for side_coords in [
                np.array([[xs[i], ys[0]]], dtype=np.float32),
                np.array([[xs[i], ys[-1]]], dtype=np.float32),
            ]:
                bv = float(np.asarray(operator.boundary_value(side_coords)).flat[0])
                if side_coords[0, 1] == ys[0]:
                    u_full[i, 0] = bv
                else:
                    u_full[i, -1] = bv
        for j in range(n + 2):
            for side_coords in [
                np.array([[xs[0], ys[j]]], dtype=np.float32),
                np.array([[xs[-1], ys[j]]], dtype=np.float32),
            ]:
                bv = float(np.asarray(operator.boundary_value(side_coords)).flat[0])
                if side_coords[0, 0] == xs[0]:
                    u_full[0, j] = bv
                else:
                    u_full[-1, j] = bv

        return (
            u_full.ravel().astype(np.float64),
            grid_full.astype(np.float64),
        )
