"""Multigrid and direct Poisson solver baselines.

Provides two solvers used by ``config/proposals/doe_ascr_c59.yaml``:

* :class:`DirectPoissonSolver` — sparse direct factorisation via
  ``scipy.sparse.linalg.spsolve``.  Pure scipy, always available.
* :class:`MultigridPoissonSolver` — algebraic multigrid via
  ``pyamg``.  Optional dependency: when ``pyamg`` is unavailable the
  registry receives a stub that raises a clear :class:`ImportError`
  at construction time, so the global coverage gate stays green.

Both solvers act on the same uniform-grid 5-point Laplacian assembled
from operator-level ``boundary_value`` / ``source_term`` calls.
Configuration is fully Pydantic — no hardcoded numerical constants.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray
from pydantic import Field

from src.pde.operators import PDEOperator
from src.research.baselines import (
    SOLVER_REGISTRY,
    BaseSolver,
    SolverConfig,
    SolverResult,
)
from src.research.extra_solvers._optional import make_optional_dependency_stub

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic configs
# ---------------------------------------------------------------------------


class DirectPoissonConfig(SolverConfig):
    """Configuration for :class:`DirectPoissonSolver`."""

    min_grid_points: int = Field(
        default=3,
        ge=2,
        description="Minimum interior nodes per dimension.",
    )


class MultigridPoissonConfig(SolverConfig):
    """Configuration for :class:`MultigridPoissonSolver`."""

    min_grid_points: int = Field(
        default=4,
        ge=2,
        description="Minimum interior nodes per dimension.",
    )
    max_levels: int = Field(
        default=10,
        ge=2,
        description="Maximum number of multigrid levels.",
    )
    cycle: str = Field(
        default="V",
        description="Multigrid cycle type: 'V', 'W', or 'F'.",
    )
    presmoother: str = Field(
        default="gauss_seidel",
        description="Presmoother (passed through to pyamg.smoothed_aggregation_solver).",
    )
    postsmoother: str = Field(
        default="gauss_seidel",
        description="Postsmoother (passed through to pyamg).",
    )


# ---------------------------------------------------------------------------
# Shared assembly: uniform 5-point Laplacian
# ---------------------------------------------------------------------------


def _assemble_poisson_2d(
    operator: PDEOperator,
    n_per_side: int,
) -> tuple[Any, NDArray[np.float64], NDArray[np.float64]]:
    """Assemble the 2D 5-point Laplacian for a uniform-grid Poisson op.

    Returns
    -------
    A
        ``scipy.sparse.csr_matrix`` of shape (N, N) where N=n_per_side**2.
    rhs
        Right-hand side vector of shape (N,).
    grid_points
        Coordinate array of shape (N, 2) for downstream error reporting.

    """
    from scipy import sparse

    n = int(n_per_side)
    x = np.linspace(
        float(operator.domain_min[0]),
        float(operator.domain_max[0]),
        n + 2,
        dtype=np.float64,
    )
    y = np.linspace(
        float(operator.domain_min[1]),
        float(operator.domain_max[1]),
        n + 2,
        dtype=np.float64,
    )
    h = float(x[1] - x[0])

    interior_x = x[1:-1]
    interior_y = y[1:-1]
    xx, yy = np.meshgrid(interior_x, interior_y, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    main = np.full(n * n, 4.0)
    horiz = np.full(n * n - 1, -1.0)
    horiz[(np.arange(1, n * n) % n) == 0] = 0.0  # break wrap on row boundaries
    vert = np.full(n * (n - 1), -1.0)

    A = sparse.diags(
        [vert, horiz, main, horiz, vert],
        offsets=[-n, -1, 0, 1, n],
        shape=(n * n, n * n),
        format="csr",
    )

    f = np.asarray(
        operator.source_term(coords.astype(np.float32)),
        dtype=np.float64,
    ).reshape(n, n)
    rhs = (h**2) * f.ravel()

    # Boundary contributions: subtract the appropriate column of A applied
    # to the Dirichlet data on the four sides.
    bc_left = np.asarray(
        operator.boundary_value(
            np.stack([np.full(n, x[0], dtype=np.float32), interior_y.astype(np.float32)], axis=-1)
        ),
        dtype=np.float64,
    )
    bc_right = np.asarray(
        operator.boundary_value(
            np.stack(
                [np.full(n, x[-1], dtype=np.float32), interior_y.astype(np.float32)],
                axis=-1,
            )
        ),
        dtype=np.float64,
    )
    bc_bottom = np.asarray(
        operator.boundary_value(
            np.stack(
                [interior_x.astype(np.float32), np.full(n, y[0], dtype=np.float32)],
                axis=-1,
            )
        ),
        dtype=np.float64,
    )
    bc_top = np.asarray(
        operator.boundary_value(
            np.stack(
                [interior_x.astype(np.float32), np.full(n, y[-1], dtype=np.float32)],
                axis=-1,
            )
        ),
        dtype=np.float64,
    )

    rhs_grid = rhs.reshape(n, n)
    rhs_grid[0, :] += bc_left  # i=0 column gets contribution from x = x_min wall
    rhs_grid[-1, :] += bc_right
    rhs_grid[:, 0] += bc_bottom
    rhs_grid[:, -1] += bc_top

    full_grid = np.stack(
        np.meshgrid(x, y, indexing="ij"),
        axis=-1,
    )
    return A, rhs_grid.ravel(), full_grid.reshape(-1, 2)


def _resolve_grid_size(operator: PDEOperator, n_dof: int, floor: int) -> int:
    """Choose ``n_per_side`` so ``n_per_side**2 == n_dof`` (rounded)."""
    if operator.dim != 2:
        raise NotImplementedError(
            f"This solver currently supports 2D Poisson-type problems only "
            f"(operator.dim={operator.dim})."
        )
    n_per_side = int(round(np.sqrt(max(int(n_dof), floor**2))))
    return max(n_per_side, floor)


# ---------------------------------------------------------------------------
# Direct solver
# ---------------------------------------------------------------------------


class DirectPoissonSolver(BaseSolver):
    """Direct sparse solver for 2D Poisson-type PDEs."""

    name = "direct_solver"
    description = "Sparse-direct solve via scipy.sparse.linalg.spsolve"

    def __init__(self, config: DirectPoissonConfig | None = None) -> None:
        self.config = config or DirectPoissonConfig()

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        try:
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise ImportError(
                "DirectPoissonSolver requires scipy. Install with: pip install scipy"
            ) from exc

        n_per_side = _resolve_grid_size(operator, n_dof, self.config.min_grid_points)
        log = logger.bind(solver=self.name, n_per_side=n_per_side, n_dof=n_per_side**2)
        log.info("direct_solve_start")
        t0 = time.perf_counter()

        A, rhs, grid = _assemble_poisson_2d(operator, n_per_side)
        # Reshape rhs back to interior-only for the solve (boundary already
        # baked into rhs via Dirichlet contributions in ``_assemble_poisson_2d``).
        n = n_per_side
        u_interior = spsolve(A.tocsc(), rhs)

        # Embed interior solution into the full grid (Dirichlet BCs on edges).
        solution_grid = np.zeros((n + 2, n + 2), dtype=np.float64)
        solution_grid[1:-1, 1:-1] = u_interior.reshape(n, n)

        wall_time = time.perf_counter() - t0
        l2_err = self._compute_l2_error(
            solution=solution_grid.ravel(),
            coords=grid.astype(np.float32),
            operator=operator,
        )
        log.info("direct_solve_done", wall_time=wall_time, l2_error=l2_err)
        return SolverResult(
            solution=solution_grid.ravel(),
            grid_points=grid,
            n_dof=int(n_per_side**2),
            wall_time_seconds=float(wall_time),
            l2_error=l2_err,
            metadata={"method": "scipy_spsolve", "n_per_side": int(n_per_side)},
        )


SOLVER_REGISTRY.setdefault("direct_solver", DirectPoissonSolver)


# ---------------------------------------------------------------------------
# Multigrid solver (optional dep)
# ---------------------------------------------------------------------------


try:  # pragma: no cover - exercised only when pyamg is installed
    import pyamg  # type: ignore[import-not-found]

    _PYAMG_AVAILABLE = True
except ImportError:
    _PYAMG_AVAILABLE = False


if _PYAMG_AVAILABLE:

    class MultigridPoissonSolver(BaseSolver):  # type: ignore[no-redef]
        """Algebraic multigrid Poisson solver (pyamg-backed)."""

        name = "multigrid"
        description = "Algebraic Multigrid (smoothed aggregation, pyamg)"

        def __init__(self, config: MultigridPoissonConfig | None = None) -> None:
            self.config = config or MultigridPoissonConfig()

        def solve(
            self,
            operator: PDEOperator,
            n_dof: int,
            **kwargs: Any,
        ) -> SolverResult:
            n_per_side = _resolve_grid_size(
                operator, n_dof, self.config.min_grid_points
            )
            log = logger.bind(
                solver=self.name, n_per_side=n_per_side, cycle=self.config.cycle
            )
            log.info("multigrid_solve_start")
            t0 = time.perf_counter()

            A, rhs, grid = _assemble_poisson_2d(operator, n_per_side)
            ml = pyamg.smoothed_aggregation_solver(
                A.tocsr(),
                max_levels=self.config.max_levels,
                presmoother=self.config.presmoother,
                postsmoother=self.config.postsmoother,
            )
            u_interior = ml.solve(
                rhs,
                tol=self.config.tolerance,
                maxiter=self.config.max_iterations,
                cycle=self.config.cycle,
            )

            n = n_per_side
            solution_grid = np.zeros((n + 2, n + 2), dtype=np.float64)
            solution_grid[1:-1, 1:-1] = u_interior.reshape(n, n)
            wall_time = time.perf_counter() - t0
            l2_err = self._compute_l2_error(
                solution=solution_grid.ravel(),
                coords=grid.astype(np.float32),
                operator=operator,
            )
            log.info("multigrid_solve_done", wall_time=wall_time, l2_error=l2_err)
            return SolverResult(
                solution=solution_grid.ravel(),
                grid_points=grid,
                n_dof=int(n_per_side**2),
                wall_time_seconds=float(wall_time),
                l2_error=l2_err,
                metadata={
                    "method": "pyamg_smoothed_aggregation",
                    "cycle": self.config.cycle,
                    "n_per_side": int(n_per_side),
                    "n_levels": int(len(ml.levels)),
                },
            )

    SOLVER_REGISTRY.setdefault("multigrid", MultigridPoissonSolver)
else:
    MultigridPoissonSolver = make_optional_dependency_stub(  # type: ignore[assignment, misc]
        name="multigrid",
        description="Algebraic Multigrid Poisson solver",
        dependency="pyamg",
        install_hint="pip install pyamg",
    )
    SOLVER_REGISTRY.setdefault("multigrid", MultigridPoissonSolver)
