"""Multigrid and direct Poisson solver baselines.

Provides two solvers used by ``config/proposals/doe_ascr_c59.yaml``:

* :class:`DirectPoissonSolver` — sparse direct factorisation via
  ``scipy.sparse.linalg.spsolve``.  Pure scipy, always available.
* :class:`MultigridPoissonSolver` — algebraic multigrid via
  ``pyamg``.  ``pyamg`` is loaded lazily inside :meth:`solve`, so the
  class always exists in the registry; missing dep surfaces as a
  clear :class:`ImportError` at solve time with the install hint.

Both solvers act on the same uniform-grid 5-point Laplacian assembled
from operator-level ``boundary_value`` / ``source_term`` calls.
Configuration is fully Pydantic — no hardcoded numerical constants.
"""

from __future__ import annotations

import time
from typing import Any, Literal

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
    cycle: Literal["V", "W", "F"] = Field(
        default="V",
        description="Multigrid cycle type.",
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
# Shared assembly: uniform N-D (2*dim+1)-point Laplacian
# ---------------------------------------------------------------------------
#
# The discretization is dimension-generic via a Kronecker-sum Laplacian:
#   A = sum_axis ( I (x) ... (x) L1 (x) ... (x) I ),  L1 = tridiag(-1, 2, -1)
# which is the 3-point (1D), 5-point (2D), or 7-point (3D) stencil with a main
# diagonal of 2*dim. For dim==2 this reproduces the previous hand-rolled
# 5-point matrix exactly. The grid is assumed isotropic (equal spacing per
# axis); exact on the unit cube used by the manufactured-solution benchmarks.

# Maximum spatial dimension supported by the Kronecker assembly.
MAX_SUPPORTED_DIM: int = 3


def _grid_axes(operator: PDEOperator, n: int) -> list[NDArray[np.float64]]:
    """Per-axis coordinate arrays of length ``n + 2`` (interior + 2 boundaries)."""
    return [
        np.linspace(
            float(operator.domain_min[d]),
            float(operator.domain_max[d]),
            n + 2,
            dtype=np.float64,
        )
        for d in range(operator.dim)
    ]


def _laplacian_nd(n: int, dim: int) -> Any:
    """Assemble the uniform (2*dim+1)-point Laplacian via a Kronecker sum."""
    from scipy import sparse

    l1 = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    identity = sparse.identity(n, format="csr")
    a_matrix: Any | None = None
    for axis in range(dim):
        term: Any | None = None
        for k in range(dim):
            mat = l1 if k == axis else identity
            term = mat if term is None else sparse.kron(term, mat, format="csr")
        a_matrix = term if a_matrix is None else (a_matrix + term)
    assert a_matrix is not None, "dim >= 1 guarantees the Kronecker sum is built"
    return a_matrix.tocsr()


def _assemble_poisson_nd(
    operator: PDEOperator,
    n_per_side: int,
) -> tuple[Any, NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """Assemble the N-D Laplacian system for a uniform-grid Poisson operator.

    Returns:
        ``(A, rhs, full_grid_coords, axes)`` where ``A`` is the sparse
        ``(n**dim, n**dim)`` interior Laplacian, ``rhs`` the right-hand side
        with Dirichlet boundary contributions folded in, ``full_grid_coords``
        the ``((n+2)**dim, dim)`` coordinate array (interior + boundary) for
        error reporting, and ``axes`` the per-axis coordinate arrays.

    """
    n = int(n_per_side)
    dim = operator.dim
    axes = _grid_axes(operator, n)
    h = float(axes[0][1] - axes[0][0])

    interior = [ax[1:-1] for ax in axes]
    interior_mesh = np.meshgrid(*interior, indexing="ij")
    interior_coords = np.stack([m.ravel() for m in interior_mesh], axis=-1)

    a_matrix = _laplacian_nd(n, dim)

    source = np.asarray(
        operator.source_term(interior_coords.astype(np.float32)),
        dtype=np.float64,
    )
    rhs_grid = ((h**2) * source).reshape((n,) * dim)

    # Dirichlet boundary contributions: the interior nodes adjacent to each of
    # the 2*dim faces gain the boundary value of their off-grid neighbour.
    for axis in range(dim):
        for slot, wall in ((0, axes[axis][0]), (n - 1, axes[axis][-1])):
            face_coords = _face_coords(interior, axis, wall, dim)
            bc = np.asarray(operator.boundary_value(face_coords), dtype=np.float64)
            index: list[Any] = [slice(None)] * dim
            index[axis] = slot
            rhs_grid[tuple(index)] += bc.reshape(rhs_grid[tuple(index)].shape)

    full_mesh = np.meshgrid(*axes, indexing="ij")
    full_grid = np.stack([m.ravel() for m in full_mesh], axis=-1)
    return a_matrix, rhs_grid.ravel(), full_grid, axes


def _face_coords(
    interior: list[NDArray[np.float64]],
    axis: int,
    wall: float,
    dim: int,
) -> NDArray[np.float32]:
    """Coordinates of one boundary face: interior transverse grid, axis fixed at ``wall``."""
    transverse = [interior[k] for k in range(dim) if k != axis]
    if not transverse:  # 1D: the face is a single point
        coords = np.array([[wall]], dtype=np.float32)
        return coords
    mesh = np.meshgrid(*transverse, indexing="ij")
    columns: list[NDArray[np.float64]] = []
    cursor = 0
    for k in range(dim):
        if k == axis:
            columns.append(np.full(mesh[0].size, wall, dtype=np.float64))
        else:
            columns.append(mesh[cursor].ravel())
            cursor += 1
    return np.stack(columns, axis=-1).astype(np.float32)


def _resolve_grid_size(operator: PDEOperator, n_dof: int, floor: int) -> int:
    """Choose ``n_per_side`` so ``n_per_side**dim ≈ n_dof`` (rounded).

    Supports 1D/2D/3D Poisson-type operators.
    """
    dim = operator.dim
    if dim < 1 or dim > MAX_SUPPORTED_DIM:
        raise NotImplementedError(
            f"This solver supports 1D/2D/3D Poisson-type problems only "
            f"(operator.dim={dim}; max supported is {MAX_SUPPORTED_DIM})."
        )
    target = max(int(n_dof), floor**dim)
    n_per_side = int(round(target ** (1.0 / dim)))
    return max(n_per_side, floor)


def _fill_boundary(
    solution_grid: NDArray[np.float64],
    operator: PDEOperator,
    n_per_side: int,
) -> None:
    """Populate the boundary faces of *solution_grid* in-place from operator BCs.

    Dimension-generic: assigns ``operator.boundary_value`` to every node lying
    on any face of the ``(n+2)**dim`` grid. For homogeneous Dirichlet BCs this
    is a no-op (zeros stay zeros); non-homogeneous values are applied correctly.
    """
    dim = operator.dim
    axes = _grid_axes(operator, n_per_side)
    shape = solution_grid.shape

    full_mesh = np.meshgrid(*axes, indexing="ij")
    coords = np.stack([m.ravel() for m in full_mesh], axis=-1)

    indices = np.indices(shape)
    boundary_mask = np.zeros(shape, dtype=bool)
    for axis in range(dim):
        boundary_mask |= (indices[axis] == 0) | (indices[axis] == shape[axis] - 1)
    flat_mask = boundary_mask.ravel()

    boundary_vals = np.asarray(
        operator.boundary_value(coords[flat_mask].astype(np.float32)),
        dtype=np.float64,
    )
    flat = solution_grid.ravel()
    flat[flat_mask] = boundary_vals
    solution_grid[...] = flat.reshape(shape)


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
        dim = operator.dim
        n = n_per_side
        log = logger.bind(solver=self.name, n_per_side=n_per_side, dim=dim, n_dof=n**dim)
        log.info("direct_solve_start")
        t0 = time.perf_counter()

        A, rhs, grid, _axes = _assemble_poisson_nd(operator, n_per_side)
        # Boundary already folded into rhs via Dirichlet contributions.
        u_interior = spsolve(A.tocsc(), rhs)

        # Embed interior solution into the full grid; populate boundary faces
        # from operator BCs so non-homogeneous Dirichlet values are correct.
        solution_grid = np.zeros((n + 2,) * dim, dtype=np.float64)
        interior_index = tuple(slice(1, -1) for _ in range(dim))
        solution_grid[interior_index] = u_interior.reshape((n,) * dim)
        _fill_boundary(solution_grid, operator, n)

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
            n_dof=int(n**dim),
            wall_time_seconds=float(wall_time),
            l2_error=l2_err,
            metadata={"method": "scipy_spsolve", "n_per_side": int(n), "dim": int(dim)},
        )


SOLVER_REGISTRY.setdefault("direct_solver", DirectPoissonSolver)


# ---------------------------------------------------------------------------
# Multigrid solver (lazy-imports pyamg)
# ---------------------------------------------------------------------------


class MultigridPoissonSolver(BaseSolver):
    """Algebraic multigrid Poisson solver (pyamg-backed).

    The :mod:`pyamg` dependency is loaded lazily inside :meth:`solve`
    so the class itself always exists in the registry.  When the
    dependency is missing, ``solve`` raises a clear :class:`ImportError`
    pointing the user at the correct install command — same UX as the
    explicit stub from :mod:`._optional`, but without an import-time
    branch that confuses both type-checkers and coverage tooling.
    """

    name = "multigrid"
    description = "Algebraic Multigrid (smoothed aggregation, pyamg)"
    _MISSING_DEP_HINT = "pip install pyamg"

    def __init__(self, config: MultigridPoissonConfig | None = None) -> None:
        self.config = config or MultigridPoissonConfig()
        self._log = logger.bind(solver=self.name)

    def solve(
        self,
        operator: PDEOperator,
        n_dof: int,
        **kwargs: Any,
    ) -> SolverResult:
        """Solve the 2D Poisson problem with smoothed-aggregation AMG.

        Args:
            operator: 2D Poisson-type operator.
            n_dof: Approximate target DOF (rounded to a perfect square).
            **kwargs: Reserved for the :class:`BaseSolver` protocol.

        Returns:
            :class:`SolverResult` with the multigrid-cycle solution,
            timing, and L2 error vs ``operator.exact_solution`` if any.

        Raises:
            ImportError: If ``pyamg`` is not installed.

        """
        # Validate the operator shape before the (optional) pyamg
        # import so dim-mismatch errors do not get masked by a missing
        # dependency on environments without pyamg installed.
        n_per_side = _resolve_grid_size(operator, n_dof, self.config.min_grid_points)

        try:
            import pyamg
        except ImportError as exc:
            raise ImportError(
                f"Solver '{self.name}' requires the optional package "
                f"'pyamg'.  Install it with: {self._MISSING_DEP_HINT}"
            ) from exc
        dim = operator.dim
        n = n_per_side
        log = self._log.bind(n_per_side=n_per_side, dim=dim, cycle=self.config.cycle)
        log.debug("multigrid_solve_start", n_dof=n_dof)
        t0 = time.perf_counter()

        A, rhs, grid, _axes = _assemble_poisson_nd(operator, n_per_side)
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

        solution_grid = np.zeros((n + 2,) * dim, dtype=np.float64)
        interior_index = tuple(slice(1, -1) for _ in range(dim))
        solution_grid[interior_index] = u_interior.reshape((n,) * dim)
        _fill_boundary(solution_grid, operator, n)
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
            n_dof=int(n**dim),
            wall_time_seconds=float(wall_time),
            l2_error=l2_err,
            metadata={
                "method": "pyamg_smoothed_aggregation",
                "cycle": self.config.cycle,
                "n_per_side": int(n),
                "dim": int(dim),
                "n_levels": int(len(ml.levels)),
            },
        )


SOLVER_REGISTRY.setdefault("multigrid", MultigridPoissonSolver)
