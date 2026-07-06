"""Classical PDE solver baselines for benchmarking.

Provides reference implementations against which AlphaGalerkin's
MCTS-guided approach is compared in SBIR proposals.

Baselines:
- UniformFDMSolver: Finite difference on uniform grid (scipy.sparse)
- DorflerAMRSolver: Dorfler marking adaptive mesh refinement
- SimplePINNSolver: Physics-Informed Neural Network baseline
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.pde.config import PDEType
from src.pde.operators import PDEOperator
from src.poc.device import resolve_device
from src.research.gpu_profiler import GpuUtilizationProfiler

logger = structlog.get_logger(__name__)

# Optional geometry predicate for masked (non-rectangular) domains.  Given
# node coordinates of shape ``(N, 2)`` it returns a boolean mask ``(N,)`` that
# is True for nodes *inside* the physical domain.  ``None`` (the default on
# every parameter below) means "full bounding box" — the historical behaviour,
# so every existing caller and test is byte-for-byte unchanged.
InsidePredicate = Callable[[NDArray[np.float64]], NDArray[np.bool_]]


@dataclass
class SolverResult:
    """Result from a baseline solver run.

    Attributes:
        solution: Solution values at grid points.
        grid_points: Grid coordinates (N, dim).
        n_dof: Degrees of freedom used.
        wall_time_seconds: Wall-clock solve time.
        l2_error: L2 error vs exact solution, if available.
        h1_error: H1 error vs exact solution, if available.
        metadata: Extra solver-specific info.

    """

    solution: NDArray[np.float64]
    grid_points: NDArray[np.float64]
    n_dof: int
    wall_time_seconds: float
    l2_error: float | None = None
    h1_error: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary (excluding large arrays)."""
        return {
            "n_dof": self.n_dof,
            "wall_time_seconds": self.wall_time_seconds,
            "l2_error": self.l2_error,
            "h1_error": self.h1_error,
            "metadata": self.metadata,
        }


class SolverConfig(BaseModel):
    """Configuration for baseline solvers."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0, description="Random seed")
    max_iterations: int = Field(default=10000, ge=1, description="Max solver iterations")
    tolerance: float = Field(default=1e-10, gt=0, description="Convergence tolerance")


class FDMConfig(SolverConfig):
    """Configuration for finite difference solvers."""

    min_grid_points: int = Field(
        default=3,
        ge=2,
        description="Minimum grid points per dimension",
    )


class AMRConfig(SolverConfig):
    """Configuration for adaptive mesh refinement solvers."""

    marking_fraction: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Dorfler bulk-chasing marking fraction (theta). Default raised "
            "from 0.3 to 0.5 so AMR on sharply-concentrated indicators "
            "(e.g. Burgers shock) marks several elements per step instead "
            "of one, reaching the target DOF within max_refinements."
        ),
    )
    max_refinements: int = Field(
        default=30,
        ge=1,
        description=(
            "Maximum number of refinement iterations. Default raised from 10 "
            "to 30 so 1D AMR with conservative marking (theta=0.3) can reach "
            "8K+ DOF on smooth Burgers/Poisson cases without saturating."
        ),
    )
    initial_dof_divisor: int = Field(
        default=2,
        ge=1,
        description=(
            "Divisor for initial DOF count (n_dof // divisor). Default 2 means "
            "the initial 1D grid starts at half the target DOF (capped by "
            "max_initial_points_1d) so refinement only needs ~1 doubling to "
            "reach the target on smooth problems."
        ),
    )
    max_initial_points_1d: int = Field(
        default=256,
        ge=2,
        description=(
            "Maximum initial grid points for 1D AMR. Default raised from 8 "
            "to 256 so high-target n_dof requests (>=128) start with a "
            "mesh dense enough that target-aware refinement reaches the "
            "requested DOF count within max_refinements steps."
        ),
    )
    min_initial_points: int = Field(
        default=4,
        ge=2,
        description="Minimum initial grid points (1D) or per-side (2D)",
    )
    initial_side_divisor_2d: int = Field(
        default=2,
        ge=1,
        description="Divisor for initial 2D side length (sqrt(n_dof) // divisor)",
    )
    min_initial_side_2d: int = Field(
        default=3,
        ge=2,
        description="Minimum initial side grid points for 2D AMR",
    )


class PINNConfig(SolverConfig):
    """Configuration for PINN solver."""

    hidden_dim: int = Field(default=64, ge=8, description="Hidden layer dimension")
    n_layers: int = Field(default=3, ge=1, description="Number of hidden layers")
    n_epochs: int = Field(default=2000, ge=1, description="Training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    n_collocation: int = Field(default=1000, ge=10, description="Interior collocation points")
    bc_loss_weight: float = Field(default=10.0, gt=0, description="Boundary condition loss weight")
    n_boundary_points: int = Field(default=50, ge=4, description="Boundary points per epoch")
    log_interval: int = Field(default=500, ge=1, description="Logging interval (epochs)")
    device: str = Field(
        default="auto",
        description=(
            "Device preference: 'auto' (CUDA if available else CPU), 'cpu', "
            "'cuda', or 'cuda:N'. Default 'auto' matches the project's "
            "GPU-preferred policy while keeping CI green on no-GPU runners."
        ),
    )
    vector_pde: bool | None = Field(
        default=None,
        description=(
            "Override for vector-valued PDE handling. None auto-detects from "
            "operator.pde_type (Navier-Stokes => 2-channel network). True/False "
            "forces 2-channel/scalar output respectively."
        ),
    )


class NavierStokesConfig(SolverConfig):
    """Configuration for Navier-Stokes FDM solver."""

    dt: float = Field(default=0.01, gt=0, description="Time step size")
    t_final: float = Field(default=1.0, gt=0, description="Final simulation time")
    default_viscosity: float = Field(
        default=0.01,
        gt=0,
        description="Fallback viscosity if operator lacks viscosity attribute",
    )
    min_grid_points: int = Field(default=4, ge=2, description="Minimum grid points per side")
    cfl_safety: float = Field(
        default=0.25,
        gt=0,
        le=1.0,
        description="CFL stability factor for diffusion (dt <= cfl_safety * h^2 / nu)",
    )
    viscosity_floor: float = Field(
        default=1e-12,
        gt=0,
        description="Minimum viscosity for CFL denominator to avoid division by zero",
    )
    log_fraction: int = Field(
        default=10,
        ge=1,
        description="Log every n_steps // log_fraction steps",
    )


class BaseSolver(ABC):
    """Protocol for classical PDE solvers.

    All baselines implement this interface so that the benchmark runner
    can treat them uniformly.
    """

    name: str = "abstract"
    description: str = "Abstract base solver"

    @abstractmethod
    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve the PDE problem.

        Args:
            operator: PDE operator defining the problem.
            n_dof: Target degrees of freedom.
            **kwargs: Solver-specific options.

        Returns:
            SolverResult with solution, timing, and error metrics.

        """
        ...

    def _compute_l2_error(
        self,
        solution: NDArray[np.float64],
        coords: NDArray[np.float64],
        operator: PDEOperator,
    ) -> float | None:
        """Compute L2 error against the exact solution, if available."""
        exact = operator.exact_solution(coords.astype(np.float32))
        if exact is None:
            return None
        if isinstance(exact, torch.Tensor):
            exact = exact.detach().cpu().numpy()
        exact = np.asarray(exact, dtype=np.float64)
        diff = solution.flatten() - exact.flatten()
        n = len(diff)
        return float(np.sqrt(np.sum(diff**2) / n)) if n > 0 else None


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


class DorflerAMRSolver(BaseSolver):
    """Dorfler marking adaptive mesh refinement.

    Uses residual-based error indicators to mark elements for
    refinement following the Dorfler bulk-chasing strategy.
    Implemented with scipy.sparse on simplex/rectangular meshes.
    """

    name = "dorfler_amr"
    description = "Dorfler marking adaptive mesh refinement"

    def __init__(
        self,
        marking_fraction: float | None = None,
        max_refinements: int | None = None,
        config: AMRConfig | None = None,
    ) -> None:
        self.config = config or AMRConfig()
        self.marking_fraction = (
            marking_fraction if marking_fraction is not None else self.config.marking_fraction
        )
        self.max_refinements = (
            max_refinements if max_refinements is not None else self.config.max_refinements
        )
        if not 0.0 < self.marking_fraction < 1.0:
            raise ValueError(f"marking_fraction must be in (0,1), got {self.marking_fraction}")

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve with adaptive refinement via Dorfler marking.

        Starts with a coarse uniform grid and refines cells where the
        residual-based error indicator is largest, until n_dof is reached
        or max_refinements is exhausted.
        """
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise ImportError(
                "DorflerAMRSolver requires scipy. Install with: pip install scipy"
            ) from exc

        if operator.dim not in (1, 2):
            raise NotImplementedError(
                f"DorflerAMRSolver supports dim=1 and dim=2, got dim={operator.dim}"
            )

        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("amr_solve_start")
        t0 = time.perf_counter()

        if operator.dim == 1:
            result = self._solve_amr_1d(operator, n_dof, sparse, spsolve, log)
        else:
            result = self._solve_amr_2d(operator, n_dof, sparse, spsolve, log)

        wall_time = time.perf_counter() - t0
        solution, grid, n_refinements = result
        l2_err = self._compute_l2_error(solution, grid, operator)

        log.info("amr_solve_done", wall_time=wall_time, l2_error=l2_err, final_dof=len(solution))
        return SolverResult(
            solution=solution,
            grid_points=grid,
            n_dof=len(solution),
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={
                "marking_fraction": self.marking_fraction,
                "n_refinements": n_refinements,
                "dim": operator.dim,
            },
        )

    def _solve_amr_1d(
        self,
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
        log: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run 1D AMR loop. Returns (solution, grid, n_refinements)."""
        n_start = max(
            min(n_dof // self.config.initial_dof_divisor, self.config.max_initial_points_1d),
            self.config.min_initial_points,
        )
        x = np.linspace(
            float(operator.domain_min[0]),
            float(operator.domain_max[0]),
            n_start,
            dtype=np.float64,
        )

        step = 0
        for step in range(self.max_refinements):
            u, _ = self._solve_on_grid(x, operator, sparse, spsolve)
            if len(x) >= n_dof:
                break
            indicators = self._compute_indicators(x, u, operator)
            marked = self._dorfler_mark(indicators)
            x = self._refine_grid(x, marked)
            log.debug(
                "amr_step",
                step=step,
                n_points=len(x),
                max_indicator=float(np.max(indicators)),
            )

        u, _ = self._solve_on_grid(x, operator, sparse, spsolve)
        grid = x.reshape(-1, 1).astype(np.float64)
        return u, grid, step + 1

    def _solve_amr_2d(
        self,
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
        log: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
        """Run 2D AMR loop with element-wise refinement.

        Uses a quadrilateral mesh with element-wise bisection.
        Returns (solution, grid, n_refinements).
        """
        # Start with coarse grid
        n_side = max(
            int(np.sqrt(n_dof)) // self.config.initial_side_divisor_2d,
            self.config.min_initial_side_2d,
        )
        x_lo, x_hi = float(operator.domain_min[0]), float(operator.domain_max[0])
        y_lo, y_hi = float(operator.domain_min[1]), float(operator.domain_max[1])
        xs = np.linspace(x_lo, x_hi, n_side + 1, dtype=np.float64)
        ys = np.linspace(y_lo, y_hi, n_side + 1, dtype=np.float64)

        step = 0
        for step in range(self.max_refinements):
            u, grid = self._solve_on_grid_2d(xs, ys, operator, sparse, spsolve)
            current_dof = len(xs) * len(ys)
            if current_dof >= n_dof:
                break

            indicators = self._compute_indicators_2d(xs, ys, u, operator)
            marked_x, marked_y = self._dorfler_mark_2d(indicators, xs, ys)

            xs = self._refine_grid(xs, marked_x)
            ys = self._refine_grid(ys, marked_y)

            log.debug(
                "amr_step_2d",
                step=step,
                n_x=len(xs),
                n_y=len(ys),
                n_dof=len(xs) * len(ys),
                max_indicator=float(np.max(indicators)),
            )

        u, grid = self._solve_on_grid_2d(xs, ys, operator, sparse, spsolve)
        return u, grid, step + 1

    # ------------------------------------------------------------------
    @staticmethod
    def _solve_on_grid(
        x: NDArray[np.float64],
        operator: PDEOperator,
        sparse: Any,
        spsolve: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve 1D Poisson on a (possibly non-uniform) grid."""
        n = len(x)
        interior = x[1:-1]
        h_left = np.diff(x[:-1])  # h_{i-1}
        h_right = np.diff(x[1:])  # h_i

        # Variable-coefficient FD stencil on non-uniform grid
        a_left = 2.0 / (h_left * (h_left + h_right))
        a_right = 2.0 / (h_right * (h_left + h_right))
        a_center = a_left + a_right

        ni = n - 2
        A = sparse.diags(
            [
                -a_left[1:],
                a_center,
                -a_right[:-1],
            ],
            [-1, 0, 1],
            shape=(ni, ni),
            format="csc",
        )

        coords = interior.reshape(-1, 1).astype(np.float32)
        f = np.asarray(operator.source_term(coords), dtype=np.float64).flatten()

        bc_l = float(
            np.asarray(operator.boundary_value(np.array([[x[0]]], dtype=np.float32))).flat[0]
        )
        bc_r = float(
            np.asarray(operator.boundary_value(np.array([[x[-1]]], dtype=np.float32))).flat[0]
        )

        rhs = f.copy()
        rhs[0] += a_left[0] * bc_l
        rhs[-1] += a_right[-1] * bc_r

        u_inner = spsolve(A, rhs)
        u_full = np.concatenate([[bc_l], u_inner, [bc_r]])
        return u_full.astype(np.float64), x

    def _compute_indicators(
        self,
        x: NDArray[np.float64],
        u: NDArray[np.float64],
        operator: PDEOperator,
    ) -> NDArray[np.float64]:
        """Compute residual-based error indicators per element."""
        n_elem = len(x) - 1
        indicators = np.zeros(n_elem, dtype=np.float64)

        for i in range(n_elem):
            h = x[i + 1] - x[i]
            mid = np.array([[(x[i] + x[i + 1]) / 2]], dtype=np.float32)
            f_mid = float(np.asarray(operator.source_term(mid)).flat[0])

            # Approximate second derivative at midpoint
            if 0 < i < n_elem - 1:
                u_xx = (u[i] - 2 * u[i + 1] + u[i + 2]) / (h**2) if h > 0 else 0.0
            else:
                u_xx = 0.0

            residual = abs(-u_xx - f_mid) if not np.isnan(u_xx) else 0.0
            indicators[i] = h * residual

        return indicators

    def _dorfler_mark(self, indicators: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Mark elements using Dorfler bulk-chasing strategy."""
        total = np.sum(indicators**2)
        threshold = self.marking_fraction * total

        sorted_idx = np.argsort(indicators)[::-1]
        cumsum = np.cumsum(indicators[sorted_idx] ** 2)

        n_mark = int(np.searchsorted(cumsum, threshold)) + 1
        marked = np.zeros(len(indicators), dtype=bool)
        marked[sorted_idx[:n_mark]] = True
        return marked

    @staticmethod
    def _refine_grid(
        x: NDArray[np.float64],
        marked: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Refine marked elements by bisection."""
        new_points = []
        for i, is_marked in enumerate(marked):
            if is_marked:
                new_points.append((x[i] + x[i + 1]) / 2)
        if new_points:
            x = np.sort(np.concatenate([x, new_points]))
        return x

    # ------------------------------------------------------------------
    # 2D AMR helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_on_grid_2d(
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        operator: PDEOperator,
        sparse: Any,
        spsolve: Any,
        inside: InsidePredicate | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve 2D Poisson-type PDE on a (possibly non-uniform) tensor-product grid.

        Args:
            xs: Grid node x-coordinates (monotone, includes both endpoints).
            ys: Grid node y-coordinates (monotone, includes both endpoints).
            operator: PDE operator supplying ``source_term`` and ``boundary_value``.
            sparse: The ``scipy.sparse`` module.
            spsolve: The ``scipy.sparse.linalg.spsolve`` callable.
            inside: Optional geometry predicate for non-rectangular (e.g.
                L-shaped) domains. When supplied, interior grid nodes whose
                coordinates fall *outside* the physical domain are pinned to
                their Dirichlet boundary value (identity row), which imposes
                the reentrant-edge boundary condition one grid layer in. When
                ``None`` (default) the full bounding box is solved — the
                historical behaviour, unchanged byte-for-byte.

        """
        nx = len(xs) - 2  # interior x points
        ny = len(ys) - 2  # interior y points
        if nx < 1 or ny < 1:
            # Grid too coarse — return zeros
            XX, YY = np.meshgrid(xs, ys, indexing="ij")
            grid = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float64)
            return np.zeros(len(grid), dtype=np.float64), grid

        hx = np.diff(xs)
        hy = np.diff(ys)

        xi = xs[1:-1]
        yi = ys[1:-1]
        XX, YY = np.meshgrid(xi, yi, indexing="ij")
        coords_flat = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)

        # Per-interior-node domain membership (i-major, idx = i*ny + j to match
        # the assembly loop below). All-True when no mask is supplied.
        if inside is not None:
            node_inside = np.asarray(inside(coords_flat.astype(np.float64)), dtype=bool).reshape(-1)
        else:
            node_inside = np.ones(nx * ny, dtype=bool)

        n_interior = nx * ny

        # Build sparse matrix with variable spacing
        # For each interior point (i,j), stencil:
        #   -u(i-1,j)/hx_l*hx_avg - u(i+1,j)/hx_r*hx_avg
        #   -u(i,j-1)/hy_l*hy_avg - u(i,j+1)/hy_r*hy_avg
        #   + u(i,j) * (1/hx_l + 1/hx_r)/hx_avg + (1/hy_l + 1/hy_r)/hy_avg)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        rhs = np.zeros(n_interior, dtype=np.float64)

        for i in range(nx):
            hx_l = hx[i]
            hx_r = hx[i + 1]
            hx_avg = (hx_l + hx_r) / 2.0
            for j in range(ny):
                hy_l = hy[j]
                hy_r = hy[j + 1]
                hy_avg = (hy_l + hy_r) / 2.0

                idx = i * ny + j

                # Masked (out-of-domain) node: pin to its Dirichlet value via an
                # identity row. Neighbouring in-domain equations still reference
                # this column, so the pinned value acts as the interior boundary
                # condition on the reentrant edge. The right-hand side entry is
                # set *after* the ``rhs += f`` step below so the source term does
                # not pollute it. No-op when ``inside is None``.
                if not node_inside[idx]:
                    rows.append(idx)
                    cols.append(idx)
                    vals.append(1.0)
                    continue

                cx = (1.0 / hx_l + 1.0 / hx_r) / hx_avg
                cy = (1.0 / hy_l + 1.0 / hy_r) / hy_avg
                rows.append(idx)
                cols.append(idx)
                vals.append(cx + cy)

                # x-neighbors
                if i > 0:
                    rows.append(idx)
                    cols.append((i - 1) * ny + j)
                    vals.append(-1.0 / (hx_l * hx_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xs[0], yi[j]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hx_l * hx_avg)

                if i < nx - 1:
                    rows.append(idx)
                    cols.append((i + 1) * ny + j)
                    vals.append(-1.0 / (hx_r * hx_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xs[-1], yi[j]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hx_r * hx_avg)

                # y-neighbors
                if j > 0:
                    rows.append(idx)
                    cols.append(i * ny + (j - 1))
                    vals.append(-1.0 / (hy_l * hy_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xi[i], ys[0]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hy_l * hy_avg)

                if j < ny - 1:
                    rows.append(idx)
                    cols.append(i * ny + (j + 1))
                    vals.append(-1.0 / (hy_r * hy_avg))
                else:
                    bc = float(
                        np.asarray(
                            operator.boundary_value(np.array([[xi[i], ys[-1]]], dtype=np.float32))
                        ).flat[0]
                    )
                    rhs[idx] += bc / (hy_r * hy_avg)

        A = sparse.csc_matrix(
            (vals, (rows, cols)),
            shape=(n_interior, n_interior),
        )

        f = np.asarray(operator.source_term(coords_flat), dtype=np.float64).flatten()
        rhs += f

        # Overwrite masked-node right-hand sides with their pinned Dirichlet
        # value *after* the source term is applied, so neither ``f`` nor any
        # boundary contribution above pollutes the identity rows. Vectorised —
        # one ``boundary_value`` call for all masked nodes. No-op when unmasked.
        if not node_inside.all():
            masked = ~node_inside
            masked_coords = coords_flat[masked].astype(np.float32)
            masked_bvals = np.asarray(
                operator.boundary_value(masked_coords), dtype=np.float64
            ).reshape(-1)
            rhs[masked] = masked_bvals

        u_inner = spsolve(A, rhs)

        # Build full grid including boundary
        XX_full, YY_full = np.meshgrid(xs, ys, indexing="ij")
        grid_full = np.stack([XX_full.ravel(), YY_full.ravel()], axis=-1).astype(np.float64)

        u_full = np.zeros((len(xs), len(ys)), dtype=np.float64)
        u_full[1:-1, 1:-1] = u_inner.reshape(nx, ny)

        # Fill boundaries
        for i in range(len(xs)):
            for y_val in [ys[0], ys[-1]]:
                bc = float(
                    np.asarray(
                        operator.boundary_value(np.array([[xs[i], y_val]], dtype=np.float32))
                    ).flat[0]
                )
                j_idx = 0 if y_val == ys[0] else len(ys) - 1
                u_full[i, j_idx] = bc
        for j in range(len(ys)):
            for x_val in [xs[0], xs[-1]]:
                bc = float(
                    np.asarray(
                        operator.boundary_value(np.array([[x_val, ys[j]]], dtype=np.float32))
                    ).flat[0]
                )
                i_idx = 0 if x_val == xs[0] else len(xs) - 1
                u_full[i_idx, j] = bc

        return u_full.ravel().astype(np.float64), grid_full

    @staticmethod
    def _compute_indicators_2d(
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        u: NDArray[np.float64],
        operator: PDEOperator,
        inside: InsidePredicate | None = None,
    ) -> NDArray[np.float64]:
        """Compute element-wise residual error indicators on 2D grid.

        Returns a 2D array of shape (n_elem_x, n_elem_y) with residual indicators.

        Args:
            xs: Grid node x-coordinates.
            ys: Grid node y-coordinates.
            u: Flattened solution over the full ``(len(xs), len(ys))`` grid.
            operator: PDE operator supplying ``source_term``.
            inside: Optional geometry predicate. When supplied, elements whose
                centre falls outside the physical domain get a zero indicator so
                they are never marked for refinement. ``None`` (default) treats
                the full bounding box as the domain — historical behaviour.

        """
        nx = len(xs) - 1
        ny = len(ys) - 1
        u_grid = u.reshape(len(xs), len(ys))
        indicators = np.zeros((nx, ny), dtype=np.float64)

        # Evaluate the domain predicate for every element centre in a single
        # vectorised call (mirrors how _solve_on_grid_2d tests its nodes) instead
        # of once per element. The centres are built through the identical
        # float32 -> float64 round-trip used by the per-element ``mid`` below so
        # the predicate input is byte-for-byte the same.
        elem_inside: NDArray[np.bool_] | None = None
        if inside is not None:
            cx_all = 0.5 * (xs[:-1] + xs[1:])
            cy_all = 0.5 * (ys[:-1] + ys[1:])
            cx_grid, cy_grid = np.meshgrid(cx_all, cy_all, indexing="ij")
            centres = (
                np.stack([cx_grid.ravel(), cy_grid.ravel()], axis=-1)
                .astype(np.float32)
                .astype(np.float64)
            )
            elem_inside = np.asarray(inside(centres), dtype=bool).reshape(nx, ny)

        for i in range(nx):
            hx = xs[i + 1] - xs[i]
            for j in range(ny):
                hy = ys[j + 1] - ys[j]
                # Element wholly outside the physical domain: never refine it.
                if elem_inside is not None and not elem_inside[i, j]:
                    continue
                mid = np.array(
                    [[(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2]],
                    dtype=np.float32,
                )
                f_mid = float(np.asarray(operator.source_term(mid)).flat[0])

                # Approximate Laplacian at element center using surrounding values
                ci = min(i + 1, len(xs) - 2)
                cj = min(j + 1, len(ys) - 2)
                ci = max(ci, 1)
                cj = max(cj, 1)

                u_xx = 0.0
                if 0 < ci < len(xs) - 1 and hx > 0:
                    u_xx = (u_grid[ci - 1, cj] - 2 * u_grid[ci, cj] + u_grid[ci + 1, cj]) / (hx**2)
                u_yy = 0.0
                if 0 < cj < len(ys) - 1 and hy > 0:
                    u_yy = (u_grid[ci, cj - 1] - 2 * u_grid[ci, cj] + u_grid[ci, cj + 1]) / (hy**2)

                laplacian = u_xx + u_yy
                has_nan = np.isnan(laplacian) or np.isnan(f_mid)
                residual = 0.0 if has_nan else abs(-laplacian - f_mid)
                h_elem = np.sqrt(hx * hy)
                indicators[i, j] = h_elem * residual

        return indicators

    def _dorfler_mark_2d(
        self,
        indicators: NDArray[np.float64],
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Dorfler marking on 2D grid. Returns marked arrays for x and y element edges."""
        flat = indicators.ravel()
        total = np.sum(flat**2)
        threshold = self.marking_fraction * total

        sorted_idx = np.argsort(flat)[::-1]
        cumsum = np.cumsum(flat[sorted_idx] ** 2)
        n_mark = int(np.searchsorted(cumsum, threshold)) + 1

        ny_elem = indicators.shape[1]
        marked_x = np.zeros(len(xs) - 1, dtype=bool)
        marked_y = np.zeros(len(ys) - 1, dtype=bool)

        for flat_idx in sorted_idx[:n_mark]:
            i = flat_idx // ny_elem
            j = flat_idx % ny_elem
            marked_x[i] = True
            marked_y[j] = True

        return marked_x, marked_y


class SimplePINNSolver(BaseSolver):
    """Simple Physics-Informed Neural Network baseline.

    Trains a small MLP with physics-informed loss (PDE residual + BC)
    for comparison with classical and AlphaGalerkin approaches.
    """

    name = "pinn"
    description = "Physics-Informed Neural Network baseline"

    def __init__(
        self,
        hidden_dim: int | None = None,
        n_layers: int | None = None,
        n_epochs: int | None = None,
        learning_rate: float | None = None,
        n_collocation: int | None = None,
        bc_loss_weight: float | None = None,
        device: str | None = None,
        vector_pde: bool | None = None,
        config: PINNConfig | None = None,
    ) -> None:
        self.config = config or PINNConfig()
        c = self.config
        self.hidden_dim = hidden_dim if hidden_dim is not None else c.hidden_dim
        self.n_layers = n_layers if n_layers is not None else c.n_layers
        self.n_epochs = n_epochs if n_epochs is not None else c.n_epochs
        self.learning_rate = learning_rate if learning_rate is not None else c.learning_rate
        self.n_collocation = n_collocation if n_collocation is not None else c.n_collocation
        self.bc_loss_weight = bc_loss_weight if bc_loss_weight is not None else c.bc_loss_weight
        self.device_preference = device if device is not None else c.device
        self.vector_pde_override = vector_pde if vector_pde is not None else c.vector_pde

    def _resolve_vector_pde(self, operator: PDEOperator) -> bool:
        """Decide whether to build a vector-valued network for this operator."""
        if self.vector_pde_override is not None:
            return self.vector_pde_override
        return getattr(operator, "pde_type", None) == PDEType.NAVIER_STOKES

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve by training a PINN."""
        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("pinn_solve_start")
        t0 = time.perf_counter()

        device = resolve_device(self.device_preference, context=f"pinn[{self.name}]")
        is_vector = self._resolve_vector_pde(operator)
        output_dim = 2 if is_vector else 1
        net = self._build_network(operator.dim, output_dim=output_dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        rng = np.random.default_rng(self.config.seed)

        # Profile GPU utilisation when running on CUDA — provides SBIR
        # proposal-grade telemetry on whether the workload is compute-bound
        # or memory-bandwidth-bound. No-ops cleanly on CPU or no-nvidia-smi
        # hosts.
        # When ``device.index is None`` (bare ``torch.device("cuda")``) the
        # tensor goes to whatever ``torch.cuda.current_device()`` returns,
        # which is **not** always 0 — third-party libraries or earlier
        # ``torch.cuda.set_device(N)`` calls can shift it. Sample dmon on
        # the same index so the report matches the actual workload.
        if device.type == "cuda":
            gpu_idx = device.index if device.index is not None else torch.cuda.current_device()
            profile_indices: list[int] = [gpu_idx]
        else:
            profile_indices = []
        with GpuUtilizationProfiler(gpu_indices=profile_indices) as profiler:
            for epoch in range(self.n_epochs):
                optimizer.zero_grad()

                # Interior collocation points
                coords_np = rng.uniform(
                    operator.domain_min,
                    operator.domain_max,
                    size=(self.n_collocation, operator.dim),
                ).astype(np.float32)
                coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
                coords.requires_grad_(True)

                u_raw = net(coords)  # (N, output_dim)

                if is_vector:
                    # Vector PDE (NS): PDE residual = sum of per-component
                    # Laplacians. Source-term forcing is treated as zero for
                    # the momentum residual (Taylor-Green has no body force).
                    #
                    # NOTE — physics simplification. Full Navier-Stokes
                    # momentum is ``du/dt + (u·∇)u + ∇p − ν∇²u = f``; the
                    # advection, pressure-gradient, and continuity terms are
                    # not in this loss. This matches the previous P40 fork's
                    # behaviour and is sufficient for the Taylor-Green decay
                    # benchmark (where the analytical IC dominates), but the
                    # PINN row is *not* a fully-physics-informed solver.
                    # Delegating to ``operator.residual()`` for proper NS
                    # physics is tracked as future work; doing it correctly
                    # requires (a) extending the PINN to predict pressure
                    # alongside velocity and (b) wiring divergence-free
                    # constraints — out of scope for this PR.
                    loss_pde = torch.zeros((), device=device)
                    for c_idx in range(output_dim):
                        uc = u_raw[:, c_idx]
                        lap_c = self._compute_laplacian(uc, coords, operator.dim)
                        loss_pde = loss_pde + torch.mean(lap_c**2)
                else:
                    u = u_raw.squeeze(-1)
                    lap = self._compute_laplacian(u, coords, operator.dim)
                    f = operator.source_term(coords)
                    if isinstance(f, np.ndarray):
                        f = torch.tensor(f, dtype=torch.float32, device=device)
                    elif isinstance(f, torch.Tensor) and f.device != device:
                        f = f.to(device)
                    pde_residual = lap + f  # For Poisson: -lap = f => lap+f=0
                    loss_pde = torch.mean(pde_residual**2)

                # Boundary loss
                bc_coords_np = operator.generate_boundary_points(
                    self.config.n_boundary_points,
                    seed=None,
                )
                bc_coords = torch.tensor(bc_coords_np, dtype=torch.float32, device=device)
                u_bc_raw = net(bc_coords)  # (N, output_dim)
                bc_vals = operator.boundary_value(bc_coords_np)
                if isinstance(bc_vals, np.ndarray):
                    bc_vals = torch.tensor(bc_vals, dtype=torch.float32, device=device)
                elif isinstance(bc_vals, torch.Tensor) and bc_vals.device != device:
                    bc_vals = bc_vals.to(device)

                if is_vector:
                    if bc_vals.dim() == 1:
                        bc_vals = bc_vals.unsqueeze(-1).expand_as(u_bc_raw)
                    loss_bc = torch.mean((u_bc_raw - bc_vals) ** 2)
                else:
                    u_bc = u_bc_raw.squeeze(-1)
                    if bc_vals.dim() > 1:
                        bc_vals = bc_vals.squeeze(-1)
                    loss_bc = torch.mean((u_bc - bc_vals) ** 2)

                loss = loss_pde + self.bc_loss_weight * loss_bc
                loss.backward()
                optimizer.step()

                if epoch % self.config.log_interval == 0:
                    log.debug(
                        "pinn_epoch",
                        epoch=epoch,
                        loss_pde=float(loss_pde),
                        loss_bc=float(loss_bc),
                    )

        # Evaluate on uniform grid
        eval_coords_np = operator.generate_collocation_points(
            n_dof, method="uniform", seed=self.config.seed
        )
        eval_coords = torch.tensor(eval_coords_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            u_eval_raw = net(eval_coords).cpu().numpy()
        u_eval = u_eval_raw if is_vector else u_eval_raw.squeeze(-1)

        wall_time = time.perf_counter() - t0
        grid = eval_coords_np.astype(np.float64)
        l2_err = self._compute_l2_error(u_eval.astype(np.float64), grid, operator)

        gpu_profile = profiler.report.to_dict() if profiler.report is not None else None
        log.info(
            "pinn_solve_done",
            wall_time=wall_time,
            l2_error=l2_err,
            device=str(device),
            gpu_samples=(profiler.report.total_samples if profiler.report else 0),
        )
        return SolverResult(
            solution=u_eval.astype(np.float64),
            grid_points=grid,
            n_dof=len(u_eval),
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={
                "hidden_dim": self.hidden_dim,
                "n_layers": self.n_layers,
                "n_epochs": self.n_epochs,
                "n_collocation": self.n_collocation,
                "device": str(device),
                "vector_pde": is_vector,
                "gpu_profile": gpu_profile,
            },
        )

    def _build_network(self, input_dim: int, output_dim: int = 1) -> torch.nn.Module:
        """Build a simple MLP for the PINN.

        Args:
            input_dim: Input coordinate dimension (e.g. 1, 2, 3).
            output_dim: Output dimension (1 for scalar PDEs, 2 for 2D vector
                PDEs like Navier-Stokes velocity).

        """
        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(self.n_layers):
            layers.append(torch.nn.Linear(in_dim, self.hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(torch.nn.Linear(in_dim, output_dim))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _compute_laplacian(u: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute Laplacian of u w.r.t. coords using autograd."""
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(u, coords, grad_outputs=grad_outputs, create_graph=True)[0]

        laplacian = torch.zeros_like(u)
        for d in range(dim):
            grad_d = grad_u[:, d]
            grad2 = torch.autograd.grad(
                grad_d,
                coords,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
            )[0]
            laplacian = laplacian + grad2[:, d]

        return laplacian


class NavierStokesFDMSolver(BaseSolver):
    """Projection method FDM solver for 2D incompressible Navier-Stokes.

    Uses the Chorin projection method (fractional step):
    1. Advection-diffusion step (explicit for advection, implicit for diffusion)
    2. Pressure Poisson solve for pressure correction
    3. Velocity projection to enforce divergence-free constraint

    Supports Taylor-Green vortex benchmark with exact analytical solution.
    """

    name = "navier_stokes_fdm"
    description = "Projection method FDM for 2D incompressible Navier-Stokes"

    def __init__(
        self,
        dt: float | None = None,
        t_final: float | None = None,
        config: NavierStokesConfig | None = None,
    ) -> None:
        self.config = config or NavierStokesConfig()
        self.dt = dt if dt is not None else self.config.dt
        self.t_final = t_final if t_final is not None else self.config.t_final

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve 2D NS using Chorin projection method."""
        try:
            from scipy import sparse
            from scipy.sparse.linalg import spsolve
        except ImportError as exc:
            raise ImportError("NavierStokesFDMSolver requires scipy.") from exc

        if operator.dim != 2:
            raise NotImplementedError(
                f"NavierStokesFDMSolver requires dim=2, got dim={operator.dim}"
            )

        log = logger.bind(solver=self.name, n_dof=n_dof)
        log.info("ns_fdm_solve_start")
        t0 = time.perf_counter()

        # Extract viscosity from operator
        viscosity = getattr(operator, "viscosity", self.config.default_viscosity)

        # Grid setup
        n = max(int(np.sqrt(n_dof / 2)), self.config.min_grid_points)
        x_min, x_max = float(operator.domain_min[0]), float(operator.domain_max[0])
        y_min, y_max = float(operator.domain_min[1]), float(operator.domain_max[1])
        xs = np.linspace(x_min, x_max, n, dtype=np.float64)
        ys = np.linspace(y_min, y_max, n, dtype=np.float64)
        h = xs[1] - xs[0]

        # Initialize velocity with exact initial condition if available
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        ux = np.zeros((n, n), dtype=np.float64)
        uy = np.zeros((n, n), dtype=np.float64)

        if hasattr(operator, "initial_condition"):
            coords_init = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
            ic = operator.initial_condition(coords_init)
            if isinstance(ic, torch.Tensor):
                ic = ic.detach().cpu().numpy()
            ic = np.asarray(ic, dtype=np.float64)
            if ic.ndim == 2 and ic.shape[-1] >= 2:
                ux = np.asarray(ic[:, 0].reshape(n, n), dtype=np.float64)  # type: ignore[assignment]
                uy = np.asarray(ic[:, 1].reshape(n, n), dtype=np.float64)  # type: ignore[assignment]

        # Time stepping via Chorin projection
        cfl_dt = self.config.cfl_safety * h**2 / max(viscosity, self.config.viscosity_floor)
        dt = min(self.dt, cfl_dt)
        n_steps = int(self.t_final / dt)

        # Build Laplacian for pressure Poisson solve (interior only)
        ni = n - 2
        if ni < 1:
            # Too coarse
            grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
            return SolverResult(
                solution=np.zeros(2 * n * n, dtype=np.float64),
                grid_points=grid,
                n_dof=2 * n * n,
                wall_time_seconds=time.perf_counter() - t0,
                l2_error=None,
            )

        I_n = sparse.eye(ni, format="csc")
        T = sparse.diags(
            [np.full(ni - 1, 1.0), np.full(ni, -4.0), np.full(ni - 1, 1.0)],
            [-1, 0, 1],
            format="csc",
        )
        L = sparse.kron(I_n, T) + sparse.kron(
            sparse.diags([np.full(ni - 1, 1.0), np.full(ni - 1, 1.0)], [-1, 1], format="csc"),
            I_n,
        )
        L = L / (h**2)

        for step_idx in range(n_steps):
            # 1. Advection-diffusion (explicit Euler for simplicity)
            ux_star = ux.copy()
            uy_star = uy.copy()

            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    # Advection (central differences)
                    dux_dx = (ux[i + 1, j] - ux[i - 1, j]) / (2.0 * h)
                    dux_dy = (ux[i, j + 1] - ux[i, j - 1]) / (2.0 * h)
                    duy_dx = (uy[i + 1, j] - uy[i - 1, j]) / (2.0 * h)
                    duy_dy = (uy[i, j + 1] - uy[i, j - 1]) / (2.0 * h)

                    advection_x = ux[i, j] * dux_dx + uy[i, j] * dux_dy
                    advection_y = ux[i, j] * duy_dx + uy[i, j] * duy_dy

                    # Diffusion (5-point Laplacian)
                    lap_ux = (
                        ux[i + 1, j] + ux[i - 1, j] + ux[i, j + 1] + ux[i, j - 1] - 4.0 * ux[i, j]
                    ) / (h**2)
                    lap_uy = (
                        uy[i + 1, j] + uy[i - 1, j] + uy[i, j + 1] + uy[i, j - 1] - 4.0 * uy[i, j]
                    ) / (h**2)

                    ux_star[i, j] = ux[i, j] + dt * (-advection_x + viscosity * lap_ux)
                    uy_star[i, j] = uy[i, j] + dt * (-advection_y + viscosity * lap_uy)

            # 2. Pressure Poisson solve: nabla^2 p = (1/dt) * div(u*)
            div = np.zeros((ni, ni), dtype=np.float64)
            for i in range(ni):
                for j in range(ni):
                    gi, gj = i + 1, j + 1
                    div[i, j] = (ux_star[gi + 1, gj] - ux_star[gi - 1, gj]) / (2.0 * h) + (
                        uy_star[gi, gj + 1] - uy_star[gi, gj - 1]
                    ) / (2.0 * h)

            rhs_p = div.ravel() / dt
            p_inner = spsolve(L, rhs_p)
            p = np.zeros((n, n), dtype=np.float64)
            p[1:-1, 1:-1] = p_inner.reshape(ni, ni)

            # 3. Projection step: u^{n+1} = u* - dt * grad(p)
            for i in range(1, n - 1):
                for j in range(1, n - 1):
                    dp_dx = (p[i + 1, j] - p[i - 1, j]) / (2.0 * h)
                    dp_dy = (p[i, j + 1] - p[i, j - 1]) / (2.0 * h)
                    ux[i, j] = ux_star[i, j] - dt * dp_dx
                    uy[i, j] = uy_star[i, j] - dt * dp_dy

            if step_idx % max(n_steps // self.config.log_fraction, 1) == 0:
                log.debug("ns_step", step=step_idx, max_ux=float(np.max(np.abs(ux))))

        wall_time = time.perf_counter() - t0

        # Build output
        grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
        solution = np.concatenate([ux.ravel(), uy.ravel()])

        # Compute L2 error against exact solution at t_final
        l2_err = self._compute_ns_l2_error(ux, uy, xx, yy, operator, self.t_final)

        log.info("ns_fdm_solve_done", wall_time=wall_time, l2_error=l2_err)
        return SolverResult(
            solution=solution,
            grid_points=grid,
            n_dof=2 * n * n,
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={
                "method": "chorin_projection",
                "dt": dt,
                "n_steps": n_steps,
                "grid_size": n,
            },
        )

    @staticmethod
    def _compute_ns_l2_error(
        ux: NDArray[np.float64],
        uy: NDArray[np.float64],
        xx: NDArray[np.float64],
        yy: NDArray[np.float64],
        operator: PDEOperator,
        t_final: float,
    ) -> float | None:
        """Compute L2 error of velocity against exact solution."""
        if not hasattr(operator, "exact_solution"):
            return None
        coords = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        exact = operator.exact_solution(coords, time=t_final)
        if exact is None:
            return None
        if isinstance(exact, torch.Tensor):
            exact = exact.detach().cpu().numpy()
        exact = np.asarray(exact, dtype=np.float64)
        if exact.ndim == 2 and exact.shape[-1] >= 2:
            exact_ux = exact[:, 0].reshape(xx.shape)
            exact_uy = exact[:, 1].reshape(xx.shape)
        else:
            return None
        err_ux = ux - exact_ux
        err_uy = uy - exact_uy
        n_pts = ux.size
        return float(np.sqrt((np.sum(err_ux**2) + np.sum(err_uy**2)) / (2 * n_pts)))


# ---------------------------------------------------------------------------
# Registry of available solvers
# ---------------------------------------------------------------------------
SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {
    "uniform_fdm": UniformFDMSolver,
    "dorfler_amr": DorflerAMRSolver,
    "pinn": SimplePINNSolver,
    "navier_stokes_fdm": NavierStokesFDMSolver,
}


def get_solver(name: str, **kwargs: Any) -> BaseSolver:
    """Get a solver instance by name.

    Args:
        name: Registered solver name.
        **kwargs: Solver-specific constructor arguments.

    Returns:
        Instantiated solver.

    Raises:
        KeyError: If solver name is not registered.

    """
    cls = SOLVER_REGISTRY.get(name)
    if cls is None:
        available = sorted(SOLVER_REGISTRY.keys())
        raise KeyError(f"Unknown solver '{name}'. Available: {available}")
    return cls(**kwargs)


def list_solvers() -> list[str]:
    """List all registered solver names."""
    return sorted(SOLVER_REGISTRY.keys())
