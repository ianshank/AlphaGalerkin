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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.pde.operators import PDEOperator

logger = structlog.get_logger(__name__)


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

    def __init__(self, config: SolverConfig | None = None) -> None:
        self.config = config or SolverConfig()

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
            raise NotImplementedError(
                f"UniformFDMSolver does not support dim={operator.dim}"
            )

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
    @staticmethod
    def _solve_1d(
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = max(n_dof, 3)
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
        bc_left = operator.boundary_value(
            np.array([[operator.domain_min[0]]], dtype=np.float32)
        )
        bc_right = operator.boundary_value(
            np.array([[operator.domain_max[0]]], dtype=np.float32)
        )
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
    @staticmethod
    def _solve_2d(
        operator: PDEOperator,
        n_dof: int,
        sparse: Any,
        spsolve: Any,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = max(int(np.sqrt(n_dof)), 3)
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

        N = n * n  # interior unknowns

        # 5-point Laplacian stencil
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
        marking_fraction: float = 0.3,
        max_refinements: int = 10,
        config: SolverConfig | None = None,
    ) -> None:
        if not 0.0 < marking_fraction < 1.0:
            raise ValueError(f"marking_fraction must be in (0,1), got {marking_fraction}")
        self.marking_fraction = marking_fraction
        self.max_refinements = max_refinements
        self.config = config or SolverConfig()

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

        if operator.dim != 1:
            raise NotImplementedError("DorflerAMRSolver currently supports dim=1 only")

        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("amr_solve_start")
        t0 = time.perf_counter()

        # Start with a coarse grid
        n_start = max(min(n_dof // 4, 8), 4)
        x = np.linspace(
            float(operator.domain_min[0]),
            float(operator.domain_max[0]),
            n_start,
            dtype=np.float64,
        )

        for step in range(self.max_refinements):
            # Solve on current grid
            u, _ = self._solve_on_grid(x, operator, sparse, spsolve)

            if len(x) >= n_dof:
                break

            # Compute element-wise error indicators (residual jump)
            indicators = self._compute_indicators(x, u, operator)

            # Dorfler marking: mark smallest set with fraction of total indicator
            marked = self._dorfler_mark(indicators)

            # Refine marked elements (bisection)
            x = self._refine_grid(x, marked)

            log.debug(
                "amr_step",
                step=step,
                n_points=len(x),
                max_indicator=float(np.max(indicators)),
            )

        # Final solve
        u, _ = self._solve_on_grid(x, operator, sparse, spsolve)
        wall_time = time.perf_counter() - t0

        grid = x.reshape(-1, 1).astype(np.float64)
        l2_err = self._compute_l2_error(u, grid, operator)

        log.info("amr_solve_done", wall_time=wall_time, l2_error=l2_err, final_dof=len(u))
        return SolverResult(
            solution=u,
            grid_points=grid,
            n_dof=len(u),
            wall_time_seconds=wall_time,
            l2_error=l2_err,
            metadata={
                "marking_fraction": self.marking_fraction,
                "n_refinements": step + 1 if step is not None else 0,  # type: ignore[possibly-undefined]
            },
        )

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
        h_left = np.diff(x[:-1])   # h_{i-1}
        h_right = np.diff(x[1:])   # h_i

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
            np.asarray(
                operator.boundary_value(np.array([[x[0]]], dtype=np.float32))
            ).flat[0]
        )
        bc_r = float(
            np.asarray(
                operator.boundary_value(np.array([[x[-1]]], dtype=np.float32))
            ).flat[0]
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


class SimplePINNSolver(BaseSolver):
    """Simple Physics-Informed Neural Network baseline.

    Trains a small MLP with physics-informed loss (PDE residual + BC)
    for comparison with classical and AlphaGalerkin approaches.
    """

    name = "pinn"
    description = "Physics-Informed Neural Network baseline"

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_epochs: int = 2000,
        learning_rate: float = 1e-3,
        n_collocation: int = 1000,
        config: SolverConfig | None = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.n_collocation = n_collocation
        self.config = config or SolverConfig()

    def solve(self, operator: PDEOperator, n_dof: int, **kwargs: Any) -> SolverResult:
        """Solve by training a PINN."""
        log = logger.bind(solver=self.name, n_dof=n_dof, dim=operator.dim)
        log.info("pinn_solve_start")
        t0 = time.perf_counter()

        device = torch.device("cpu")
        net = self._build_network(operator.dim).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)

        rng = np.random.default_rng(self.config.seed)

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

            u = net(coords).squeeze(-1)

            # PDE residual loss (Laplacian)
            lap = self._compute_laplacian(u, coords, operator.dim)
            f = operator.source_term(coords)
            if isinstance(f, np.ndarray):
                f = torch.tensor(f, dtype=torch.float32, device=device)
            pde_residual = lap + f  # For Poisson: -lap = f => lap + f = 0
            loss_pde = torch.mean(pde_residual**2)

            # Boundary loss
            bc_coords_np = operator.generate_boundary_points(50, seed=None)
            bc_coords = torch.tensor(bc_coords_np, dtype=torch.float32, device=device)
            u_bc = net(bc_coords).squeeze(-1)
            bc_vals = operator.boundary_value(bc_coords_np)
            if isinstance(bc_vals, np.ndarray):
                bc_vals = torch.tensor(bc_vals, dtype=torch.float32, device=device)
            loss_bc = torch.mean((u_bc - bc_vals) ** 2)

            loss = loss_pde + 10.0 * loss_bc
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
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
            u_eval = net(eval_coords).squeeze(-1).cpu().numpy()

        wall_time = time.perf_counter() - t0
        grid = eval_coords_np.astype(np.float64)
        l2_err = self._compute_l2_error(u_eval.astype(np.float64), grid, operator)

        log.info("pinn_solve_done", wall_time=wall_time, l2_error=l2_err)
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
            },
        )

    def _build_network(self, input_dim: int) -> torch.nn.Module:
        """Build a simple MLP for the PINN."""
        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(self.n_layers):
            layers.append(torch.nn.Linear(in_dim, self.hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = self.hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _compute_laplacian(u: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute Laplacian of u w.r.t. coords using autograd."""
        grad_outputs = torch.ones_like(u)
        grad_u = torch.autograd.grad(
            u, coords, grad_outputs=grad_outputs, create_graph=True
        )[0]

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


# ---------------------------------------------------------------------------
# Registry of available solvers
# ---------------------------------------------------------------------------
SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {
    "uniform_fdm": UniformFDMSolver,
    "dorfler_amr": DorflerAMRSolver,
    "pinn": SimplePINNSolver,
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
