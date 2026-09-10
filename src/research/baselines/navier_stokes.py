"""Chorin-projection finite-difference Navier–Stokes baseline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray

from src.pde.operators import PDEOperator
from src.research.baselines.base import BaseSolver
from src.research.baselines.schemas import NavierStokesConfig, SolverResult

logger = structlog.get_logger(__name__)


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
                ux = np.asarray(ic[:, 0].reshape(n, n), dtype=np.float64)
                uy = np.asarray(ic[:, 1].reshape(n, n), dtype=np.float64)

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
