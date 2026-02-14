"""Advection-diffusion equation physics module on [0,1]^2 (steady-state)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import structlog

from src.alphagalerkin.core.types import PDEType
from src.alphagalerkin.physics.base import (
    BoundaryCondition,
    ManufacturedSolution,
    SolveResult,
)
from src.alphagalerkin.physics.registry import register_physics

logger = structlog.get_logger("physics.advdiff")

# Default physical parameters
_DEFAULT_DIFFUSIVITY: float = 0.1
_DEFAULT_VELOCITY: tuple[float, float] = (1.0, 0.0)


@register_physics("advdiff_2d")
class AdvectionDiffusionModule:
    """Steady-state advection-diffusion: -D*laplacian(u) + v.grad(u) = f.

    On [0,1]^2 with homogeneous Dirichlet BCs.

    Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    With velocity v = (vx, vy) and diffusivity D:
        -D*laplacian(u) = 2*D*pi^2*sin(pi*x)*sin(pi*y)
        v.grad(u) = vx*pi*cos(pi*x)*sin(pi*y) + vy*pi*sin(pi*x)*cos(pi*y)
        f = 2*D*pi^2*sin(pi*x)*sin(pi*y) + vx*pi*cos(pi*x)*sin(pi*y)
            + vy*pi*sin(pi*x)*cos(pi*y)
    """

    name: str = "advdiff_2d"
    pde_type: PDEType = PDEType.MIXED

    def __init__(
        self,
        diffusivity: float = _DEFAULT_DIFFUSIVITY,
        velocity: tuple[float, float] = _DEFAULT_VELOCITY,
    ) -> None:
        self._diffusivity = diffusivity
        self._velocity = velocity

    @property
    def diffusivity(self) -> float:
        """Diffusion coefficient D."""
        return self._diffusivity

    @property
    def velocity(self) -> tuple[float, float]:
        """Advection velocity (vx, vy)."""
        return self._velocity

    def weak_form(
        self,
        trial: Any,
        test: Any,
        mesh: Any,
    ) -> Any:
        """Weak form: D*integral(grad(u).grad(v)) + integral((v.grad(u))*v_test) = integral(f*v)."""
        logger.debug("physics.advdiff.weak_form")
        return None  # Placeholder for FEM backend integration

    def boundary_conditions(
        self,
        mesh: Any = None,
        config: Any = None,
    ) -> list[BoundaryCondition]:
        """Homogeneous Dirichlet BCs on all boundaries."""
        return [
            BoundaryCondition(
                bc_type="dirichlet", value=0.0, region="all",
            ),
        ]

    def manufactured_solution(
        self,
        config: Any = None,
    ) -> ManufacturedSolution:
        """MMS: u = sin(pi*x)*sin(pi*y)."""
        d = self._diffusivity
        vx, vy = self._velocity

        def exact(points: np.ndarray) -> np.ndarray:
            x, y = points[:, 0], points[:, 1]
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def forcing(points: np.ndarray) -> np.ndarray:
            x, y = points[:, 0], points[:, 1]
            # -D*laplacian(u) term
            diffusion = (
                2.0
                * d
                * np.pi**2
                * np.sin(np.pi * x)
                * np.sin(np.pi * y)
            )
            # v.grad(u) term
            advection_x = (
                vx
                * np.pi
                * np.cos(np.pi * x)
                * np.sin(np.pi * y)
            )
            advection_y = (
                vy
                * np.pi
                * np.sin(np.pi * x)
                * np.cos(np.pi * y)
            )
            return diffusion + advection_x + advection_y

        def boundary(points: np.ndarray) -> np.ndarray:
            return np.zeros(len(points))

        return ManufacturedSolution(
            exact_solution=exact,
            forcing=forcing,
            boundary_data=boundary,
            expected_convergence_order=2.0,
            name="advdiff_sinsin",
        )

    def reward_function(
        self,
        state: Any,
        action: Any,
        next_state: Any,
    ) -> float:
        """Reward based on residual reduction and DOF efficiency."""
        return 0.0  # Placeholder

    def state_features(self, discretization: Any) -> Any:
        """Per-element features for the GNN encoder."""
        return None  # Placeholder

    def action_validators(self) -> list[Any]:
        """Advection-diffusion-specific action validators."""
        return []

    def default_config(self) -> dict[str, Any]:
        """Default parameters for advection-diffusion problems."""
        return {
            "domain": {
                "type": "rectangle",
                "bounds": [[0.0, 1.0], [0.0, 1.0]],
            },
            "diffusivity": self._diffusivity,
            "velocity": list(self._velocity),
        }

    def solve_on_grid(self, n: int) -> SolveResult:
        """Solve advection-diffusion on uniform n x n grid via FD.

        Uses 5-point stencil for diffusion and central differences
        for advection.
        """
        start = time.perf_counter()

        d = self._diffusivity
        vx, vy = self._velocity
        h = 1.0 / (n + 1)
        total_dofs = n * n

        # Build 5-point stencil diffusion operator: -D*laplacian
        main_diag = 4.0 * d * np.ones(total_dofs)
        off_diag_1 = -d * np.ones(total_dofs - 1)
        off_diag_n = -d * np.ones(total_dofs - n)

        # Zero out connections across row boundaries
        for i in range(1, n):
            off_diag_1[i * n - 1] = 0.0

        diffusion_op = sp.diags(
            [
                off_diag_n,
                off_diag_1,
                main_diag,
                off_diag_1,
                off_diag_n,
            ],
            [-n, -1, 0, 1, n],
            shape=(total_dofs, total_dofs),
            format="csc",
        ) / (h * h)

        # Build advection operator: v.grad(u) via central differences
        # du/dx ~ (u_{i+1,j} - u_{i-1,j}) / (2h)
        adv_x_upper = (vx / (2.0 * h)) * np.ones(total_dofs - 1)
        adv_x_lower = -(vx / (2.0 * h)) * np.ones(total_dofs - 1)

        # Zero out connections across row boundaries
        for i in range(1, n):
            adv_x_upper[i * n - 1] = 0.0
            adv_x_lower[i * n - 1] = 0.0

        advection_x_op = sp.diags(
            [adv_x_lower, adv_x_upper],
            [-1, 1],
            shape=(total_dofs, total_dofs),
            format="csc",
        )

        # du/dy ~ (u_{i,j+1} - u_{i,j-1}) / (2h)
        adv_y_upper = (vy / (2.0 * h)) * np.ones(total_dofs - n)
        adv_y_lower = -(vy / (2.0 * h)) * np.ones(total_dofs - n)

        advection_y_op = sp.diags(
            [adv_y_lower, adv_y_upper],
            [-n, n],
            shape=(total_dofs, total_dofs),
            format="csc",
        )

        # Combined operator: -D*laplacian + v.grad
        operator = diffusion_op + advection_x_op + advection_y_op

        # RHS from manufactured solution
        x_grid = np.linspace(h, 1.0 - h, n)
        y_grid = np.linspace(h, 1.0 - h, n)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        points = np.column_stack(
            [grid_x.ravel(), grid_y.ravel()],
        )

        mms = self.manufactured_solution()
        rhs = mms.forcing(points)

        # Solve
        u = scipy.sparse.linalg.spsolve(operator, rhs)

        # Compute residual
        residual = operator @ u - rhs
        residual_norm = float(np.linalg.norm(residual))

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "physics.advdiff.solved",
            grid_size=n,
            diffusivity=d,
            velocity=list(self._velocity),
            residual_norm=residual_norm,
            solve_time_ms=elapsed_ms,
        )

        return SolveResult(
            solution=u,
            residual_norm=residual_norm,
            condition_number=1.0,  # Skip expensive computation
            solve_time_ms=elapsed_ms,
            converged=True,
        )
