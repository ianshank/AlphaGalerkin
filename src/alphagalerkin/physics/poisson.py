"""Poisson equation physics module: -nabla^2 u = f on [0,1]^2."""

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

logger = structlog.get_logger("physics.poisson")


@register_physics("poisson_2d")
class PoissonModule:
    """Elliptic PDE: -nabla^2 u = f with Dirichlet BCs.

    Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    which gives f = 2*pi^2*sin(pi*x)*sin(pi*y).
    """

    name: str = "poisson_2d"
    pde_type: PDEType = PDEType.ELLIPTIC

    def weak_form(
        self,
        trial: Any,
        test: Any,
        mesh: Any,
    ) -> Any:
        """Weak form: integral(grad(u).grad(v)) = integral(f*v)."""
        logger.debug("physics.poisson.weak_form")
        return None  # Placeholder for FEM backend integration

    def boundary_conditions(
        self,
        mesh: Any = None,
        config: Any = None,
    ) -> list[BoundaryCondition]:
        """Homogeneous Dirichlet BCs on all boundaries."""
        return [
            BoundaryCondition(
                bc_type="dirichlet",
                value=0.0,
                region="all",
            ),
        ]

    def manufactured_solution(
        self,
        config: Any = None,
    ) -> ManufacturedSolution:
        """MMS: u = sin(pi*x)*sin(pi*y)."""

        def exact(points: np.ndarray) -> np.ndarray:
            x, y = points[:, 0], points[:, 1]
            return np.sin(np.pi * x) * np.sin(np.pi * y)

        def forcing(points: np.ndarray) -> np.ndarray:
            x, y = points[:, 0], points[:, 1]
            return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

        def boundary(points: np.ndarray) -> np.ndarray:
            return np.zeros(len(points))

        return ManufacturedSolution(
            exact_solution=exact,
            forcing=forcing,
            boundary_data=boundary,
            expected_convergence_order=2.0,
            name="poisson_sinsin",
        )

    def reward_function(
        self,
        state: Any,
        action: Any,
        next_state: Any,
    ) -> float:
        """Reward based on residual reduction and DOF efficiency."""
        return 0.0  # Placeholder

    def state_features(self, _discretization: Any) -> Any:
        """Per-element features for the GNN encoder."""
        return None  # Placeholder

    def action_validators(self) -> list[Any]:
        """Poisson-specific action validators."""
        return []

    def default_config(self) -> dict[str, Any]:
        """Default parameters for Poisson problems."""
        return {
            "domain": {
                "type": "rectangle",
                "bounds": [[0.0, 1.0], [0.0, 1.0]],
            },
            "diffusivity": 1.0,
        }

    def solve_on_grid(self, n: int) -> SolveResult:
        """Solve Poisson on uniform n x n grid via FD.

        Uses 5-point stencil for validation/testing.
        """
        start = time.perf_counter()

        h = 1.0 / (n + 1)
        total_dofs = n * n

        # Build 5-point stencil Laplacian
        main_diag = 4.0 * np.ones(total_dofs)
        off_diag_1 = -np.ones(total_dofs - 1)
        off_diag_n = -np.ones(total_dofs - n)

        # Zero out connections across row boundaries
        for i in range(1, n):
            off_diag_1[i * n - 1] = 0.0

        laplacian = sp.diags(
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
        u = scipy.sparse.linalg.spsolve(laplacian, rhs)

        # Compute residual
        residual = laplacian @ u - rhs
        residual_norm = float(np.linalg.norm(residual))

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "physics.poisson.solved",
            grid_size=n,
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
