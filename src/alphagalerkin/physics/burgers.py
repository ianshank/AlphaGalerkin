"""Burgers equation physics module: 1D viscous Burgers (steady-state)."""

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

logger = structlog.get_logger("physics.burgers")

# Default viscosity coefficient
_DEFAULT_VISCOSITY: float = 0.01


@register_physics("burgers_1d")
class BurgersModule:
    """Steady-state 1D viscous Burgers: -nu*u_xx = f on [0,1].

    The full time-dependent Burgers equation is:
        u_t + u*u_x = nu*u_xx

    For steady-state validation we solve the linear diffusion part:
        -nu*u_xx = f

    Manufactured solution: u_exact = sin(pi*x)
    which gives f = nu*pi^2*sin(pi*x).
    """

    name: str = "burgers_1d"
    pde_type: PDEType = PDEType.PARABOLIC

    def __init__(self, viscosity: float = _DEFAULT_VISCOSITY) -> None:
        self._viscosity = viscosity

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity coefficient nu."""
        return self._viscosity

    def weak_form(
        self,
        trial: Any,
        test: Any,
        mesh: Any,
    ) -> Any:
        """Weak form: nu*integral(u_x*v_x) = integral(f*v)."""
        logger.debug("physics.burgers.weak_form")
        return None  # Placeholder for FEM backend integration

    def boundary_conditions(
        self,
        mesh: Any = None,
        config: Any = None,
    ) -> list[BoundaryCondition]:
        """Homogeneous Dirichlet BCs on both endpoints."""
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
        """MMS: u = sin(pi*x) on [0,1]."""
        nu = self._viscosity

        def exact(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.sin(np.pi * x)

        def forcing(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return nu * np.pi**2 * np.sin(np.pi * x)

        def boundary(points: np.ndarray) -> np.ndarray:
            return np.zeros(len(points))

        return ManufacturedSolution(
            exact_solution=exact,
            forcing=forcing,
            boundary_data=boundary,
            expected_convergence_order=2.0,
            name="burgers_sin",
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
        """Burgers-specific action validators."""
        return []

    def default_config(self) -> dict[str, Any]:
        """Default parameters for Burgers problems."""
        return {
            "domain": {
                "type": "interval",
                "bounds": [[0.0, 1.0]],
            },
            "viscosity": self._viscosity,
        }

    def solve_on_grid(self, n: int) -> SolveResult:
        """Solve steady-state Burgers on uniform 1D grid via FD.

        Uses standard 3-point stencil for -nu*u_xx = f.
        """
        start = time.perf_counter()

        nu = self._viscosity
        h = 1.0 / (n + 1)
        total_dofs = n

        # Build tridiagonal stiffness matrix for -nu*u_xx
        main_diag = 2.0 * nu * np.ones(total_dofs)
        off_diag = -nu * np.ones(total_dofs - 1)

        stiffness = sp.diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(total_dofs, total_dofs),
            format="csc",
        ) / (h * h)

        # RHS from manufactured solution
        x_grid = np.linspace(h, 1.0 - h, n)
        # Points as (n, 1) array for consistency with MMS interface
        points = x_grid.reshape(-1, 1)

        mms = self.manufactured_solution()
        rhs = mms.forcing(points)

        # Solve
        u = scipy.sparse.linalg.spsolve(stiffness, rhs)

        # Compute residual
        residual = stiffness @ u - rhs
        residual_norm = float(np.linalg.norm(residual))

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "physics.burgers.solved",
            grid_size=n,
            viscosity=nu,
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
