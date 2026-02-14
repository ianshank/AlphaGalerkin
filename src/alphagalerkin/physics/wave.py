"""Wave equation physics module: 1D wave equation (steady-state)."""

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
    PhysicsModuleBase,
    SolveResult,
)
from src.alphagalerkin.physics.registry import register_physics

logger = structlog.get_logger("physics.wave")

# Default wave speed
_DEFAULT_WAVE_SPEED: float = 1.0


@register_physics("wave_1d")
class WaveModule(PhysicsModuleBase):
    """Steady-state 1D wave equation: -c^2*u_xx = f on [0,1].

    The full time-dependent wave equation is:
        u_tt = c^2*u_xx

    For steady-state validation we solve:
        -c^2*u_xx = f

    Manufactured solution: u_exact = sin(pi*x)
    which gives f = c^2*pi^2*sin(pi*x).
    """

    name: str = "wave_1d"
    pde_type: PDEType = PDEType.HYPERBOLIC

    def __init__(
        self,
        wave_speed: float = _DEFAULT_WAVE_SPEED,
    ) -> None:
        self._wave_speed = wave_speed

    @property
    def wave_speed(self) -> float:
        """Wave propagation speed c."""
        return self._wave_speed

    def weak_form(
        self,
        trial: Any,
        test: Any,
        mesh: Any,
    ) -> Any:
        """Weak form: c^2*integral(u_x*v_x) = integral(f*v)."""
        logger.debug("physics.wave.weak_form")
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
        c = self._wave_speed

        def exact(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.sin(np.pi * x)

        def forcing(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return c**2 * np.pi**2 * np.sin(np.pi * x)

        def boundary(points: np.ndarray) -> np.ndarray:
            return np.zeros(len(points))

        return ManufacturedSolution(
            exact_solution=exact,
            forcing=forcing,
            boundary_data=boundary,
            expected_convergence_order=2.0,
            name="wave_sin",
        )

    def default_config(self) -> dict[str, Any]:
        """Default parameters for wave equation problems."""
        return {
            "domain": {
                "type": "interval",
                "bounds": [[0.0, 1.0]],
            },
            "wave_speed": self._wave_speed,
        }

    def solve_on_grid(self, n: int) -> SolveResult:
        """Solve steady-state wave equation on uniform 1D grid via FD.

        Uses standard 3-point stencil for -c^2*u_xx = f.
        """
        start = time.perf_counter()

        c = self._wave_speed
        h = 1.0 / (n + 1)
        total_dofs = n

        # Build tridiagonal stiffness matrix for -c^2*u_xx
        main_diag = 2.0 * c**2 * np.ones(total_dofs)
        off_diag = -(c**2) * np.ones(total_dofs - 1)

        stiffness = sp.diags(
            [off_diag, main_diag, off_diag],
            [-1, 0, 1],
            shape=(total_dofs, total_dofs),
            format="csc",
        ) / (h * h)

        # RHS from manufactured solution
        x_grid = np.linspace(h, 1.0 - h, n)
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
            "physics.wave.solved",
            grid_size=n,
            wave_speed=c,
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
