"""Residual computation and convergence monitoring.

Tracks solution residuals over time to determine when
the solver has converged to a steady state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass
class ResidualHistory:
    """Tracks convergence history for monitoring."""

    density_l2: list[float] = field(default_factory=list)
    density_linf: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)

    @property
    def n_records(self) -> int:
        return len(self.steps)

    @property
    def last_l2(self) -> float:
        return self.density_l2[-1] if self.density_l2 else float("inf")

    @property
    def converged_ratio(self) -> float:
        """Ratio of current to initial residual (lower = more converged)."""
        if len(self.density_l2) < 2:
            return 1.0
        return self.density_l2[-1] / max(self.density_l2[0], 1e-30)


class ResidualMonitor:
    """Monitors solver convergence via residual norms.

    Computes the change in conservative variables between time steps
    as a measure of convergence toward steady state.
    """

    def __init__(self, log_interval: int = 100) -> None:
        self.log_interval = log_interval
        self.history = ResidualHistory()
        self._prev_density: NDArray[np.float64] | None = None

    def update(
        self,
        density: NDArray[np.float64],
        step: int,
        dt: float,
    ) -> float:
        """Compute and record residual for current step.

        Args:
            density: Current density field (ny, nx).
            step: Current time step number.
            dt: Current timestep size.

        Returns:
            L2 norm of density residual.

        """
        if self._prev_density is None:
            self._prev_density = density.copy()
            l2 = float(np.sqrt(np.mean(density**2)))
            self.history.density_l2.append(l2)
            self.history.density_linf.append(float(np.max(np.abs(density))))
            self.history.steps.append(step)
            return l2

        # Compute change
        d_rho = density - self._prev_density
        l2 = float(np.sqrt(np.mean(d_rho**2)) / max(dt, 1e-30))
        linf = float(np.max(np.abs(d_rho)) / max(dt, 1e-30))

        self.history.density_l2.append(l2)
        self.history.density_linf.append(linf)
        self.history.steps.append(step)
        self._prev_density = density.copy()

        if step % self.log_interval == 0:
            logger.info(
                "residual_update",
                step=step,
                l2=l2,
                linf=linf,
                dt=dt,
                converged_ratio=self.history.converged_ratio,
            )

        return l2

    def is_converged(self, tolerance: float) -> bool:
        """Check if residual has dropped below tolerance."""
        return self.history.last_l2 < tolerance

    def reset(self) -> None:
        """Reset monitor for a new simulation."""
        self.history = ResidualHistory()
        self._prev_density = None
