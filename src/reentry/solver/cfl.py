"""CFL number computation and adaptive timestep control.

The CFL (Courant-Friedrichs-Lewy) condition restricts the timestep
for explicit schemes: dt <= CFL * dx / (|u| + a) where a is sound speed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class CFLController:
    """Adaptive timestep control based on CFL condition.

    Supports CFL ramping during startup to improve stability
    when initial transients are strong.
    """

    def __init__(
        self,
        cfl_target: float = 0.5,
        cfl_ramp_start: float = 0.1,
        cfl_ramp_steps: int = 100,
        adaptive: bool = True,
    ) -> None:
        self.cfl_target = cfl_target
        self.cfl_ramp_start = cfl_ramp_start
        self.cfl_ramp_steps = cfl_ramp_steps
        self.adaptive = adaptive

    def current_cfl(self, step: int) -> float:
        """Get effective CFL number at current step (with ramping)."""
        if not self.adaptive or step >= self.cfl_ramp_steps:
            return self.cfl_target

        # Linear ramp from start to target
        frac = step / max(self.cfl_ramp_steps, 1)
        return self.cfl_ramp_start + frac * (self.cfl_target - self.cfl_ramp_start)

    def compute_timestep(
        self,
        wave_speed: NDArray[np.float64],
        dx: NDArray[np.float64],
        dy: NDArray[np.float64],
        step: int = 0,
    ) -> float:
        """Compute stable timestep from CFL condition.

        dt = CFL * min(dx / s_x, dy / s_y) over all cells.

        Args:
            wave_speed: Maximum wave speed |u|+a at each cell (ny, nx).
            dx: Cell widths (ny, nx).
            dy: Cell heights (ny, nx).
            step: Current time step for CFL ramping.

        Returns:
            Stable timestep in seconds.

        """
        cfl = self.current_cfl(step)

        # Wave speed in each direction (conservative: use max wave speed)
        s = np.maximum(wave_speed, 1e-30)

        # Minimum dt from both directions
        dt_x = dx / s
        dt_y = dy / s

        dt_min = min(float(dt_x.min()), float(dt_y.min()))

        return cfl * dt_min

    @staticmethod
    def wave_speed(
        velocity_x: NDArray[np.float64],
        velocity_y: NDArray[np.float64],
        sound_speed: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute maximum wave speed |u| + a for CFL condition.

        Args:
            velocity_x: x-velocity (ny, nx).
            velocity_y: y-velocity (ny, nx).
            sound_speed: Local sound speed (ny, nx).

        Returns:
            Maximum wave speed at each cell (ny, nx).

        """
        return np.abs(velocity_x) + np.abs(velocity_y) + sound_speed
