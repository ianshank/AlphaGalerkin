"""Fire solver configuration.

Defines grid, timestep, tolerance, and convergence parameters
for the coupled heat + advection-diffusion fire spread solver.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig


class FireTimeIntegration(str, Enum):
    """Time integration schemes for fire solver."""

    FORWARD_EULER = "forward_euler"
    RK4 = "rk4"
    CRANK_NICOLSON = "crank_nicolson"


class FireSolverConfig(BaseModuleConfig):
    """Configuration for the fire spread solver."""

    # Grid
    nx: int = Field(default=100, ge=4, description="Grid cells in x direction.")
    ny: int = Field(default=100, ge=4, description="Grid cells in y direction.")
    domain_size_x_m: float = Field(default=1000.0, gt=0.0, description="Domain size in x (meters).")
    domain_size_y_m: float = Field(default=1000.0, gt=0.0, description="Domain size in y (meters).")

    # Time
    dt_s: float = Field(default=0.5, gt=0.0, description="Timestep in seconds.")
    time_integration: FireTimeIntegration = Field(
        default=FireTimeIntegration.RK4,
        description="Time integration method.",
    )
    prediction_horizon_s: float = Field(
        default=1800.0,
        gt=0.0,
        description="Prediction horizon in seconds (default 30 min).",
    )

    # Diffusion
    thermal_diffusivity_m2_s: float = Field(
        default=1e-5,
        gt=0.0,
        description="Effective thermal diffusivity in m^2/s.",
    )

    # Convergence
    max_steps: int = Field(default=100000, ge=1, description="Maximum timesteps.")
    energy_conservation_rtol: float = Field(
        default=1e-6,
        gt=0.0,
        description="Relative tolerance for energy conservation check.",
    )

    @model_validator(mode="after")
    def _validate_grid(self) -> FireSolverConfig:
        dx = self.domain_size_x_m / self.nx
        dy = self.domain_size_y_m / self.ny
        # CFL-like stability check for explicit diffusion
        max_dt = 0.25 * min(dx, dy) ** 2 / self.thermal_diffusivity_m2_s
        if self.time_integration == FireTimeIntegration.FORWARD_EULER:
            if self.dt_s > max_dt:
                msg = (
                    f"dt_s={self.dt_s} exceeds diffusion stability limit "
                    f"{max_dt:.4f} for explicit scheme"
                )
                raise ValueError(msg)
        return self
