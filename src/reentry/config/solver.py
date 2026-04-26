"""Solver configuration for compressible Navier-Stokes.

Defines flux scheme, limiter, time integration, and convergence
parameters for the reentry flow solver.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig


class FluxScheme(str, Enum):
    """Supported numerical flux schemes."""

    ROE = "roe"
    HLLC = "hllc"
    AUSM_PLUS = "ausm_plus"


class LimiterType(str, Enum):
    """Supported slope limiters for MUSCL reconstruction."""

    MINMOD = "minmod"
    VAN_LEER = "van_leer"
    VAN_ALBADA = "van_albada"
    SUPERBEE = "superbee"
    NONE = "none"


class TimeIntegration(str, Enum):
    """Time integration schemes."""

    FORWARD_EULER = "forward_euler"
    RK4 = "rk4"
    SSPRK3 = "ssprk3"
    IMEX = "imex"


class ReentrySolverConfig(BaseModuleConfig):
    """Configuration for the compressible flow solver."""

    # Spatial discretization
    flux_scheme: FluxScheme = Field(
        default=FluxScheme.ROE,
        description="Numerical flux scheme for inviscid fluxes.",
    )
    limiter: LimiterType = Field(
        default=LimiterType.VAN_LEER,
        description="Slope limiter for MUSCL reconstruction.",
    )
    reconstruction_order: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Spatial reconstruction order (1=first-order, 2=MUSCL).",
    )
    enable_h_correction: bool = Field(
        default=True,
        description="Enable H-correction for carbuncle suppression (Roe only).",
    )

    # Time integration
    time_integration: TimeIntegration = Field(
        default=TimeIntegration.SSPRK3,
        description="Time integration scheme.",
    )
    cfl: float = Field(
        default=0.5,
        gt=0.0,
        le=10.0,
        description="CFL number for explicit time stepping.",
    )
    adaptive_cfl: bool = Field(
        default=True,
        description="Enable adaptive CFL ramping during convergence.",
    )
    cfl_ramp_start: float = Field(
        default=0.1,
        gt=0.0,
        description="Initial CFL for ramping.",
    )
    cfl_ramp_steps: int = Field(
        default=100,
        ge=0,
        description="Number of steps to ramp from cfl_ramp_start to cfl.",
    )

    # Convergence
    max_iterations: int = Field(
        default=10000,
        ge=1,
        description="Maximum number of time steps.",
    )
    residual_tolerance: float = Field(
        default=1e-8,
        gt=0.0,
        description="Convergence criterion on density residual L2 norm.",
    )

    # Viscous terms
    enable_viscous: bool = Field(
        default=True,
        description="Enable viscous (Navier-Stokes) terms.",
    )

    # Shock detection
    shock_detection_threshold: float = Field(
        default=0.3,
        gt=0.0,
        le=1.0,
        description="Pressure gradient threshold for shock detection.",
    )

    @model_validator(mode="after")
    def _validate_cfl_ramp(self) -> ReentrySolverConfig:
        if self.adaptive_cfl and self.cfl_ramp_start >= self.cfl:
            msg = "cfl_ramp_start must be less than cfl when adaptive_cfl is enabled"
            raise ValueError(msg)
        return self
