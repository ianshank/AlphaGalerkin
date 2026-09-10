"""Pydantic configs and result types for classical PDE solver baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


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


class FDMConfig(SolverConfig):
    """Configuration for finite difference solvers."""

    min_grid_points: int = Field(
        default=3,
        ge=2,
        description="Minimum grid points per dimension",
    )


class AMRConfig(SolverConfig):
    """Configuration for adaptive mesh refinement solvers."""

    marking_fraction: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Dorfler bulk-chasing marking fraction (theta). Default raised "
            "from 0.3 to 0.5 so AMR on sharply-concentrated indicators "
            "(e.g. Burgers shock) marks several elements per step instead "
            "of one, reaching the target DOF within max_refinements."
        ),
    )
    max_refinements: int = Field(
        default=30,
        ge=1,
        description=(
            "Maximum number of refinement iterations. Default raised from 10 "
            "to 30 so 1D AMR with conservative marking (theta=0.3) can reach "
            "8K+ DOF on smooth Burgers/Poisson cases without saturating."
        ),
    )
    initial_dof_divisor: int = Field(
        default=2,
        ge=1,
        description=(
            "Divisor for initial DOF count (n_dof // divisor). Default 2 means "
            "the initial 1D grid starts at half the target DOF (capped by "
            "max_initial_points_1d) so refinement only needs ~1 doubling to "
            "reach the target on smooth problems."
        ),
    )
    max_initial_points_1d: int = Field(
        default=256,
        ge=2,
        description=(
            "Maximum initial grid points for 1D AMR. Default raised from 8 "
            "to 256 so high-target n_dof requests (>=128) start with a "
            "mesh dense enough that target-aware refinement reaches the "
            "requested DOF count within max_refinements steps."
        ),
    )
    min_initial_points: int = Field(
        default=4,
        ge=2,
        description="Minimum initial grid points (1D) or per-side (2D)",
    )
    initial_side_divisor_2d: int = Field(
        default=2,
        ge=1,
        description="Divisor for initial 2D side length (sqrt(n_dof) // divisor)",
    )
    min_initial_side_2d: int = Field(
        default=3,
        ge=2,
        description="Minimum initial side grid points for 2D AMR",
    )


class PINNConfig(SolverConfig):
    """Configuration for PINN solver."""

    hidden_dim: int = Field(default=64, ge=8, description="Hidden layer dimension")
    n_layers: int = Field(default=3, ge=1, description="Number of hidden layers")
    n_epochs: int = Field(default=2000, ge=1, description="Training epochs")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    n_collocation: int = Field(default=1000, ge=10, description="Interior collocation points")
    bc_loss_weight: float = Field(default=10.0, gt=0, description="Boundary condition loss weight")
    n_boundary_points: int = Field(default=50, ge=4, description="Boundary points per epoch")
    log_interval: int = Field(default=500, ge=1, description="Logging interval (epochs)")
    device: str = Field(
        default="auto",
        description=(
            "Device preference: 'auto' (CUDA if available else CPU), 'cpu', "
            "'cuda', or 'cuda:N'. Default 'auto' matches the project's "
            "GPU-preferred policy while keeping CI green on no-GPU runners."
        ),
    )
    vector_pde: bool | None = Field(
        default=None,
        description=(
            "Override for vector-valued PDE handling. None auto-detects from "
            "operator.pde_type (Navier-Stokes => 2-channel network). True/False "
            "forces 2-channel/scalar output respectively."
        ),
    )


class NavierStokesConfig(SolverConfig):
    """Configuration for Navier-Stokes FDM solver."""

    dt: float = Field(default=0.01, gt=0, description="Time step size")
    t_final: float = Field(default=1.0, gt=0, description="Final simulation time")
    default_viscosity: float = Field(
        default=0.01,
        gt=0,
        description="Fallback viscosity if operator lacks viscosity attribute",
    )
    min_grid_points: int = Field(default=4, ge=2, description="Minimum grid points per side")
    cfl_safety: float = Field(
        default=0.25,
        gt=0,
        le=1.0,
        description="CFL stability factor for diffusion (dt <= cfl_safety * h^2 / nu)",
    )
    viscosity_floor: float = Field(
        default=1e-12,
        gt=0,
        description="Minimum viscosity for CFL denominator to avoid division by zero",
    )
    log_fraction: int = Field(
        default=10,
        ge=1,
        description="Log every n_steps // log_fraction steps",
    )
