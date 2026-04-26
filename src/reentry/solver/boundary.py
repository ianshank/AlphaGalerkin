"""Boundary conditions for compressible flow.

Implements ghost-cell-based boundary conditions for:
- Freestream (supersonic inflow)
- No-slip wall (isothermal/adiabatic)
- Symmetry (reflect normal velocity)
- Supersonic outflow (zero-order extrapolation)

Ghost cells are populated before each RHS evaluation so that
the interior stencil sees correct neighbor values at boundaries.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.reentry.config.freestream import FreestreamConfig
from src.reentry.config.wall import WallConfig, WallThermalModel


class BoundaryFace(str, Enum):
    """Boundary face identifiers for 2D structured mesh."""

    WEST = "west"  # x_min (i=0)
    EAST = "east"  # x_max (i=nx-1)
    SOUTH = "south"  # y_min (j=0) — typically the wall
    NORTH = "north"  # y_max (j=ny-1)


@runtime_checkable
class BoundaryCondition(Protocol):
    """Boundary condition interface for ghost cell population."""

    def apply(
        self,
        q: NDArray[np.float64],
        face: BoundaryFace,
        n_ghost: int,
        gamma: float,
    ) -> None:
        """Populate ghost cells for the given boundary face.

        Args:
            q: Conservative variable array (total_ny, total_nx, n_vars).
               Modified in-place.
            face: Which boundary face to apply.
            n_ghost: Number of ghost cell layers.
            gamma: Ratio of specific heats.

        """
        ...


class FreestreamBC:
    """Supersonic freestream inflow boundary condition.

    Sets ghost cells to freestream state. Valid for supersonic inflow
    where all characteristics enter the domain.
    """

    def __init__(
        self,
        config: FreestreamConfig,
        gamma: float = 1.4,
        r_specific: float = 287.058,
    ) -> None:
        self.config = config
        # Precompute conservative state from freestream
        rho = config.density_kg_m3
        u = config.velocity_m_s
        p = rho * config.temperature_K * r_specific

        gm1 = gamma - 1.0
        e = p / (gm1 * rho) + 0.5 * u**2
        self._q_inf = np.array([rho, rho * u, 0.0, rho * e])

    def apply(
        self,
        q: NDArray[np.float64],
        face: BoundaryFace,
        n_ghost: int,
        gamma: float,
    ) -> None:
        n_vars = q.shape[2]
        q_inf = self._q_inf[:n_vars]

        if face == BoundaryFace.WEST:
            for g in range(n_ghost):
                q[:, g, :] = q_inf
        elif face == BoundaryFace.EAST:
            for g in range(n_ghost):
                q[:, -(g + 1), :] = q_inf
        elif face == BoundaryFace.SOUTH:
            for g in range(n_ghost):
                q[g, :, :] = q_inf
        elif face == BoundaryFace.NORTH:
            for g in range(n_ghost):
                q[-(g + 1), :, :] = q_inf


class WallBC:
    """No-slip wall boundary condition with isothermal or adiabatic options.

    Reflects momentum (no-slip) and sets temperature or heat flux
    based on the wall thermal model.
    """

    def __init__(
        self,
        config: WallConfig,
        gamma: float = 1.4,
        r_specific: float = 287.058,
    ) -> None:
        self.config = config
        self.gamma = gamma
        self.r_specific = r_specific

    def apply(
        self,
        q: NDArray[np.float64],
        face: BoundaryFace,
        n_ghost: int,
        gamma: float,
    ) -> None:
        gm1 = gamma - 1.0

        if face == BoundaryFace.SOUTH:
            for g in range(n_ghost):
                gi = n_ghost - 1 - g  # Ghost index from wall
                ii = n_ghost + g  # Interior mirror index

                # Copy density
                q[gi, :, 0] = q[ii, :, 0]
                # Reflect x-velocity (no-slip: u_ghost = -u_interior)
                q[gi, :, 1] = -q[ii, :, 1]
                # Reflect y-velocity
                q[gi, :, 2] = -q[ii, :, 2]

                if self.config.thermal_model == WallThermalModel.ISOTHERMAL:
                    # Set wall temperature
                    rho = q[gi, :, 0]
                    t_wall = self.config.wall_temperature_K
                    p_wall = rho * self.r_specific * t_wall
                    e_wall = p_wall / (gm1 * rho)
                    q[gi, :, 3] = rho * e_wall  # No kinetic energy at wall
                else:
                    # Adiabatic: extrapolate energy
                    q[gi, :, 3] = q[ii, :, 3]

                # Species: zero gradient
                if q.shape[2] > 4:
                    q[gi, :, 4:] = q[ii, :, 4:]

        elif face == BoundaryFace.NORTH:
            ny_total = q.shape[0]
            for g in range(n_ghost):
                gi = ny_total - 1 - (n_ghost - 1 - g)
                ii = ny_total - 1 - (n_ghost + g)
                q[gi, :, 0] = q[ii, :, 0]
                q[gi, :, 1] = -q[ii, :, 1]
                q[gi, :, 2] = -q[ii, :, 2]
                q[gi, :, 3] = q[ii, :, 3]
                if q.shape[2] > 4:
                    q[gi, :, 4:] = q[ii, :, 4:]


class SymmetryBC:
    """Symmetry boundary condition (reflect normal velocity component)."""

    def apply(
        self,
        q: NDArray[np.float64],
        face: BoundaryFace,
        n_ghost: int,
        gamma: float,
    ) -> None:
        if face == BoundaryFace.SOUTH:
            for g in range(n_ghost):
                gi = n_ghost - 1 - g
                ii = n_ghost + g
                q[gi, :, :] = q[ii, :, :]
                q[gi, :, 2] = -q[ii, :, 2]  # Reflect v
        elif face == BoundaryFace.NORTH:
            ny = q.shape[0]
            for g in range(n_ghost):
                gi = ny - 1 - (n_ghost - 1 - g)
                ii = ny - 1 - (n_ghost + g)
                q[gi, :, :] = q[ii, :, :]
                q[gi, :, 2] = -q[ii, :, 2]
        elif face == BoundaryFace.WEST:
            for g in range(n_ghost):
                gi = n_ghost - 1 - g
                ii = n_ghost + g
                q[:, gi, :] = q[:, ii, :]
                q[:, gi, 1] = -q[:, ii, 1]  # Reflect u
        elif face == BoundaryFace.EAST:
            nx = q.shape[1]
            for g in range(n_ghost):
                gi = nx - 1 - (n_ghost - 1 - g)
                ii = nx - 1 - (n_ghost + g)
                q[:, gi, :] = q[:, ii, :]
                q[:, gi, 1] = -q[:, ii, 1]


class SupersonicOutflowBC:
    """Supersonic outflow: zero-order extrapolation.

    All characteristics leave the domain, so all variables
    are extrapolated from the interior.
    """

    def apply(
        self,
        q: NDArray[np.float64],
        face: BoundaryFace,
        n_ghost: int,
        gamma: float,
    ) -> None:
        if face == BoundaryFace.EAST:
            nx = q.shape[1]
            for g in range(n_ghost):
                q[:, nx - 1 - g, :] = q[:, nx - 1 - n_ghost, :]
        elif face == BoundaryFace.NORTH:
            ny = q.shape[0]
            for g in range(n_ghost):
                q[ny - 1 - g, :, :] = q[ny - 1 - n_ghost, :, :]
        elif face == BoundaryFace.WEST:
            for g in range(n_ghost):
                q[:, g, :] = q[:, n_ghost, :]
        elif face == BoundaryFace.SOUTH:
            for g in range(n_ghost):
                q[g, :, :] = q[n_ghost, :, :]
