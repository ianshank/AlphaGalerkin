"""Stagnation point analysis for blunt body flows.

Extracts stagnation point quantities and compares against
analytical/semi-empirical correlations (Fay-Riddell, Sutton-Graves).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class StagnationResult:
    """Stagnation point quantities.

    Attributes:
        x_stag: x-coordinate of stagnation point.
        p_stag: Stagnation pressure in Pa.
        q_stag: Stagnation heat flux in W/m^2.
        t_stag: Post-shock temperature in K.
        standoff: Shock standoff distance in meters.

    """

    x_stag: float
    p_stag: float
    q_stag: float
    t_stag: float
    standoff: float


def find_stagnation_point(
    pressure: NDArray[np.float64],
    x_cell: NDArray[np.float64],
    y_cell: NDArray[np.float64],
    heat_flux: NDArray[np.float64],
    temperature: NDArray[np.float64],
    density: NDArray[np.float64],
    wall_row: int = 0,
) -> StagnationResult:
    """Find stagnation point as location of maximum surface pressure.

    Args:
        pressure: Pressure (ny, nx).
        x_cell: Cell x-coords (ny, nx).
        y_cell: Cell y-coords (ny, nx).
        heat_flux: Surface heat flux (nx,) at wall.
        temperature: Temperature (ny, nx).
        density: Density (ny, nx).
        wall_row: Row index of wall.

    Returns:
        StagnationResult with stagnation quantities.

    """
    p_wall = pressure[wall_row, :]
    i_stag = int(np.argmax(p_wall))

    x_stag = float(x_cell[wall_row, i_stag])
    p_stag = float(p_wall[i_stag])
    q_stag = float(heat_flux[i_stag])
    t_stag = float(temperature[wall_row + 1, i_stag]) if pressure.shape[0] > 1 else 0.0

    # Estimate shock standoff from density jump
    rho_col = density[:, i_stag]
    j_shock = int(np.argmax(rho_col))
    standoff = float(abs(y_cell[j_shock, i_stag] - y_cell[wall_row, i_stag]))

    return StagnationResult(
        x_stag=x_stag,
        p_stag=p_stag,
        q_stag=q_stag,
        t_stag=t_stag,
        standoff=standoff,
    )


def sutton_graves_heat_flux(
    nose_radius_m: float,
    freestream_velocity_m_s: float,
    freestream_density_kg_m3: float,
    k_sg: float = 1.7415e-4,
) -> float:
    """Sutton-Graves stagnation point heating correlation.

    q_s = k * sqrt(rho_inf / R_n) * V_inf^3

    Args:
        nose_radius_m: Vehicle nose radius.
        freestream_velocity_m_s: Freestream velocity.
        freestream_density_kg_m3: Freestream density.
        k_sg: Sutton-Graves constant (default for Earth air).

    Returns:
        Stagnation heat flux in W/m^2.

    """
    return k_sg * np.sqrt(freestream_density_kg_m3 / nose_radius_m) * freestream_velocity_m_s**3


def fay_riddell_correction(
    heat_flux_frozen: float,
    lewis_number: float = 1.4,
    wall_enthalpy_ratio: float = 0.1,
) -> float:
    """Fay-Riddell correction for catalytic wall effects.

    Modifies frozen-flow heat flux for finite-rate surface catalysis.

    Args:
        heat_flux_frozen: Heat flux with frozen chemistry.
        lewis_number: Lewis number (diffusion/thermal).
        wall_enthalpy_ratio: h_w / h_e ratio.

    Returns:
        Corrected heat flux in W/m^2.

    """
    correction = 1.0 + (lewis_number**0.52 - 1.0) * (1.0 - wall_enthalpy_ratio)
    return heat_flux_frozen * correction
