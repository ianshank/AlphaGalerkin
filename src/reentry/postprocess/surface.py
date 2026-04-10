"""Surface quantity extraction from 2D flow solutions.

Extracts wall-surface quantities for validation:
- Surface pressure coefficient Cp
- Surface heat flux (convective + radiative)
- Skin friction coefficient Cf
- Stanton number St
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SurfaceData:
    """Extracted surface quantities along the vehicle wall.

    Attributes:
        x: Surface x-coordinates (n_surface,).
        pressure: Surface static pressure (n_surface,) in Pa.
        heat_flux: Surface heat flux (n_surface,) in W/m^2.
        cp: Pressure coefficient (n_surface,).
        cf: Skin friction coefficient (n_surface,).
        stanton: Stanton number (n_surface,).

    """

    x: NDArray[np.float64]
    pressure: NDArray[np.float64]
    heat_flux: NDArray[np.float64]
    cp: NDArray[np.float64]
    cf: NDArray[np.float64]
    stanton: NDArray[np.float64]


def extract_surface(
    density: NDArray[np.float64],
    velocity_x: NDArray[np.float64],
    velocity_y: NDArray[np.float64],
    pressure: NDArray[np.float64],
    temperature: NDArray[np.float64],
    x_cell: NDArray[np.float64],
    y_cell: NDArray[np.float64],
    wall_row: int = 0,
    freestream_rho: float = 1.225,
    freestream_u: float = 300.0,
    freestream_p: float = 101325.0,
    freestream_t: float = 300.0,
    wall_temperature: float = 300.0,
    gamma: float = 1.4,
    r_specific: float = 287.058,
    prandtl: float = 0.72,
    mu_ref: float = 1.716e-5,
) -> SurfaceData:
    """Extract surface quantities from 2D solution at wall boundary.

    Args:
        density: Density field (ny, nx).
        velocity_x: x-velocity (ny, nx).
        velocity_y: y-velocity (ny, nx).
        pressure: Pressure field (ny, nx).
        temperature: Temperature field (ny, nx).
        x_cell: Cell center x-coords (ny, nx).
        y_cell: Cell center y-coords (ny, nx).
        wall_row: Row index of the wall boundary (default 0 = south).
        freestream_rho: Freestream density.
        freestream_u: Freestream velocity magnitude.
        freestream_p: Freestream pressure.
        freestream_t: Freestream temperature.
        wall_temperature: Wall temperature in K.
        gamma: Ratio of specific heats.
        r_specific: Specific gas constant.
        prandtl: Prandtl number.
        mu_ref: Reference viscosity for Sutherland's law.

    Returns:
        SurfaceData with all surface quantities.

    """
    # Surface coordinates
    x_surf = x_cell[wall_row, :]
    p_surf = pressure[wall_row, :]
    t_surf = temperature[wall_row, :]

    # Dynamic pressure
    q_inf = 0.5 * freestream_rho * freestream_u**2

    # Pressure coefficient
    cp = (p_surf - freestream_p) / max(q_inf, 1e-30)

    # Velocity gradient at wall (first-order one-sided)
    if density.shape[0] > 1:
        dy_wall = y_cell[1, 0] - y_cell[0, 0]
        du_dy = velocity_x[1, :] / max(dy_wall, 1e-30)
    else:
        du_dy = np.zeros_like(x_surf)

    # Viscosity at wall (Sutherland)
    s_const = 110.4
    t_ref = 273.15
    t_wall = np.maximum(t_surf, 1.0)
    mu_wall = mu_ref * (t_wall / t_ref) ** 1.5 * (t_ref + s_const) / (t_wall + s_const)

    # Wall shear stress
    tau_wall = mu_wall * du_dy

    # Skin friction
    cf = tau_wall / max(q_inf, 1e-30)

    # Heat flux (Fourier's law at wall, first-order gradient)
    if density.shape[0] > 1:
        dt_dy = (temperature[1, :] - t_surf) / max(dy_wall, 1e-30)
    else:
        dt_dy = np.zeros_like(x_surf)

    cp_gas = gamma * r_specific / (gamma - 1.0)
    k_wall = mu_wall * cp_gas / prandtl
    heat_flux = -k_wall * dt_dy

    # Stanton number
    rho_inf_u_inf_cp = freestream_rho * freestream_u * cp_gas
    stanton = np.abs(heat_flux) / max(rho_inf_u_inf_cp * (freestream_t - wall_temperature), 1e-30)

    return SurfaceData(
        x=x_surf,
        pressure=p_surf,
        heat_flux=heat_flux,
        cp=cp,
        cf=cf,
        stanton=stanton,
    )
