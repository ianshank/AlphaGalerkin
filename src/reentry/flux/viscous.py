"""Viscous flux computation for compressible Navier-Stokes equations.

Computes the viscous stress tensor and heat flux using central
differences for the velocity and temperature gradients.

Viscous flux vector:
    F_v = [0, tau_xx, tau_xy, u*tau_xx + v*tau_xy - q_x]
    G_v = [0, tau_xy, tau_yy, u*tau_xy + v*tau_yy - q_y]

where tau is the Newtonian stress tensor and q is Fourier heat flux.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ViscousFlux:
    """Computes viscous fluxes for 2D compressible Navier-Stokes.

    Uses second-order central differences for velocity and temperature
    gradients, evaluated at cell centers.
    """

    def __init__(
        self,
        gamma: float = 1.4,
        prandtl: float = 0.72,
        r_specific: float = 287.058,
    ) -> None:
        self.gamma = gamma
        self.gm1 = gamma - 1.0
        self.prandtl = prandtl
        self.r_specific = r_specific

    def compute(
        self,
        rho: NDArray[np.float64],
        u: NDArray[np.float64],
        v: NDArray[np.float64],
        p: NDArray[np.float64],
        mu: NDArray[np.float64],
        dx: NDArray[np.float64],
        dy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute viscous flux divergence.

        Args:
            rho: Density (ny, nx).
            u: x-velocity (ny, nx).
            v: y-velocity (ny, nx).
            p: Pressure (ny, nx).
            mu: Dynamic viscosity (ny, nx).
            dx: Cell widths (ny, nx).
            dy: Cell heights (ny, nx).

        Returns:
            Viscous source term (ny, nx, 4): contribution to dQ/dt.

        """
        ny, nx = rho.shape
        rhs = np.zeros((ny, nx, 4), dtype=np.float64)

        # Temperature for heat flux
        temp = p / (rho * self.r_specific)

        # Velocity gradients (central differences)
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        dt_dx = np.zeros_like(temp)
        dt_dy = np.zeros_like(temp)

        if nx > 2:
            du_dx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx[:, 1:-1])
            dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * dx[:, 1:-1])
            dt_dx[:, 1:-1] = (temp[:, 2:] - temp[:, :-2]) / (2.0 * dx[:, 1:-1])

        if ny > 2:
            du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dy[1:-1, :])
            dv_dy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dy[1:-1, :])
            dt_dy[1:-1, :] = (temp[2:, :] - temp[:-2, :]) / (2.0 * dy[1:-1, :])

        # Divergence of velocity
        div_u = du_dx + dv_dy

        # Stress tensor components (Stokes hypothesis: lambda = -2/3 mu)
        tau_xx = mu * (2.0 * du_dx - (2.0 / 3.0) * div_u)
        tau_yy = mu * (2.0 * dv_dy - (2.0 / 3.0) * div_u)
        tau_xy = mu * (du_dy + dv_dx)

        # Heat flux (Fourier's law)
        # k = mu * cp / Pr, cp = gamma * R / (gamma - 1)
        cp = self.gamma * self.r_specific / self.gm1
        k = mu * cp / self.prandtl
        qx = -k * dt_dx
        qy = -k * dt_dy

        # Viscous flux divergence
        # d(tau_xx)/dx + d(tau_xy)/dy for x-momentum
        if nx > 2:
            rhs[:, 1:-1, 1] += (tau_xx[:, 2:] - tau_xx[:, :-2]) / (2.0 * dx[:, 1:-1])
            rhs[:, 1:-1, 2] += (tau_xy[:, 2:] - tau_xy[:, :-2]) / (2.0 * dx[:, 1:-1])
        if ny > 2:
            rhs[1:-1, :, 1] += (tau_xy[2:, :] - tau_xy[:-2, :]) / (2.0 * dy[1:-1, :])
            rhs[1:-1, :, 2] += (tau_yy[2:, :] - tau_yy[:-2, :]) / (2.0 * dy[1:-1, :])

        # Energy: d(u*tau_xx + v*tau_xy - qx)/dx + d(u*tau_xy + v*tau_yy - qy)/dy
        work_x = u * tau_xx + v * tau_xy - qx
        work_y = u * tau_xy + v * tau_yy - qy

        if nx > 2:
            rhs[:, 1:-1, 3] += (work_x[:, 2:] - work_x[:, :-2]) / (2.0 * dx[:, 1:-1])
        if ny > 2:
            rhs[1:-1, :, 3] += (work_y[2:, :] - work_y[:-2, :]) / (2.0 * dy[1:-1, :])

        return rhs
