"""HLLC approximate Riemann solver for compressible Euler equations.

Harten-Lax-van Leer-Contact solver that resolves the contact
discontinuity exactly, providing better accuracy than HLL while
being more robust than Roe (no carbuncle instability).

Reference: Toro, E.F. (1994) "Restoration of the contact surface in
the HLL-Riemann solver." Shock Waves 4:25-34.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class HLLCFlux:
    """HLLC approximate Riemann solver.

    Three-wave model (SL, S*, SR) that captures contacts exactly.
    More robust than Roe for strong shocks; no entropy fix needed.
    """

    def __init__(self, gamma: float = 1.4) -> None:
        self.gamma = gamma

    def compute(
        self,
        rho_l: NDArray[np.float64],
        u_l: NDArray[np.float64],
        p_l: NDArray[np.float64],
        rho_r: NDArray[np.float64],
        u_r: NDArray[np.float64],
        p_r: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute HLLC flux for 1D Euler equations.

        Args:
            rho_l: Left density (N,).
            u_l: Left velocity (N,).
            p_l: Left pressure (N,).
            rho_r: Right density (N,).
            u_r: Right velocity (N,).
            p_r: Right pressure (N,).

        Returns:
            Numerical flux (N, 3) for [mass, momentum, energy].

        """
        gm1 = self.gamma - 1.0

        # Sound speeds
        a_l = np.sqrt(self.gamma * p_l / np.maximum(rho_l, 1e-30))
        a_r = np.sqrt(self.gamma * p_r / np.maximum(rho_r, 1e-30))

        # Wave speed estimates (Davis)
        s_l = np.minimum(u_l - a_l, u_r - a_r)
        s_r = np.maximum(u_l + a_l, u_r + a_r)

        # Contact wave speed (Toro eq. 10.37)
        num = p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)
        den = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
        s_star = num / np.maximum(np.abs(den), 1e-30) * np.sign(den + 1e-30)

        # Conservative states
        e_l = p_l / (gm1 * rho_l) + 0.5 * u_l**2
        e_r = p_r / (gm1 * rho_r) + 0.5 * u_r**2

        # Physical fluxes
        f_l = np.column_stack(
            [
                rho_l * u_l,
                rho_l * u_l**2 + p_l,
                u_l * (rho_l * e_l + p_l),
            ]
        )
        f_r = np.column_stack(
            [
                rho_r * u_r,
                rho_r * u_r**2 + p_r,
                u_r * (rho_r * e_r + p_r),
            ]
        )

        # Star states
        denom_l = np.maximum(np.abs(s_l - s_star), 1e-30) * np.sign(s_l - s_star + 1e-30)
        rho_l_star = rho_l * (s_l - u_l) / denom_l
        denom_r = np.maximum(np.abs(s_r - s_star), 1e-30) * np.sign(s_r - s_star + 1e-30)
        rho_r_star = rho_r * (s_r - u_r) / denom_r

        # Pressure in star region (used for contact wave flux)
        _p_star = p_l + rho_l * (s_l - u_l) * (s_star - u_l)  # noqa: F841

        # Star region conservative states
        u_l_star = np.column_stack(
            [
                rho_l_star,
                rho_l_star * s_star,
                rho_l_star * (e_l + (s_star - u_l) * (s_star + p_l / (rho_l * (s_l - u_l)))),
            ]
        )
        u_r_star = np.column_stack(
            [
                rho_r_star,
                rho_r_star * s_star,
                rho_r_star * (e_r + (s_star - u_r) * (s_star + p_r / (rho_r * (s_r - u_r)))),
            ]
        )

        # HLLC fluxes
        u_l_cons = np.column_stack([rho_l, rho_l * u_l, rho_l * e_l])
        u_r_cons = np.column_stack([rho_r, rho_r * u_r, rho_r * e_r])

        f_l_star = f_l + s_l[:, np.newaxis] * (u_l_star - u_l_cons)
        f_r_star = f_r + s_r[:, np.newaxis] * (u_r_star - u_r_cons)

        # Select flux based on wave speeds
        flux = np.where(
            s_l[:, np.newaxis] >= 0,
            f_l,
            np.where(
                s_star[:, np.newaxis] >= 0,
                f_l_star,
                np.where(
                    s_r[:, np.newaxis] >= 0,
                    f_r_star,
                    f_r,
                ),
            ),
        )

        return flux

    def max_wave_speed(
        self,
        rho: NDArray[np.float64],
        u: NDArray[np.float64],
        p: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximum wave speed for CFL computation."""
        a = np.sqrt(self.gamma * p / np.maximum(rho, 1e-30))
        return np.abs(u) + a
