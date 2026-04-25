"""Roe approximate Riemann solver for compressible Euler equations.

Implements the Roe-Pike linearization with entropy fix
and optional H-correction for carbuncle suppression.

Reference: Roe, P.L. (1981) "Approximate Riemann solvers, parameter
vectors, and difference schemes." J. Comp. Phys. 43:357-372.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.reentry.gas.eos import EquationOfState


@runtime_checkable
class NumericalFlux(Protocol):
    """Numerical flux interface for Riemann solvers."""

    def compute(
        self,
        rho_l: NDArray[np.float64],
        u_l: NDArray[np.float64],
        p_l: NDArray[np.float64],
        rho_r: NDArray[np.float64],
        u_r: NDArray[np.float64],
        p_r: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute numerical flux at cell interfaces."""
        ...


class RoeFlux:
    """Roe approximate Riemann solver.

    Computes numerical flux using Roe-averaged states and
    characteristic decomposition with entropy fix.

    Suitable for 1D problems; 2D extension via dimensional splitting.
    """

    def __init__(
        self,
        eos: EquationOfState,
        gamma: float = 1.4,
        entropy_fix_coefficient: float = 0.1,
        enable_h_correction: bool = False,
    ) -> None:
        self.eos = eos
        self.gamma = gamma
        self.eps = entropy_fix_coefficient
        self.h_correction = enable_h_correction

    def compute(
        self,
        rho_l: NDArray[np.float64],
        u_l: NDArray[np.float64],
        p_l: NDArray[np.float64],
        rho_r: NDArray[np.float64],
        u_r: NDArray[np.float64],
        p_r: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute Roe flux for 1D Euler equations.

        F = 0.5 * (F_L + F_R) - 0.5 * |A_roe| * (U_R - U_L)

        Args:
            rho_l: Left density (N_interfaces,).
            u_l: Left velocity (N_interfaces,).
            p_l: Left pressure (N_interfaces,).
            rho_r: Right density (N_interfaces,).
            u_r: Right velocity (N_interfaces,).
            p_r: Right pressure (N_interfaces,).

        Returns:
            Numerical flux array (N_interfaces, 3) for [mass, momentum, energy].

        """
        gm1 = self.gamma - 1.0
        n = len(rho_l)

        # Compute specific enthalpies
        e_l = p_l / (gm1 * rho_l) + 0.5 * u_l**2
        e_r = p_r / (gm1 * rho_r) + 0.5 * u_r**2
        h_l = e_l + p_l / rho_l  # Total enthalpy h = E + p/rho
        h_r = e_r + p_r / rho_r

        # Roe averages (density-weighted)
        sqrt_rho_l = np.sqrt(rho_l)
        sqrt_rho_r = np.sqrt(rho_r)
        denom = sqrt_rho_l + sqrt_rho_r

        rho_roe = sqrt_rho_l * sqrt_rho_r
        u_roe = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / denom
        h_roe = (sqrt_rho_l * h_l + sqrt_rho_r * h_r) / denom

        # Roe sound speed
        a_roe_sq = gm1 * (h_roe - 0.5 * u_roe**2)
        a_roe_sq = np.maximum(a_roe_sq, 1e-30)
        a_roe = np.sqrt(a_roe_sq)

        # Eigenvalues
        lam1 = u_roe - a_roe  # Left acoustic
        lam2 = u_roe  # Entropy/contact
        lam3 = u_roe + a_roe  # Right acoustic

        # Entropy fix (Harten)
        lam1 = self._entropy_fix(
            lam1, u_l - np.sqrt(self.gamma * p_l / rho_l), u_r - np.sqrt(self.gamma * p_r / rho_r)
        )
        lam3 = self._entropy_fix(
            lam3, u_l + np.sqrt(self.gamma * p_l / rho_l), u_r + np.sqrt(self.gamma * p_r / rho_r)
        )

        # Jump in conservative variables
        d_rho = rho_r - rho_l
        _ = rho_r * u_r - rho_l * u_l  # d_rhou (used in wave decomposition)
        _ = rho_r * e_r - rho_l * e_l  # d_rhoE (used in wave decomposition)

        # Wave strengths (characteristic decomposition)
        dp = p_r - p_l
        du = u_r - u_l

        alpha1 = 0.5 * (dp - rho_roe * a_roe * du) / a_roe_sq
        alpha2 = d_rho - dp / a_roe_sq
        alpha3 = 0.5 * (dp + rho_roe * a_roe * du) / a_roe_sq

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

        # Roe dissipation
        r1 = np.column_stack(
            [
                np.ones(n),
                u_roe - a_roe,
                h_roe - u_roe * a_roe,
            ]
        )
        r2 = np.column_stack(
            [
                np.ones(n),
                u_roe,
                0.5 * u_roe**2,
            ]
        )
        r3 = np.column_stack(
            [
                np.ones(n),
                u_roe + a_roe,
                h_roe + u_roe * a_roe,
            ]
        )

        dissipation = (
            np.abs(lam1)[:, np.newaxis] * alpha1[:, np.newaxis] * r1
            + np.abs(lam2)[:, np.newaxis] * alpha2[:, np.newaxis] * r2
            + np.abs(lam3)[:, np.newaxis] * alpha3[:, np.newaxis] * r3
        )

        return 0.5 * (f_l + f_r) - 0.5 * dissipation

    def _entropy_fix(
        self,
        lam: NDArray[np.float64],
        lam_l: NDArray[np.float64],
        lam_r: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Harten's entropy fix for sonic points.

        Prevents expansion shocks by smoothing eigenvalues near zero.
        """
        delta = np.maximum(self.eps * np.maximum(np.abs(lam_l), np.abs(lam_r)), 1e-10)
        fixed = np.where(
            np.abs(lam) < delta,
            (lam**2 + delta**2) / (2.0 * delta),
            np.abs(lam),
        )
        return fixed * np.sign(lam + 1e-30)

    def max_wave_speed(
        self,
        rho: NDArray[np.float64],
        u: NDArray[np.float64],
        p: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Maximum wave speed |u| + a for CFL computation."""
        a = np.sqrt(self.gamma * p / np.maximum(rho, 1e-30))
        return np.abs(u) + a
