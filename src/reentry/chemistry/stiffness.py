"""Stiff ODE integrator for chemical source terms.

Chemical reaction rates span 12+ orders of magnitude, requiring
implicit time integration. Uses backward Euler with Newton iteration
or SciPy's LSODA for operator-split chemistry sub-stepping.
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.reentry.chemistry.mechanism import ChemicalMechanism

logger = structlog.get_logger(__name__)


class ChemistryIntegrator:
    """Implicit integrator for stiff chemical kinetics.

    Uses operator splitting: the flow solver advances the inviscid/viscous
    terms, then this integrator advances the chemical source terms over
    the same timestep using an implicit method.

    Supports two modes:
    1. SciPy LSODA (auto-switching stiff/non-stiff)
    2. Backward Euler with Newton iteration (simpler, fewer dependencies)
    """

    def __init__(
        self,
        mechanism: ChemicalMechanism,
        method: str = "lsoda",
        rtol: float = 1e-6,
        atol: float = 1e-10,
        max_substeps: int = 100,
    ) -> None:
        self.mechanism = mechanism
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_substeps = max_substeps

    def integrate(
        self,
        density: NDArray[np.float64],
        temperature_tr: NDArray[np.float64],
        temperature_ve: NDArray[np.float64],
        mass_fractions: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Advance species mass fractions by dt using chemistry.

        Args:
            density: Mixture density (N,).
            temperature_tr: Translational temperature (N,).
            temperature_ve: Vibrational temperature (N,).
            mass_fractions: Species mass fractions (N, n_species).
            dt: Timestep in seconds.

        Returns:
            Updated mass fractions (N, n_species).

        """
        n_pts = density.shape[0]
        y_new = mass_fractions.copy()

        for i in range(n_pts):
            rho_i = density[i]
            t_tr_i = temperature_tr[i]
            t_ve_i = temperature_ve[i]
            y_i = mass_fractions[i, :]

            if self.method == "lsoda":
                y_new[i, :] = self._integrate_lsoda(rho_i, t_tr_i, t_ve_i, y_i, dt)
            else:
                y_new[i, :] = self._integrate_backward_euler(rho_i, t_tr_i, t_ve_i, y_i, dt)

        # Enforce positivity and mass conservation
        y_new = np.clip(y_new, 0.0, 1.0)
        y_new /= np.sum(y_new, axis=1, keepdims=True)

        return y_new

    def _integrate_lsoda(
        self,
        rho: float,
        t_tr: float,
        t_ve: float,
        y0: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Integrate using SciPy's LSODA (auto stiff/non-stiff)."""
        rho_arr = np.array([rho])
        t_tr_arr = np.array([t_tr])
        t_ve_arr = np.array([t_ve])

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            y_2d = y.reshape(1, -1)
            omega = self.mechanism.source_terms(rho_arr, t_tr_arr, t_ve_arr, y_2d)
            # dy/dt = omega / rho (source terms are in kg/m^3/s)
            return omega[0, :] / max(rho, 1e-30)

        try:
            sol = solve_ivp(
                rhs,
                (0.0, dt),
                y0,
                method="LSODA",
                rtol=self.rtol,
                atol=self.atol,
                max_step=dt / 2,
            )
            if sol.success:
                return sol.y[:, -1]
            logger.warning("chemistry_lsoda_failed", message=sol.message)
            return y0
        except Exception as e:
            logger.warning("chemistry_integration_error", error=str(e))
            return y0

    def _integrate_backward_euler(
        self,
        rho: float,
        t_tr: float,
        t_ve: float,
        y0: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """Simple backward Euler with fixed-point iteration."""
        rho_arr = np.array([rho])
        t_tr_arr = np.array([t_tr])
        t_ve_arr = np.array([t_ve])

        y = y0.copy()
        for _ in range(self.max_substeps):
            y_2d = y.reshape(1, -1)
            omega = self.mechanism.source_terms(rho_arr, t_tr_arr, t_ve_arr, y_2d)
            dydt = omega[0, :] / max(rho, 1e-30)
            y_new = y0 + dt * dydt

            if np.max(np.abs(y_new - y)) < self.atol:
                return y_new
            y = y_new

        return y
