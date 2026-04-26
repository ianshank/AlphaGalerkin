"""2D compressible Navier-Stokes solver extending the Euler solver.

Adds viscous flux contributions (stress tensor + heat conduction)
to the inviscid Euler solver via operator splitting.

NS equations:
    dU/dt + dF_inv/dx + dG_inv/dy = dF_vis/dx + dG_vis/dy
"""

from __future__ import annotations

import numpy as np
import structlog
from numpy.typing import NDArray

from src.reentry.config.solver import ReentrySolverConfig
from src.reentry.flux.viscous import ViscousFlux
from src.reentry.mesh.structured import StructuredMesh2D
from src.reentry.solver.euler_2d import Euler2DResult, Euler2DSolver

logger = structlog.get_logger(__name__)


class NavierStokes2DSolver(Euler2DSolver):
    """2D compressible Navier-Stokes solver.

    Extends Euler2DSolver by adding viscous flux contributions
    (stress tensor + Fourier heat conduction) after each inviscid step.

    Viscosity model: constant mu or Sutherland's law (configurable).
    """

    def __init__(
        self,
        config: ReentrySolverConfig,
        mesh: StructuredMesh2D,
        gamma: float = 1.4,
        mu_ref: float = 1.716e-5,
        t_ref: float = 273.15,
        s_const: float = 110.4,
        prandtl: float = 0.72,
        r_specific: float = 287.058,
    ) -> None:
        super().__init__(config, mesh, gamma)
        self.mu_ref = mu_ref
        self.t_ref = t_ref
        self.s_const = s_const
        self.prandtl = prandtl
        self.r_specific = r_specific
        self.viscous_flux = ViscousFlux(
            gamma=gamma,
            prandtl=prandtl,
            r_specific=r_specific,
        )

    def _rhs(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Override RHS to add viscous terms."""
        # Inviscid contribution from parent
        rhs = super()._rhs(q)

        if not self.config.enable_viscous:
            return rhs

        # Extract primitives for viscous flux
        si, sj = self.mesh.interior_slice
        qi = q[si, sj, :]
        rho = np.maximum(qi[:, :, 0], 1e-30)
        u = qi[:, :, 1] / rho
        v = qi[:, :, 2] / rho
        e_int = qi[:, :, 3] / rho - 0.5 * (u**2 + v**2)
        p = self.gm1 * rho * np.maximum(e_int, 1e-30)

        # Compute viscosity (Sutherland's law)
        temp = p / (rho * self.r_specific)
        mu = self._sutherland_viscosity(temp)

        # Compute viscous flux divergence
        metrics = self.mesh.metrics
        visc_rhs = self.viscous_flux.compute(
            rho,
            u,
            v,
            p,
            mu,
            metrics.dx,
            metrics.dy,
        )

        # Add viscous contribution to interior cells
        rhs[si, sj, :] += visc_rhs

        return rhs

    def _sutherland_viscosity(self, temperature: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sutherland's law for dynamic viscosity.

        mu = mu_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S)

        where S = 110.4 K for air.
        """
        s_const = self.s_const
        t_clipped = np.maximum(temperature, 1.0)
        return (
            self.mu_ref
            * (t_clipped / self.t_ref) ** 1.5
            * (self.t_ref + s_const)
            / (t_clipped + s_const)
        )

    def solve(
        self,
        q0: NDArray[np.float64],
        t_final: float,
    ) -> Euler2DResult:
        """Run the Navier-Stokes solver (delegates to Euler with viscous RHS)."""
        logger.info(
            "ns_2d_solver_start",
            nx=self.mesh.nx,
            ny=self.mesh.ny,
            mu_ref=self.mu_ref,
            prandtl=self.prandtl,
        )
        return super().solve(q0, t_final)
