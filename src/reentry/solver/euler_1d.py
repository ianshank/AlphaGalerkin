"""1D compressible Euler solver for shock tube problems.

Solves the 1D Euler equations using finite-volume method with
MUSCL reconstruction and configurable Riemann solver (Roe/HLLC).

Used for validation against analytical shock tube solutions
(Sod, Lax, Shu-Osher).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
from src.reentry.flux.hllc import HLLCFlux
from src.reentry.flux.reconstruction import MUSCLReconstruction
from src.reentry.flux.roe import RoeFlux
from src.reentry.gas.eos import CaloricallyPerfectEOS

logger = structlog.get_logger(__name__)


@dataclass
class ShockTubeIC:
    """Initial conditions for a shock tube problem.

    Defines left and right states separated at x_diaphragm.
    """

    rho_l: float
    u_l: float
    p_l: float
    rho_r: float
    u_r: float
    p_r: float
    x_diaphragm: float = 0.5
    x_min: float = 0.0
    x_max: float = 1.0

    @classmethod
    def sod(cls) -> ShockTubeIC:
        """Sod shock tube (1978) — classic validation case."""
        return cls(rho_l=1.0, u_l=0.0, p_l=1.0, rho_r=0.125, u_r=0.0, p_r=0.1)

    @classmethod
    def lax(cls) -> ShockTubeIC:
        """Lax shock tube — stronger shock, tests robustness."""
        return cls(rho_l=0.445, u_l=0.698, p_l=3.528, rho_r=0.5, u_r=0.0, p_r=0.571)

    @classmethod
    def shu_osher(cls) -> ShockTubeIC:
        """Shu-Osher problem — shock-entropy wave interaction."""
        return cls(
            rho_l=3.857143,
            u_l=2.629369,
            p_l=10.33333,
            rho_r=1.0,
            u_r=0.0,
            p_r=1.0,
            x_diaphragm=-4.0,
            x_min=-5.0,
            x_max=5.0,
        )


@dataclass
class Euler1DResult:
    """Result of a 1D Euler simulation."""

    x: NDArray[np.float64]
    density: NDArray[np.float64]
    velocity: NDArray[np.float64]
    pressure: NDArray[np.float64]
    time: float
    n_steps: int
    final_residual: float


class Euler1DSolver:
    """1D compressible Euler solver using finite-volume method.

    Spatial: MUSCL reconstruction with TVD limiter + Roe/HLLC flux.
    Temporal: SSPRK3 (Strong Stability Preserving Runge-Kutta).
    """

    def __init__(
        self,
        config: ReentrySolverConfig,
        gamma: float = 1.4,
        n_cells: int = 200,
    ) -> None:
        self.config = config
        self.gamma = gamma
        self.n_cells = n_cells

        # Initialize reconstruction
        self.reconstruction = MUSCLReconstruction(config.limiter)

        # Initialize flux solver
        from src.reentry.config.gas import GasConfig

        gas_config = GasConfig(
            name="euler_1d_gas",
            species=["N2"],
            gamma=gamma,
            molecular_weights={"N2": 0.0280134},
            formation_enthalpies={"N2": 0.0},
            theta_v={"N2": 0.0},
        )
        eos = CaloricallyPerfectEOS(gas_config)

        if config.flux_scheme == FluxScheme.ROE:
            self.flux_solver = RoeFlux(
                eos=eos,
                gamma=gamma,
                enable_h_correction=config.enable_h_correction,
            )
        elif config.flux_scheme == FluxScheme.HLLC:
            self.flux_solver = HLLCFlux(gamma=gamma)
        else:
            msg = f"Unsupported flux scheme for 1D: {config.flux_scheme}"
            raise ValueError(msg)

    def solve(
        self,
        ic: ShockTubeIC,
        t_final: float,
    ) -> Euler1DResult:
        """Run the 1D Euler solver to t_final.

        Args:
            ic: Initial conditions (shock tube setup).
            t_final: Final simulation time.

        Returns:
            Euler1DResult with solution profiles.

        """
        gm1 = self.gamma - 1.0
        n = self.n_cells
        dx = (ic.x_max - ic.x_min) / n
        x = np.linspace(ic.x_min + 0.5 * dx, ic.x_max - 0.5 * dx, n)

        # Initialize conservative variables: [rho, rho*u, rho*E]
        q = np.zeros((n, 3), dtype=np.float64)
        left = x <= ic.x_diaphragm

        # Shu-Osher has a sinusoidal density perturbation on the right
        if ic.x_diaphragm < 0:
            # Shu-Osher style
            rho_right = ic.rho_r + 0.2 * np.sin(5.0 * x)
            rho_right[left] = ic.rho_l
            q[:, 0] = rho_right
        else:
            q[:, 0] = np.where(left, ic.rho_l, ic.rho_r)

        u_init = np.where(left, ic.u_l, ic.u_r)
        p_init = np.where(left, ic.p_l, ic.p_r)

        q[:, 1] = q[:, 0] * u_init
        q[:, 2] = p_init / gm1 + 0.5 * q[:, 0] * u_init**2

        # Time integration
        t = 0.0
        step = 0
        max_steps = self.config.max_iterations

        while t < t_final and step < max_steps:
            # Compute timestep from CFL
            rho = q[:, 0]
            u = q[:, 1] / np.maximum(rho, 1e-30)
            e_int = q[:, 2] / np.maximum(rho, 1e-30) - 0.5 * u**2
            p = gm1 * rho * np.maximum(e_int, 1e-30)
            a = np.sqrt(self.gamma * np.maximum(p, 1e-30) / np.maximum(rho, 1e-30))
            s_max = np.max(np.abs(u) + a)
            dt = self.config.cfl * dx / max(s_max, 1e-30)
            dt = min(dt, t_final - t)

            # SSPRK3 time step
            q1 = q + dt * self._rhs(q, dx)
            q2 = 0.75 * q + 0.25 * (q1 + dt * self._rhs(q1, dx))
            q = (q + 2.0 * (q2 + dt * self._rhs(q2, dx))) / 3.0

            # Enforce positivity
            q[:, 0] = np.maximum(q[:, 0], 1e-10)
            e_int_check = q[:, 2] / q[:, 0] - 0.5 * (q[:, 1] / q[:, 0]) ** 2
            neg_mask = e_int_check < 0
            if np.any(neg_mask):
                # Reset to minimum internal energy
                u_fix = q[neg_mask, 1] / q[neg_mask, 0]
                q[neg_mask, 2] = q[neg_mask, 0] * (1e-10 + 0.5 * u_fix**2)

            t += dt
            step += 1

        # Extract final primitives
        rho = q[:, 0]
        u = q[:, 1] / np.maximum(rho, 1e-30)
        e_int = q[:, 2] / np.maximum(rho, 1e-30) - 0.5 * u**2
        p = gm1 * rho * np.maximum(e_int, 1e-30)

        logger.info(
            "euler_1d_complete",
            steps=step,
            time=t,
            density_range=(float(rho.min()), float(rho.max())),
        )

        return Euler1DResult(
            x=x,
            density=rho,
            velocity=u,
            pressure=p,
            time=t,
            n_steps=step,
            final_residual=0.0,
        )

    def _rhs(self, q: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
        """Compute right-hand side: dQ/dt = -1/dx * (F_{i+1/2} - F_{i-1/2})."""
        gm1 = self.gamma - 1.0
        n = q.shape[0]

        # Extract primitives
        rho = np.maximum(q[:, 0], 1e-30)
        u = q[:, 1] / rho
        e_int = q[:, 2] / rho - 0.5 * u**2
        p = gm1 * rho * np.maximum(e_int, 1e-30)

        # Add ghost cells for reconstruction
        rho_ext = np.concatenate([[rho[0]], rho, [rho[-1]]])
        u_ext = np.concatenate([[u[0]], u, [u[-1]]])
        p_ext = np.concatenate([[p[0]], p, [p[-1]]])

        # MUSCL reconstruction
        rho_l, rho_r = self.reconstruction.reconstruct(rho_ext)
        u_l, u_r = self.reconstruction.reconstruct(u_ext)
        p_l, p_r = self.reconstruction.reconstruct(p_ext)

        # Enforce positivity after reconstruction
        rho_l = np.maximum(rho_l, 1e-10)
        rho_r = np.maximum(rho_r, 1e-10)
        p_l = np.maximum(p_l, 1e-10)
        p_r = np.maximum(p_r, 1e-10)

        # Compute numerical flux at all interfaces
        flux = self.flux_solver.compute(rho_l, u_l, p_l, rho_r, u_r, p_r)

        # Finite volume update: dQ/dt = -(F_{i+1/2} - F_{i-1/2}) / dx
        rhs = np.zeros_like(q)
        rhs = -(flux[1 : n + 1, :] - flux[:n, :]) / dx

        return rhs
