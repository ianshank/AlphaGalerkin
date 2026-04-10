"""2D compressible Euler solver using finite-volume method.

Solves the 2D inviscid compressible Euler equations:
    dU/dt + dF/dx + dG/dy = 0

where U = [rho, rho*u, rho*v, rho*E] are conservative variables,
F and G are the x- and y-direction fluxes.

Uses dimensional splitting: sweep x-direction, then y-direction.
Each sweep uses MUSCL reconstruction + Roe/HLLC numerical flux.
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
from src.reentry.mesh.structured import StructuredMesh2D
from src.reentry.solver.boundary import BoundaryCondition, BoundaryFace
from src.reentry.solver.cfl import CFLController
from src.reentry.solver.residual import ResidualMonitor

logger = structlog.get_logger(__name__)

N_EULER_VARS = 4  # rho, rho*u, rho*v, rho*E


@dataclass
class Euler2DResult:
    """Result of a 2D Euler simulation."""

    density: NDArray[np.float64]
    velocity_x: NDArray[np.float64]
    velocity_y: NDArray[np.float64]
    pressure: NDArray[np.float64]
    mach: NDArray[np.float64]
    time: float
    n_steps: int
    final_residual: float
    converged: bool


class Euler2DSolver:
    """2D compressible Euler solver on structured mesh.

    Uses dimensional splitting with MUSCL-Hancock reconstruction
    and configurable Riemann solver. Time integration via SSPRK3.
    """

    def __init__(
        self,
        config: ReentrySolverConfig,
        mesh: StructuredMesh2D,
        gamma: float = 1.4,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.gamma = gamma
        self.gm1 = gamma - 1.0
        self.ng = mesh.n_ghost

        # Reconstruction
        self.recon = MUSCLReconstruction(config.limiter)

        # Flux solver
        from src.reentry.config.gas import GasConfig

        gas_config = GasConfig(
            name="euler_2d_gas",
            species=["N2"],
            gamma=gamma,
            molecular_weights={"N2": 0.0280134},
            formation_enthalpies={"N2": 0.0},
            theta_v={"N2": 0.0},
        )
        eos = CaloricallyPerfectEOS(gas_config)
        if config.flux_scheme == FluxScheme.ROE:
            self._flux_1d = RoeFlux(
                eos=eos, gamma=gamma, enable_h_correction=config.enable_h_correction
            )
        else:
            self._flux_1d = HLLCFlux(gamma=gamma)

        # CFL controller
        self.cfl_ctrl = CFLController(
            cfl_target=config.cfl,
            cfl_ramp_start=config.cfl_ramp_start if config.adaptive_cfl else config.cfl,
            cfl_ramp_steps=config.cfl_ramp_steps if config.adaptive_cfl else 0,
            adaptive=config.adaptive_cfl,
        )

        # Boundary conditions (set externally)
        self.bcs: dict[BoundaryFace, BoundaryCondition] = {}

        # Residual monitor
        self.residual_monitor = ResidualMonitor()

    def set_bc(self, face: BoundaryFace, bc: BoundaryCondition) -> None:
        """Set boundary condition for a given face."""
        self.bcs[face] = bc

    def initialize_uniform(
        self,
        rho: float,
        u: float,
        v: float,
        p: float,
    ) -> NDArray[np.float64]:
        """Initialize with uniform flow state.

        Returns:
            Conservative variable array (total_ny, total_nx, 4).

        """
        q = self.mesh.allocate_field(N_EULER_VARS)
        e = p / (self.gm1 * rho) + 0.5 * (u**2 + v**2)
        q[:, :, 0] = rho
        q[:, :, 1] = rho * u
        q[:, :, 2] = rho * v
        q[:, :, 3] = rho * e
        return q

    def solve(
        self,
        q0: NDArray[np.float64],
        t_final: float,
    ) -> Euler2DResult:
        """Run the 2D Euler solver.

        Args:
            q0: Initial conservative state (total_ny, total_nx, 4).
            t_final: Final simulation time.

        Returns:
            Euler2DResult with flow field and convergence info.

        """
        q = q0.copy()
        t = 0.0
        step = 0
        max_steps = self.config.max_iterations

        self.residual_monitor.reset()

        while t < t_final and step < max_steps:
            # Apply boundary conditions
            self._apply_bcs(q)

            # Compute timestep
            dt = self._compute_dt(q, step)
            dt = min(dt, t_final - t)

            # SSPRK3 time step
            rhs0 = self._rhs(q)
            q1 = q + dt * rhs0

            self._apply_bcs(q1)
            rhs1 = self._rhs(q1)
            q2 = 0.75 * q + 0.25 * (q1 + dt * rhs1)

            self._apply_bcs(q2)
            rhs2 = self._rhs(q2)
            q = (q + 2.0 * (q2 + dt * rhs2)) / 3.0

            # Enforce positivity
            self._enforce_positivity(q)

            # Track residual
            si, sj = self.mesh.interior_slice
            self.residual_monitor.update(q[si, sj, 0], step, dt)

            t += dt
            step += 1

            # Check convergence
            if self.residual_monitor.is_converged(self.config.residual_tolerance):
                logger.info("euler_2d_converged", step=step, time=t)
                break

        return self._extract_result(q, t, step)

    def _rhs(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute dQ/dt using dimensional splitting."""
        rhs = np.zeros_like(q)
        si, sj = self.mesh.interior_slice
        ny_int = self.mesh.ny
        nx_int = self.mesh.nx
        metrics = self.mesh.metrics

        # X-sweep: for each row j, compute F_{i+1/2}
        for j in range(ny_int):
            jg = j + self.ng
            rhs_row = self._sweep_x(q[jg, :, :], metrics.dx[j, :])
            rhs[jg, self.ng : self.ng + nx_int, :] += rhs_row

        # Y-sweep: for each column i, compute G_{j+1/2}
        for i in range(nx_int):
            ig = i + self.ng
            rhs_col = self._sweep_y(q[:, ig, :], metrics.dy[:, i])
            rhs[self.ng : self.ng + ny_int, ig, :] += rhs_col

        return rhs

    def _sweep_x(
        self,
        q_row: NDArray[np.float64],
        dx: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute x-direction flux contribution for one row.

        Args:
            q_row: (total_nx, 4) conservative variables for this row.
            dx: Cell widths for interior cells (nx,).

        Returns:
            RHS contribution (nx, 4) for interior cells.

        """
        nx_int = self.mesh.nx
        ng = self.ng

        # Extract primitives
        rho = np.maximum(q_row[:, 0], 1e-30)
        u = q_row[:, 1] / rho
        v = q_row[:, 2] / rho
        e_int = q_row[:, 3] / rho - 0.5 * (u**2 + v**2)
        p = self.gm1 * rho * np.maximum(e_int, 1e-30)

        # MUSCL reconstruction on primitives
        rho_l, rho_r = self.recon.reconstruct(rho)
        u_l, u_r = self.recon.reconstruct(u)
        p_l, p_r = self.recon.reconstruct(p)

        rho_l = np.maximum(rho_l, 1e-10)
        rho_r = np.maximum(rho_r, 1e-10)
        p_l = np.maximum(p_l, 1e-10)
        p_r = np.maximum(p_r, 1e-10)

        # Compute 1D flux (in x-direction, use u as normal velocity)
        flux = self._flux_1d.compute(rho_l, u_l, p_l, rho_r, u_r, p_r)

        # Convert back to 2D: add v-momentum flux
        # The 1D flux gives [rho*u, rho*u^2+p, u*(rho*E+p)]
        # For 2D, we need [rho*u, rho*u^2+p, rho*u*v, u*(rho*E+p)]
        v_avg = 0.5 * (
            q_row[:-1, 2] / np.maximum(q_row[:-1, 0], 1e-30)
            + q_row[1:, 2] / np.maximum(q_row[1:, 0], 1e-30)
        )
        flux_4 = np.zeros((flux.shape[0], 4))
        flux_4[:, 0] = flux[:, 0]  # Mass
        flux_4[:, 1] = flux[:, 1]  # x-momentum
        flux_4[:, 2] = flux[:, 0] * v_avg  # y-momentum transport
        flux_4[:, 3] = flux[:, 2]  # Energy

        # Finite volume update for interior cells
        rhs = np.zeros((nx_int, 4), dtype=np.float64)
        for k in range(4):
            fi = flux_4[ng - 1 : ng - 1 + nx_int + 1, k]
            rhs[:, k] = -(fi[1:] - fi[:-1]) / dx

        return rhs

    def _sweep_y(
        self,
        q_col: NDArray[np.float64],
        dy: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute y-direction flux contribution for one column.

        Same logic as _sweep_x but with v as the normal velocity
        and u as the transverse velocity.
        """
        ny_int = self.mesh.ny
        ng = self.ng

        rho = np.maximum(q_col[:, 0], 1e-30)
        u = q_col[:, 1] / rho
        v = q_col[:, 2] / rho
        e_int = q_col[:, 3] / rho - 0.5 * (u**2 + v**2)
        p = self.gm1 * rho * np.maximum(e_int, 1e-30)

        # Reconstruct in y-direction: v is the normal velocity
        rho_l, rho_r = self.recon.reconstruct(rho)
        v_l, v_r = self.recon.reconstruct(v)
        p_l, p_r = self.recon.reconstruct(p)

        rho_l = np.maximum(rho_l, 1e-10)
        rho_r = np.maximum(rho_r, 1e-10)
        p_l = np.maximum(p_l, 1e-10)
        p_r = np.maximum(p_r, 1e-10)

        flux = self._flux_1d.compute(rho_l, v_l, p_l, rho_r, v_r, p_r)

        # Transpose for y-direction
        u_avg = 0.5 * (
            q_col[:-1, 1] / np.maximum(q_col[:-1, 0], 1e-30)
            + q_col[1:, 1] / np.maximum(q_col[1:, 0], 1e-30)
        )
        flux_4 = np.zeros((flux.shape[0], 4))
        flux_4[:, 0] = flux[:, 0]  # Mass
        flux_4[:, 1] = flux[:, 0] * u_avg  # x-momentum transport
        flux_4[:, 2] = flux[:, 1]  # y-momentum
        flux_4[:, 3] = flux[:, 2]  # Energy

        rhs = np.zeros((ny_int, 4), dtype=np.float64)
        for k in range(4):
            gi = flux_4[ng - 1 : ng - 1 + ny_int + 1, k]
            rhs[:, k] = -(gi[1:] - gi[:-1]) / dy

        return rhs

    def _compute_dt(self, q: NDArray[np.float64], step: int) -> float:
        """Compute timestep from CFL condition."""
        si, sj = self.mesh.interior_slice
        rho = np.maximum(q[si, sj, 0], 1e-30)
        u = q[si, sj, 1] / rho
        v = q[si, sj, 2] / rho
        e_int = q[si, sj, 3] / rho - 0.5 * (u**2 + v**2)
        p = self.gm1 * rho * np.maximum(e_int, 1e-30)
        a = np.sqrt(self.gamma * p / rho)

        ws = CFLController.wave_speed(u, v, a)
        return self.cfl_ctrl.compute_timestep(ws, self.mesh.metrics.dx, self.mesh.metrics.dy, step)

    def _apply_bcs(self, q: NDArray[np.float64]) -> None:
        """Apply all boundary conditions."""
        for face, bc in self.bcs.items():
            bc.apply(q, face, self.ng, self.gamma)

    def _enforce_positivity(self, q: NDArray[np.float64]) -> None:
        """Clamp density and pressure to positive values."""
        q[:, :, 0] = np.maximum(q[:, :, 0], 1e-10)
        rho = q[:, :, 0]
        u = q[:, :, 1] / rho
        v = q[:, :, 2] / rho
        e_int = q[:, :, 3] / rho - 0.5 * (u**2 + v**2)
        neg = e_int < 0
        if np.any(neg):
            q[neg, 3] = rho[neg] * (1e-10 + 0.5 * (u[neg] ** 2 + v[neg] ** 2))

    def _extract_result(
        self,
        q: NDArray[np.float64],
        t: float,
        step: int,
    ) -> Euler2DResult:
        """Extract primitives from final solution."""
        si, sj = self.mesh.interior_slice
        qi = q[si, sj, :]
        rho = np.maximum(qi[:, :, 0], 1e-30)
        u = qi[:, :, 1] / rho
        v = qi[:, :, 2] / rho
        e_int = qi[:, :, 3] / rho - 0.5 * (u**2 + v**2)
        p = self.gm1 * rho * np.maximum(e_int, 1e-30)
        a = np.sqrt(self.gamma * p / rho)
        mach = np.sqrt(u**2 + v**2) / a

        return Euler2DResult(
            density=rho,
            velocity_x=u,
            velocity_y=v,
            pressure=p,
            mach=mach,
            time=t,
            n_steps=step,
            final_residual=self.residual_monitor.history.last_l2,
            converged=self.residual_monitor.is_converged(self.config.residual_tolerance),
        )
