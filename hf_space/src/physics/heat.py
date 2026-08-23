"""Heat Equation Solver for Synthetic Physics Data Generation.

Solves the 2D Heat equation: ∂u/∂t = α∇²u
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.physics.solver import DiffEqSolver, PhysicsSample, generate_random_field

logger = structlog.get_logger(__name__)


@dataclass
class HeatSample(PhysicsSample[NDArray[np.float32], NDArray[np.float32]]):
    """A single Heat equation sample.

    Attributes:
        input_field: Initial temperature distribution u(t=0).
        output_field: Final temperature distribution u(t=T).

    """

    pass


class HeatSolver(DiffEqSolver[NDArray[np.float32], NDArray[np.float32]]):
    """Solves the 2D Heat equation on a grid."""

    def __init__(
        self,
        alpha: float = 0.01,
        time_step: float = 0.001,
        total_time: float = 1.0,
        resolution: int = 32,
    ) -> None:
        """Initialize Heat solver.

        Args:
            alpha: Thermal diffusivity.
            time_step: Time step for simulation (dt).
            total_time: Total simulation time (T).
            resolution: Grid resolution.

        """
        super().__init__(resolution=resolution)
        self.alpha = alpha
        self.time_step = time_step
        self.total_time = total_time

    def solve(self, input_field: NDArray[np.float32]) -> NDArray[np.float32]:
        """Solve heat equation from initial condition u0."""
        logger.debug(
            "heat_solve_start",
            resolution=input_field.shape,
            alpha=self.alpha,
            total_time=self.total_time,
        )

        # Use spectral method (FFT) for fast solving with periodic BCs
        # u(x, t) = IFFT( FFT(u0) * exp(-alpha * k^2 * t) )

        u0 = input_field
        n = u0.shape[0]

        # Wavenumbers
        freqs = np.fft.fftfreq(n) * n
        kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
        k2 = kx**2 + ky**2

        # Decay factor in Fourier domain
        # exp(-alpha * k^2 * T)
        # Note: We assume domain is [0, 1]^2, so k needs scaling by 2pi?
        # Standard FFT assumes [0, 2pi] or integer modes over domain.
        # If domain is [0, 1], modes are 2*pi*k.

        k2_scaled = k2 * (2 * np.pi) ** 2
        decay = np.exp(-self.alpha * k2_scaled * self.total_time)

        # Solve
        u0_hat = np.fft.fft2(u0)
        u_final_hat = u0_hat * decay
        u_final = np.real(np.fft.ifft2(u_final_hat))

        logger.debug(
            "heat_solve_complete",
            output_range=(float(u_final.min()), float(u_final.max())),
        )

        return u_final.astype(np.float32)

    def generate_sample(self, seed: int | None = None) -> HeatSample:
        """Generate a random Heat sample."""
        resolution = self.resolution

        # Initial condition: Random field
        u0 = generate_random_field(
            grid_size=resolution,
            n_sources=None,  # Continuous field often strictly better for heat
            smooth=True,
            seed=seed,
        )

        # Solve
        u_final = self.solve(u0)

        coords = self._get_grid_coords(resolution)

        return HeatSample(
            input_field=u0.flatten().astype(np.float32),
            output_field=u_final.flatten().astype(np.float32),
            coords=coords,
            grid_size=resolution,
            metadata={"alpha": self.alpha, "total_time": self.total_time},
        )
