"""Linear Elasticity Solver for Synthetic Physics Data Generation.

Solves the static linear elasticity equation:
∇ ⋅ σ + F = 0
where σ is the stress tensor and F is the external body force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.physics.solver import DiffEqSolver, PhysicsSample, generate_random_field

logger = structlog.get_logger(__name__)


@dataclass
class ElasticitySample(PhysicsSample[NDArray[np.float32], NDArray[np.float32]]):
    """A single Elasticity sample.

    Attributes:
        input_field: Body force F(x) = (Fx, Fy) -> Shape (N*N, 2).
        output_field: Displacement u(x) = (ux, uy) -> Shape (N*N, 2).
    """

    pass


class ElasticitySolver(DiffEqSolver[NDArray[np.float32], NDArray[np.float32]]):
    """Solves Linear Elasticity on a 2D grid."""

    def __init__(
        self,
        young_modulus: float = 1.0,
        poisson_ratio: float = 0.3,
        resolution: int = 32,
    ) -> None:
        """Initialize Elasticity solver.

        Args:
            young_modulus: E.
            poisson_ratio: ν.
            resolution: Grid resolution.
        """
        super().__init__(resolution=resolution)
        self.E = young_modulus
        self.nu = poisson_ratio
        
        # Lamé parameters
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))

    def solve(self, input_field: NDArray[np.float32]) -> NDArray[np.float32]:
        """Solve for displacement given body force F.
        
        Input: F (N*N, 2)
        Output: u (N*N, 2)
        
        Uses spectral method assuming Periodic BCs for simplicity in this demo,
        or we can implement FD/FEM. Periodic is much faster and cleaner for "resolution independence" demos.
        """
        resolution = int(np.sqrt(input_field.shape[0]))
        
        logger.debug(
            "elasticity_solve_start",
            resolution=resolution,
            E=self.E,
            nu=self.nu,
        )
        
        F = input_field.reshape(resolution, resolution, 2)
        Fx, Fy = F[..., 0], F[..., 1]
        
        n = resolution
        
        # Fourier transform of forces
        Fx_hat = np.fft.fft2(Fx)
        Fy_hat = np.fft.fft2(Fy)
        
        # Wavenumbers
        freqs = np.fft.fftfreq(n) * n * 2 * np.pi  # Scaled to domain size
        kx, ky = np.meshgrid(freqs, freqs, indexing="ij")
        
        # Avoid zero mode (rigid body motion)
        k2 = kx**2 + ky**2
        k2[0, 0] = 1.0
        
        # Navier-Cauchy equations in Fourier domain:
        # (lambda + mu) grad(div u) + mu Laplacian u + F = 0
        #
        # In k-space:
        # - (lambda + mu) k (k . u_hat) - mu |k|^2 u_hat + F_hat = 0
        # M u_hat = F_hat
        
        # We need to invert the matrix M for each k
        
        mu = self.mu
        lam = self.lam

        # Efficient vectorization?
        # M[0,0] = (lam+mu)kx*kx + mu*k2
        # M[0,1] = (lam+mu)kx*ky
        # M[1,0] = (lam+mu)ky*kx
        # M[1,1] = (lam+mu)ky*ky + mu*k2
        
        A = lam + mu
        
        M00 = A * kx * kx + mu * k2
        M01 = A * kx * ky
        M10 = A * ky * kx
        M11 = A * ky * ky + mu * k2
        
        det = M00 * M11 - M01 * M10
        
        # Handle singular zero mode
        det[0, 0] = 1.0
        
        # Inverse M * F_hat = u_hat
        # [ u0 ]   1  [ M11  -M01 ] [ F0 ]
        # [ u1 ] = --- [ -M10  M00 ] [ F1 ]
        #          det
        
        ux_hat = (M11 * Fx_hat - M01 * Fy_hat) / det
        uy_hat = (-M10 * Fx_hat + M00 * Fy_hat) / det
        
        # Zero mode: no displacement for balanced force
        ux_hat[0, 0] = 0.0
        uy_hat[0, 0] = 0.0
        
        ux = np.real(np.fft.ifft2(ux_hat))
        uy = np.real(np.fft.ifft2(uy_hat))
        
        u = np.stack([ux, uy], axis=-1)
        
        logger.debug(
            "elasticity_solve_complete",
            output_range_x=(float(ux.min()), float(ux.max())),
            output_range_y=(float(uy.min()), float(uy.max())),
        )
        
        return u.reshape(-1, 2).astype(np.float32)

    def generate_sample(self, seed: int | None = None) -> ElasticitySample:
        """Generate a random Elasticity sample."""
        resolution = self.resolution
        
        # Generate random forces
        Fx = generate_random_field(resolution, smooth=True, seed=seed)
        Fy = generate_random_field(resolution, smooth=True, seed=seed if seed is None else seed + 1)
        
        # Enforce zero mean force for periodic stability (equilibrium)
        Fx -= np.mean(Fx)
        Fy -= np.mean(Fy)
        
        F = np.stack([Fx, Fy], axis=-1).reshape(-1, 2).astype(np.float32)
        
        # Solve
        u = self.solve(F)
        
        coords = self._get_grid_coords(resolution)
        
        return ElasticitySample(
            input_field=F,
            output_field=u,
            coords=coords,
            grid_size=resolution,
            metadata={
                "E": self.E,
                "nu": self.nu
            }
        )
