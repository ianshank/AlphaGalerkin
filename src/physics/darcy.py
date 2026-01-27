"""Darcy Flow Solver for Synthetic Physics Data Generation.

Solves the steady-state Darcy flow equation:
-∇⋅(a(x) ∇u(x)) = f(x)
where a(x) is the permeability field, u(x) is the pressure, and f(x) is the forcing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.physics.solver import DiffEqSolver, PhysicsSample, generate_random_field

logger = structlog.get_logger(__name__)


@dataclass
class DarcySample(PhysicsSample[NDArray[np.float32], NDArray[np.float32]]):
    """A single Darcy flow sample.

    Attributes:
        input_field: Permeability field a(x).
        output_field: Pressure field u(x).
    """

    pass


class DarcyFlowSolver(DiffEqSolver[NDArray[np.float32], NDArray[np.float32]]):
    """Solves Darcy flow equation on a 2D grid."""

    def __init__(
        self,
        forcing: float = 1.0,
        resolution: int = 32,
    ) -> None:
        """Initialize Darcy solver.

        Args:
            forcing: Constant forcing term f(x).
            resolution: Grid resolution.
        """
        super().__init__(resolution=resolution)
        self.forcing = forcing

    def solve(self, input_field: NDArray[np.float32]) -> NDArray[np.float32]:
        """Solve Darcy flow for permeability map a(x).
        
        Uses finite difference method.
        -∇⋅(a ∇u) = f
        
        Discretized on grid.
        """
        logger.debug(
            "darcy_solve_start",
            resolution=input_field.shape,
            forcing=self.forcing,
            permeability_range=(float(input_field.min()), float(input_field.max())),
        )
        
        permeability = input_field
        n = permeability.shape[0]
        h = 1.0 / (n - 1)
        
        # Flattened grid size
        N = n * n
        
        # Build sparse matrix A
        # We'll use scipy.sparse for efficiency
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
        
        A = lil_matrix((N, N))
        b = np.full(N, self.forcing * h * h)
        
        # Helper to get index
        def idx(i, j):
            return i * n + j
            
        for i in range(n):
            for j in range(n):
                k = idx(i, j)
                
                # Dirichlet BCs: u = 0 on boundary
                if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                    A[k, k] = 1.0
                    b[k] = 0.0
                    continue
                
                # Harmonic average of permeability at interfaces
                # a_left = 2 / (1/a[i,j] + 1/a[i,j-1])
                # Simplified: arithmetic average for demo speed
                a_curr = permeability[i, j]
                a_left = (permeability[i, j-1] + a_curr) / 2
                a_right = (permeability[i, j+1] + a_curr) / 2
                a_top = (permeability[i-1, j] + a_curr) / 2
                a_bottom = (permeability[i+1, j] + a_curr) / 2
                
                A[k, k] = a_left + a_right + a_top + a_bottom
                A[k, idx(i, j-1)] = -a_left
                A[k, idx(i, j+1)] = -a_right
                A[k, idx(i-1, j)] = -a_top
                A[k, idx(i+1, j)] = -a_bottom
                
        # Solve linear system
        A_csr = A.tocsr()
        u = spsolve(A_csr, b)
        
        logger.debug(
            "darcy_solve_complete",
            output_range=(float(u.min()), float(u.max())),
        )

        return u.reshape(n, n).astype(np.float32)

    def generate_sample(self, seed: int | None = None) -> DarcySample:
        """Generate a random Darcy flow sample."""
        resolution = self.resolution
        
        # Permeability: Random field, usually strictly positive
        # Log-normal distribution is common for permeability
        log_k = generate_random_field(
            grid_size=resolution,
            smooth=True,
            seed=seed,
            source_std=2.0
        )
        permeability = np.exp(log_k)
        
        # Clip to ensure numerical stability
        permeability = np.clip(permeability, 0.1, 100.0)
        
        u = self.solve(permeability)
        
        coords = self._get_grid_coords(resolution)
        
        return DarcySample(
            input_field=permeability.flatten().astype(np.float32),
            output_field=u.flatten().astype(np.float32),
            coords=coords,
            grid_size=resolution,
             metadata={
                "forcing": self.forcing
            }
        )

    def generate_batch(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> list[DarcySample]:
        """Generate a batch of samples efficiently.

        Args:
            n_samples: Number of samples to generate.
            seed: Base random seed.

        Returns:
            List of DarcySample objects.
        """
        logger.info(
            "darcy_batch_generation_start",
            n_samples=n_samples,
            resolution=self.resolution,
        )
        
        samples = []
        for i in range(n_samples):
            sample_seed = seed + i if seed is not None else None
            samples.append(self.generate_sample(seed=sample_seed))
            
            if (i + 1) % 100 == 0:
                logger.debug("batch_progress", completed=i + 1, total=n_samples)
        
        logger.info("darcy_batch_generation_complete", n_samples=n_samples)
        return samples
