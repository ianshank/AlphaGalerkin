"""Poisson Equation Solver for Synthetic Physics Data Generation.

Solves the 2D Poisson equation: ∇²φ = ρ
where φ is the potential (influence field) and ρ is the charge density.

This provides ground truth data for validating the neural operator's
ability to approximate the Green's function of the Laplacian.

The Green's function G(x,y) satisfies: ∇²G = δ(x-y)
Solution: φ(x) = ∫ G(x,y) ρ(y) dy

For the 2D Laplacian with zero boundary conditions:
G(x,y) ~ -log|x-y| / (2π)  (in free space)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


@dataclass
class PoissonSample:
    """A single Poisson equation sample.

    Attributes:
        coords: Grid coordinates (N, 2) normalized to [0, 1].
        charges: Charge density at each point (N,).
        potential: Ground truth potential/influence field (N,).
        grid_size: Original grid resolution.

    """

    coords: NDArray[np.float32]  # (N, 2)
    charges: NDArray[np.float32]  # (N,)
    potential: NDArray[np.float32]  # (N,)
    grid_size: int


class PoissonSolver:
    """Solves the 2D Poisson equation on a grid.

    Uses finite difference discretization with Dirichlet boundary conditions.
    The solution is computed using the discrete Green's function approach
    or iterative methods depending on grid size.
    """

    def __init__(
        self,
        boundary_value: float = 0.0,
        use_spectral: bool = True,
        regularization: float = 1e-6,
    ) -> None:
        """Initialize Poisson solver.

        Args:
            boundary_value: Value at domain boundaries (Dirichlet BC).
            use_spectral: Use spectral method (FFT) for faster solving.
            regularization: Small value to stabilize division.

        """
        self.boundary_value = boundary_value
        self.use_spectral = use_spectral
        self.regularization = regularization

    def solve(
        self,
        charges: NDArray[np.float32],
        grid_size: int | None = None,
    ) -> NDArray[np.float32]:
        """Solve Poisson equation for given charge distribution.

        Args:
            charges: Charge density, either (N, N) grid or (N,) flattened.
            grid_size: Grid size if charges is flattened.

        Returns:
            Potential field with same shape as input.

        """
        # Handle input shape
        if charges.ndim == 1:
            if grid_size is None:
                grid_size = int(np.sqrt(len(charges)))
            charges_2d = charges.reshape(grid_size, grid_size)
        else:
            charges_2d = charges
            grid_size = charges_2d.shape[0]

        if self.use_spectral:
            potential_2d = self._solve_spectral(charges_2d)
        else:
            potential_2d = self._solve_iterative(charges_2d)

        # Return in same shape as input
        if charges.ndim == 1:
            return potential_2d.flatten().astype(np.float32)
        return potential_2d.astype(np.float32)

    def _solve_spectral(
        self,
        charges: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Solve using spectral method (DST - Discrete Sine Transform).

        For Poisson with zero Dirichlet BC, use DST Type I.
        The eigenvalues of the discrete Laplacian are known analytically.
        """
        n = charges.shape[0]
        h = 1.0 / (n + 1)  # Grid spacing

        # Compute DST-I of the right-hand side
        # DST-I is its own inverse (up to normalization)
        rhs_hat = self._dst2d(charges)

        # Eigenvalues of the discrete Laplacian with DST basis
        i_indices = np.arange(1, n + 1)
        j_indices = np.arange(1, n + 1)
        eigenvalues = (
            2 * (np.cos(np.pi * i_indices / (n + 1)) - 1)[:, None]
            + 2 * (np.cos(np.pi * j_indices / (n + 1)) - 1)[None, :]
        ) / (h * h)

        # Avoid division by zero
        eigenvalues = np.where(
            np.abs(eigenvalues) < self.regularization,
            self.regularization,
            eigenvalues,
        )

        # Solve in spectral domain
        potential_hat = rhs_hat / eigenvalues

        # Inverse transform
        potential = self._idst2d(potential_hat)

        return potential

    def _dst2d(self, x: NDArray) -> NDArray:
        """2D Discrete Sine Transform (Type I)."""
        # DST-I can be computed via FFT of extended signal
        n = x.shape[0]

        # Extend to 2(n+1) and use FFT
        # For DST-I: x_ext[k] = 0, x[0:n], 0, -x[n-1::-1]
        x_ext = np.zeros((2 * (n + 1), 2 * (n + 1)))
        x_ext[1 : n + 1, 1 : n + 1] = x
        x_ext[n + 2 : 2 * n + 2, 1 : n + 1] = -x[::-1, :]
        x_ext[1 : n + 1, n + 2 : 2 * n + 2] = -x[:, ::-1]
        x_ext[n + 2 : 2 * n + 2, n + 2 : 2 * n + 2] = x[::-1, ::-1]

        # FFT and extract imaginary part
        fft_result = np.fft.fft2(x_ext)
        dst = -fft_result[1 : n + 1, 1 : n + 1].imag / 2

        return dst

    def _idst2d(self, x: NDArray) -> NDArray:
        """Inverse 2D DST (Type I is self-inverse up to normalization)."""
        n = x.shape[0]
        # DST-I is its own inverse with factor 2/(n+1)
        return self._dst2d(x) * (2.0 / (n + 1)) ** 2

    def _solve_iterative(
        self,
        charges: NDArray[np.float32],
        max_iter: int = 10000,
        tol: float = 1e-6,
    ) -> NDArray[np.float64]:
        """Solve using Gauss-Seidel iteration (fallback method)."""
        n = charges.shape[0]
        h = 1.0 / (n + 1)
        h2 = h * h

        # Initialize potential
        potential = np.zeros_like(charges, dtype=np.float64)

        for iteration in range(max_iter):
            potential_old = potential.copy()

            # Gauss-Seidel update
            for i in range(n):
                for j in range(n):
                    # Neighbors (with zero BC at boundaries)
                    left = potential[i, j - 1] if j > 0 else self.boundary_value
                    right = potential[i, j + 1] if j < n - 1 else self.boundary_value
                    down = potential[i - 1, j] if i > 0 else self.boundary_value
                    up = potential[i + 1, j] if i < n - 1 else self.boundary_value

                    potential[i, j] = (left + right + down + up - h2 * charges[i, j]) / 4

            # Check convergence
            diff = np.max(np.abs(potential - potential_old))
            if diff < tol:
                logger.debug("poisson_converged", iterations=iteration + 1)
                break

        return potential


def generate_random_charges(
    grid_size: int,
    n_charges: int | None = None,
    charge_std: float = 1.0,
    smooth: bool = True,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Generate random charge distribution.

    Args:
        grid_size: Size of the grid (N x N).
        n_charges: Number of point charges (None for continuous field).
        charge_std: Standard deviation of charge magnitudes.
        smooth: Apply Gaussian smoothing for smoother fields.
        seed: Random seed for reproducibility.

    Returns:
        Charge density array of shape (grid_size, grid_size).

    """
    rng = np.random.default_rng(seed)

    if n_charges is not None:
        # Sparse point charges
        charges = np.zeros((grid_size, grid_size), dtype=np.float32)
        positions = rng.integers(0, grid_size, size=(n_charges, 2))
        magnitudes = rng.normal(0, charge_std, size=n_charges)

        for (i, j), mag in zip(positions, magnitudes, strict=True):
            charges[i, j] += mag
    else:
        # Continuous random field
        charges = rng.normal(0, charge_std, size=(grid_size, grid_size)).astype(
            np.float32
        )

    if smooth:
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter

        sigma = max(1, grid_size / 10)
        charges = gaussian_filter(charges, sigma=sigma).astype(np.float32)

    return charges


def generate_influence_field(
    grid_size: int,
    n_charges: int | None = None,
    charge_std: float = 1.0,
    seed: int | None = None,
) -> PoissonSample:
    """Generate a complete Poisson sample with ground truth.

    Args:
        grid_size: Size of the grid.
        n_charges: Number of point charges (None for continuous).
        charge_std: Standard deviation of charges.
        seed: Random seed.

    Returns:
        PoissonSample with coordinates, charges, and potential.

    """
    solver = PoissonSolver()

    # Generate charges
    charges_2d = generate_random_charges(
        grid_size=grid_size,
        n_charges=n_charges,
        charge_std=charge_std,
        seed=seed,
    )

    # Solve for potential
    potential_2d = solver.solve(charges_2d)

    # Create coordinate grid normalized to [0, 1]
    x = np.linspace(0, 1, grid_size, dtype=np.float32)
    y = np.linspace(0, 1, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Flatten to (N, 2) coordinates
    coords = np.stack([xx.flatten(), yy.flatten()], axis=-1).astype(np.float32)

    return PoissonSample(
        coords=coords,
        charges=charges_2d.flatten().astype(np.float32),
        potential=potential_2d.flatten().astype(np.float32),
        grid_size=grid_size,
    )


class PoissonDataset:
    """Dataset of Poisson equation samples for training/evaluation.

    Generates samples on-the-fly or from a cached set.
    Supports different grid sizes for zero-shot transfer testing.
    """

    def __init__(
        self,
        grid_size: int = 9,
        n_samples: int = 1000,
        n_charges: int | None = 5,
        charge_std: float = 1.0,
        cache_samples: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialize dataset.

        Args:
            grid_size: Grid resolution (N x N).
            n_samples: Number of samples to generate.
            n_charges: Point charges per sample (None for continuous).
            charge_std: Charge magnitude standard deviation.
            cache_samples: Whether to cache generated samples.
            seed: Base random seed.

        """
        self.grid_size = grid_size
        self.n_samples = n_samples
        self.n_charges = n_charges
        self.charge_std = charge_std
        self.cache_samples = cache_samples
        self.seed = seed

        self._cache: list[PoissonSample] | None = None
        if cache_samples:
            self._generate_cache()

    def _generate_cache(self) -> None:
        """Pre-generate and cache all samples."""
        logger.info(
            "generating_poisson_dataset",
            grid_size=self.grid_size,
            n_samples=self.n_samples,
        )

        self._cache = []
        for i in range(self.n_samples):
            sample = generate_influence_field(
                grid_size=self.grid_size,
                n_charges=self.n_charges,
                charge_std=self.charge_std,
                seed=self.seed + i,
            )
            self._cache.append(sample)

    def __len__(self) -> int:
        """Get dataset size."""
        return self.n_samples

    def __getitem__(self, idx: int) -> PoissonSample:
        """Get sample by index."""
        if self._cache is not None:
            return self._cache[idx]

        return generate_influence_field(
            grid_size=self.grid_size,
            n_charges=self.n_charges,
            charge_std=self.charge_std,
            seed=self.seed + idx,
        )

    def __iter__(self) -> Iterator[PoissonSample]:
        """Iterate over samples."""
        for i in range(self.n_samples):
            yield self[i]

    def get_statistics(self) -> dict[str, float]:
        """Compute dataset statistics."""
        if self._cache is None:
            self._generate_cache()

        assert self._cache is not None

        all_potentials = np.concatenate([s.potential for s in self._cache])
        all_charges = np.concatenate([s.charges for s in self._cache])

        return {
            "potential_mean": float(np.mean(all_potentials)),
            "potential_std": float(np.std(all_potentials)),
            "potential_min": float(np.min(all_potentials)),
            "potential_max": float(np.max(all_potentials)),
            "charge_mean": float(np.mean(all_charges)),
            "charge_std": float(np.std(all_charges)),
        }


def create_poisson_dataloader(
    grid_size: int = 9,
    n_samples: int = 1000,
    batch_size: int = 32,
    shuffle: bool = True,
    **dataset_kwargs: object,
) -> Iterator[dict[str, NDArray[np.float32]]]:
    """Create a data loader for Poisson samples.

    Yields batches as dictionaries compatible with neural network training.

    Args:
        grid_size: Grid resolution.
        n_samples: Total samples.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        **dataset_kwargs: Additional PoissonDataset arguments.

    Yields:
        Batches with keys: coords, charges, potential, grid_size.

    """
    dataset = PoissonDataset(grid_size=grid_size, n_samples=n_samples, **dataset_kwargs)

    indices = list(range(len(dataset)))
    if shuffle:
        import random

        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        yield {
            "coords": np.stack([s.coords for s in samples]),
            "charges": np.stack([s.charges for s in samples]),
            "potential": np.stack([s.potential for s in samples]),
            "grid_size": samples[0].grid_size,
        }
