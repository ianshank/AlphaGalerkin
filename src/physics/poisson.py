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

from src.physics.solver import DiffEqSolver, PhysicsSample

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Numerical-stability defaults. Surfaced as named module constants so callers
# and reviewers can reason about them without grepping the function bodies.
# These are deliberately not Pydantic Fields because PoissonSolver pre-dates
# the project's Pydantic-config pattern and is constructed positionally in
# ~15 call sites; introducing a config class would be a breaking refactor
# out of scope for the magic-number-externalization sweep. Future work:
# wrap behind PoissonSolverConfig if/when more knobs are added.
# ---------------------------------------------------------------------------

# Small additive term used to stabilize spectral-method division when the
# DST denominator approaches zero (corner Fourier modes). 1e-6 is a tiny
# fraction of typical |phi| and won't bias the solution at any meaningful
# resolution.
DEFAULT_POISSON_REGULARIZATION = 1e-6

# Convergence tolerance (max-norm update) for the Gauss-Seidel iterative
# fallback used when ``use_spectral=False``. Matches the spectral-path's
# regularization scale; lowering it past ~1e-8 hits float32 noise.
DEFAULT_GAUSS_SEIDEL_TOL = 1e-6

# Maximum Gauss-Seidel sweeps before giving up. 10_000 is generous: the
# iteration converges in ~1k sweeps on typical 32x32 grids; the cap is a
# safety net against pathological problems where every sweep makes no
# progress (e.g., a degenerate configuration or a tolerance set below
# float-noise floor). Linked to DEFAULT_GAUSS_SEIDEL_TOL — both control
# the same iterative fallback, so callers tuning one usually tune the
# other in lockstep.
DEFAULT_GAUSS_SEIDEL_MAX_ITER = 10_000


@dataclass
class PoissonSample(PhysicsSample[NDArray[np.float32], NDArray[np.float32]]):
    """A single Poisson equation sample.

    Attributes:
        input_field: Charge density (N,).
        output_field: Potential/influence field (N,).
        coords: Grid coordinates (N, 2).
        grid_size: Original grid resolution.

    """

    @property
    def charges(self) -> NDArray[np.float32]:
        """Alias for input_field for backward compatibility."""
        return self.input_field

    @property
    def potential(self) -> NDArray[np.float32]:
        """Alias for output_field for backward compatibility."""
        return self.output_field


class PoissonSolver(DiffEqSolver[NDArray[np.float32], NDArray[np.float32]]):
    """Solves the 2D Poisson equation on a grid.

    Uses finite difference discretization with Dirichlet boundary conditions.
    The solution is computed using the discrete Green's function approach
    or iterative methods depending on grid size.
    """

    def __init__(
        self,
        boundary_value: float = 0.0,
        use_spectral: bool = True,
        regularization: float = DEFAULT_POISSON_REGULARIZATION,
        resolution: int = 32,
    ) -> None:
        """Initialize Poisson solver.

        Args:
            boundary_value: Value at domain boundaries (Dirichlet BC).
            use_spectral: Use spectral method (FFT) for faster solving.
            regularization: Small value to stabilize division.
            resolution: Default grid resolution.

        """
        super().__init__(resolution=resolution)
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

        logger.debug(
            "poisson_solve_start",
            grid_size=grid_size,
            use_spectral=self.use_spectral,
            charge_range=(float(charges_2d.min()), float(charges_2d.max())),
        )

        if self.use_spectral:
            potential_2d = self._solve_spectral(charges_2d)
        else:
            potential_2d = self._solve_iterative(charges_2d)

        logger.debug(
            "poisson_solve_complete",
            potential_range=(float(potential_2d.min()), float(potential_2d.max())),
        )

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
        from scipy.fft import dstn, idstn

        n = charges.shape[0]
        h = 1.0 / (n + 1)  # Grid spacing

        # Compute DST-I of the right-hand side
        # Using scipy's DST for numerical stability
        rhs_hat = dstn(charges.astype(np.float64), type=1)

        # Eigenvalues of the discrete Laplacian with DST basis
        i_indices = np.arange(1, n + 1)
        j_indices = np.arange(1, n + 1)
        eigenvalues = (
            2 * (np.cos(np.pi * i_indices / (n + 1)) - 1)[:, None]
            + 2 * (np.cos(np.pi * j_indices / (n + 1)) - 1)[None, :]
        ) / (h * h)

        # Avoid division by zero
        n_regularized = np.sum(np.abs(eigenvalues) < self.regularization)
        eigenvalues = np.where(
            np.abs(eigenvalues) < self.regularization,
            self.regularization,
            eigenvalues,
        )

        if n_regularized > 0:
            logger.debug(
                "poisson_spectral_regularized",
                n_regularized=int(n_regularized),
                regularization=self.regularization,
            )

        # Solve in spectral domain
        potential_hat = rhs_hat / eigenvalues

        # Inverse transform with proper normalization
        # DST-I inverse has normalization factor 1/(2*(n+1))^2 which idstn handles
        potential = idstn(potential_hat, type=1)

        return potential

    def _dst2d(self, x: NDArray) -> NDArray:
        """2D Discrete Sine Transform (Type I) using scipy."""
        from scipy.fft import dstn

        return dstn(x.astype(np.float64), type=1)

    def _idst2d(self, x: NDArray) -> NDArray:
        """Inverse 2D DST (Type I) using scipy."""
        from scipy.fft import idstn

        return idstn(x.astype(np.float64), type=1)

    def _solve_iterative(
        self,
        charges: NDArray[np.float32],
        max_iter: int = DEFAULT_GAUSS_SEIDEL_MAX_ITER,
        tol: float = DEFAULT_GAUSS_SEIDEL_TOL,
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

    def generate_sample(
        self,
        seed: int | None = None,
        n_charges: int | None = None,
        charge_std: float = 1.0,
    ) -> PoissonSample:
        """Generate a random complete Poisson sample.

        Args:
            seed: Random seed.
            n_charges: Number of point charges (None for continuous).
            charge_std: Standard deviation of charges.

        Returns:
            PoissonSample with coordinates, charges, and potential.

        """
        # Define grid size (use instance resolution if not generic)
        grid_size = self.resolution

        # Generate charges
        charges_2d = generate_random_charges(
            grid_size=grid_size,
            n_charges=n_charges,
            charge_std=charge_std,
            seed=seed,
        )

        # Solve for potential
        potential_2d = self.solve(charges_2d)

        # Create coordinate grid
        coords = self._get_grid_coords(grid_size)

        return PoissonSample(
            input_field=charges_2d.flatten().astype(np.float32),
            output_field=potential_2d.flatten().astype(np.float32),
            coords=coords,
            grid_size=grid_size,
        )


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
        charges = rng.normal(0, charge_std, size=(grid_size, grid_size)).astype(np.float32)

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
        input_field=charges_2d.flatten().astype(np.float32),
        output_field=potential_2d.flatten().astype(np.float32),
        coords=coords,
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
