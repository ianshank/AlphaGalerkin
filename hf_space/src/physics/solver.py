"""Base interface for differential equation solvers.

Provides a standardized API for generating synthetic data for various
physics problems (Poisson, Heat, Darcy, Elasticity).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


@dataclass
class PhysicsSample(Generic[T_Input, T_Output]):
    """A generic physics sample (input -> output).

    Attributes:
        input_field: The driving field (e.g., charge, permeability, initial state).
        output_field: The solution field (e.g., potential, pressure, final state).
        coords: Grid coordinates (N, D).
        grid_size: Resolution of the grid.
        metadata: specific problem parameters (e.g., time t, viscosity).

    """

    input_field: T_Input
    output_field: T_Output
    coords: NDArray[np.float32]
    grid_size: int
    metadata: dict[str, float] | None = None


class DiffEqSolver(abc.ABC, Generic[T_Input, T_Output]):
    """Abstract base class for differential equation solvers."""

    def __init__(self, resolution: int = 32) -> None:
        """Initialize solver.

        Args:
            resolution: Default grid resolution.

        """
        self.resolution = resolution

    @abc.abstractmethod
    def solve(self, input_field: T_Input) -> T_Output:
        """Solve the differential equation for the given input.

        Args:
            input_field: The forcing term or coefficients.

        Returns:
            The solution field.

        """
        pass

    @abc.abstractmethod
    def generate_sample(self, seed: int | None = None) -> PhysicsSample[T_Input, T_Output]:
        """Generate a random sample pair (input, solution).

        Args:
            seed: Random seed for reproducibility.

        Returns:
            A generic PhysicsSample.

        """
        pass

    def _get_grid_coords(self, grid_size: int) -> NDArray[np.float32]:
        """Generate normalized grid coordinates [0, 1]^2.

        Args:
            grid_size: Resolution.

        Returns:
            (N, 2) array of coordinates.

        """
        x = np.linspace(0, 1, grid_size, dtype=np.float32)
        y = np.linspace(0, 1, grid_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        return np.stack([xx.flatten(), yy.flatten()], axis=-1)


def generate_random_field(
    grid_size: int,
    n_sources: int | None = None,
    source_std: float = 1.0,
    smooth: bool = True,
    seed: int | None = None,
) -> NDArray[np.float32]:
    """Generate a random 2D field.

    Args:
        grid_size: Size of the grid (N x N).
        n_sources: Number of point sources (None for continuous field).
        source_std: Standard deviation of source magnitudes.
        smooth: Apply Gaussian smoothing.
        seed: Random seed.

    Returns:
        Field array of shape (grid_size, grid_size).

    """
    rng = np.random.default_rng(seed)

    if n_sources is not None:
        # Sparse point sources
        field = np.zeros((grid_size, grid_size), dtype=np.float32)
        positions = rng.integers(0, grid_size, size=(n_sources, 2))
        magnitudes = rng.normal(0, source_std, size=n_sources)

        for (i, j), mag in zip(positions, magnitudes):
            field[i, j] += mag
    else:
        # Continuous random field
        field = rng.normal(0, source_std, size=(grid_size, grid_size)).astype(np.float32)

    if smooth:
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter

        sigma = max(1, grid_size / 10)
        field = gaussian_filter(field, sigma=sigma).astype(np.float32)

    return field
