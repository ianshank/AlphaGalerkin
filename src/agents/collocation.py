"""Adaptive collocation point allocation strategies.

Provides allocation strategies that distribute collocation points
across the domain, with adaptive strategies concentrating points
where the PDE residual is high.

Example:
    from src.agents.collocation import create_collocation_allocator
    from src.agents.config import CollocationConfig, CollocationStrategy

    config = CollocationConfig(
        name="adaptive",
        strategy=CollocationStrategy.ADAPTIVE,
        n_points=500,
    )
    allocator = create_collocation_allocator(config)
    points = allocator.allocate(
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
        dim=2,
    )

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.templates.logging import create_logger_class
from src.templates.registry import create_registry

if TYPE_CHECKING:
    from src.agents.config import CollocationConfig

CollocationLogger = create_logger_class("Collocation")


class CollocationAllocator(ABC):
    """Abstract base for collocation point allocation strategies.

    Args:
        config: Allocation configuration.

    """

    def __init__(self, config: CollocationConfig) -> None:
        self.config = config
        self._logger = CollocationLogger("allocator")

    @abstractmethod
    def allocate(
        self,
        domain_min: list[float],
        domain_max: list[float],
        dim: int,
        residuals: NDArray[np.float32] | None = None,
        coords: NDArray[np.float32] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Allocate collocation points within the domain.

        Args:
            domain_min: Minimum coordinates of the domain.
            domain_max: Maximum coordinates of the domain.
            dim: Spatial dimension.
            residuals: Optional residual values at existing points.
            coords: Optional existing point coordinates.
            seed: Optional random seed for reproducibility.

        Returns:
            Array of shape (n_points, dim) with allocated coordinates.

        """

    def reallocate(
        self,
        current_coords: NDArray[np.float32],
        residuals: NDArray[np.float32],
        domain_min: list[float],
        domain_max: list[float],
    ) -> NDArray[np.float32]:
        """Reallocate points using residual information.

        Convenience wrapper that passes existing coordinates and
        residuals to ``allocate``.

        Args:
            current_coords: Existing point coordinates (N, dim).
            residuals: Residual values at existing points (N,).
            domain_min: Minimum coordinates of the domain.
            domain_max: Maximum coordinates of the domain.

        Returns:
            New point coordinates (n_points, dim).

        """
        dim = current_coords.shape[1] if current_coords.ndim > 1 else 1
        return self.allocate(
            domain_min=domain_min,
            domain_max=domain_max,
            dim=dim,
            residuals=residuals,
            coords=current_coords,
            seed=self.config.seed,
        )


CollocationRegistry, register_collocation = create_registry(
    "Collocation",
    CollocationAllocator,
)


@register_collocation("uniform")
class UniformAllocator(CollocationAllocator):
    """Uniform grid-based collocation point allocation."""

    def allocate(
        self,
        domain_min: list[float],
        domain_max: list[float],
        dim: int,
        residuals: NDArray[np.float32] | None = None,
        coords: NDArray[np.float32] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate uniformly spaced points via random sampling."""
        rng = np.random.default_rng(seed)
        points = rng.uniform(
            low=domain_min,
            high=domain_max,
            size=(self.config.n_points, dim),
        ).astype(np.float32)
        return points


@register_collocation("adaptive")
class AdaptiveAllocator(CollocationAllocator):
    """Residual-weighted adaptive collocation allocation.

    Places more points in regions where the residual magnitude is high.
    Falls back to uniform allocation when no residual data is available.
    """

    def allocate(
        self,
        domain_min: list[float],
        domain_max: list[float],
        dim: int,
        residuals: NDArray[np.float32] | None = None,
        coords: NDArray[np.float32] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Allocate points weighted by residual magnitude."""
        rng = np.random.default_rng(seed)

        if residuals is None or coords is None:
            return UniformAllocator(self.config).allocate(domain_min, domain_max, dim, seed=seed)

        abs_residuals = np.abs(residuals).astype(np.float64)
        total = abs_residuals.sum()
        if total < 1e-12:
            return UniformAllocator(self.config).allocate(domain_min, domain_max, dim, seed=seed)

        weights = abs_residuals / total

        n_adaptive = int(self.config.n_points * self.config.adaptation_rate)
        n_uniform = self.config.n_points - n_adaptive

        adaptive_indices = rng.choice(len(coords), size=n_adaptive, replace=True, p=weights)
        perturbation_scale = (
            np.array(
                [domain_max[d] - domain_min[d] for d in range(dim)],
                dtype=np.float32,
            )
            * self.config.perturbation_fraction
        )
        adaptive_points = coords[adaptive_indices].copy()
        adaptive_points += rng.normal(0, perturbation_scale, size=adaptive_points.shape).astype(
            np.float32
        )

        for d in range(dim):
            adaptive_points[:, d] = np.clip(adaptive_points[:, d], domain_min[d], domain_max[d])

        uniform_points = UniformAllocator(self.config).allocate(
            domain_min, domain_max, dim, seed=seed
        )[:n_uniform]

        return np.concatenate([adaptive_points, uniform_points], axis=0).astype(np.float32)


@register_collocation("importance_weighted")
class ImportanceWeightedAllocator(CollocationAllocator):
    """Importance-weighted allocation proportional to |residual|^p."""

    def allocate(
        self,
        domain_min: list[float],
        domain_max: list[float],
        dim: int,
        residuals: NDArray[np.float32] | None = None,
        coords: NDArray[np.float32] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Allocate points proportional to |residual|^p."""
        rng = np.random.default_rng(seed)

        if residuals is None or coords is None:
            return UniformAllocator(self.config).allocate(domain_min, domain_max, dim, seed=seed)

        importance = np.abs(residuals).astype(np.float64) ** self.config.importance_exponent
        total = importance.sum()
        if total < 1e-12:
            return UniformAllocator(self.config).allocate(domain_min, domain_max, dim, seed=seed)

        weights = importance / total
        indices = rng.choice(len(coords), size=self.config.n_points, replace=True, p=weights)
        return coords[indices].copy().astype(np.float32)


@register_collocation("error_guided")
class ErrorGuidedAllocator(CollocationAllocator):
    """Error-guided refinement that adds points near high-error regions.

    Keeps a fraction of existing points and adds new points concentrated
    near high-error locations via local perturbation.
    """

    def allocate(
        self,
        domain_min: list[float],
        domain_max: list[float],
        dim: int,
        residuals: NDArray[np.float32] | None = None,
        coords: NDArray[np.float32] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Refine existing points near high-error regions."""
        rng = np.random.default_rng(seed)

        if residuals is None or coords is None:
            return UniformAllocator(self.config).allocate(domain_min, domain_max, dim, seed=seed)

        abs_res = np.abs(residuals)
        threshold = np.percentile(abs_res, 100.0 * (1.0 - self.config.adaptation_rate))
        high_error_mask = abs_res >= threshold
        high_error_coords = coords[high_error_mask]

        if len(high_error_coords) == 0:
            return UniformAllocator(self.config).allocate(domain_min, domain_max, dim, seed=seed)

        n_refined = min(
            self.config.n_points,
            len(high_error_coords) * self.config.refined_oversampling_factor,
        )
        n_uniform = self.config.n_points - n_refined

        indices = rng.choice(len(high_error_coords), size=n_refined, replace=True)
        perturbation_scale = (
            np.array(
                [domain_max[d] - domain_min[d] for d in range(dim)],
                dtype=np.float32,
            )
            * self.config.perturbation_fraction
        )
        refined_points = high_error_coords[indices].copy()
        refined_points += rng.normal(0, perturbation_scale, size=refined_points.shape).astype(
            np.float32
        )

        for d in range(dim):
            refined_points[:, d] = np.clip(refined_points[:, d], domain_min[d], domain_max[d])

        if n_uniform > 0:
            uniform_points = UniformAllocator(self.config).allocate(
                domain_min, domain_max, dim, seed=seed
            )[:n_uniform]
            return np.concatenate([refined_points, uniform_points], axis=0).astype(np.float32)

        return refined_points[: self.config.n_points].astype(np.float32)


def create_collocation_allocator(config: CollocationConfig) -> CollocationAllocator:
    """Factory function to create a collocation allocator from config.

    Args:
        config: Collocation configuration specifying strategy.

    Returns:
        Configured allocator instance.

    Raises:
        KeyError: If the strategy is not registered.

    """
    allocator_cls = CollocationRegistry().get_or_raise(config.strategy.value)
    return allocator_cls(config)
