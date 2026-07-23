"""Generic PyTorch Dataset wrapper for physics solvers.

Provides a unified interface for generating training data from any
DiffEqSolver implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import structlog
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from src.physics.solver import DiffEqSolver, PhysicsSample

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound="PhysicsSample")


class PhysicsDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset for physics solver samples.

    Wraps any DiffEqSolver to generate training data on-demand or from cache.

    Example:
        >>> from src.physics.darcy import DarcyFlowSolver
        >>> solver = DarcyFlowSolver(resolution=16)
        >>> dataset = PhysicsDataset(solver, n_samples=1000, seed=42)
        >>> sample = dataset[0]
        >>> print(sample["input"].shape, sample["output"].shape)

    """

    def __init__(
        self,
        solver: DiffEqSolver,
        n_samples: int = 1000,
        seed: int = 42,
        cache: bool = True,
        cache_dir: Path | str | None = None,
        normalize: bool = True,
    ) -> None:
        """Initialize physics dataset.

        Args:
            solver: Any DiffEqSolver instance.
            n_samples: Number of samples to generate.
            seed: Base random seed for reproducibility.
            cache: Whether to cache generated samples.
            cache_dir: Directory for caching (None = in-memory only).
            normalize: Whether to normalize input/output fields.

        """
        self.solver = solver
        self.n_samples = n_samples
        self.seed = seed
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self._cache: list[PhysicsSample] | None = None
        self._stats: dict[str, tuple[float, float]] | None = None

        if cache:
            self._generate_cache()
            if normalize:
                self._compute_stats()

        logger.info(
            "physics_dataset_initialized",
            solver=type(solver).__name__,
            n_samples=n_samples,
            resolution=solver.resolution,
            seed=seed,
            normalize=normalize,
        )

    def _generate_cache(self) -> None:
        """Generate and cache all samples."""
        logger.debug("generating_samples", n_samples=self.n_samples)

        self._cache = []
        for i in range(self.n_samples):
            sample = self.solver.generate_sample(seed=self.seed + i)
            self._cache.append(sample)

            if (i + 1) % 100 == 0:
                logger.debug("generation_progress", completed=i + 1, total=self.n_samples)

    def _compute_stats(self) -> None:
        """Compute normalization statistics from cached data."""
        if self._cache is None:
            return

        inputs = np.stack([s.input_field for s in self._cache])
        outputs = np.stack([s.output_field for s in self._cache])

        self._stats = {
            "input_mean": float(np.mean(inputs)),
            "input_std": float(np.std(inputs)),
            "output_mean": float(np.mean(outputs)),
            "output_std": float(np.std(outputs)),
        }

        logger.debug("normalization_stats", **self._stats)

    def __len__(self) -> int:
        """Return dataset size."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample as tensors.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys: input, output, coords, grid_size.

        """
        if self._cache is not None:
            sample = self._cache[idx]
        else:
            sample = self.solver.generate_sample(seed=self.seed + idx)

        input_field = sample.input_field.astype(np.float32)
        output_field = sample.output_field.astype(np.float32)

        # Normalize if stats available
        if self.normalize and self._stats is not None:
            input_field = (input_field - self._stats["input_mean"]) / (
                self._stats["input_std"] + 1e-8
            )
            output_field = (output_field - self._stats["output_mean"]) / (
                self._stats["output_std"] + 1e-8
            )

        return {
            "input": torch.from_numpy(input_field),
            "output": torch.from_numpy(output_field),
            "coords": torch.from_numpy(sample.coords),
            "grid_size": sample.grid_size,
        }

    def get_stats(self) -> dict[str, float]:
        """Return normalization statistics."""
        if self._stats is None:
            raise ValueError(
                "Statistics not computed. Set normalize=True or call _compute_stats()."
            )
        return self._stats

    @staticmethod
    def create_splits(
        solver: DiffEqSolver,
        n_train: int = 1000,
        n_val: int = 100,
        n_test: int = 100,
        seed: int = 42,
        **kwargs,
    ) -> tuple[PhysicsDataset, PhysicsDataset, PhysicsDataset]:
        """Create train/val/test splits with disjoint seeds.

        Args:
            solver: Solver instance.
            n_train: Training samples.
            n_val: Validation samples.
            n_test: Test samples.
            seed: Base seed.
            **kwargs: Additional PhysicsDataset arguments.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        """
        # Use disjoint seed ranges
        train_ds = PhysicsDataset(solver, n_samples=n_train, seed=seed, **kwargs)
        val_ds = PhysicsDataset(solver, n_samples=n_val, seed=seed + n_train, **kwargs)
        test_ds = PhysicsDataset(solver, n_samples=n_test, seed=seed + n_train + n_val, **kwargs)

        logger.info(
            "dataset_splits_created",
            train=n_train,
            val=n_val,
            test=n_test,
        )

        return train_ds, val_ds, test_ds
