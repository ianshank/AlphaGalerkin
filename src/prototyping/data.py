"""Data generation for rapid prototyping.

Provides synthetic data generators for quick
experimentation without real data dependencies.
"""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SyntheticData:
    """Container for synthetic data.

    Attributes:
        data_id: Unique identifier.
        inputs: Input data.
        targets: Target data.
        n_samples: Number of samples.
        input_shape: Shape of inputs.
        target_shape: Shape of targets.
        metadata: Additional metadata.

    """

    data_id: str
    inputs: list[Any]
    targets: list[Any]
    n_samples: int
    input_shape: tuple[int, ...] = field(default_factory=tuple)
    target_shape: tuple[int, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self) -> Iterator[tuple[Any, Any]]:
        """Iterate over (input, target) pairs."""
        for inp, target in zip(self.inputs, self.targets):
            yield inp, target

    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> tuple[SyntheticData, SyntheticData]:
        """Split data into train and test sets.

        Args:
            train_ratio: Ratio of training data.
            shuffle: Whether to shuffle before split.
            seed: Random seed.

        Returns:
            (train_data, test_data).

        """
        if seed is not None:
            random.seed(seed)

        indices = list(range(self.n_samples))
        if shuffle:
            random.shuffle(indices)

        split_idx = int(self.n_samples * train_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_data = SyntheticData(
            data_id=f"{self.data_id}_train",
            inputs=[self.inputs[i] for i in train_indices],
            targets=[self.targets[i] for i in train_indices],
            n_samples=len(train_indices),
            input_shape=self.input_shape,
            target_shape=self.target_shape,
            metadata={**self.metadata, "split": "train"},
        )

        test_data = SyntheticData(
            data_id=f"{self.data_id}_test",
            inputs=[self.inputs[i] for i in test_indices],
            targets=[self.targets[i] for i in test_indices],
            n_samples=len(test_indices),
            input_shape=self.input_shape,
            target_shape=self.target_shape,
            metadata={**self.metadata, "split": "test"},
        )

        return train_data, test_data

    def batch(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[tuple[list[Any], list[Any]]]:
        """Iterate in batches.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            drop_last: Whether to drop incomplete batch.

        Yields:
            (batch_inputs, batch_targets).

        """
        indices = list(range(self.n_samples))
        if shuffle:
            random.shuffle(indices)

        for i in range(0, self.n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            if drop_last and len(batch_indices) < batch_size:
                break
            yield (
                [self.inputs[j] for j in batch_indices],
                [self.targets[j] for j in batch_indices],
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without data)."""
        return {
            "data_id": self.data_id,
            "n_samples": self.n_samples,
            "input_shape": self.input_shape,
            "target_shape": self.target_shape,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class DataGenerator:
    """Generator for synthetic data.

    Provides various data generation patterns for
    quick experimentation.

    Attributes:
        seed: Random seed for reproducibility.

    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize generator.

        Args:
            seed: Random seed.

        """
        self.seed = seed
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._generators: dict[str, Callable[..., SyntheticData]] = {
            "linear": self._generate_linear,
            "polynomial": self._generate_polynomial,
            "sinusoidal": self._generate_sinusoidal,
            "classification": self._generate_classification,
            "board": self._generate_board,
            "poisson": self._generate_poisson,
        }
        self._logger = logger.bind(generator="DataGenerator")

    def register_generator(
        self,
        name: str,
        fn: Callable[..., SyntheticData],
    ) -> None:
        """Register a custom generator.

        Args:
            name: Generator name.
            fn: Generator function.

        """
        self._generators[name] = fn
        self._logger.info("registered_generator", name=name)

    def generate(
        self,
        generator_type: str,
        n_samples: int,
        **kwargs: Any,
    ) -> SyntheticData:
        """Generate synthetic data.

        Args:
            generator_type: Type of data to generate.
            n_samples: Number of samples.
            **kwargs: Generator-specific arguments.

        Returns:
            Generated SyntheticData.

        """
        if generator_type not in self._generators:
            raise ValueError(f"Unknown generator: {generator_type}")

        self._logger.info(
            "generating_data",
            type=generator_type,
            n_samples=n_samples,
        )

        data = self._generators[generator_type](n_samples, **kwargs)
        return data

    def _generate_linear(
        self,
        n_samples: int,
        n_features: int = 10,
        noise_std: float = 0.1,
    ) -> SyntheticData:
        """Generate linear regression data.

        Args:
            n_samples: Number of samples.
            n_features: Number of input features.
            noise_std: Noise standard deviation.

        Returns:
            SyntheticData with linear relationship.

        """
        # Generate random weights
        weights = [self._rng.gauss(0, 1) for _ in range(n_features)]
        bias = self._rng.gauss(0, 1)

        inputs = []
        targets = []

        for _ in range(n_samples):
            x = [self._rng.gauss(0, 1) for _ in range(n_features)]
            y = sum(w * xi for w, xi in zip(weights, x)) + bias
            y += self._rng.gauss(0, noise_std)
            inputs.append(x)
            targets.append([y])

        return SyntheticData(
            data_id=str(uuid.uuid4())[:8],
            inputs=inputs,
            targets=targets,
            n_samples=n_samples,
            input_shape=(n_features,),
            target_shape=(1,),
            metadata={
                "type": "linear",
                "weights": weights,
                "bias": bias,
                "noise_std": noise_std,
            },
        )

    def _generate_polynomial(
        self,
        n_samples: int,
        degree: int = 3,
        noise_std: float = 0.1,
    ) -> SyntheticData:
        """Generate polynomial regression data.

        Args:
            n_samples: Number of samples.
            degree: Polynomial degree.
            noise_std: Noise standard deviation.

        Returns:
            SyntheticData with polynomial relationship.

        """
        # Generate random coefficients
        coeffs = [self._rng.gauss(0, 1) for _ in range(degree + 1)]

        inputs = []
        targets = []

        for _ in range(n_samples):
            x = self._rng.uniform(-2, 2)
            y = sum(c * (x ** i) for i, c in enumerate(coeffs))
            y += self._rng.gauss(0, noise_std)
            inputs.append([x])
            targets.append([y])

        return SyntheticData(
            data_id=str(uuid.uuid4())[:8],
            inputs=inputs,
            targets=targets,
            n_samples=n_samples,
            input_shape=(1,),
            target_shape=(1,),
            metadata={
                "type": "polynomial",
                "degree": degree,
                "coefficients": coeffs,
                "noise_std": noise_std,
            },
        )

    def _generate_sinusoidal(
        self,
        n_samples: int,
        frequency: float = 1.0,
        amplitude: float = 1.0,
        noise_std: float = 0.1,
    ) -> SyntheticData:
        """Generate sinusoidal data.

        Args:
            n_samples: Number of samples.
            frequency: Signal frequency.
            amplitude: Signal amplitude.
            noise_std: Noise standard deviation.

        Returns:
            SyntheticData with sinusoidal relationship.

        """
        inputs = []
        targets = []

        for i in range(n_samples):
            x = (i / n_samples) * 4 * math.pi
            y = amplitude * math.sin(frequency * x)
            y += self._rng.gauss(0, noise_std)
            inputs.append([x])
            targets.append([y])

        return SyntheticData(
            data_id=str(uuid.uuid4())[:8],
            inputs=inputs,
            targets=targets,
            n_samples=n_samples,
            input_shape=(1,),
            target_shape=(1,),
            metadata={
                "type": "sinusoidal",
                "frequency": frequency,
                "amplitude": amplitude,
                "noise_std": noise_std,
            },
        )

    def _generate_classification(
        self,
        n_samples: int,
        n_classes: int = 2,
        n_features: int = 10,
        separation: float = 1.0,
    ) -> SyntheticData:
        """Generate classification data.

        Args:
            n_samples: Number of samples.
            n_classes: Number of classes.
            n_features: Number of features.
            separation: Class separation factor.

        Returns:
            SyntheticData for classification.

        """
        # Generate class centers
        centers = [
            [self._rng.gauss(0, separation) for _ in range(n_features)]
            for _ in range(n_classes)
        ]

        inputs = []
        targets = []

        for _ in range(n_samples):
            cls = self._rng.randint(0, n_classes - 1)
            center = centers[cls]
            x = [c + self._rng.gauss(0, 0.5) for c in center]
            inputs.append(x)
            targets.append([cls])

        return SyntheticData(
            data_id=str(uuid.uuid4())[:8],
            inputs=inputs,
            targets=targets,
            n_samples=n_samples,
            input_shape=(n_features,),
            target_shape=(1,),
            metadata={
                "type": "classification",
                "n_classes": n_classes,
                "separation": separation,
            },
        )

    def _generate_board(
        self,
        n_samples: int,
        board_size: int = 9,
        density: float = 0.3,
    ) -> SyntheticData:
        """Generate board-like data for Go.

        Args:
            n_samples: Number of samples.
            board_size: Board size.
            density: Stone density.

        Returns:
            SyntheticData with board patterns.

        """
        inputs = []
        targets = []

        for _ in range(n_samples):
            # Generate random board state
            board = [
                [0.0 for _ in range(board_size)]
                for _ in range(board_size)
            ]
            # Place random stones
            n_stones = int(board_size * board_size * density)
            for _ in range(n_stones):
                r = self._rng.randint(0, board_size - 1)
                c = self._rng.randint(0, board_size - 1)
                board[r][c] = self._rng.choice([-1.0, 1.0])

            # Flatten board
            flat_board = [v for row in board for v in row]
            inputs.append(flat_board)

            # Target: influence map (simplified)
            influence = [0.0 for _ in range(board_size * board_size)]
            for i, v in enumerate(flat_board):
                if v != 0:
                    influence[i] = v
            targets.append(influence)

        return SyntheticData(
            data_id=str(uuid.uuid4())[:8],
            inputs=inputs,
            targets=targets,
            n_samples=n_samples,
            input_shape=(board_size * board_size,),
            target_shape=(board_size * board_size,),
            metadata={
                "type": "board",
                "board_size": board_size,
                "density": density,
            },
        )

    def _generate_poisson(
        self,
        n_samples: int,
        grid_size: int = 9,
        n_sources: int = 3,
    ) -> SyntheticData:
        """Generate Poisson-like field data.

        Args:
            n_samples: Number of samples.
            grid_size: Grid size.
            n_sources: Number of sources.

        Returns:
            SyntheticData with field patterns.

        """
        inputs = []
        targets = []

        for _ in range(n_samples):
            # Generate random source distribution
            source = [[0.0] * grid_size for _ in range(grid_size)]
            for _ in range(n_sources):
                r = self._rng.randint(0, grid_size - 1)
                c = self._rng.randint(0, grid_size - 1)
                source[r][c] = self._rng.uniform(-1, 1)

            # Compute simplified "solution" (diffusion-like)
            solution = [[0.0] * grid_size for _ in range(grid_size)]
            for i in range(grid_size):
                for j in range(grid_size):
                    for si in range(grid_size):
                        for sj in range(grid_size):
                            if source[si][sj] != 0:
                                dist = math.sqrt((i - si) ** 2 + (j - sj) ** 2) + 1
                                solution[i][j] += source[si][sj] / dist

            flat_source = [v for row in source for v in row]
            flat_solution = [v for row in solution for v in row]

            inputs.append(flat_source)
            targets.append(flat_solution)

        return SyntheticData(
            data_id=str(uuid.uuid4())[:8],
            inputs=inputs,
            targets=targets,
            n_samples=n_samples,
            input_shape=(grid_size * grid_size,),
            target_shape=(grid_size * grid_size,),
            metadata={
                "type": "poisson",
                "grid_size": grid_size,
                "n_sources": n_sources,
            },
        )


def create_data_generator(seed: int | None = None) -> DataGenerator:
    """Create a data generator.

    Args:
        seed: Random seed.

    Returns:
        Configured DataGenerator.

    """
    return DataGenerator(seed=seed)
