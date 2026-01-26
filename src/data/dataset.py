"""PyTorch Dataset classes for AlphaGalerkin training.

Provides efficient data loading from replay buffers and game databases.
Supports variable board sizes through padding and masking.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

from src.training.replay_buffer import Experience

if TYPE_CHECKING:
    from src.training.replay_buffer import ReplayBuffer


class ReplayDataset(Dataset):
    """Dataset wrapper around replay buffer.

    Provides indexed access to experiences in the buffer.
    Note: The buffer contents may change during iteration.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        transform: callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            buffer: Replay buffer to wrap.
            transform: Optional transform to apply to experiences.

        """
        self.buffer = buffer
        self.transform = transform

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.buffer)

    def __getitem__(self, idx: int) -> Experience:
        """Get experience by index.

        Note: This samples randomly from the buffer, not by index.
        For true indexed access, use a list-backed dataset.

        Args:
            idx: Index (used for compatibility, not actual indexing).

        Returns:
            Random experience from buffer.

        """
        # Sample single experience
        samples = self.buffer.sample(1)
        if not samples:
            raise IndexError("Buffer is empty")

        experience = samples[0]

        if self.transform is not None:
            experience = self.transform(experience)

        return experience


class StreamingReplayDataset(IterableDataset):
    """Streaming dataset that continuously samples from replay buffer.

    More efficient than indexed access for training as it avoids
    the overhead of creating index lists.
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        batch_size: int = 32,
        transform: callable | None = None,
    ) -> None:
        """Initialize streaming dataset.

        Args:
            buffer: Replay buffer to sample from.
            batch_size: Number of samples per iteration.
            transform: Optional transform function.

        """
        self.buffer = buffer
        self.batch_size = batch_size
        self.transform = transform

    def __iter__(self) -> Iterator[Experience]:
        """Iterate over experiences.

        Yields:
            Experiences sampled from buffer.

        """
        while True:
            samples = self.buffer.sample(self.batch_size)
            for experience in samples:
                if self.transform is not None:
                    experience = self.transform(experience)
                yield experience


class BoardSizeBatchSampler(Sampler):
    """Batch sampler that groups experiences by board size.

    Ensures each batch contains experiences from the same board size,
    avoiding padding overhead within batches.
    """

    def __init__(
        self,
        experiences: list[Experience],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ) -> None:
        """Initialize batch sampler.

        Args:
            experiences: List of experiences.
            batch_size: Batch size.
            drop_last: Whether to drop incomplete batches.
            shuffle: Whether to shuffle within size groups.

        """
        self.experiences = experiences
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group indices by board size
        self.size_to_indices: dict[int, list[int]] = {}
        for idx, exp in enumerate(experiences):
            size = exp.board_size
            if size not in self.size_to_indices:
                self.size_to_indices[size] = []
            self.size_to_indices[size].append(idx)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches.

        Yields:
            Lists of indices forming each batch.

        """
        all_batches = []

        for _size, indices in self.size_to_indices.items():
            if self.shuffle:
                import random
                indices = indices.copy()
                random.shuffle(indices)

            # Create batches for this size
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle batches (not within batches)
        if self.shuffle:
            import random
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        """Get number of batches."""
        total = 0
        for indices in self.size_to_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total += n_batches
        return total


class ExperienceListDataset(Dataset):
    """Dataset backed by a list of experiences.

    Provides true indexed access, useful for evaluation or
    when buffer contents should be frozen.
    """

    def __init__(
        self,
        experiences: list[Experience],
        transform: callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            experiences: List of experiences.
            transform: Optional transform function.

        """
        self.experiences = experiences
        self.transform = transform

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.experiences)

    def __getitem__(self, idx: int) -> Experience:
        """Get experience by index.

        Args:
            idx: Index.

        Returns:
            Experience at index.

        """
        experience = self.experiences[idx]

        if self.transform is not None:
            experience = self.transform(experience)

        return experience


class AugmentedExperience:
    """Data augmentation for Go board experiences.

    Applies symmetry transformations (rotations, reflections)
    to increase data diversity.
    """

    def __init__(
        self,
        use_rotations: bool = True,
        use_reflections: bool = True,
    ) -> None:
        """Initialize augmentation.

        Args:
            use_rotations: Apply 90/180/270 degree rotations.
            use_reflections: Apply horizontal/vertical reflections.

        """
        self.use_rotations = use_rotations
        self.use_reflections = use_reflections

        # Build transformation list
        self.transforms: list[callable] = [lambda x: x]  # Identity

        if use_rotations:
            self.transforms.extend([
                lambda x: torch.rot90(x, 1, [-2, -1]),
                lambda x: torch.rot90(x, 2, [-2, -1]),
                lambda x: torch.rot90(x, 3, [-2, -1]),
            ])

        if use_reflections:
            self.transforms.extend([
                lambda x: torch.flip(x, [-1]),  # Horizontal
                lambda x: torch.flip(x, [-2]),  # Vertical
            ])

    def __call__(self, experience: Experience) -> Experience:
        """Apply random augmentation.

        Args:
            experience: Original experience.

        Returns:
            Augmented experience.

        """
        import random

        transform = random.choice(self.transforms)

        # Transform board state
        board_state = transform(experience.board_state)

        # Transform policy (reshape, transform, flatten)
        board_size = experience.board_size
        n_positions = board_size ** 2

        # Separate position policy and pass move
        position_policy = experience.target_policy[:n_positions]
        pass_policy = experience.target_policy[n_positions:]

        # Reshape to 2D, transform, flatten
        position_policy_2d = position_policy.view(board_size, board_size)
        position_policy_2d = transform(position_policy_2d.unsqueeze(0)).squeeze(0)
        position_policy = position_policy_2d.flatten()

        # Recombine
        target_policy = torch.cat([position_policy, pass_policy])

        return Experience(
            board_state=board_state,
            board_size=experience.board_size,
            target_policy=target_policy,
            target_value=experience.target_value,
            metadata=experience.metadata,
        )
