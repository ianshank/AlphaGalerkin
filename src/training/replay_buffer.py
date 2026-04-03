"""Replay buffer for storing and sampling training experiences.

Implements:
- Circular buffer with configurable capacity
- Priority-based sampling (proportional prioritization)
- Support for variable board sizes
- Thread-safe operations
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Any

from src.constants import DEFAULT_PER_ALPHA, DEFAULT_PER_BETA, DEFAULT_PER_BETA_INCREMENT, NUMERIC_EPSILON

import numpy as np
import structlog
import torch
from jaxtyping import Float
from torch import Tensor

logger = structlog.get_logger(__name__)


@dataclass
class Experience:
    """Single training experience from self-play.

    Attributes:
        board_state: Board representation (channels, height, width).
        board_size: Original board size (for batching).
        target_policy: MCTS visit distribution (actions,).
        target_value: Game outcome from current player's perspective.
        metadata: Optional additional information.

    """

    board_state: Float[Tensor, "channels height width"]
    board_size: int
    target_policy: Float[Tensor, actions]
    target_value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> Experience:
        """Move tensors to device.

        Args:
            device: Target device.

        Returns:
            New Experience with tensors on device.

        """
        return Experience(
            board_state=self.board_state.to(device),
            board_size=self.board_size,
            target_policy=self.target_policy.to(device),
            target_value=self.target_value,
            metadata=self.metadata.copy(),
        )


class SumTree:
    """Binary tree for efficient priority-based sampling.

    Each leaf stores a priority value, and internal nodes store the sum
    of their children. Enables O(log n) sampling proportional to priorities.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize sum tree.

        Args:
            capacity: Maximum number of elements (leaves).

        """
        self.capacity = capacity
        # Tree has 2*capacity - 1 nodes (capacity leaves + capacity-1 internal)
        self.tree = np.zeros(2 * capacity - 1)
        self.data: list[Experience | None] = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree.

        Args:
            idx: Index of changed node.
            change: Change in priority value.

        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf with cumulative sum >= s.

        Args:
            idx: Current node index.
            s: Target cumulative sum.

        Returns:
            Index of the leaf.

        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        """Get total priority sum."""
        return float(self.tree[0])

    def add(self, priority: float, data: Experience) -> None:
        """Add data with given priority.

        Args:
            priority: Priority value.
            data: Experience to store.

        """
        idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.update(idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at index.

        Args:
            idx: Tree index.
            priority: New priority value.

        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, Experience | None]:
        """Get data by cumulative priority value.

        Args:
            s: Target cumulative sum.

        Returns:
            Tuple of (tree_index, priority, data).

        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Replay buffer with prioritized experience replay.

    Implements proportional prioritization where the probability of
    sampling experience i is P(i) = p_i^alpha / sum(p_j^alpha).

    Also supports importance sampling weights for unbiased gradient updates.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = DEFAULT_PER_ALPHA,
        beta: float = DEFAULT_PER_BETA,
        beta_increment: float = DEFAULT_PER_BETA_INCREMENT,
        epsilon: float = NUMERIC_EPSILON,
    ) -> None:
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences.
            alpha: Priority exponent (0 = uniform, 1 = full prioritization).
            beta: Importance sampling exponent (annealed to 1).
            beta_increment: Beta increment per sample call.
            epsilon: Small constant added to priorities.

        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = 1.0

        # Thread safety
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return self.tree.n_entries

    def add(
        self,
        experience: Experience,
        priority: float | None = None,
    ) -> None:
        """Add experience to buffer.

        Args:
            experience: Experience to add.
            priority: Optional priority (defaults to max_priority).

        """
        with self._lock:
            if priority is None:
                priority = self.max_priority

            # Apply alpha exponent
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.add(priority, experience)

    def add_batch(
        self,
        experiences: list[Experience],
        priorities: list[float] | None = None,
    ) -> None:
        """Add batch of experiences.

        Args:
            experiences: Experiences to add.
            priorities: Optional priorities per experience.

        """
        if priorities is None:
            priorities = [self.max_priority] * len(experiences)

        for exp, prio in zip(experiences, priorities, strict=False):
            self.add(exp, prio)

    def sample(
        self,
        batch_size: int,
        return_weights: bool = False,
    ) -> tuple[list[Experience], list[int]] | tuple[list[Experience], list[int], Tensor]:
        """Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample.
            return_weights: Whether to return importance sampling weights.

        Returns:
            Tuple of (experiences, indices) or (experiences, indices, weights).

        """
        with self._lock:
            batch_size = min(batch_size, len(self))
            if batch_size == 0:
                if return_weights:
                    return [], [], torch.tensor([])
                return [], []

            experiences: list[Experience] = []
            indices: list[int] = []
            priorities: list[float] = []

            # Divide priority range into segments
            segment = self.tree.total / batch_size

            # Anneal beta
            self.beta = min(1.0, self.beta + self.beta_increment)

            for i in range(batch_size):
                # Sample uniformly within segment
                low = segment * i
                high = segment * (i + 1)
                s = random.uniform(low, high)

                idx, priority, data = self.tree.get(s)

                if data is not None:
                    experiences.append(data)
                    indices.append(idx)
                    priorities.append(priority)

            if return_weights:
                # Compute importance sampling weights
                # w_i = (N * P(i))^(-beta) / max(w_j)
                priorities_arr = np.array(priorities)
                probs = priorities_arr / self.tree.total
                weights = (len(self) * probs) ** (-self.beta)
                weights = weights / weights.max()  # Normalize
                weights_tensor = torch.tensor(weights, dtype=torch.float32)
                return experiences, indices, weights_tensor

            return experiences, indices

    def update_priorities(
        self,
        indices: list[int],
        priorities: list[float],
    ) -> None:
        """Update priorities for sampled experiences.

        Args:
            indices: Tree indices from sampling.
            priorities: New priority values (e.g., TD errors).

        """
        with self._lock:
            for idx, priority in zip(indices, priorities, strict=False):
                # Apply alpha exponent
                priority = (priority + self.epsilon) ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)


class UniformReplayBuffer:
    """Simple replay buffer with uniform sampling.

    Thread-safe circular buffer for storing experiences.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize uniform replay buffer.

        Args:
            capacity: Maximum number of experiences.

        """
        self.capacity = capacity
        self.buffer: list[Experience] = []
        self.position = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        """Get number of experiences in buffer."""
        return len(self.buffer)

    def add(self, experience: Experience) -> None:
        """Add experience to buffer.

        Args:
            experience: Experience to add.

        """
        with self._lock:
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity

    def add_batch(self, experiences: list[Experience]) -> None:
        """Add batch of experiences.

        Args:
            experiences: Experiences to add.

        """
        for exp in experiences:
            self.add(exp)

    def sample(self, batch_size: int) -> list[Experience]:
        """Sample batch of experiences uniformly.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of sampled experiences.

        """
        with self._lock:
            batch_size = min(batch_size, len(self.buffer))
            if batch_size == 0:
                return []
            return random.sample(self.buffer, batch_size)

    def clear(self) -> None:
        """Clear all experiences."""
        with self._lock:
            self.buffer.clear()
            self.position = 0

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats.

        """
        with self._lock:
            if not self.buffer:
                return {
                    "size": 0,
                    "capacity": self.capacity,
                    "fill_ratio": 0.0,
                }

            board_sizes = [exp.board_size for exp in self.buffer]
            values = [exp.target_value for exp in self.buffer]

            return {
                "size": len(self.buffer),
                "capacity": self.capacity,
                "fill_ratio": len(self.buffer) / self.capacity,
                "board_sizes": {
                    "min": min(board_sizes),
                    "max": max(board_sizes),
                    "unique": len(set(board_sizes)),
                },
                "values": {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                },
            }


# Default buffer type for convenience
ReplayBuffer = UniformReplayBuffer


def create_replay_buffer(
    capacity: int,
    prioritized: bool = False,
    alpha: float = DEFAULT_PER_ALPHA,
    beta: float = DEFAULT_PER_BETA,
) -> UniformReplayBuffer | PrioritizedReplayBuffer:
    """Factory function to create replay buffer.

    Args:
        capacity: Maximum number of experiences.
        prioritized: Whether to use prioritized replay.
        alpha: Priority exponent (for prioritized).
        beta: Importance sampling exponent (for prioritized).

    Returns:
        Configured replay buffer.

    """
    if prioritized:
        return PrioritizedReplayBuffer(capacity, alpha=alpha, beta=beta)
    return UniformReplayBuffer(capacity)
