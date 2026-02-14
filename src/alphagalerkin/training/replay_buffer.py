"""Prioritized experience replay buffer with staleness weighting."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from src.alphagalerkin.core.config import ReplayConfig

logger = structlog.get_logger("training.replay")


@dataclass
class Experience:
    """A single training experience from self-play.

    Attributes:
        state_features: Per-element feature array from the
            discretization state.
        policy_target: Action-name to probability mapping
            representing the MCTS-derived policy.
        value_target: Scalar outcome value used as the training
            target for the value head.
        iteration: Training iteration at which this experience
            was generated (used for staleness decay).
        priority: Sampling priority for prioritized replay.

    """

    state_features: np.ndarray
    policy_target: dict[str, float]
    value_target: float
    iteration: int = 0
    priority: float = 1.0


class ReplayBuffer:
    """Replay buffer with optional prioritization.

    Supports uniform and prioritized sampling with
    importance-sampling weights. Priority exponent and
    beta annealing are controlled via ``ReplayConfig``.

    Parameters
    ----------
    config:
        Replay buffer configuration specifying capacity,
        priority parameters, and minimum fill requirements.

    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config
        self._buffer: deque[Experience] = deque(
            maxlen=config.capacity,
        )
        self._priorities: list[float] = []

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of stored experiences."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """Whether buffer has enough samples for training."""
        return self.size >= self._config.min_size_to_train

    # ---------------------------------------------------------------
    # Insertion
    # ---------------------------------------------------------------

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer.

        New experiences receive the current maximum priority
        to ensure they are sampled at least once.
        """
        max_priority = (
            max(self._priorities) if self._priorities else 1.0
        )
        experience.priority = max_priority
        self._buffer.append(experience)
        self._priorities.append(max_priority)

        # Keep priority list in sync with deque evictions
        while len(self._priorities) > len(self._buffer):
            self._priorities.pop(0)

    def add_batch(self, experiences: list[Experience]) -> None:
        """Add a batch of experiences."""
        for exp in experiences:
            self.add(exp)

    # ---------------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
    ) -> list[Experience]:
        """Sample a batch of experiences.

        Uses prioritized sampling if ``priority_alpha > 0``,
        otherwise falls back to uniform random sampling.

        Parameters
        ----------
        batch_size:
            Desired number of experiences to sample.

        Returns
        -------
        list[Experience]
            Sampled experiences (length may be smaller than
            *batch_size* when the buffer is not full enough).

        Raises
        ------
        RuntimeError
            If the buffer has not reached ``min_size_to_train``.

        """
        if not self.is_ready:
            msg = (
                f"Buffer not ready: {self.size} < "
                f"{self._config.min_size_to_train}"
            )
            raise RuntimeError(msg)

        actual_batch = min(batch_size, self.size)

        if (
            self._config.prioritized
            and self._config.priority_alpha > 0
        ):
            return self._prioritized_sample(actual_batch)
        return self._uniform_sample(actual_batch)

    def _uniform_sample(
        self,
        batch_size: int,
    ) -> list[Experience]:
        """Uniform random sampling without replacement."""
        indices = random.sample(range(self.size), batch_size)
        return [self._buffer[i] for i in indices]

    def _prioritized_sample(
        self,
        batch_size: int,
    ) -> list[Experience]:
        """Prioritized sampling based on TD-error priorities.

        Priorities are raised to the power ``alpha`` and
        normalized into a probability distribution.
        """
        alpha = self._config.priority_alpha
        priorities = np.array(
            self._priorities[: self.size],
        )
        probs = priorities ** alpha
        total = probs.sum()
        if total == 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total

        indices = np.random.choice(
            self.size,
            size=batch_size,
            p=probs,
            replace=False,
        )
        return [self._buffer[i] for i in indices]

    # ---------------------------------------------------------------
    # Priority management
    # ---------------------------------------------------------------

    def update_priorities(
        self,
        indices: list[int],
        priorities: list[float],
    ) -> None:
        """Update priorities for sampled experiences.

        Parameters
        ----------
        indices:
            Buffer indices of the sampled experiences.
        priorities:
            New priority values (typically ``|TD-error| + eps``).

        """
        for idx, priority in zip(indices, priorities, strict=True):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = max(priority, 1e-6)

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def clear(self) -> None:
        """Remove all experiences from the buffer."""
        self._buffer.clear()
        self._priorities.clear()

    def get_state(self) -> dict[str, Any]:
        """Serialize buffer state for checkpointing."""
        return {
            "experiences": list(self._buffer),
            "priorities": list(self._priorities),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load buffer state from a checkpoint.

        Parameters
        ----------
        state:
            Dictionary previously returned by ``get_state``.

        """
        self._buffer = deque(
            state["experiences"],
            maxlen=self._config.capacity,
        )
        self._priorities = list(state["priorities"])
