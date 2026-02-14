"""Dirichlet noise injection for exploration."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("mcts.noise")


class DirichletNoise:
    """Adds Dirichlet noise at the root node for exploration.

    At the root of each MCTS search the prior probabilities are
    mixed with samples from a symmetric Dirichlet distribution::

        new_prior = (1 - epsilon) * prior + epsilon * noise

    This prevents the search from collapsing onto the network's
    favoured actions too early.

    Args:
        alpha: Concentration parameter of the Dirichlet
            distribution.  Smaller values produce sparser noise.
        epsilon: Mixing weight in ``[0, 1]``.  ``0`` disables
            noise; ``1`` replaces priors entirely.

    """

    def __init__(
        self,
        alpha: float = 0.3,
        epsilon: float = 0.25,
    ) -> None:
        if alpha <= 0:
            msg = f"alpha must be positive, got {alpha}"
            raise ValueError(msg)
        if not 0.0 <= epsilon <= 1.0:
            msg = f"epsilon must be in [0, 1], got {epsilon}"
            raise ValueError(msg)

        self._alpha = alpha
        self._epsilon = epsilon

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Dirichlet concentration parameter."""
        return self._alpha

    @property
    def epsilon(self) -> float:
        """Noise mixing weight."""
        return self._epsilon

    # ---------------------------------------------------------------
    # Core
    # ---------------------------------------------------------------

    def apply(
        self,
        priors: dict[Any, float],
        rng: np.random.Generator | None = None,
    ) -> dict[Any, float]:
        """Mix Dirichlet noise into prior probabilities.

        Args:
            priors: Mapping of action to prior probability.
            rng: Optional NumPy random generator for
                reproducibility.

        Returns:
            New dict with the same keys and noised values.

        """
        if not priors:
            return priors

        if rng is None:
            rng = np.random.default_rng()

        actions = list(priors.keys())
        noise = rng.dirichlet(
            [self._alpha] * len(actions),
        )

        result: dict[Any, float] = {}
        for i, action in enumerate(actions):
            result[action] = (1 - self._epsilon) * priors[action] + self._epsilon * noise[i]

        logger.debug(
            "mcts.noise.applied",
            n_actions=len(actions),
            alpha=self._alpha,
            epsilon=self._epsilon,
        )
        return result
