"""Reward composition factory for discretization games.

:class:`RewardComposer` aggregates multiple scalar reward components
(accuracy, efficiency, stability, ...) into a single scalar via
configurable weights.  Helper methods translate raw metrics (residual
norms, DOF counts, condition numbers) into normalised ``[0, 1]``
reward signals.

All weights are injected at construction time through
:data:`~alphagalerkin.core.types.RewardWeights` -- nothing is
hardcoded.
"""
from __future__ import annotations

import numpy as np
import structlog

from src.alphagalerkin.core.types import RewardWeights

logger = structlog.get_logger("env.rewards")


class RewardComposer:
    """Compose reward signals from multiple components.

    Parameters
    ----------
    weights:
        Mapping from component name to scalar weight.  Components
        whose name does not appear in the map receive weight 0.
        If ``None``, sensible defaults are used.

    """

    def __init__(
        self, weights: RewardWeights | None = None
    ) -> None:
        self._weights: RewardWeights = weights or {
            "accuracy": 1.0,
            "efficiency": 0.5,
            "stability": 0.3,
        }

    # -- main entry point --------------------------------------------

    def compute(
        self,
        accuracy: float = 0.0,
        efficiency: float = 0.0,
        stability: float = 0.0,
        **extra_components: float,
    ) -> float:
        """Compute the weighted sum of reward components.

        Named arguments correspond to the three built-in components;
        additional components can be passed as keyword arguments and
        will be looked up in the weight map.
        """
        components: dict[str, float] = {
            "accuracy": accuracy,
            "efficiency": efficiency,
            "stability": stability,
            **extra_components,
        }

        total = 0.0
        for name, value in components.items():
            weight = self._weights.get(name, 0.0)
            total += weight * value

        logger.debug(
            "rewards.computed",
            total=total,
            components=components,
        )
        return total

    # -- component helpers -------------------------------------------

    def accuracy_reward(
        self,
        residual_norm: float,
        prev_residual: float | None = None,
    ) -> float:
        """Reward based on residual reduction.

        If *prev_residual* is available the reward is the fractional
        reduction ``1 - r_new / r_old`` (clamped to ``[0, 1]``).
        Otherwise the reward is ``1 - r_new`` (assuming the norm is
        already small).
        """
        if prev_residual is not None and prev_residual > 0:
            return max(0.0, 1.0 - residual_norm / prev_residual)
        return max(0.0, 1.0 - residual_norm)

    def efficiency_reward(
        self,
        dof_count: int,
        max_dof: int,
    ) -> float:
        """Reward for DOF efficiency (fewer DOFs is better).

        Returns ``1 - dof_count / max_dof``, clamped to ``[0, 1]``.
        """
        return max(0.0, 1.0 - dof_count / max_dof)

    def stability_reward(
        self,
        condition_number: float,
        threshold: float = 1e6,
    ) -> float:
        """Reward based on matrix conditioning.

        Uses a log-scale comparison: ``1 - log10(kappa) / log10(T)``
        where *T* is the acceptable threshold.  Returns 0 when the
        condition number exceeds the threshold.
        """
        if condition_number <= 0:
            return 0.0
        log_cond = float(np.log10(max(1.0, condition_number)))
        log_thresh = float(np.log10(threshold))
        return max(0.0, 1.0 - log_cond / log_thresh)
