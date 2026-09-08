"""Untrained MCTS evaluator for ``SubstrateRefinementGame``.

Policy prior: softmax of residual indicators over legal actions (the search
objective's *where to refine* signal). Leaf value: the trailing error-per-DOF
channel of ``SubstrateRefinementGame.to_tensor``, **not** ``state[0]``.

``EncodedValueEvaluator`` reads ``state.reshape(-1)[0]`` as value, which on this
encoding is the residual of element 0. ``RandomEvaluator`` returns value 0.0.
Neither is an honest look-ahead arm for the MCTS-vs-classical arena.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.mcts.evaluator import EvaluationResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

#: Same floor as ``src.mcts.evaluator._SOFTMAX_NORMALIZER_FLOOR``.
_SOFTMAX_NORMALIZER_FLOOR: float = 1e-8


class ResidualPriorErrorValueEvaluator:
    """Residual-weighted legal prior + error-per-DOF leaf value.

    Structurally satisfies ``src.mcts.evaluator.Evaluator`` (duck-typed). Lives
    under ``src.research`` so the MCTS engine stays domain-free.
    """

    def __init__(self, n_actions: int, *, prior_temperature: float = 1.0) -> None:
        """Initialise.

        Args:
            n_actions: Fixed action-index range (``game.action_space_size``).
            prior_temperature: Softmax temperature over residual logits. Must
                be strictly positive.

        Raises:
            ValueError: If ``n_actions < 1`` or temperature is not positive.

        """
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        if prior_temperature <= 0.0:
            raise ValueError(f"prior_temperature must be > 0, got {prior_temperature}")
        self.n_actions = n_actions
        self.prior_temperature = float(prior_temperature)

    def evaluate(
        self,
        state: NDArray[np.float32],
        legal_actions: list[int],
    ) -> EvaluationResult:
        """Softmax residual prior on legal actions; value from the last slot."""
        policy = np.zeros(self.n_actions, dtype=np.float32)
        flat = np.asarray(state, dtype=np.float64).reshape(-1)
        indicators = flat[:-1] if flat.size > 1 else flat
        if legal_actions:
            logits = np.array(
                [
                    float(indicators[action]) if 0 <= action < indicators.shape[0] else 0.0
                    for action in legal_actions
                ],
                dtype=np.float64,
            )
            logits = np.where(np.isfinite(logits), np.clip(logits, 0.0, None), 0.0)
            if float(logits.sum()) <= 0.0:
                uniform = 1.0 / len(legal_actions)
                for action in legal_actions:
                    policy[action] = uniform
            else:
                scaled = logits / self.prior_temperature
                shifted = scaled - float(np.max(scaled))
                weights = np.exp(shifted)
                weights = weights / (float(weights.sum()) + _SOFTMAX_NORMALIZER_FLOOR)
                for action, weight in zip(legal_actions, weights, strict=True):
                    if 0 <= action < self.n_actions:
                        policy[action] = np.float32(weight)
        value = float(flat[-1]) if flat.size else 0.0
        if np.isfinite(value):
            value = max(-1.0, min(1.0, value))
        else:
            value = 0.0
        return EvaluationResult(policy=policy, value=value)

    def evaluate_batch(
        self,
        states: list[NDArray[np.float32]],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        """Evaluate a batch by delegating to :meth:`evaluate`."""
        return [
            self.evaluate(state, legal)
            for state, legal in zip(states, legal_actions_batch, strict=True)
        ]


__all__ = ["ResidualPriorErrorValueEvaluator"]
