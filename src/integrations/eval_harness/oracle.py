"""Greedy 1-step basis-selection oracle (the labelled-dataset ground truth).

``BasisSelectionGame.apply_action`` deterministically computes the post-action L2
``error_estimate`` (least-squares fit against the operator's exact solution). So
the *greedy* best next basis at a state is just the legal action whose resulting
error is smallest — exact, not heuristic.

Caveat (documented and load-bearing): this is a **1-step myopic** oracle. MCTS's
value is multi-step lookahead, so high agreement with the greedy ranking is a
sanity/alignment signal, not a proof of optimality — and a strong policy may
*beat* greedy. The residual scorer, not this oracle, is the outcome metric.

This module imports nothing heavy; ``game.apply_action`` pulls torch/numpy at
runtime, so the oracle is exercised only in the full environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pde.game import PDEState
    from src.pde.games.basis_selection import BasisSelectionGame


def greedy_basis_oracle(game: BasisSelectionGame, state: PDEState) -> dict[str, Any]:
    """Rank legal actions at ``state`` by ascending post-action L2 error.

    Args:
        game: The basis-selection game (provides legal actions + ``apply_action``).
        state: The state to rank actions from (typically the initial empty-basis
            state).

    Returns:
        A label dict with ``greedy_action`` (best legal action or ``None`` when
        no legal actions), ``greedy_residual`` (its resulting error), and the full
        ``ranked_actions`` / ``ranked_residuals`` lists (ascending by error).

    """
    legal_actions = game.get_valid_actions(state)
    scored: list[tuple[int, float]] = []
    for action in legal_actions:
        next_state = game.apply_action(state, action)
        scored.append((int(action), float(next_state.error_estimate)))
    scored.sort(key=lambda pair: pair[1])
    ranked_actions = [action for action, _ in scored]
    ranked_residuals = [residual for _, residual in scored]
    return {
        "greedy_action": ranked_actions[0] if ranked_actions else None,
        "greedy_residual": ranked_residuals[0] if ranked_residuals else None,
        "ranked_actions": ranked_actions,
        "ranked_residuals": ranked_residuals,
    }


__all__ = ["greedy_basis_oracle"]
