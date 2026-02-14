"""Selection strategies for MCTS child traversal.

Each strategy function scores a child node relative to its parent
and returns a float.  The tree manager picks the child with the
highest score at every interior node during the selection phase.

Registry
--------
``SELECTION_STRATEGIES`` maps :class:`SelectionPolicy` enum values
to callables with signature
``(child, cpuct, parent_visits) -> float``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

from src.alphagalerkin.core.types import SelectionPolicy

if TYPE_CHECKING:
    from src.alphagalerkin.mcts.node import MCTSNode

logger = structlog.get_logger("mcts.selection")


# -------------------------------------------------------------------
# Strategy implementations
# -------------------------------------------------------------------


def puct_score(
    child: MCTSNode,
    cpuct: float,
    parent_visits: int,
) -> float:
    """Predictor + Upper Confidence Bound for Trees (PUCT).

    score = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent)
            / (1 + N_child)

    This is the default AlphaZero-style selection formula.

    Args:
        child: The candidate child node.
        cpuct: Exploration constant.
        parent_visits: Visit count of the parent.

    Returns:
        PUCT score for the child.

    """
    if child.visit_count == 0:
        return float("inf")

    exploitation = child.mean_value
    exploration = cpuct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return exploitation + exploration


def ucb1_score(
    child: MCTSNode,
    cpuct: float,
    parent_visits: int,
) -> float:
    """Classic UCB1 selection (ignores prior probability).

    score = Q(s,a) + c * sqrt(ln(N_parent) / N_child)

    Falls back to ``+inf`` for unvisited children so they are
    explored first.

    Args:
        child: The candidate child node.
        cpuct: Exploration constant (used in place of ``c``).
        parent_visits: Visit count of the parent.

    Returns:
        UCB1 score for the child.

    """
    if child.visit_count == 0:
        return float("inf")

    exploitation = child.mean_value
    exploration = cpuct * math.sqrt(math.log(max(1, parent_visits)) / child.visit_count)
    return exploitation + exploration


def rave_score(
    child: MCTSNode,
    cpuct: float,
    parent_visits: int,
) -> float:
    """Rapid Action Value Estimation (RAVE).

    Uses the standard PUCT formula as a baseline.  A full RAVE
    implementation requires AMAF statistics that are maintained
    outside the node; this entry point provides the selection
    hook so that callers can swap in an enhanced version later.

    Args:
        child: The candidate child node.
        cpuct: Exploration constant.
        parent_visits: Visit count of the parent.

    Returns:
        RAVE score (currently identical to PUCT).

    """
    # Placeholder: falls back to PUCT.  A proper RAVE
    # implementation would blend Q_RAVE with Q here.
    return puct_score(child, cpuct, parent_visits)


# -------------------------------------------------------------------
# Strategy type alias
# -------------------------------------------------------------------

SelectionFn = Callable[["MCTSNode", float, int], float]
"""Callable[[child, cpuct, parent_visits], score]."""


# -------------------------------------------------------------------
# Strategy registry
# -------------------------------------------------------------------

SELECTION_STRATEGIES: dict[SelectionPolicy, SelectionFn] = {
    SelectionPolicy.PUCT: puct_score,
    SelectionPolicy.UCB1: ucb1_score,
    SelectionPolicy.RAVE: rave_score,
}
"""Maps :class:`SelectionPolicy` to the corresponding scorer."""


def get_selection_fn(policy: SelectionPolicy) -> SelectionFn:
    """Look up a selection function by policy enum.

    Args:
        policy: The desired selection policy.

    Returns:
        Matching selection function.

    Raises:
        KeyError: If *policy* is not registered.

    """
    try:
        return SELECTION_STRATEGIES[policy]
    except KeyError:
        available = ", ".join(p.value for p in SELECTION_STRATEGIES)
        msg = f"Unknown selection policy {policy.value!r}. Available: {available}"
        raise KeyError(msg) from None
