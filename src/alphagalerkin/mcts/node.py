"""MCTS node for discretization search tree."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.alphagalerkin.env.actions import Action
    from src.alphagalerkin.env.state import DiscretizationState


class MCTSNode:
    """A node in the MCTS search tree.

    Each node stores a discretization state, visit statistics,
    and references to parent/children.
    """

    def __init__(
        self,
        state: DiscretizationState,
        prior: float = 0.0,
        parent: MCTSNode | None = None,
        action_from_parent: Action | None = None,
    ) -> None:
        self._state = state
        self._prior = prior
        self._parent = parent
        self._action_from_parent = action_from_parent
        self._children: dict[Action, MCTSNode] = {}
        self._visit_count: int = 0
        self._total_value: float = 0.0
        self._is_terminal: bool = False
        self._max_value: float = float("-inf")

    # ---------------------------------------------------------------
    # Read-only properties
    # ---------------------------------------------------------------

    @property
    def state(self) -> DiscretizationState:
        """The discretization state at this node."""
        return self._state

    @property
    def prior(self) -> float:
        """Prior probability assigned by the policy network."""
        return self._prior

    @property
    def parent(self) -> MCTSNode | None:
        """Parent node, or ``None`` for the root."""
        return self._parent

    @property
    def action_from_parent(self) -> Action | None:
        """Action that was taken to reach this node."""
        return self._action_from_parent

    @property
    def children(self) -> dict[Action, MCTSNode]:
        """Mapping from action to child node."""
        return self._children

    @property
    def visit_count(self) -> int:
        """Number of times this node has been visited."""
        return self._visit_count

    @property
    def total_value(self) -> float:
        """Cumulative value backpropagated through this node."""
        return self._total_value

    @property
    def is_leaf(self) -> bool:
        """Whether the node has no expanded children."""
        return len(self._children) == 0

    @property
    def is_terminal(self) -> bool:
        """Whether the node represents a terminal state."""
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value: bool) -> None:
        self._is_terminal = value

    @property
    def mean_value(self) -> float:
        """Average value across all visits (0.0 if unvisited)."""
        if self._visit_count == 0:
            return 0.0
        return self._total_value / self._visit_count

    @property
    def max_value(self) -> float:
        """Maximum backed-up value seen at this node."""
        return self._max_value if self.visit_count > 0 else 0.0

    # ---------------------------------------------------------------
    # UCB scoring
    # ---------------------------------------------------------------

    def ucb_score(
        self,
        cpuct: float,
        parent_visits: int,
    ) -> float:
        """Compute PUCT score for child selection.

        score = Q(s,a) + cpuct * P(s,a) * sqrt(N_parent)
                / (1 + N_child)

        An unvisited child returns ``+inf`` so it is selected
        first.

        Args:
            cpuct: Exploration constant.
            parent_visits: Total visit count of the parent node.

        Returns:
            Combined exploitation + exploration score.

        """
        if self._visit_count == 0:
            return float("inf")

        exploitation = self.mean_value
        exploration = cpuct * self._prior * math.sqrt(parent_visits) / (1 + self._visit_count)
        return exploitation + exploration

    # ---------------------------------------------------------------
    # Tree operations
    # ---------------------------------------------------------------

    def expand(
        self,
        action_priors: dict[Action, float],
    ) -> None:
        """Expand this node with child nodes for each action.

        Args:
            action_priors: Mapping of action to prior probability.

        Raises:
            ValueError: If *action_priors* is empty.
            RuntimeError: If the node is already expanded.

        """
        if not action_priors:
            msg = "Cannot expand with empty priors: at least one action required"
            raise ValueError(msg)

        if self._children:
            msg = "Node already expanded"
            raise RuntimeError(msg)

        for action, prior in action_priors.items():
            child_state = self._state.apply_action(action)
            child = MCTSNode(
                state=child_state,
                prior=prior,
                parent=self,
                action_from_parent=action,
            )
            self._children[action] = child

    def backup(self, value: float) -> None:
        """Record one visit and accumulate *value*."""
        self._visit_count += 1
        self._total_value += value
        self._max_value = max(self._max_value, value)

    def select_best_child(self, cpuct: float) -> MCTSNode:
        """Select the child with the highest UCB score.

        Args:
            cpuct: PUCT exploration constant.

        Returns:
            Child node with the best score.

        Raises:
            RuntimeError: If no children exist.

        """
        if not self._children:
            msg = "No children to select from"
            raise RuntimeError(msg)

        best_score = float("-inf")
        best_child: MCTSNode | None = None
        parent_visits = self._visit_count

        for child in self._children.values():
            score = child.ucb_score(cpuct, parent_visits)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:  # pragma: no cover
            msg = "Failed to select best child"
            raise RuntimeError(msg)

        return best_child
