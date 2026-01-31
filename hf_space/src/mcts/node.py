"""MCTS Node implementation for AlphaGalerkin.

Each node represents a game state in the search tree.
Uses PUCT (Predictor + Upper Confidence bounds for Trees) selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MCTSNode:
    """Node in the Monte Carlo Tree Search.

    Attributes:
        state: Game state representation (optional, for root node).
        parent: Parent node (None for root).
        action: Action that led to this node.
        prior: Prior probability from policy network.
        children: Dictionary of child nodes keyed by action.
        visit_count: Number of times this node was visited (N).
        total_value: Sum of values from all visits (W).
        virtual_loss: Virtual loss for parallel MCTS.

    """

    parent: MCTSNode | None = None
    action: int | None = None
    prior: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    virtual_loss: float = 0.0

    # Optional state storage (mainly for root)
    _state: NDArray[np.float32] | None = field(default=None, repr=False)

    @property
    def q_value(self) -> float:
        """Mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def q_value_with_virtual_loss(self) -> float:
        """Q value adjusted for virtual loss (for parallel MCTS)."""
        if self.visit_count + self.virtual_loss == 0:
            return 0.0
        return (self.total_value - self.virtual_loss) / (
            self.visit_count + self.virtual_loss
        )

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (unexpanded)."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this node is the root."""
        return self.parent is None

    def ucb_score(
        self,
        c_puct: float,
        parent_visits: int,
    ) -> float:
        """Compute UCB score for node selection.

        UCB(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N_parent) / (1 + N(s, a))

        Args:
            c_puct: Exploration constant.
            parent_visits: Total visits to parent node.

        Returns:
            UCB score.

        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (
            1 + self.visit_count + self.virtual_loss
        )
        return self.q_value_with_virtual_loss + exploration

    def select_child(
        self,
        c_puct: float,
    ) -> MCTSNode:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant.

        Returns:
            Selected child node.

        Raises:
            ValueError: If node has no children.

        """
        if not self.children:
            raise ValueError("Cannot select from node with no children")

        best_score = float("-inf")
        best_child = None
        parent_visits = self.visit_count

        for child in self.children.values():
            score = child.ucb_score(c_puct, parent_visits)
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("No child selected - node has no children to select from")
        return best_child

    def expand(
        self,
        action_priors: dict[int, float],
    ) -> None:
        """Expand node with action priors from policy network.

        Args:
            action_priors: Dictionary mapping actions to prior probabilities.

        """
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(
                    parent=self,
                    action=action,
                    prior=prior,
                )

    def backup(
        self,
        value: float,
    ) -> None:
        """Backup value through the tree.

        Propagates value from leaf to root, alternating sign
        for two-player games.

        Args:
            value: Value estimate from neural network or terminal state.

        """
        node: MCTSNode | None = self
        current_value = value

        while node is not None:
            node.visit_count += 1
            node.total_value += current_value
            # Remove virtual loss if it was applied
            node.virtual_loss = max(0, node.virtual_loss - 1)
            # Flip value for opponent's perspective
            current_value = -current_value
            node = node.parent

    def add_virtual_loss(
        self,
        amount: float = 1.0,
    ) -> None:
        """Add virtual loss for parallel MCTS.

        Virtual loss discourages other threads from selecting
        the same path while this one is being evaluated.

        Args:
            amount: Virtual loss amount.

        """
        self.virtual_loss += amount

    def remove_virtual_loss(
        self,
        amount: float = 1.0,
    ) -> None:
        """Remove virtual loss after evaluation.

        Args:
            amount: Virtual loss amount to remove.

        """
        self.virtual_loss = max(0, self.virtual_loss - amount)

    def get_visit_distribution(
        self,
        temperature: float = 1.0,
    ) -> dict[int, float]:
        """Get action probability distribution based on visit counts.

        Args:
            temperature: Temperature for distribution. Higher = more uniform.
                        0 = deterministic (select most visited).

        Returns:
            Dictionary mapping actions to probabilities.

        """
        if not self.children:
            return {}

        actions = list(self.children.keys())
        visits = np.array([
            self.children[a].visit_count for a in actions
        ], dtype=np.float32)

        if temperature == 0:
            # Deterministic: select most visited
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Softmax with temperature
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                probs = visits_temp / total
            else:
                # Uniform distribution if no visits
                probs = np.ones_like(visits) / len(visits)
            # Ensure probabilities sum to exactly 1.0 for np.random.choice
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            # Note: prob_sum == 0 case already handled by uniform fallback above

        return {a: float(p) for a, p in zip(actions, probs)}

    def get_best_action(self) -> int:
        """Get action with highest visit count.

        Returns:
            Best action.

        Raises:
            ValueError: If node has no children.

        """
        if not self.children:
            raise ValueError("Cannot get best action from node with no children")

        return max(self.children.keys(), key=lambda a: self.children[a].visit_count)

    def get_pv(
        self,
        max_depth: int = 10,
    ) -> list[int]:
        """Get principal variation (best line).

        Args:
            max_depth: Maximum depth to traverse.

        Returns:
            List of actions in the principal variation.

        """
        pv = []
        node = self

        for _ in range(max_depth):
            if node.is_leaf:
                break

            best_action = node.get_best_action()
            pv.append(best_action)
            node = node.children[best_action]

        return pv

    def prune_except(
        self,
        action: int,
    ) -> MCTSNode | None:
        """Prune all children except one, returning new root.

        Used to advance the tree after making a move.

        Args:
            action: Action to keep as new root.

        Returns:
            Child node for the action, or None if not found.

        """
        if action not in self.children:
            return None

        new_root = self.children[action]
        new_root.parent = None

        # Clear references to allow garbage collection
        self.children = {}

        return new_root

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MCTSNode(action={self.action}, "
            f"N={self.visit_count}, Q={self.q_value:.3f}, "
            f"P={self.prior:.3f}, children={len(self.children)})"
        )
