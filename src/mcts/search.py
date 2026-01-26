"""Monte Carlo Tree Search implementation for AlphaGalerkin.

Implements PUCT-based MCTS with:
- Neural network policy and value guidance
- Batch leaf evaluation for efficiency
- Dirichlet noise for exploration
- Virtual loss for parallel search
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.mcts.node import MCTSNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.mcts.evaluator import Evaluator


class GameInterface(Protocol):
    """Protocol for game environment interface."""

    def get_state(self) -> NDArray[np.float32]:
        """Get current game state as tensor."""
        ...

    def get_legal_actions(self) -> list[int]:
        """Get list of legal action indices."""
        ...

    def apply_action(self, action: int) -> None:
        """Apply action to game state."""
        ...

    def is_terminal(self) -> bool:
        """Check if game is over."""
        ...

    def get_winner(self) -> int:
        """Get winner: 1 for current player, -1 for opponent, 0 for draw."""
        ...

    def clone(self) -> GameInterface:
        """Create a deep copy of the game state."""
        ...


class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Uses PUCT selection formula:
        a* = argmax_a [Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))]
    """

    def __init__(
        self,
        evaluator: Evaluator,
        c_puct: float = 1.5,
        n_simulations: int = 800,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        virtual_loss: float = 3.0,
    ) -> None:
        """Initialize MCTS.

        Args:
            evaluator: Neural network evaluator.
            c_puct: PUCT exploration constant.
            n_simulations: Number of simulations per search.
            dirichlet_alpha: Dirichlet noise concentration.
            dirichlet_epsilon: Dirichlet noise mixing weight.
            virtual_loss: Virtual loss for parallel search.

        """
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.virtual_loss = virtual_loss

        # Reusable root node (for tree reuse)
        self._root: MCTSNode | None = None

    def search(
        self,
        game: GameInterface,
        add_noise: bool = True,
    ) -> dict[int, float]:
        """Run MCTS and return action distribution.

        Args:
            game: Game interface at current position.
            add_noise: Whether to add Dirichlet noise at root.

        Returns:
            Dictionary mapping actions to visit probabilities.

        """
        # Initialize or reuse root
        root = self._get_or_create_root(game)

        # Add Dirichlet noise to root for exploration
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.n_simulations):
            self._simulate(root, game.clone())

        # Return visit distribution
        return root.get_visit_distribution(temperature=1.0)

    def get_action(
        self,
        game: GameInterface,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> int:
        """Run MCTS and select an action.

        Args:
            game: Game interface.
            temperature: Selection temperature (0 = deterministic).
            add_noise: Whether to add exploration noise.

        Returns:
            Selected action.

        """
        # Run search
        self.search(game, add_noise)

        assert self._root is not None

        # Get visit distribution with temperature
        distribution = self._root.get_visit_distribution(temperature)

        # Sample action
        if temperature == 0:
            action = self._root.get_best_action()
        else:
            actions = list(distribution.keys())
            probs = list(distribution.values())
            action = np.random.choice(actions, p=probs)

        return int(action)

    def advance(
        self,
        action: int,
    ) -> None:
        """Advance the tree after making a move.

        Reuses the subtree rooted at the chosen action.

        Args:
            action: Action that was taken.

        """
        if self._root is not None:
            self._root = self._root.prune_except(action)

    def reset(self) -> None:
        """Reset the search tree."""
        self._root = None

    def _get_or_create_root(
        self,
        game: GameInterface,
    ) -> MCTSNode:
        """Get existing root or create new one.

        Args:
            game: Current game state.

        Returns:
            Root node.

        """
        if self._root is None:
            self._root = MCTSNode()

        # Expand root if needed
        if self._root.is_leaf:
            self._expand_node(self._root, game)

        return self._root

    def _simulate(
        self,
        root: MCTSNode,
        game: GameInterface,
    ) -> float:
        """Run one simulation from root.

        Args:
            root: Root node.
            game: Game state (will be modified).

        Returns:
            Value estimate from the simulation.

        """
        node = root
        path: list[MCTSNode] = [node]

        # Selection: traverse tree to leaf
        while not node.is_leaf:
            node = node.select_child(self.c_puct)
            path.append(node)

            # Apply virtual loss
            node.add_virtual_loss(self.virtual_loss)

            # Apply action to game
            if node.action is not None:
                game.apply_action(node.action)

        # Check for terminal state
        if game.is_terminal():
            value = float(game.get_winner())
        else:
            # Expansion and evaluation
            value = self._expand_and_evaluate(node, game)

        # Backup: propagate value through path
        # Remove virtual loss and update values
        for n in reversed(path[1:]):  # Skip root
            n.remove_virtual_loss(self.virtual_loss)

        node.backup(value)

        return value

    def _expand_node(
        self,
        node: MCTSNode,
        game: GameInterface,
    ) -> float:
        """Expand node and return value estimate.

        Args:
            node: Node to expand.
            game: Game state at node.

        Returns:
            Value estimate from neural network.

        """
        state = game.get_state()
        legal_actions = game.get_legal_actions()

        if not legal_actions:
            # No legal moves - game is over
            return float(game.get_winner())

        # Get neural network evaluation
        result = self.evaluator.evaluate(state, legal_actions)

        # Create action priors
        action_priors = {
            a: float(result.policy[a])
            for a in legal_actions
        }

        # Expand node
        node.expand(action_priors)

        return result.value

    def _expand_and_evaluate(
        self,
        node: MCTSNode,
        game: GameInterface,
    ) -> float:
        """Expand leaf node and return value.

        Args:
            node: Leaf node to expand.
            game: Game state at node.

        Returns:
            Value estimate.

        """
        return self._expand_node(node, game)

    def _add_dirichlet_noise(
        self,
        root: MCTSNode,
    ) -> None:
        """Add Dirichlet noise to root prior probabilities.

        P'(a) = (1 - epsilon) * P(a) + epsilon * noise(a)

        Args:
            root: Root node.

        """
        n_actions = len(root.children)
        if n_actions == 0:
            return

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * n_actions)

        # Mix noise into priors
        for i, child in enumerate(root.children.values()):
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * noise[i]
            )

    def get_pv(self) -> list[int]:
        """Get principal variation from current root.

        Returns:
            List of best actions from root.

        """
        if self._root is None:
            return []
        return self._root.get_pv()

    def get_root_value(self) -> float:
        """Get value estimate at root.

        Returns:
            Root Q-value.

        """
        if self._root is None:
            return 0.0
        return self._root.q_value


class BatchMCTS(MCTS):
    """MCTS with batch leaf evaluation for efficiency.

    Collects multiple leaf nodes and evaluates them together,
    improving GPU utilization.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        batch_size: int = 8,
        **kwargs: object,
    ) -> None:
        """Initialize batch MCTS.

        Args:
            evaluator: Neural network evaluator.
            batch_size: Number of leaves to batch.
            **kwargs: Additional MCTS arguments.

        """
        super().__init__(evaluator, **kwargs)
        self.batch_size = batch_size

    def search(
        self,
        game: GameInterface,
        add_noise: bool = True,
    ) -> dict[int, float]:
        """Run batched MCTS search.

        Args:
            game: Game interface.
            add_noise: Whether to add exploration noise.

        Returns:
            Action visit distribution.

        """
        root = self._get_or_create_root(game)

        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Run simulations in batches
        remaining = self.n_simulations

        while remaining > 0:
            batch_size = min(self.batch_size, remaining)
            self._simulate_batch(root, game, batch_size)
            remaining -= batch_size

        return root.get_visit_distribution(temperature=1.0)

    def _simulate_batch(
        self,
        root: MCTSNode,
        game: GameInterface,
        batch_size: int,
    ) -> None:
        """Run a batch of simulations.

        Args:
            root: Root node.
            game: Base game state.
            batch_size: Number of simulations.

        """
        # Collect leaves to evaluate
        leaves: list[tuple[MCTSNode, GameInterface]] = []
        paths: list[list[MCTSNode]] = []

        for _ in range(batch_size):
            game_copy = game.clone()
            node = root
            path = [node]

            # Selection
            while not node.is_leaf:
                node = node.select_child(self.c_puct)
                path.append(node)
                node.add_virtual_loss(self.virtual_loss)

                if node.action is not None:
                    game_copy.apply_action(node.action)

            if not game_copy.is_terminal():
                leaves.append((node, game_copy))
            paths.append(path)

        # Batch evaluate leaves
        if leaves:
            states = [g.get_state() for _, g in leaves]
            legal_actions = [g.get_legal_actions() for _, g in leaves]

            results = self.evaluator.evaluate_batch(states, legal_actions)

            # Expand leaves with results
            for (node, game_state), result, la in zip(
                leaves, results, legal_actions
            ):
                action_priors = {a: float(result.policy[a]) for a in la}
                node.expand(action_priors)

        # Backup
        for i, path in enumerate(paths):
            if i < len(leaves):
                value = results[i].value
            else:
                # Terminal state
                game_copy = game.clone()
                for node in path[1:]:
                    if node.action is not None:
                        game_copy.apply_action(node.action)
                value = float(game_copy.get_winner())

            # Remove virtual loss
            for node in path[1:]:
                node.remove_virtual_loss(self.virtual_loss)

            # Backup value
            path[-1].backup(value)
