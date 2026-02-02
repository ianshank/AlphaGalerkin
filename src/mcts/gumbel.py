"""Gumbel AlphaZero MCTS implementation.

This module implements the Gumbel AlphaZero algorithm which improves
upon standard MCTS by using Gumbel sampling for action selection
and sequential halving for efficient search.

Reference:
    "Policy improvement by planning with Gumbel"
    Danihelka et al., ICLR 2022

Features:
    - Gumbel-Top-k sampling for root selection
    - Sequential halving for efficient simulations
    - Improved policy target computation
    - Better exploration through completed Q-values
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.games.interface import GameInterface
    from src.games.state import GameState
    from src.modeling.model import AlphaGalerkinModel

logger = structlog.get_logger(__name__)


class GumbelMCTSConfig(BaseModel):
    """Configuration for Gumbel MCTS.

    Attributes:
        n_simulations: Total number of simulations to run.
        max_num_considered_actions: Maximum actions to consider at root.
        c_visit: Visit count scaling constant.
        c_scale: Value scale for completed Q.
        use_mixed_value: Use mixed value estimates.
        gumbel_scale: Scale for Gumbel noise.
        discount: Value discount factor.

    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Core parameters
    n_simulations: int = Field(
        default=800,
        ge=1,
        description="Total number of MCTS simulations",
    )
    max_num_considered_actions: int = Field(
        default=16,
        ge=1,
        description="Maximum actions to consider at root",
    )

    # Gumbel parameters
    gumbel_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale for Gumbel noise",
    )

    # Value parameters
    c_visit: float = Field(
        default=50.0,
        gt=0,
        description="Visit count scaling constant",
    )
    c_scale: float = Field(
        default=1.0,
        gt=0,
        description="Value scale for completed Q",
    )
    use_mixed_value: bool = Field(
        default=True,
        description="Use mixed value estimates",
    )

    # Search parameters
    discount: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="Value discount factor",
    )
    root_dirichlet_alpha: float = Field(
        default=0.03,
        gt=0,
        description="Dirichlet noise alpha for exploration",
    )
    root_exploration_fraction: float = Field(
        default=0.25,
        ge=0,
        le=1,
        description="Fraction of root policy from Dirichlet noise",
    )

    # Performance
    batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for leaf evaluation",
    )


@dataclass
class GumbelNode:
    """Node in the Gumbel MCTS tree.

    Attributes:
        state: Game state at this node.
        prior: Prior probability from policy network.
        gumbel: Gumbel noise value for this action.
        visit_count: Number of times node was visited.
        value_sum: Sum of values backpropagated through this node.
        children: Dictionary mapping actions to child nodes.

    """

    state: GameState | None = None
    prior: float = 0.0
    gumbel: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, GumbelNode] = field(default_factory=dict)

    # For leaf storage before expansion
    _is_expanded: bool = False
    _terminal_value: float | None = None

    @property
    def value(self) -> float:
        """Get mean value estimate.

        Returns:
            Average value or 0 if no visits.

        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def is_expanded(self) -> bool:
        """Check if node has been expanded.

        Returns:
            True if node is expanded.

        """
        return self._is_expanded

    @property
    def is_terminal(self) -> bool:
        """Check if node is terminal.

        Returns:
            True if terminal.

        """
        return self._terminal_value is not None

    def compute_completed_q(
        self,
        c_visit: float,
        c_scale: float,
    ) -> float:
        """Compute completed Q-value for this node.

        Uses the formula from the Gumbel MuZero paper.

        Args:
            c_visit: Visit count scaling.
            c_scale: Value scaling.

        Returns:
            Completed Q-value.

        """
        if self.visit_count == 0:
            # Return value estimate or prior-weighted value
            return 0.0

        # Completed Q = value + sigma(prior, visits)
        sigma = c_scale * np.sqrt(c_visit) / (c_visit + self.visit_count)

        return self.value + sigma * self.prior


@dataclass
class GumbelSearchResult:
    """Result from Gumbel MCTS search."""

    action: int
    policy: np.ndarray
    value: float
    root_value: float
    visit_counts: np.ndarray
    q_values: np.ndarray
    n_simulations: int


class GumbelMCTS:
    """Gumbel AlphaZero MCTS implementation.

    Implements the improved MCTS algorithm from the Gumbel MuZero
    paper, featuring:
    - Gumbel-Top-k trick for root action selection
    - Sequential halving for efficient simulation allocation
    - Improved policy targets via completed Q-values

    Attributes:
        config: MCTS configuration.
        game: Game interface.
        model: Neural network for evaluation.

    """

    def __init__(
        self,
        config: GumbelMCTSConfig,
        game: GameInterface,
        model: AlphaGalerkinModel,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize Gumbel MCTS.

        Args:
            config: MCTS configuration.
            game: Game interface.
            model: Neural network model.
            device: Device for inference.

        """
        self.config = config
        self.game = game
        self.model = model

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self._logger = structlog.get_logger(__name__).bind(
            n_simulations=config.n_simulations,
            max_actions=config.max_num_considered_actions,
        )

    def search(
        self,
        root_state: GameState,
    ) -> GumbelSearchResult:
        """Run Gumbel MCTS search from root state.

        Args:
            root_state: Starting game state.

        Returns:
            GumbelSearchResult with action selection and statistics.

        """
        # Get policy and value for root
        policy, value = self._evaluate(root_state)
        action_mask = self.game.get_action_mask(root_state)

        # Add Dirichlet noise for exploration
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * action_mask.num_legal)
        noise_full = np.zeros_like(policy)
        noise_full[action_mask.mask] = noise

        policy = (
            1 - self.config.root_exploration_fraction
        ) * policy + self.config.root_exploration_fraction * noise_full

        # Mask illegal actions
        policy = policy * action_mask.mask.astype(np.float32)
        policy = policy / (policy.sum() + 1e-8)

        # Sample Gumbel noise for legal actions
        gumbels = np.random.gumbel(size=len(policy)) * self.config.gumbel_scale

        # Compute scores: log(prior) + gumbel
        log_policy = np.log(policy + 1e-8)
        scores = log_policy + gumbels

        # Apply mask to scores
        scores[~action_mask.mask] = -np.inf

        # Select top-k actions
        k = min(self.config.max_num_considered_actions, action_mask.num_legal)
        top_actions = np.argsort(scores)[-k:][::-1]

        # Create root node
        root = GumbelNode(state=root_state)
        root._is_expanded = True

        # Initialize children for top actions
        for action in top_actions:
            root.children[action] = GumbelNode(
                prior=policy[action],
                gumbel=gumbels[action],
            )

        # Run sequential halving
        selected_action, visit_counts = self._sequential_halving(
            root,
            top_actions.tolist(),
            self.config.n_simulations,
        )

        # Compute final policy from visit counts
        final_policy = np.zeros(len(policy))
        for action, node in root.children.items():
            final_policy[action] = node.visit_count
        final_policy = final_policy / (final_policy.sum() + 1e-8)

        # Compute Q-values
        q_values = np.zeros(len(policy))
        for action, node in root.children.items():
            q_values[action] = node.compute_completed_q(
                self.config.c_visit,
                self.config.c_scale,
            )

        return GumbelSearchResult(
            action=selected_action,
            policy=final_policy,
            value=root.value,
            root_value=value,
            visit_counts=final_policy * self.config.n_simulations,
            q_values=q_values,
            n_simulations=self.config.n_simulations,
        )

    def _sequential_halving(
        self,
        root: GumbelNode,
        actions: list[int],
        total_simulations: int,
    ) -> tuple[int, dict[int, int]]:
        """Run sequential halving to allocate simulations.

        Args:
            root: Root node.
            actions: Actions to consider.
            total_simulations: Total simulation budget.

        Returns:
            Tuple of (best action, visit counts).

        """
        remaining_actions = actions.copy()
        remaining_budget = total_simulations
        visit_counts: dict[int, int] = dict.fromkeys(actions, 0)

        while len(remaining_actions) > 1 and remaining_budget > 0:
            # Allocate simulations uniformly
            sims_per_action = max(1, remaining_budget // (2 * len(remaining_actions)))

            for action in remaining_actions:
                for _ in range(sims_per_action):
                    if remaining_budget <= 0:
                        break

                    # Simulate from this action
                    child = root.children[action]

                    if child.state is None:
                        # First visit - expand
                        child.state = self.game.apply_action(root.state, action)

                    # Run simulation
                    value = self._simulate(child)

                    # Backpropagate
                    child.visit_count += 1
                    child.value_sum += value
                    visit_counts[action] += 1
                    remaining_budget -= 1

            # Compute scores and halve
            scores = []
            for action in remaining_actions:
                node = root.children[action]
                q = node.compute_completed_q(
                    self.config.c_visit,
                    self.config.c_scale,
                )
                score = node.gumbel + np.log(node.prior + 1e-8) + q
                scores.append((score, action))

            scores.sort(reverse=True)
            remaining_actions = [a for _, a in scores[: len(scores) // 2]]

        # Select best action
        best_action = max(
            actions,
            key=lambda a: root.children[a].gumbel
            + np.log(root.children[a].prior + 1e-8)
            + root.children[a].compute_completed_q(
                self.config.c_visit,
                self.config.c_scale,
            ),
        )

        return best_action, visit_counts

    def _simulate(self, node: GumbelNode) -> float:
        """Simulate from node to get value estimate.

        Args:
            node: Node to simulate from.

        Returns:
            Value estimate.

        """
        if node.is_terminal:
            return node._terminal_value

        # Check for terminal state
        if self.game.is_terminal(node.state):
            winner = self.game.get_winner(node.state)
            if winner is None:
                value = 0.0
            elif winner == node.state.current_player:
                value = 1.0
            else:
                value = -1.0
            node._terminal_value = value
            return value

        # Evaluate with neural network
        _, value = self._evaluate(node.state)

        return value

    def _evaluate(
        self,
        state: GameState,
    ) -> tuple[np.ndarray, float]:
        """Evaluate state with neural network.

        Args:
            state: Game state to evaluate.

        Returns:
            Tuple of (policy array, value).

        """
        self.model.eval()

        with torch.no_grad():
            tensor = self.game.to_tensor(state).unsqueeze(0).to(self.device)
            output = self.model(tensor)

            policy = torch.softmax(output.policy_logits, dim=-1)
            policy = policy[0].cpu().numpy()
            value = output.value[0].item()

        return policy, value

    def get_improved_policy(
        self,
        root_state: GameState,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Get improved policy from search.

        Args:
            root_state: Root game state.
            temperature: Temperature for policy sharpening.

        Returns:
            Improved policy distribution.

        """
        result = self.search(root_state)

        if temperature == 0:
            # Deterministic: select most visited
            policy = np.zeros_like(result.policy)
            policy[result.action] = 1.0
            return policy

        # Apply temperature
        visit_counts = result.visit_counts
        visit_counts = np.power(visit_counts, 1.0 / temperature)
        return visit_counts / (visit_counts.sum() + 1e-8)


def create_gumbel_mcts(
    game: GameInterface,
    model: AlphaGalerkinModel,
    n_simulations: int = 800,
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> GumbelMCTS:
    """Factory function to create Gumbel MCTS.

    Args:
        game: Game interface.
        model: Neural network model.
        n_simulations: Number of simulations.
        device: Inference device.
        **kwargs: Additional configuration.

    Returns:
        Configured GumbelMCTS instance.

    """
    config = GumbelMCTSConfig(n_simulations=n_simulations, **kwargs)
    return GumbelMCTS(config=config, game=game, model=model, device=device)
