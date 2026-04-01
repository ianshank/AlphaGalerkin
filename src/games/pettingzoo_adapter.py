"""PettingZoo adapter for AlphaGalerkin game interfaces.

Bridges the GameInterface protocol to PettingZoo's ParallelEnv API,
enabling multi-agent RL training on any registered game.

PettingZoo is an optional dependency. When not installed, this module
provides a stub base class and sets HAS_PETTINGZOO = False. All
functionality gracefully degrades.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Optional dependency - graceful degradation
try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv

    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False
    ParallelEnv = object  # type: ignore[assignment,misc]

import structlog

from src.games.interface import GameInterface
from src.games.state import GameState

logger = structlog.get_logger(__name__)


class PettingZooAdapter(ParallelEnv if HAS_PETTINGZOO else object):  # type: ignore[misc]
    """Wraps AlphaGalerkin GameInterface as a PettingZoo ParallelEnv.

    Each agent takes an action simultaneously. The adapter maps between
    PettingZoo's multi-agent dict-based API and GameInterface's
    sequential action model.

    Attributes:
        metadata: PettingZoo environment metadata.
        game: The underlying GameInterface instance.
        n_agents: Number of agents in the environment.

    """

    metadata: dict[str, Any] = {"render_modes": ["human"], "name": "alphagalerkin_v0"}

    def __init__(
        self,
        game: GameInterface,
        n_agents: int = 2,
        board_size: int | None = None,
    ) -> None:
        """Initialize PettingZoo adapter.

        Args:
            game: AlphaGalerkin GameInterface implementation.
            n_agents: Number of agents.
            board_size: Board size override (uses game default if None).

        Raises:
            RuntimeError: If pettingzoo is not installed.

        """
        if not HAS_PETTINGZOO:
            raise RuntimeError(
                "pettingzoo and gymnasium are required for PettingZooAdapter. "
                "Install with: pip install pettingzoo gymnasium"
            )

        self.game = game
        self.n_agents = n_agents
        self._board_size = board_size or game.default_board_size

        # PettingZoo agent IDs
        self.possible_agents: list[str] = [f"agent_{i}" for i in range(n_agents)]
        self.agents: list[str] = []

        # Internal state
        self._state: GameState | None = None
        self._cumulative_rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}

        logger.info(
            "pettingzoo_adapter_created",
            game=game.name,
            n_agents=n_agents,
            board_size=self._board_size,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed (unused, for API compatibility).
            options: Additional options (unused).

        Returns:
            Tuple of (observations, infos) dicts keyed by agent ID.

        """
        self.agents = list(self.possible_agents)
        self._state = self.game.initial_state(self._board_size)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self._terminations = dict.fromkeys(self.agents, False)
        self._truncations = dict.fromkeys(self.agents, False)

        obs = self._get_observations()
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}

        logger.debug("environment_reset", n_agents=len(self.agents))

        return obs, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Execute one step by applying each agent's action sequentially.

        Iterates through agents in order; for each, validates and applies
        the action to the game state. Stops early if the game reaches a
        terminal state. Invalid actions receive a -1.0 penalty.

        Args:
            actions: Dict mapping agent IDs to action indices.

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos).

        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")

        rewards: dict[str, float] = dict.fromkeys(self.agents, 0.0)
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}

        # Apply actions sequentially (GameInterface is turn-based)
        for agent_id in self.agents:
            if agent_id not in actions:
                continue
            if self._terminations.get(agent_id, False):
                continue

            action = actions[agent_id]

            # Validate action
            if not self.game.validate_action(self._state, action):
                # Invalid action: penalize and skip
                rewards[agent_id] = -1.0
                infos[agent_id]["invalid_action"] = True
                continue

            self._state = self.game.apply_action(self._state, action)

            # Check terminal after each action
            if self.game.is_terminal(self._state):
                result = self.game.get_result(self._state)
                for a in self.agents:
                    self._terminations[a] = True
                # Assign rewards based on game result
                if result.winner is not None:
                    for i, a in enumerate(self.agents):
                        player = 1 if i % 2 == 0 else -1
                        rewards[a] = 1.0 if result.winner == player else -1.0
                break

        observations = self._get_observations()

        # Remove terminated agents
        self.agents = [a for a in self.agents if not self._terminations.get(a, False)]

        return observations, rewards, self._terminations, self._truncations, infos

    def observation_space(self, agent: str) -> Any:
        """Get observation space for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            gymnasium Box space for observations.

        """
        shape = self.game.get_observation_shape(self._board_size)
        return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)

    def action_space(self, agent: str) -> Any:
        """Get action space for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            gymnasium Discrete space for actions.

        """
        return spaces.Discrete(self.game.action_space_size)

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Build observation dict for all active agents.

        Returns:
            Dict mapping agent IDs to observation arrays.

        """
        if self._state is None:
            return {}

        tensor = self.game.to_tensor(self._state)
        obs_np = tensor.numpy()

        return {agent: obs_np.copy() for agent in self.agents}

    def __repr__(self) -> str:
        """String representation."""
        return f"PettingZooAdapter(game={self.game.name}, n_agents={self.n_agents})"
