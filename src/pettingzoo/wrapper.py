"""Generic PettingZoo AEC wrapper for AlphaGalerkin games.

Adapts any GameInterface implementation to the PettingZoo AEC
(Agent Environment Cycle) API, enabling standard multi-agent RL
benchmarking with action masking, configurable rewards, and
structured logging.

Design Principles:
    - Generic: Works with any GameInterface subclass (Go, Othello, Hex, etc.)
    - Configurable: All behavior controlled via PettingZooConfig (no hardcoded values)
    - Observable: Structured logging for debugging and monitoring
    - Compatible: Full PettingZoo AEC API compliance
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from gymnasium import spaces
from pettingzoo.utils.agent_selector import AgentSelector

from pettingzoo import AECEnv
from src.games.interface import GameInterface
from src.games.state import GameState
from src.pettingzoo.config import PettingZooConfig

logger = structlog.get_logger(__name__)

# Player constants matching GameInterface convention
_PLAYER_1 = 1
_PLAYER_2 = -1


class AlphaGalerkinAECEnv(AECEnv):
    """PettingZoo AEC environment wrapping any AlphaGalerkin GameInterface.

    This wrapper translates between the GameInterface contract (immutable
    states, action indices, ±1 player encoding) and PettingZoo's AEC API
    (agent iteration, observation dicts with action masks, per-step rewards).

    The wrapper is fully generic — it accepts any GameInterface at construction
    time and derives observation/action spaces from the game's properties.

    Attributes:
        metadata: PettingZoo environment metadata.
        game: The underlying GameInterface implementation.
        config: Wrapper configuration.

    """

    metadata: dict[str, Any] = {
        "render_modes": ["ansi"],
        "name": "alphagalerkin_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        game: GameInterface,
        config: PettingZooConfig | None = None,
    ) -> None:
        """Initialize the AEC wrapper.

        Args:
            game: An AlphaGalerkin GameInterface implementation.
            config: Wrapper configuration (uses defaults if None).

        """
        super().__init__()
        self.game = game
        self.config = config or PettingZooConfig()
        self._log = logger.bind(game=game.name)

        # Resolve board size
        self._board_size = self.config.board_size or game.default_board_size

        # Agent setup (2-player games)
        self.possible_agents = [self.config.agent_name(i) for i in range(game.n_players)]
        self._agent_to_player = {
            self.possible_agents[0]: _PLAYER_1,
            self.possible_agents[1]: _PLAYER_2,
        }
        self._player_to_agent = {v: k for k, v in self._agent_to_player.items()}

        # Compute action space size for current board size
        # For games with variable action space, we need the game to tell us
        self._action_space_size = self._compute_action_space_size()

        # Observation: dict with 'observation' array and 'action_mask'
        obs_shape = self._compute_observation_shape()
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=obs_shape,
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self._action_space_size,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Discrete(self._action_space_size) for agent in self.possible_agents
        }

        # Internal state (initialized in reset)
        self._state: GameState | None = None
        self._cumulative_rewards: dict[str, float] = {}
        self._step_count = 0

        self._log.debug(
            "env_initialized",
            board_size=self._board_size,
            action_space_size=self._action_space_size,
            obs_shape=obs_shape,
            agents=self.possible_agents,
        )

    def _compute_action_space_size(self) -> int:
        """Compute action space size for the configured board size.

        Returns:
            Number of discrete actions.

        """
        # For games where action_space_size depends on board_size (Go, Othello, Hex),
        # we need to set the internal board size first
        if hasattr(self.game, "_board_size"):
            self.game._board_size = self._board_size
        return self.game.action_space_size

    def _compute_observation_shape(self) -> tuple[int, int, int]:
        """Compute observation tensor shape.

        Returns (H, W, C) for PettingZoo convention (channels-last).

        Returns:
            Observation shape tuple.

        """
        channels, h, w = self.game.get_observation_shape(self._board_size)
        # PettingZoo convention: (H, W, C)
        return (h, w, channels)

    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent.

        Args:
            agent: Agent name.

        Returns:
            Gymnasium Dict space.

        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent.

        Args:
            agent: Agent name.

        Returns:
            Gymnasium Discrete space.

        """
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (supports 'board_size' override).

        """
        if seed is not None:
            np.random.seed(seed)

        # Allow board size override via options
        board_size = self._board_size
        if options and "board_size" in options:
            board_size = options["board_size"]
            self._board_size = board_size

        self._state = self.game.initial_state(board_size)
        self.agents = list(self.possible_agents)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)
        self.rewards = dict.fromkeys(self.agents, 0.0)
        self.terminations = dict.fromkeys(self.agents, False)
        self.truncations = dict.fromkeys(self.agents, False)
        self.infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        self._step_count = 0

        # Set up agent selection order
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._log.debug(
            "env_reset",
            board_size=board_size,
            seed=seed,
            first_agent=self.agent_selection,
        )

    def observe(self, agent: str) -> dict[str, np.ndarray] | None:
        """Get observation for the specified agent.

        Returns a dict with 'observation' (H, W, C float32 array) and
        'action_mask' (int8 binary vector of legal actions).

        Args:
            agent: Agent to observe for.

        Returns:
            Observation dict or None if agent is done.

        """
        if self._state is None:
            return None

        if self.terminations.get(agent, True) or self.truncations.get(agent, True):
            return None

        # Get observation tensor from game (C, H, W) and convert to (H, W, C)
        tensor = self.game.to_tensor(self._state)
        obs_chw = tensor.cpu().numpy()
        obs_hwc = np.transpose(obs_chw, (1, 2, 0)).astype(np.float32)

        # Build action mask
        mask = np.zeros(self._action_space_size, dtype=np.int8)
        if agent == self.agent_selection:
            action_mask = self.game.get_action_mask(self._state)
            mask[: len(action_mask.mask)] = action_mask.mask.astype(np.int8)

        return {"observation": obs_hwc, "action_mask": mask}

    def step(self, action: int | None) -> None:
        """Apply action for the current agent.

        Args:
            action: Action index, or None if agent is terminated/truncated.

        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping")

        agent = self.agent_selection

        # Handle terminated/truncated agent
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Reset per-step rewards
        self.rewards = dict.fromkeys(self.agents, 0.0)

        # Handle None action (agent is done)
        if action is None:
            self._log.warning("none_action", agent=agent)
            self._terminate_with_illegal(agent)
            self._advance_agent()
            return

        # Validate action bounds
        if not (0 <= action < self._action_space_size):
            self._log.warning(
                "out_of_bounds_action",
                agent=agent,
                action=action,
                max_action=self._action_space_size,
            )
            if self.config.terminate_on_illegal:
                self._terminate_with_illegal(agent)
                self._advance_agent()
                return

        # Check legality
        legal_actions = self.game.get_legal_actions(self._state)
        if action not in legal_actions:
            self._log.debug(
                "illegal_action",
                agent=agent,
                action=action,
                n_legal=len(legal_actions),
            )
            if self.config.terminate_on_illegal:
                self._terminate_with_illegal(agent)
                self._advance_agent()
                return

        # Apply legal action
        self._state = self.game.apply_action(self._state, action)
        self._step_count += 1

        # Check terminal
        if self.game.is_terminal(self._state):
            self._handle_terminal()
        elif self.config.max_cycles and self._step_count >= self.config.max_cycles:
            self._handle_truncation()

        self._advance_agent()
        self._accumulate_rewards()

    def _terminate_with_illegal(self, agent: str) -> None:
        """Handle illegal move by terminating the game.

        Args:
            agent: Agent that made the illegal move.

        """
        self.rewards[agent] = self.config.reward_illegal
        opponent = self._get_opponent(agent)
        self.rewards[opponent] = self.config.reward_win

        for a in self.agents:
            self.terminations[a] = True

        self._log.info(
            "game_terminated_illegal",
            agent=agent,
            step=self._step_count,
        )

    def _handle_terminal(self) -> None:
        """Assign rewards when the game reaches a terminal state."""
        if self._state is None:
            return

        winner = self.game.get_winner(self._state)

        if winner is None:
            # Draw
            for agent in self.agents:
                self.rewards[agent] = self.config.reward_draw
        else:
            winner_agent = self._player_to_agent[winner]
            loser_agent = self._get_opponent(winner_agent)
            self.rewards[winner_agent] = self.config.reward_win
            self.rewards[loser_agent] = self.config.reward_lose

        for agent in self.agents:
            self.terminations[agent] = True

        self._log.info(
            "game_terminated",
            winner=winner,
            step=self._step_count,
            reason="terminal",
        )

    def _handle_truncation(self) -> None:
        """Handle game truncation due to max_cycles."""
        for agent in self.agents:
            self.truncations[agent] = True
            self.rewards[agent] = self.config.reward_draw

        self._log.info(
            "game_truncated",
            step=self._step_count,
            max_cycles=self.config.max_cycles,
        )

    def _advance_agent(self) -> None:
        """Advance to the next agent in the cycle."""
        self.agent_selection = self._agent_selector.next()

    def _get_opponent(self, agent: str) -> str:
        """Get the opponent of the given agent.

        Args:
            agent: Current agent name.

        Returns:
            Opponent agent name.

        """
        for a in self.agents:
            if a != agent:
                return a
        return agent  # Fallback (shouldn't happen in 2-player games)

    def _accumulate_rewards(self) -> None:
        """Add current step rewards to cumulative totals."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def render(self) -> str | None:
        """Render the current board state as text.

        Returns:
            ANSI string representation, or None if no render mode.

        """
        if self.config.render_mode != "ansi" or self._state is None:
            return None

        board = self._state.board
        size = self._state.board_size
        lines = []

        # Column labels
        col_labels = "  " + " ".join(chr(ord("A") + i) for i in range(size))
        lines.append(col_labels)

        # Board rows
        piece_map = {0: ".", 1: "X", -1: "O"}
        for row in range(size):
            row_label = f"{size - row:2d}"
            cells = " ".join(piece_map.get(int(board[row, col]), "?") for col in range(size))
            lines.append(f"{row_label} {cells}")

        # Game info
        player = self._state.current_player
        player_str = "X (player 1)" if player == _PLAYER_1 else "O (player 2)"
        lines.append(f"Move {self._state.move_number}, {player_str} to play")

        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""
        self._state = None
        self._log.debug("env_closed")

    @property
    def board_size(self) -> int:
        """Get the current board size.

        Returns:
            Board size.

        """
        return self._board_size

    @property
    def game_state(self) -> GameState | None:
        """Access the underlying GameState (for debugging/analysis).

        Returns:
            Current GameState or None if not reset.

        """
        return self._state
