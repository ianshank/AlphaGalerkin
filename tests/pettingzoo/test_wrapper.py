"""Tests for the generic PettingZoo AEC wrapper.

Tests the AlphaGalerkinAECEnv adapter with multiple game backends
(Go, Othello, Hex) to verify full PettingZoo AEC API compliance,
observation/action space correctness, reward logic, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from src.games.go import GoGame
from src.games.hex import HexGame
from src.games.othello import OthelloGame
from src.pettingzoo.config import PettingZooConfig
from src.pettingzoo.wrapper import AlphaGalerkinAECEnv


class TestPettingZooConfig:
    """Tests for PettingZooConfig."""

    def test_default_values(self):
        config = PettingZooConfig()
        assert config.board_size is None
        assert config.reward_win == 1.0
        assert config.reward_lose == -1.0
        assert config.reward_draw == 0.0
        assert config.reward_illegal == -1.0
        assert config.max_cycles is None
        assert config.agent_prefix == "player"
        assert config.terminate_on_illegal is True
        assert config.render_mode is None

    def test_custom_values(self):
        config = PettingZooConfig(
            board_size=9,
            reward_win=10.0,
            agent_prefix="agent",
            max_cycles=500,
        )
        assert config.board_size == 9
        assert config.reward_win == 10.0
        assert config.agent_prefix == "agent"
        assert config.max_cycles == 500

    def test_agent_name_generation(self):
        config = PettingZooConfig(agent_prefix="bot")
        assert config.agent_name(0) == "bot_0"
        assert config.agent_name(1) == "bot_1"


class TestWrapperInitialization:
    """Tests for AEC wrapper initialization."""

    def test_go_wrapper_creation(self):
        game = GoGame()
        env = AlphaGalerkinAECEnv(game)
        assert env.game.name == "go"
        assert len(env.possible_agents) == 2

    def test_othello_wrapper_creation(self):
        game = OthelloGame()
        env = AlphaGalerkinAECEnv(game)
        assert env.game.name == "othello"

    def test_hex_wrapper_creation(self):
        game = HexGame()
        env = AlphaGalerkinAECEnv(game)
        assert env.game.name == "hex"

    def test_custom_config(self):
        config = PettingZooConfig(board_size=9, agent_prefix="p")
        env = AlphaGalerkinAECEnv(GoGame(), config)
        assert env.board_size == 9
        assert env.possible_agents == ["p_0", "p_1"]

    def test_default_board_size(self):
        env = AlphaGalerkinAECEnv(GoGame())
        assert env.board_size == 19  # Go default

    def test_custom_board_size(self):
        config = PettingZooConfig(board_size=9)
        env = AlphaGalerkinAECEnv(GoGame(), config)
        assert env.board_size == 9


class TestWrapperSpaces:
    """Tests for observation and action spaces."""

    @pytest.fixture
    def go_env(self):
        config = PettingZooConfig(board_size=9)
        return AlphaGalerkinAECEnv(GoGame(), config)

    @pytest.fixture
    def othello_env(self):
        config = PettingZooConfig(board_size=8)
        return AlphaGalerkinAECEnv(OthelloGame(), config)

    @pytest.fixture
    def hex_env(self):
        config = PettingZooConfig(board_size=7)
        return AlphaGalerkinAECEnv(HexGame(), config)

    def test_go_action_space(self, go_env: AlphaGalerkinAECEnv):
        space = go_env.action_space("player_0")
        assert isinstance(space, spaces.Discrete)
        assert space.n == 82  # 9*9 + 1

    def test_othello_action_space(self, othello_env: AlphaGalerkinAECEnv):
        space = othello_env.action_space("player_0")
        assert isinstance(space, spaces.Discrete)
        assert space.n == 65  # 8*8 + 1

    def test_hex_action_space(self, hex_env: AlphaGalerkinAECEnv):
        space = hex_env.action_space("player_0")
        assert isinstance(space, spaces.Discrete)
        assert space.n == 49  # 7*7

    def test_observation_space_is_dict(self, go_env: AlphaGalerkinAECEnv):
        space = go_env.observation_space("player_0")
        assert isinstance(space, spaces.Dict)
        assert "observation" in space.spaces
        assert "action_mask" in space.spaces

    def test_go_observation_shape(self, go_env: AlphaGalerkinAECEnv):
        space = go_env.observation_space("player_0")
        obs_space = space.spaces["observation"]
        assert obs_space.shape == (9, 9, 17)  # (H, W, C)

    def test_othello_observation_shape(self, othello_env: AlphaGalerkinAECEnv):
        space = othello_env.observation_space("player_0")
        obs_space = space.spaces["observation"]
        assert obs_space.shape == (8, 8, 3)

    def test_hex_observation_shape(self, hex_env: AlphaGalerkinAECEnv):
        space = hex_env.observation_space("player_0")
        obs_space = space.spaces["observation"]
        assert obs_space.shape == (7, 7, 3)

    def test_action_mask_shape(self, go_env: AlphaGalerkinAECEnv):
        space = go_env.observation_space("player_0")
        mask_space = space.spaces["action_mask"]
        assert mask_space.shape == (82,)


class TestWrapperReset:
    """Tests for environment reset."""

    @pytest.fixture
    def env(self):
        config = PettingZooConfig(board_size=9)
        return AlphaGalerkinAECEnv(GoGame(), config)

    def test_reset_sets_agents(self, env: AlphaGalerkinAECEnv):
        env.reset()
        assert env.agents == ["player_0", "player_1"]

    def test_reset_sets_first_agent(self, env: AlphaGalerkinAECEnv):
        env.reset()
        assert env.agent_selection == "player_0"

    def test_reset_initializes_rewards(self, env: AlphaGalerkinAECEnv):
        env.reset()
        for agent in env.agents:
            assert env.rewards[agent] == 0.0

    def test_reset_initializes_terminations(self, env: AlphaGalerkinAECEnv):
        env.reset()
        for agent in env.agents:
            assert env.terminations[agent] is False
            assert env.truncations[agent] is False

    def test_reset_with_seed(self, env: AlphaGalerkinAECEnv):
        env.reset(seed=42)
        assert env.game_state is not None

    def test_reset_with_board_size_override(self, env: AlphaGalerkinAECEnv):
        env.reset(options={"board_size": 13})
        assert env.board_size == 13


class TestWrapperObserve:
    """Tests for observation generation."""

    @pytest.fixture
    def env(self):
        config = PettingZooConfig(board_size=9)
        env = AlphaGalerkinAECEnv(GoGame(), config)
        env.reset()
        return env

    def test_observe_returns_dict(self, env: AlphaGalerkinAECEnv):
        obs = env.observe("player_0")
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "action_mask" in obs

    def test_observation_array_shape(self, env: AlphaGalerkinAECEnv):
        obs = env.observe("player_0")
        assert obs["observation"].shape == (9, 9, 17)  # (H, W, C)
        assert obs["observation"].dtype == np.float32

    def test_action_mask_has_legal_moves(self, env: AlphaGalerkinAECEnv):
        obs = env.observe("player_0")
        mask = obs["action_mask"]
        assert mask.dtype == np.int8
        assert np.sum(mask) > 0  # At least one legal move

    def test_observe_not_current_agent(self, env: AlphaGalerkinAECEnv):
        """Non-current agent should have all-zero action mask."""
        obs = env.observe("player_1")
        if obs is not None:
            assert np.sum(obs["action_mask"]) == 0


class TestWrapperStep:
    """Tests for step execution."""

    @pytest.fixture
    def go_env(self):
        config = PettingZooConfig(board_size=9)
        env = AlphaGalerkinAECEnv(GoGame(), config)
        env.reset()
        return env

    @pytest.fixture
    def othello_env(self):
        config = PettingZooConfig(board_size=6)
        env = AlphaGalerkinAECEnv(OthelloGame(), config)
        env.reset()
        return env

    def test_step_advances_agent(self, go_env: AlphaGalerkinAECEnv):
        first_agent = go_env.agent_selection
        obs = go_env.observe(first_agent)
        mask = obs["action_mask"]
        legal_actions = np.where(mask)[0]
        go_env.step(int(legal_actions[0]))
        assert go_env.agent_selection != first_agent

    def test_step_legal_action(self, go_env: AlphaGalerkinAECEnv):
        obs = go_env.observe("player_0")
        mask = obs["action_mask"]
        legal = np.where(mask)[0]
        go_env.step(int(legal[0]))
        # Should not terminate
        assert not all(go_env.terminations.values())

    def test_step_illegal_action_terminates(self, go_env: AlphaGalerkinAECEnv):
        """Illegal action should terminate the game."""
        # Action 0 (corner) is legal in Go, but let's force illegal
        # by stepping with an out-of-range action
        go_env.step(9999)
        assert all(go_env.terminations.values())

    def test_illegal_move_rewards(self, go_env: AlphaGalerkinAECEnv):
        """Illegal move gives penalty to mover, reward to opponent."""
        go_env.step(9999)  # Illegal
        assert go_env.rewards["player_0"] == -1.0
        assert go_env.rewards["player_1"] == 1.0


class TestWrapperFullGame:
    """Tests for full game loops."""

    def test_go_random_game(self):
        config = PettingZooConfig(board_size=5, max_cycles=100)
        env = AlphaGalerkinAECEnv(GoGame(), config)
        env.reset(seed=42)

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal)) if len(legal) > 0 else None
            env.step(action)

        # Game should have ended
        assert all(env.terminations.values()) or all(env.truncations.values())
        env.close()

    def test_othello_random_game(self):
        config = PettingZooConfig(board_size=6, max_cycles=50)
        env = AlphaGalerkinAECEnv(OthelloGame(), config)
        env.reset(seed=42)

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal)) if len(legal) > 0 else None
            env.step(action)

        assert all(env.terminations.values()) or all(env.truncations.values())
        env.close()

    def test_hex_random_game(self):
        config = PettingZooConfig(board_size=5, max_cycles=50)
        env = AlphaGalerkinAECEnv(HexGame(), config)
        env.reset(seed=42)

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal)) if len(legal) > 0 else None
            env.step(action)

        assert all(env.terminations.values()) or all(env.truncations.values())
        env.close()


class TestWrapperRender:
    """Tests for ANSI rendering."""

    def test_render_ansi(self):
        config = PettingZooConfig(board_size=5, render_mode="ansi")
        env = AlphaGalerkinAECEnv(GoGame(), config)
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert "A" in output  # Column labels
        assert "." in output  # Empty cells

    def test_render_none(self):
        config = PettingZooConfig(board_size=5)
        env = AlphaGalerkinAECEnv(GoGame(), config)
        env.reset()
        assert env.render() is None


class TestWrapperTruncation:
    """Tests for max_cycles truncation."""

    def test_truncation_at_max_cycles(self):
        config = PettingZooConfig(board_size=9, max_cycles=2)
        env = AlphaGalerkinAECEnv(GoGame(), config)
        env.reset()

        # Play 2 moves (should trigger truncation on the second)
        obs = env.observe("player_0")
        legal = np.where(obs["action_mask"])[0]
        env.step(int(legal[0]))

        obs = env.observe("player_1")
        legal = np.where(obs["action_mask"])[0]
        env.step(int(legal[0]))

        assert all(env.truncations.values())


class TestWrapperMultipleResets:
    """Tests for resetting the environment multiple times."""

    def test_multiple_resets(self):
        config = PettingZooConfig(board_size=5)
        env = AlphaGalerkinAECEnv(GoGame(), config)

        for _ in range(3):
            env.reset()
            assert len(env.agents) == 2
            assert env.agent_selection == "player_0"
            obs = env.observe("player_0")
            assert obs is not None

    def test_reset_after_terminal(self):
        config = PettingZooConfig(board_size=5, max_cycles=1)
        env = AlphaGalerkinAECEnv(GoGame(), config)

        env.reset()
        obs = env.observe("player_0")
        legal = np.where(obs["action_mask"])[0]
        env.step(int(legal[0]))
        # Should be truncated now

        # Reset should work cleanly
        env.reset()
        assert not any(env.terminations.values())
        assert not any(env.truncations.values())
