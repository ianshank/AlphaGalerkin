"""Tests for PettingZoo environment factory functions.

Validates that go_env(), othello_env(), and hex_env() create
properly configured environments with correct spaces and behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pettingzoo.environments import go_env, hex_env, othello_env
from src.pettingzoo.wrapper import AlphaGalerkinAECEnv


class TestGoEnvFactory:
    """Tests for the Go environment factory."""

    def test_default_creation(self):
        env = go_env()
        assert isinstance(env, AlphaGalerkinAECEnv)
        assert env.game.name == "go"
        assert env.board_size == 19

    def test_custom_board_size(self):
        env = go_env(board_size=9)
        assert env.board_size == 9

    def test_custom_komi(self):
        env = go_env(board_size=9, komi=6.5)
        assert env.game.komi == 6.5

    def test_render_mode(self):
        env = go_env(board_size=5, render_mode="ansi")
        env.reset()
        assert env.render() is not None

    @pytest.mark.parametrize("size", [5, 9, 13, 19])
    def test_variable_sizes(self, size: int):
        env = go_env(board_size=size)
        env.reset()
        obs = env.observe("player_0")
        assert obs["observation"].shape == (size, size, 17)

    def test_full_game_loop(self):
        env = go_env(board_size=5)
        env.reset(seed=42)
        steps = 0
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal))
            env.step(action)
            steps += 1
            if steps > 100:
                break
        env.close()


class TestOthelloEnvFactory:
    """Tests for the Othello environment factory."""

    def test_default_creation(self):
        env = othello_env()
        assert isinstance(env, AlphaGalerkinAECEnv)
        assert env.game.name == "othello"
        assert env.board_size == 8

    def test_custom_board_size(self):
        env = othello_env(board_size=6)
        assert env.board_size == 6

    @pytest.mark.parametrize("size", [6, 8, 10, 12])
    def test_variable_sizes(self, size: int):
        env = othello_env(board_size=size)
        env.reset()
        obs = env.observe("player_0")
        assert obs["observation"].shape == (size, size, 3)
        # Action space = size*size + 1 (pass)
        assert obs["action_mask"].shape == (size * size + 1,)

    def test_full_game_loop(self):
        env = othello_env(board_size=6)
        env.reset(seed=42)
        steps = 0
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal))
            env.step(action)
            steps += 1
            if steps > 100:
                break
        env.close()


class TestHexEnvFactory:
    """Tests for the Hex environment factory."""

    def test_default_creation(self):
        env = hex_env()
        assert isinstance(env, AlphaGalerkinAECEnv)
        assert env.game.name == "hex"
        assert env.board_size == 11

    def test_custom_board_size(self):
        env = hex_env(board_size=7)
        assert env.board_size == 7

    @pytest.mark.parametrize("size", [5, 7, 9, 11, 13])
    def test_variable_sizes(self, size: int):
        env = hex_env(board_size=size)
        env.reset()
        obs = env.observe("player_0")
        assert obs["observation"].shape == (size, size, 3)
        # Hex has no pass — action space is exactly N²
        assert obs["action_mask"].shape == (size * size,)

    def test_full_game_loop(self):
        env = hex_env(board_size=5)
        env.reset(seed=42)
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            if term or trunc:
                action = None
            else:
                mask = obs["action_mask"]
                legal = np.where(mask)[0]
                action = int(np.random.choice(legal))
            env.step(action)
        # Hex always terminates (no draws)
        assert all(env.terminations.values())
        env.close()


class TestCrossResolutionTransfer:
    """Tests verifying observation shape consistency across board sizes.

    These tests validate the core premise: the same wrapper produces
    compatible observations at different resolutions, enabling
    zero-shot transfer experiments.
    """

    def test_go_observation_consistency(self):
        """Go observations have consistent channel count across sizes."""
        for size in [5, 9, 13, 19]:
            env = go_env(board_size=size)
            env.reset()
            obs = env.observe("player_0")
            h, w, c = obs["observation"].shape
            assert h == size
            assert w == size
            assert c == 17  # Always 17 channels

    def test_othello_observation_consistency(self):
        """Othello observations have consistent channel count across sizes."""
        for size in [6, 8, 10, 12]:
            env = othello_env(board_size=size)
            env.reset()
            obs = env.observe("player_0")
            h, w, c = obs["observation"].shape
            assert h == size
            assert w == size
            assert c == 3  # Always 3 channels

    def test_hex_observation_consistency(self):
        """Hex observations have consistent channel count across sizes."""
        for size in [5, 7, 9, 11]:
            env = hex_env(board_size=size)
            env.reset()
            obs = env.observe("player_0")
            h, w, c = obs["observation"].shape
            assert h == size
            assert w == size
            assert c == 3  # Always 3 channels

    def test_action_mask_shape_scales(self):
        """Action mask size scales with board size as expected."""
        # Go: N² + 1
        for size in [5, 9, 13]:
            env = go_env(board_size=size)
            env.reset()
            obs = env.observe("player_0")
            assert obs["action_mask"].shape == (size * size + 1,)

        # Othello: N² + 1
        for size in [6, 8, 10]:
            env = othello_env(board_size=size)
            env.reset()
            obs = env.observe("player_0")
            assert obs["action_mask"].shape == (size * size + 1,)

        # Hex: N² (no pass)
        for size in [5, 7, 9]:
            env = hex_env(board_size=size)
            env.reset()
            obs = env.observe("player_0")
            assert obs["action_mask"].shape == (size * size,)
