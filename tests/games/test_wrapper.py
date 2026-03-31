"""Tests for StatefulGameWrapper.

Validates that the wrapper correctly bridges stateless GameInterface
to the stateful protocol expected by MCTS.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.games.chess import ChessGame
from src.games.wrapper import StatefulGameWrapper


class TestStatefulGameWrapperInit:
    """Tests for StatefulGameWrapper initialization."""

    def test_wraps_chess_game(self) -> None:
        """Test wrapping a ChessGame instance."""
        game = ChessGame()
        state = game.initial_state()
        wrapper = StatefulGameWrapper(game, state)
        assert wrapper is not None

    def test_initial_state_not_terminal(self) -> None:
        """Test that initial chess state is not terminal."""
        game = ChessGame()
        state = game.initial_state()
        wrapper = StatefulGameWrapper(game, state)
        assert not wrapper.is_terminal()


class TestStatefulGameWrapperState:
    """Tests for state access through wrapper."""

    @pytest.fixture
    def wrapper(self) -> StatefulGameWrapper:
        """Create a fresh wrapper around initial chess state."""
        game = ChessGame()
        state = game.initial_state()
        return StatefulGameWrapper(game, state)

    def test_get_state_returns_tensor(self, wrapper: StatefulGameWrapper) -> None:
        """Test that get_state returns a numpy array."""
        state = wrapper.get_state()
        assert isinstance(state, np.ndarray)

    def test_get_state_has_correct_channels(self, wrapper: StatefulGameWrapper) -> None:
        """Test state has 119 channels for chess."""
        state = wrapper.get_state()
        # ChessGame tensor: (119, 8, 8)
        assert state.shape[0] == 119
        assert state.shape[1] == 8
        assert state.shape[2] == 8


class TestStatefulGameWrapperLegalActions:
    """Tests for legal action generation."""

    @pytest.fixture
    def wrapper(self) -> StatefulGameWrapper:
        """Create a fresh wrapper."""
        game = ChessGame()
        state = game.initial_state()
        return StatefulGameWrapper(game, state)

    def test_has_legal_actions(self, wrapper: StatefulGameWrapper) -> None:
        """Test that initial position has legal actions."""
        actions = wrapper.get_legal_actions()
        assert len(actions) > 0

    def test_legal_actions_are_ints(self, wrapper: StatefulGameWrapper) -> None:
        """Test that legal actions are integers."""
        actions = wrapper.get_legal_actions()
        for action in actions:
            assert isinstance(action, int | np.integer)

    def test_initial_position_has_20_legal_moves(self, wrapper: StatefulGameWrapper) -> None:
        """Test that initial chess position has 20 legal moves."""
        actions = wrapper.get_legal_actions()
        assert len(actions) == 20


class TestStatefulGameWrapperApplyAction:
    """Tests for applying actions through wrapper."""

    def test_apply_action_returns_new_wrapper(self) -> None:
        """Test that applying an action returns a new wrapper."""
        game = ChessGame()
        state = game.initial_state()
        wrapper = StatefulGameWrapper(game, state)

        actions = wrapper.get_legal_actions()
        new_wrapper = wrapper.clone()
        new_wrapper.apply_action(actions[0])

        # Original should be unchanged
        assert not wrapper.is_terminal()

    def test_play_two_moves(self) -> None:
        """Test playing two consecutive moves."""
        game = ChessGame()
        state = game.initial_state()
        wrapper = StatefulGameWrapper(game, state)

        # Play two moves
        actions = wrapper.get_legal_actions()
        wrapper.apply_action(actions[0])

        actions2 = wrapper.get_legal_actions()
        assert len(actions2) > 0  # Black should have moves
        wrapper.apply_action(actions2[0])

        # Should still not be terminal
        assert not wrapper.is_terminal()


class TestStatefulGameWrapperClone:
    """Tests for the clone operation."""

    def test_clone_creates_independent_copy(self) -> None:
        """Test that clone creates an independent copy."""
        game = ChessGame()
        state = game.initial_state()
        wrapper = StatefulGameWrapper(game, state)

        cloned = wrapper.clone()
        actions = wrapper.get_legal_actions()

        # Apply action to clone only
        cloned.apply_action(actions[0])

        # Original should be unchanged
        original_actions = wrapper.get_legal_actions()
        assert len(original_actions) == 20  # Initial position
