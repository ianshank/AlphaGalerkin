"""Tests for Go game implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.go import BLACK, EMPTY, WHITE, GoGame
from src.games.interface import GamePhase
from src.games.registry import GameRegistry
from src.games.state import GameState


class TestGoGame:
    """Tests for Go game implementation."""

    @pytest.fixture
    def game(self) -> GoGame:
        """Create Go game instance."""
        return GoGame()

    @pytest.fixture
    def small_game(self) -> GoGame:
        """Create small board Go game for faster tests."""
        game = GoGame()
        game._board_size = 9
        return game

    def test_registration(self) -> None:
        """Test that Go is registered in the registry."""
        assert GameRegistry().is_registered("go")

        game = GameRegistry().get("go")
        assert game is not None
        assert game.name == "go"

    def test_initial_state(self, game: GoGame) -> None:
        """Test initial state creation."""
        state = game.initial_state(board_size=9)

        assert state.board.shape == (9, 9)
        assert np.all(state.board == EMPTY)
        assert state.current_player == BLACK
        assert state.move_number == 0
        assert len(state.move_history) == 0

    def test_initial_state_default_size(self, game: GoGame) -> None:
        """Test default board size."""
        state = game.initial_state()

        assert state.board.shape == (19, 19)

    def test_action_space_size(self, game: GoGame) -> None:
        """Test action space includes pass move."""
        game._board_size = 9
        assert game.action_space_size == 9 * 9 + 1  # 82

        game._board_size = 19
        assert game.action_space_size == 19 * 19 + 1  # 362

    def test_legal_actions_initial(self, small_game: GoGame) -> None:
        """Test legal actions on empty board."""
        state = small_game.initial_state(board_size=9)
        legal = small_game.get_legal_actions(state)

        # All positions + pass
        assert len(legal) == 9 * 9 + 1
        assert 81 in legal  # Pass move

    def test_apply_action(self, small_game: GoGame) -> None:
        """Test applying a move."""
        state = small_game.initial_state(board_size=9)

        # Play at center (4, 4) = 4*9 + 4 = 40
        new_state = small_game.apply_action(state, 40)

        assert new_state.board[4, 4] == BLACK
        assert new_state.current_player == WHITE
        assert new_state.move_number == 1
        assert 40 in new_state.move_history

    def test_apply_pass(self, small_game: GoGame) -> None:
        """Test applying pass move."""
        state = small_game.initial_state(board_size=9)

        # Pass move
        new_state = small_game.apply_action(state, 81)

        assert np.all(new_state.board == EMPTY)
        assert new_state.current_player == WHITE
        assert new_state.metadata["consecutive_passes"] == 1

    def test_capture(self, small_game: GoGame) -> None:
        """Test stone capture."""
        state = small_game.initial_state(board_size=9)

        # Create capture situation:
        # . B .
        # B W B
        # . B .
        # White stone at (1, 1) surrounded by black
        moves = [
            (0, 1),  # Black at (0, 1)
            (1, 1),  # White at (1, 1)
            (1, 0),  # Black at (1, 0)
            (8, 8),  # White pass (somewhere else)
            (1, 2),  # Black at (1, 2)
            (8, 7),  # White pass
            (2, 1),  # Black at (2, 1) - captures white
        ]

        for row, col in moves:
            action = row * 9 + col
            state = small_game.apply_action(state, action)

        # White stone should be captured
        assert state.board[1, 1] == EMPTY
        assert state.metadata.get("captured_white", 0) == 1

    def test_suicide_illegal(self, small_game: GoGame) -> None:
        """Test that suicide moves are illegal."""
        state = small_game.initial_state(board_size=9)

        # Create suicide situation
        # . B .
        # B . B
        # . B .
        # Playing white at (1, 1) would be suicide
        state.board[0, 1] = BLACK
        state.board[1, 0] = BLACK
        state.board[1, 2] = BLACK
        state.board[2, 1] = BLACK
        state.current_player = WHITE

        legal = small_game.get_legal_actions(state)

        # (1, 1) should not be legal
        assert (1 * 9 + 1) not in legal

    def test_ko_detection(self, small_game: GoGame) -> None:
        """Test basic ko detection."""
        # Ko detection is implemented via superko
        state = small_game.initial_state(board_size=9)

        # This is a simplified test - full ko requires specific setup
        assert small_game.superko is True

    def test_is_terminal(self, small_game: GoGame) -> None:
        """Test terminal state detection."""
        state = small_game.initial_state(board_size=9)

        assert not small_game.is_terminal(state)

        # Two consecutive passes
        state = small_game.apply_action(state, 81)  # Pass
        state = small_game.apply_action(state, 81)  # Pass

        assert small_game.is_terminal(state)

    def test_get_result(self, small_game: GoGame) -> None:
        """Test game result calculation."""
        state = small_game.initial_state(board_size=9)

        # Play a few moves and end game
        state = small_game.apply_action(state, 40)  # Black center
        state = small_game.apply_action(state, 81)  # White pass
        state = small_game.apply_action(state, 81)  # Black pass

        result = small_game.get_result(state)

        # Black should win with 1 stone vs komi
        assert result.score_black >= 1
        assert result.score_white >= small_game.komi

    def test_to_tensor(self, small_game: GoGame) -> None:
        """Test tensor conversion."""
        state = small_game.initial_state(board_size=9)
        state = small_game.apply_action(state, 40)  # Black at center

        tensor = small_game.to_tensor(state)

        assert tensor.shape == (17, 9, 9)
        assert tensor.dtype == torch.float32

        # Check that black stone is encoded
        # White to play, so white stones go to plane 0
        # Black stones go to plane 8
        assert tensor[8, 4, 4] == 1.0  # Black stone
        assert tensor[16, 0, 0] == 0.0  # White to play (0)

    def test_get_symmetries(self, small_game: GoGame) -> None:
        """Test 8-fold symmetry generation."""
        state = small_game.initial_state(board_size=9)
        state = small_game.apply_action(state, 40)  # Black at center

        policy = np.zeros(82)
        policy[40] = 1.0  # All probability at center

        symmetries = small_game.get_symmetries(state, policy)

        assert len(symmetries) == 8  # 4 rotations x 2 reflections

    def test_action_to_string(self, small_game: GoGame) -> None:
        """Test action to string conversion."""
        small_game._board_size = 9

        # A9 is top-left (0, 0) = action 0
        assert small_game.action_to_string(0, 9) == "A9"

        # Pass move
        assert small_game.action_to_string(81, 9) == "pass"

    def test_string_to_action(self, small_game: GoGame) -> None:
        """Test string to action conversion."""
        small_game._board_size = 9

        assert small_game.string_to_action("A9", 9) == 0
        assert small_game.string_to_action("pass", 9) == 81

    def test_game_phase(self, small_game: GoGame) -> None:
        """Test game phase detection."""
        state = small_game.initial_state(board_size=9)

        # Opening
        assert small_game.get_phase(state) == GamePhase.OPENING

        # Play many moves to reach midgame
        for i in range(20):
            legal = small_game.get_legal_actions(state)
            # Take first non-pass action
            for action in legal:
                if action != 81:
                    state = small_game.apply_action(state, action)
                    break

        assert small_game.get_phase(state) in [GamePhase.OPENING, GamePhase.MIDGAME]


class TestGameState:
    """Tests for GameState."""

    def test_copy(self) -> None:
        """Test state copying."""
        board = np.zeros((9, 9), dtype=np.int8)
        board[4, 4] = BLACK
        state = GameState(board=board, current_player=WHITE)

        copy = state.copy()

        assert np.array_equal(copy.board, state.board)
        assert copy.current_player == state.current_player
        assert copy is not state
        assert copy.board is not state.board

    def test_with_move(self) -> None:
        """Test creating new state with move."""
        board = np.zeros((9, 9), dtype=np.int8)
        state = GameState(board=board, current_player=BLACK)

        new_board = board.copy()
        new_board[4, 4] = BLACK
        new_state = state.with_move(action=40, new_board=new_board)

        assert new_state.current_player == WHITE
        assert new_state.move_number == 1
        assert 40 in new_state.move_history

    def test_hash(self) -> None:
        """Test state hashing."""
        board1 = np.zeros((9, 9), dtype=np.int8)
        board1[4, 4] = BLACK
        state1 = GameState(board=board1, current_player=WHITE)

        board2 = np.zeros((9, 9), dtype=np.int8)
        board2[4, 4] = BLACK
        state2 = GameState(board=board2, current_player=WHITE)

        # Same states should have same hash
        assert hash(state1) == hash(state2)

        # Different states should have different hash
        board3 = np.zeros((9, 9), dtype=np.int8)
        state3 = GameState(board=board3, current_player=BLACK)
        assert hash(state1) != hash(state3)

    def test_equality(self) -> None:
        """Test state equality."""
        board = np.zeros((9, 9), dtype=np.int8)
        board[4, 4] = BLACK

        state1 = GameState(board=board.copy(), current_player=WHITE)
        state2 = GameState(board=board.copy(), current_player=WHITE)

        assert state1 == state2
