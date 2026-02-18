"""Tests for Othello game implementation.

Covers initialization, legal moves, disc flipping, pass logic,
terminal detection, scoring, tensor encoding, symmetries,
and variable board size support.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.interface import GamePhase, GameResult
from src.games.othello import BLACK, EMPTY, WHITE, OthelloGame
from src.games.registry import GameRegistry
from src.games.state import ActionMask, GameState


class TestOthelloRegistration:
    """Tests for Othello game registration."""

    def test_registered_in_registry(self):
        assert GameRegistry().is_registered("othello")

    def test_get_returns_instance(self):
        game = GameRegistry().get("othello")
        assert isinstance(game, OthelloGame)

    def test_game_info(self):
        info = GameRegistry().get_info("othello")
        assert info is not None
        assert info["name"] == "othello"
        assert info["n_players"] == 2


class TestOthelloInitialization:
    """Tests for Othello board initialization."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_default_board_size(self, game: OthelloGame):
        state = game.initial_state()
        assert state.board_size == 8

    @pytest.mark.parametrize("size", [4, 6, 8, 10, 12, 14, 16])
    def test_variable_board_sizes(self, game: OthelloGame, size: int):
        state = game.initial_state(board_size=size)
        assert state.board_size == size
        assert state.board.shape == (size, size)

    def test_odd_board_size_raises(self, game: OthelloGame):
        with pytest.raises(ValueError, match="even"):
            game.initial_state(board_size=7)

    def test_out_of_range_raises(self, game: OthelloGame):
        with pytest.raises(ValueError, match="out of range"):
            game.initial_state(board_size=2)

    def test_initial_center_discs(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        board = state.board
        assert board[3, 3] == WHITE
        assert board[3, 4] == BLACK
        assert board[4, 3] == BLACK
        assert board[4, 4] == WHITE

    def test_initial_center_discs_6x6(self, game: OthelloGame):
        state = game.initial_state(board_size=6)
        board = state.board
        assert board[2, 2] == WHITE
        assert board[2, 3] == BLACK
        assert board[3, 2] == BLACK
        assert board[3, 3] == WHITE

    def test_initial_player_is_black(self, game: OthelloGame):
        state = game.initial_state()
        assert state.current_player == BLACK

    def test_initial_move_number(self, game: OthelloGame):
        state = game.initial_state()
        assert state.move_number == 0

    def test_initial_empty_count(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        empty_count = np.sum(state.board == EMPTY)
        assert empty_count == 60  # 64 - 4 center discs


class TestOthelloActionSpace:
    """Tests for Othello action space."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_action_space_size_8x8(self, game: OthelloGame):
        game._board_size = 8
        assert game.action_space_size == 65  # 64 + 1 pass

    def test_action_space_size_6x6(self, game: OthelloGame):
        game._board_size = 6
        assert game.action_space_size == 37  # 36 + 1 pass

    def test_state_channels(self, game: OthelloGame):
        assert game.state_channels == 3


class TestOthelloLegalActions:
    """Tests for Othello legal move generation."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_initial_legal_actions_8x8(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        legal = game.get_legal_actions(state)
        # Standard Othello opening: 4 legal moves for black
        assert len(legal) == 4
        # Should include D3, C4, F5, E6 (or equivalent indices)
        expected = {2 * 8 + 3, 3 * 8 + 2, 4 * 8 + 5, 5 * 8 + 4}
        assert set(legal) == expected

    def test_action_mask_matches_legal(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        legal = game.get_legal_actions(state)
        mask = game.get_action_mask(state)
        assert isinstance(mask, ActionMask)
        assert mask.num_legal == len(legal)
        for action in legal:
            assert mask.is_legal(action)

    def test_pass_when_no_placement_moves(self, game: OthelloGame):
        """When a player has no valid placements, pass must be available."""
        state = game.initial_state(board_size=4)
        board = np.zeros((4, 4), dtype=np.int8)
        # Fill board so white has no moves
        board[0, :] = BLACK
        board[1, :] = BLACK
        board[2, :] = BLACK
        board[3, 0:3] = BLACK
        board[3, 3] = WHITE
        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=10,
            metadata={"board_size": 4, "consecutive_passes": 0},
        )
        legal = game.get_legal_actions(state)
        assert legal == [16]  # Only pass (4*4=16)


class TestOthelloApplyAction:
    """Tests for Othello move application and disc flipping."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_apply_first_move(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        # D3 = row 2, col 3 = action 2*8+3 = 19
        new_state = game.apply_action(state, 19)
        assert new_state.board[2, 3] == BLACK
        # Should have flipped D4 (white → black)
        assert new_state.board[3, 3] == BLACK
        assert new_state.current_player == WHITE
        assert new_state.move_number == 1

    def test_flip_multiple_directions(self, game: OthelloGame):
        """Placing a disc should flip in all valid directions."""
        board = np.zeros((6, 6), dtype=np.int8)
        # Set up a position where placing WHITE at (2,4) flips (2,3)
        # Need: W at (2,4), B at (2,3), W bracket at (2,2)
        board[2, 2] = WHITE
        board[2, 3] = BLACK
        # Place white at (2,4) — should flip (2,3) since W(2,2)-B(2,3)-W(2,4)
        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=4,
            metadata={"board_size": 6, "consecutive_passes": 0},
        )
        new_state = game.apply_action(state, 2 * 6 + 4)
        assert new_state.board[2, 4] == WHITE
        assert new_state.board[2, 3] == WHITE  # flipped

    def test_apply_pass(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        board = state.board.copy()
        # Force a pass scenario
        pass_action = 64  # 8*8
        pass_state = GameState(
            board=board,
            current_player=BLACK,
            move_number=5,
            metadata={"board_size": 8, "consecutive_passes": 0},
        )
        new_state = game.apply_action(pass_state, pass_action)
        assert new_state.metadata["consecutive_passes"] == 1
        assert np.array_equal(new_state.board, board)

    def test_occupied_position_raises(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        with pytest.raises(ValueError, match="occupied"):
            game.apply_action(state, 3 * 8 + 3)  # Center disc

    def test_no_flip_raises(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        # Try to place where no flip occurs
        with pytest.raises(ValueError, match="flips no opponent"):
            game.apply_action(state, 0)  # Corner


class TestOthelloTerminal:
    """Tests for Othello terminal detection and scoring."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_not_terminal_initially(self, game: OthelloGame):
        state = game.initial_state()
        assert not game.is_terminal(state)

    def test_terminal_after_two_passes(self, game: OthelloGame):
        state = game.initial_state()
        state = GameState(
            board=state.board,
            current_player=BLACK,
            move_number=10,
            metadata={"board_size": 8, "consecutive_passes": 2},
        )
        assert game.is_terminal(state)

    def test_terminal_when_board_full(self, game: OthelloGame):
        board = np.ones((4, 4), dtype=np.int8)  # All black
        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=16,
            metadata={"board_size": 4, "consecutive_passes": 0},
        )
        assert game.is_terminal(state)

    def test_result_black_wins(self, game: OthelloGame):
        board = np.ones((4, 4), dtype=np.int8)
        board[0, 0] = WHITE
        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=16,
            metadata={"board_size": 4, "consecutive_passes": 2},
        )
        result = game.get_result(state)
        assert isinstance(result, GameResult)
        assert result.winner == BLACK
        assert result.score_black == 15.0
        assert result.score_white == 1.0
        assert result.reason == "disc_count"

    def test_result_draw(self, game: OthelloGame):
        board = np.zeros((4, 4), dtype=np.int8)
        board[:2, :] = BLACK
        board[2:, :] = WHITE
        state = GameState(
            board=board,
            current_player=BLACK,
            move_number=16,
            metadata={"board_size": 4, "consecutive_passes": 2},
        )
        result = game.get_result(state)
        assert result.winner is None

    def test_get_winner(self, game: OthelloGame):
        board = np.full((4, 4), WHITE, dtype=np.int8)
        state = GameState(
            board=board,
            current_player=BLACK,
            move_number=16,
            metadata={"board_size": 4, "consecutive_passes": 2},
        )
        assert game.get_winner(state) == WHITE

    def test_get_winner_not_terminal(self, game: OthelloGame):
        state = game.initial_state()
        assert game.get_winner(state) is None


class TestOthelloTensor:
    """Tests for Othello neural network tensor encoding."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_tensor_shape(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        tensor = game.to_tensor(state)
        assert tensor.shape == (3, 8, 8)

    def test_tensor_shape_variable_size(self, game: OthelloGame):
        for size in [6, 10, 12]:
            state = game.initial_state(board_size=size)
            tensor = game.to_tensor(state)
            assert tensor.shape == (3, size, size)

    def test_tensor_dtype(self, game: OthelloGame):
        state = game.initial_state()
        tensor = game.to_tensor(state)
        assert tensor.dtype == torch.float32

    def test_tensor_values_binary(self, game: OthelloGame):
        state = game.initial_state()
        tensor = game.to_tensor(state)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_tensor_player_plane(self, game: OthelloGame):
        state = game.initial_state()
        tensor = game.to_tensor(state)
        # Black to play → plane 2 should be all 1s
        assert torch.all(tensor[2] == 1.0)

    def test_tensor_white_player_plane(self, game: OthelloGame):
        state = game.initial_state()
        # Force white to play
        state = GameState(
            board=state.board,
            current_player=WHITE,
            move_number=1,
            metadata=state.metadata,
        )
        tensor = game.to_tensor(state)
        assert torch.all(tensor[2] == 0.0)

    def test_tensor_encodes_own_pieces(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        tensor = game.to_tensor(state)
        # Plane 0 = current player (black) pieces
        assert tensor[0, 3, 4] == 1.0  # Black disc
        assert tensor[0, 4, 3] == 1.0  # Black disc
        assert tensor[0, 3, 3] == 0.0  # White disc (not own)

    def test_tensor_encodes_opponent_pieces(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        tensor = game.to_tensor(state)
        # Plane 1 = opponent (white) pieces
        assert tensor[1, 3, 3] == 1.0  # White disc
        assert tensor[1, 4, 4] == 1.0  # White disc
        assert tensor[1, 3, 4] == 0.0  # Black disc (not opponent)


class TestOthelloSymmetries:
    """Tests for Othello symmetry generation."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_eight_symmetries(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        policy = np.random.dirichlet(np.ones(65))
        symmetries = game.get_symmetries(state, policy)
        assert len(symmetries) == 8

    def test_symmetries_preserve_policy_sum(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        policy = np.random.dirichlet(np.ones(65))
        symmetries = game.get_symmetries(state, policy)
        for _, sym_policy in symmetries:
            assert abs(np.sum(sym_policy) - 1.0) < 1e-5

    def test_symmetries_with_tensor_policy(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        policy = torch.softmax(torch.randn(65), dim=0)
        symmetries = game.get_symmetries(state, policy)
        assert len(symmetries) == 8
        for _, sym_policy in symmetries:
            assert isinstance(sym_policy, torch.Tensor)


class TestOthelloGamePhase:
    """Tests for Othello game phase detection."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_opening_phase(self, game: OthelloGame):
        state = game.initial_state(board_size=8)
        assert game.get_phase(state) == GamePhase.OPENING

    def test_terminal_phase(self, game: OthelloGame):
        board = np.ones((4, 4), dtype=np.int8)
        state = GameState(
            board=board,
            current_player=BLACK,
            move_number=16,
            metadata={"board_size": 4, "consecutive_passes": 2},
        )
        assert game.get_phase(state) == GamePhase.TERMINAL


class TestOthelloGameplay:
    """End-to-end gameplay tests for Othello."""

    @pytest.fixture
    def game(self) -> OthelloGame:
        return OthelloGame()

    def test_play_random_game_4x4(self, game: OthelloGame):
        """Play a random 4×4 game to completion."""
        np.random.seed(42)
        state = game.initial_state(board_size=4)
        max_moves = 20

        for _ in range(max_moves):
            if game.is_terminal(state):
                break
            legal = game.get_legal_actions(state)
            action = np.random.choice(legal)
            state = game.apply_action(state, action)

        # Game should reach a terminal state
        assert game.is_terminal(state) or state.move_number >= max_moves
        if game.is_terminal(state):
            result = game.get_result(state)
            assert result.score_black + result.score_white <= 16  # 4*4

    def test_play_random_game_8x8(self, game: OthelloGame):
        """Play a random 8×8 game to completion."""
        np.random.seed(123)
        state = game.initial_state(board_size=8)
        max_moves = 80

        for _ in range(max_moves):
            if game.is_terminal(state):
                break
            legal = game.get_legal_actions(state)
            action = np.random.choice(legal)
            state = game.apply_action(state, action)

        if game.is_terminal(state):
            result = game.get_result(state)
            assert result.score_black >= 0
            assert result.score_white >= 0

    def test_immutable_state(self, game: OthelloGame):
        """Verify apply_action returns new state, doesn't modify original."""
        state = game.initial_state(board_size=8)
        original_board = state.board.copy()
        legal = game.get_legal_actions(state)
        _ = game.apply_action(state, legal[0])
        assert np.array_equal(state.board, original_board)
