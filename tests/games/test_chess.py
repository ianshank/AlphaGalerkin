"""Tests for Chess game implementation.

Tests the complete Chess implementation including:
- Game registration
- Initial state setup
- Legal move generation
- Special moves (castling, en passant, promotion)
- Check and checkmate detection
- Draw conditions
- Neural network tensor encoding
- Symmetry transformations
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.chess import (
    ChessGame,
    Piece,
    WHITE,
    BLACK,
    BOARD_SIZE,
    ACTION_SPACE_SIZE,
    STARTING_POSITION,
)
from src.games.interface import GameResult
from src.games.registry import GameRegistry
from src.games.state import GameState, ActionMask


class TestChessRegistration:
    """Tests for Chess game registration."""

    def test_game_registered(self) -> None:
        """Test that Chess is registered in the game registry."""
        assert GameRegistry().is_registered("chess")

    def test_get_game_from_registry(self) -> None:
        """Test getting Chess game from registry."""
        game = GameRegistry().get("chess")
        assert isinstance(game, ChessGame)

    def test_game_info(self) -> None:
        """Test game info from registry."""
        info = GameRegistry().get_info("chess")
        assert info["name"] == "chess"
        assert "description" in info


class TestChessInitialization:
    """Tests for Chess game initialization."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_initial_state(self, game: ChessGame) -> None:
        """Test initial board setup."""
        state = game.initial_state()

        # Board shape
        assert state.board.shape == (BOARD_SIZE, BOARD_SIZE)

        # Current player is white
        assert state.current_player == WHITE

        # Move number is 0
        assert state.move_number == 0

    def test_initial_piece_placement(self, game: ChessGame) -> None:
        """Test pieces are in correct starting positions."""
        state = game.initial_state()
        board = state.board

        # White pieces on rows 6-7
        assert board[7, 0] == Piece.ROOK  # White rook
        assert board[7, 4] == Piece.KING  # White king
        assert all(board[6, i] == Piece.PAWN for i in range(8))  # White pawns

        # Black pieces on rows 0-1
        assert board[0, 0] == -Piece.ROOK  # Black rook
        assert board[0, 4] == -Piece.KING  # Black king
        assert all(board[1, i] == -Piece.PAWN for i in range(8))  # Black pawns

    def test_initial_castling_rights(self, game: ChessGame) -> None:
        """Test all castling rights available initially."""
        state = game.initial_state()
        castling = state.metadata["castling_rights"]

        assert castling["K"] is True  # White kingside
        assert castling["Q"] is True  # White queenside
        assert castling["k"] is True  # Black kingside
        assert castling["q"] is True  # Black queenside

    def test_initial_no_en_passant(self, game: ChessGame) -> None:
        """Test no en passant square initially."""
        state = game.initial_state()
        assert state.metadata["en_passant_square"] is None


class TestChessProperties:
    """Tests for Chess game properties."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_action_space_size(self, game: ChessGame) -> None:
        """Test action space size is correct."""
        assert game.action_space_size == ACTION_SPACE_SIZE
        assert game.action_space_size == 8 * 8 * 73  # 4672

    def test_state_channels(self, game: ChessGame) -> None:
        """Test state channels for neural network."""
        assert game.state_channels == 119


class TestChessLegalMoves:
    """Tests for legal move generation."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_initial_legal_moves(self, game: ChessGame) -> None:
        """Test legal moves from initial position."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)

        # White should have 20 legal moves initially
        # 16 pawn moves (8 single + 8 double) + 4 knight moves
        assert len(legal) == 20

    def test_action_mask_shape(self, game: ChessGame) -> None:
        """Test action mask has correct shape."""
        state = game.initial_state()
        mask = game.get_action_mask(state)

        assert isinstance(mask, ActionMask)
        assert len(mask.mask) == ACTION_SPACE_SIZE
        assert mask.action_space_size == ACTION_SPACE_SIZE

    def test_action_mask_matches_legal_actions(self, game: ChessGame) -> None:
        """Test action mask matches legal actions list."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)
        mask = game.get_action_mask(state)

        assert mask.num_legal == len(legal)
        for action in legal:
            assert mask.is_legal(action)


class TestChessApplyAction:
    """Tests for applying moves."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_apply_pawn_move(self, game: ChessGame) -> None:
        """Test applying a pawn move."""
        state = game.initial_state()

        # Find e2-e4 move
        legal = game.get_legal_actions(state)
        e2e4 = game.string_to_action("e2e4", state)

        assert e2e4 is not None
        assert e2e4 in legal

        # Apply move
        new_state = game.apply_action(state, e2e4)

        # Pawn moved
        assert new_state.board[4, 4] == Piece.PAWN  # e4
        assert new_state.board[6, 4] == 0  # e2 empty

        # Player changed
        assert new_state.current_player == BLACK

        # En passant square set
        assert new_state.metadata["en_passant_square"] == (5, 4)  # e3

    def test_apply_knight_move(self, game: ChessGame) -> None:
        """Test applying a knight move."""
        state = game.initial_state()

        # Find Nf3 move
        nf3 = game.string_to_action("g1f3", state)
        assert nf3 is not None

        new_state = game.apply_action(state, nf3)

        # Knight moved
        assert new_state.board[5, 5] == Piece.KNIGHT  # f3
        assert new_state.board[7, 6] == 0  # g1 empty

    def test_player_alternates(self, game: ChessGame) -> None:
        """Test that player alternates after each move."""
        state = game.initial_state()
        assert state.current_player == WHITE

        # White move
        action = game.get_legal_actions(state)[0]
        state = game.apply_action(state, action)
        assert state.current_player == BLACK

        # Black move
        action = game.get_legal_actions(state)[0]
        state = game.apply_action(state, action)
        assert state.current_player == WHITE


class TestChessCastling:
    """Tests for castling."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_castling_rights_lost_on_king_move(self, game: ChessGame) -> None:
        """Test castling rights lost when king moves."""
        # Set up a position where king can move
        state = game.initial_state()

        # Clear pieces between king and make a simple game
        board = state.board.copy()
        # Clear e2 pawn for king to move
        board[6, 4] = 0

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=0,
            move_history=[],
            metadata=state.metadata.copy(),
        )

        # Find king move
        ke2 = game.string_to_action("e1e2", state)
        if ke2 and ke2 in game.get_legal_actions(state):
            new_state = game.apply_action(state, ke2)
            assert new_state.metadata["castling_rights"]["K"] is False
            assert new_state.metadata["castling_rights"]["Q"] is False


class TestChessEnPassant:
    """Tests for en passant."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_en_passant_square_set(self, game: ChessGame) -> None:
        """Test en passant square is set on double pawn move."""
        state = game.initial_state()

        # e2-e4
        e2e4 = game.string_to_action("e2e4", state)
        assert e2e4 is not None

        new_state = game.apply_action(state, e2e4)
        assert new_state.metadata["en_passant_square"] == (5, 4)  # e3

    def test_en_passant_square_cleared(self, game: ChessGame) -> None:
        """Test en passant square is cleared after other moves."""
        state = game.initial_state()

        # e2-e4
        e2e4 = game.string_to_action("e2e4", state)
        state = game.apply_action(state, e2e4)

        # Black plays e7-e6 (not double push)
        e7e6 = game.string_to_action("e7e6", state)
        if e7e6:
            state = game.apply_action(state, e7e6)
            assert state.metadata["en_passant_square"] is None


class TestChessCheck:
    """Tests for check detection."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_initial_not_in_check(self, game: ChessGame) -> None:
        """Test initial position is not in check."""
        state = game.initial_state()
        assert not game._is_in_check(state, WHITE)
        assert not game._is_in_check(state, BLACK)


class TestChessTermination:
    """Tests for game termination."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_initial_not_terminal(self, game: ChessGame) -> None:
        """Test initial position is not terminal."""
        state = game.initial_state()
        assert not game.is_terminal(state)

    def test_insufficient_material_k_vs_k(self, game: ChessGame) -> None:
        """Test King vs King is insufficient material."""
        # Create position with only two kings
        board = np.zeros((8, 8), dtype=np.int8)
        board[0, 4] = -Piece.KING  # Black king
        board[7, 4] = Piece.KING  # White king

        state = GameState(
            board=board,
            current_player=WHITE,
            move_number=0,
            move_history=[],
            metadata={"castling_rights": {}, "en_passant_square": None, "halfmove_clock": 0, "position_history": []},
        )

        assert game._is_insufficient_material(state)
        assert game.is_terminal(state)

        result = game.get_result(state)
        assert result.winner is None
        assert result.reason == "insufficient_material"


class TestChessTensorEncoding:
    """Tests for neural network tensor encoding."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_tensor_shape(self, game: ChessGame) -> None:
        """Test tensor encoding has correct shape."""
        state = game.initial_state()
        tensor = game.to_tensor(state)

        assert tensor.shape == (119, 8, 8)

    def test_tensor_dtype(self, game: ChessGame) -> None:
        """Test tensor encoding has correct dtype."""
        state = game.initial_state()
        tensor = game.to_tensor(state)

        assert tensor.dtype == torch.float32

    def test_tensor_valid_range(self, game: ChessGame) -> None:
        """Test tensor values are in valid range."""
        state = game.initial_state()
        tensor = game.to_tensor(state)

        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_piece_planes_populated(self, game: ChessGame) -> None:
        """Test piece planes are properly populated."""
        state = game.initial_state()
        tensor = game.to_tensor(state)

        # Check that piece planes have some non-zero values
        piece_planes = tensor[:12]  # First 12 planes are current piece positions
        assert piece_planes.sum() > 0


class TestChessSymmetry:
    """Tests for symmetry transformations."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_symmetry_count(self, game: ChessGame) -> None:
        """Test that Chess returns 2 symmetries (original + horizontal flip)."""
        state = game.initial_state()
        policy = np.ones(ACTION_SPACE_SIZE) / ACTION_SPACE_SIZE

        symmetries = game.get_symmetries(state, policy)
        assert len(symmetries) == 2

    def test_symmetry_preserves_policy_sum(self, game: ChessGame) -> None:
        """Test that symmetry preserves policy probability sum."""
        state = game.initial_state()
        policy = np.random.dirichlet(np.ones(ACTION_SPACE_SIZE))

        symmetries = game.get_symmetries(state, policy)

        for sym_state, sym_policy in symmetries:
            # Policy should sum to approximately 1
            assert abs(sym_policy.sum() - 1.0) < 1e-5


class TestChessMoveNotation:
    """Tests for move notation conversion."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_action_to_string(self, game: ChessGame) -> None:
        """Test action to string conversion."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)

        # All legal actions should produce valid strings
        for action in legal:
            move_str = game.action_to_string(action, state)
            assert len(move_str) >= 4
            assert move_str[0] in "abcdefgh"
            assert move_str[1] in "12345678"

    def test_string_to_action_roundtrip(self, game: ChessGame) -> None:
        """Test string to action roundtrip."""
        state = game.initial_state()

        # Test some standard opening moves
        moves = ["e2e4", "d2d4", "g1f3", "b1c3"]
        for move_str in moves:
            action = game.string_to_action(move_str, state)
            if action is not None:  # Move is legal
                recovered = game.action_to_string(action, state)
                assert recovered == move_str


class TestChessGameResult:
    """Tests for game result computation."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_ongoing_game_result(self, game: ChessGame) -> None:
        """Test result for ongoing game."""
        state = game.initial_state()
        result = game.get_result(state)

        assert result.winner is None
        assert result.reason == "game_ongoing"

    def test_get_winner_ongoing(self, game: ChessGame) -> None:
        """Test get_winner for ongoing game."""
        state = game.initial_state()
        winner = game.get_winner(state)

        assert winner is None


class TestChessMoveGeneration:
    """Tests for piece-specific move generation."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_knight_has_two_moves_initially(self, game: ChessGame) -> None:
        """Test each knight has exactly 2 moves from starting position."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)

        # Count knight moves (from g1 and b1)
        knight_moves = []
        for action in legal:
            from_row, from_col, _, _, _ = game._decode_move(action)
            piece = state.board[from_row, from_col]
            if abs(piece) == Piece.KNIGHT:
                knight_moves.append(action)

        assert len(knight_moves) == 4  # 2 moves per knight × 2 knights

    def test_pawn_has_two_moves_initially(self, game: ChessGame) -> None:
        """Test each pawn has exactly 2 moves from starting position."""
        state = game.initial_state()
        legal = game.get_legal_actions(state)

        # Count pawn moves per pawn (each should have 2: single and double push)
        pawn_moves = []
        for action in legal:
            from_row, from_col, _, _, _ = game._decode_move(action)
            piece = state.board[from_row, from_col]
            if abs(piece) == Piece.PAWN:
                pawn_moves.append(action)

        assert len(pawn_moves) == 16  # 2 moves per pawn × 8 pawns


class TestChessEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def game(self) -> ChessGame:
        """Create Chess game instance."""
        return ChessGame()

    def test_invalid_move_notation(self, game: ChessGame) -> None:
        """Test invalid move notation returns None."""
        state = game.initial_state()

        assert game.string_to_action("", state) is None
        assert game.string_to_action("xxx", state) is None
        assert game.string_to_action("z1z2", state) is None

    def test_illegal_move_notation(self, game: ChessGame) -> None:
        """Test illegal move notation returns None."""
        state = game.initial_state()

        # e1e8 is not legal from starting position
        assert game.string_to_action("e1e8", state) is None

    def test_board_size_fixed(self, game: ChessGame) -> None:
        """Test that board size is always 8x8."""
        # Board size parameter is ignored for chess
        state = game.initial_state(board_size=10)
        assert state.board.shape == (8, 8)
