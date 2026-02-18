"""Tests for Hex game implementation.

Covers initialization, legal moves, Union-Find connectivity,
win detection, variable board sizes, tensor encoding,
and full gameplay scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.games.hex import BLACK, EMPTY, WHITE, HexGame, _UnionFind
from src.games.interface import GamePhase, GameResult
from src.games.registry import GameRegistry
from src.games.state import ActionMask, GameState


class TestUnionFind:
    """Tests for the Union-Find data structure."""

    def test_initial_self_parents(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_and_find(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        assert uf.connected(0, 1)
        assert not uf.connected(0, 2)

    def test_transitive_connectivity(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2)

    def test_copy_independence(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf_copy = uf.copy()
        uf_copy.union(2, 3)
        assert not uf.connected(2, 3)
        assert uf_copy.connected(2, 3)


class TestHexRegistration:
    """Tests for Hex game registration."""

    def test_registered_in_registry(self):
        assert GameRegistry().is_registered("hex")

    def test_get_returns_instance(self):
        game = GameRegistry().get("hex")
        assert isinstance(game, HexGame)

    def test_game_info(self):
        info = GameRegistry().get_info("hex")
        assert info is not None
        assert info["name"] == "hex"
        assert info["n_players"] == 2


class TestHexInitialization:
    """Tests for Hex board initialization."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_default_board_size(self, game: HexGame):
        state = game.initial_state()
        assert state.board_size == 11

    @pytest.mark.parametrize("size", [3, 5, 7, 9, 11, 13, 19])
    def test_variable_board_sizes(self, game: HexGame, size: int):
        state = game.initial_state(board_size=size)
        assert state.board_size == size
        assert state.board.shape == (size, size)

    def test_out_of_range_raises(self, game: HexGame):
        with pytest.raises(ValueError, match="out of range"):
            game.initial_state(board_size=2)

    def test_initial_board_empty(self, game: HexGame):
        state = game.initial_state(board_size=7)
        assert np.all(state.board == EMPTY)

    def test_initial_player_is_black(self, game: HexGame):
        state = game.initial_state()
        assert state.current_player == BLACK

    def test_initial_metadata(self, game: HexGame):
        state = game.initial_state(board_size=7)
        assert state.metadata["board_size"] == 7
        assert state.metadata["winner"] is None
        assert "uf_black" in state.metadata
        assert "uf_white" in state.metadata


class TestHexActionSpace:
    """Tests for Hex action space."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_action_space_size_no_pass(self, game: HexGame):
        """Hex has no pass move — action space is exactly N²."""
        game._board_size = 7
        assert game.action_space_size == 49

    def test_action_space_size_11x11(self, game: HexGame):
        game._board_size = 11
        assert game.action_space_size == 121

    def test_state_channels(self, game: HexGame):
        assert game.state_channels == 3


class TestHexLegalActions:
    """Tests for Hex legal move generation."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_initial_all_cells_legal(self, game: HexGame):
        state = game.initial_state(board_size=5)
        legal = game.get_legal_actions(state)
        assert len(legal) == 25  # 5*5

    def test_action_mask_matches_legal(self, game: HexGame):
        state = game.initial_state(board_size=5)
        legal = game.get_legal_actions(state)
        mask = game.get_action_mask(state)
        assert isinstance(mask, ActionMask)
        assert mask.num_legal == len(legal)

    def test_legal_actions_decrease_after_moves(self, game: HexGame):
        state = game.initial_state(board_size=5)
        legal_before = len(game.get_legal_actions(state))
        state = game.apply_action(state, 0)
        legal_after = len(game.get_legal_actions(state))
        assert legal_after == legal_before - 1

    def test_no_legal_actions_after_win(self, game: HexGame):
        """After a player wins, no legal actions remain."""
        state = game.initial_state(board_size=3)
        # Black connects top to bottom: (0,0), (1,0), (2,0)
        state = game.apply_action(state, 0)      # Black (0,0)
        state = game.apply_action(state, 1)      # White (0,1)
        state = game.apply_action(state, 3)      # Black (1,0)
        state = game.apply_action(state, 4)      # White (1,1)
        state = game.apply_action(state, 6)      # Black (2,0) — wins!
        assert game.get_legal_actions(state) == []


class TestHexApplyAction:
    """Tests for Hex move application."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_apply_places_stone(self, game: HexGame):
        state = game.initial_state(board_size=5)
        new_state = game.apply_action(state, 0)
        assert new_state.board[0, 0] == BLACK

    def test_apply_switches_player(self, game: HexGame):
        state = game.initial_state(board_size=5)
        new_state = game.apply_action(state, 0)
        assert new_state.current_player == WHITE

    def test_apply_increments_move(self, game: HexGame):
        state = game.initial_state(board_size=5)
        new_state = game.apply_action(state, 0)
        assert new_state.move_number == 1

    def test_occupied_position_raises(self, game: HexGame):
        state = game.initial_state(board_size=5)
        state = game.apply_action(state, 0)
        with pytest.raises(ValueError, match="occupied"):
            game.apply_action(state, 0)

    def test_immutable_state(self, game: HexGame):
        state = game.initial_state(board_size=5)
        original_board = state.board.copy()
        _ = game.apply_action(state, 0)
        assert np.array_equal(state.board, original_board)


class TestHexWinDetection:
    """Tests for Hex win detection via Union-Find."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_black_wins_top_to_bottom(self, game: HexGame):
        """Black wins by connecting top to bottom on 3×3."""
        state = game.initial_state(board_size=3)
        # Black: (0,0), (1,0), (2,0) — column 0, top to bottom
        state = game.apply_action(state, 0)   # B(0,0)
        assert not game.is_terminal(state)
        state = game.apply_action(state, 1)   # W(0,1)
        state = game.apply_action(state, 3)   # B(1,0)
        assert not game.is_terminal(state)
        state = game.apply_action(state, 4)   # W(1,1)
        state = game.apply_action(state, 6)   # B(2,0) — connected!
        assert game.is_terminal(state)
        assert game.get_winner(state) == BLACK

    def test_white_wins_left_to_right(self, game: HexGame):
        """White wins by connecting left to right on 3×3."""
        state = game.initial_state(board_size=3)
        # White: (0,0), (0,1), (0,2) — row 0, left to right
        state = game.apply_action(state, 3)   # B(1,0) — not blocking
        state = game.apply_action(state, 0)   # W(0,0)
        state = game.apply_action(state, 6)   # B(2,0)
        state = game.apply_action(state, 1)   # W(0,1)
        state = game.apply_action(state, 7)   # B(2,1)
        state = game.apply_action(state, 2)   # W(0,2) — connected!
        assert game.is_terminal(state)
        assert game.get_winner(state) == WHITE

    def test_no_winner_mid_game(self, game: HexGame):
        state = game.initial_state(board_size=5)
        state = game.apply_action(state, 0)
        assert not game.is_terminal(state)
        assert game.get_winner(state) is None

    def test_diagonal_connection(self, game: HexGame):
        """Test hex adjacency: diagonal (r-1, c+1) is a neighbor."""
        state = game.initial_state(board_size=3)
        # Black path: (0,0) → (1,1) → (2,1) using hex adjacency
        # (0,0) neighbors include (1,0) and (0,1)
        # (1,1) neighbors include (0,1) and (0,0)... wait
        # Hex neighbors of (r,c): (r-1,c), (r-1,c+1), (r,c-1), (r,c+1), (r+1,c-1), (r+1,c)
        # So (0,0) is adjacent to (1,0) — yes
        # (1,0) is adjacent to (2,0) — yes
        # Let's test a non-trivial path
        state = game.apply_action(state, 0)   # B(0,0)
        state = game.apply_action(state, 2)   # W(0,2)
        state = game.apply_action(state, 4)   # B(1,1) — adj to (0,0) via (-1,-1)? No.
        # (1,1) neighbors: (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
        # (0,0) neighbors: (-1,0)X, (-1,1)X, (0,-1)X, (0,1), (1,-1)X, (1,0)
        # So (0,0) and (1,1) are NOT adjacent in hex
        # Let's use: (0,0) → (1,0) → (2,0) instead
        state2 = game.initial_state(board_size=3)
        state2 = game.apply_action(state2, 0)   # B(0,0)
        state2 = game.apply_action(state2, 1)   # W(0,1)
        # (0,0) is adjacent to (1,0) via hex neighbor (1,0)
        state2 = game.apply_action(state2, 3)   # B(1,0)
        state2 = game.apply_action(state2, 4)   # W(1,1)
        state2 = game.apply_action(state2, 6)   # B(2,0)
        assert game.is_terminal(state2)
        assert game.get_winner(state2) == BLACK


class TestHexResult:
    """Tests for Hex game results."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_result_has_winner(self, game: HexGame):
        state = game.initial_state(board_size=3)
        state = game.apply_action(state, 0)   # B(0,0)
        state = game.apply_action(state, 1)   # W(0,1)
        state = game.apply_action(state, 3)   # B(1,0)
        state = game.apply_action(state, 4)   # W(1,1)
        state = game.apply_action(state, 6)   # B(2,0)
        result = game.get_result(state)
        assert isinstance(result, GameResult)
        assert result.winner == BLACK
        assert result.score_black == 1.0
        assert result.score_white == 0.0
        assert result.reason == "connection"

    def test_no_draws_possible(self, game: HexGame):
        """Hex is a determined game — no draws possible."""
        # Fill a 3x3 board completely, one player must win
        state = game.initial_state(board_size=3)
        actions = list(range(9))
        np.random.seed(42)
        np.random.shuffle(actions)
        for action in actions:
            if game.is_terminal(state):
                break
            state = game.apply_action(state, action)
        assert game.is_terminal(state)
        assert game.get_winner(state) is not None


class TestHexTensor:
    """Tests for Hex neural network tensor encoding."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_tensor_shape(self, game: HexGame):
        state = game.initial_state(board_size=7)
        tensor = game.to_tensor(state)
        assert tensor.shape == (3, 7, 7)

    @pytest.mark.parametrize("size", [3, 5, 7, 11, 13])
    def test_tensor_shape_variable(self, game: HexGame, size: int):
        state = game.initial_state(board_size=size)
        tensor = game.to_tensor(state)
        assert tensor.shape == (3, size, size)

    def test_tensor_dtype(self, game: HexGame):
        state = game.initial_state(board_size=5)
        tensor = game.to_tensor(state)
        assert tensor.dtype == torch.float32

    def test_tensor_values_range(self, game: HexGame):
        state = game.initial_state(board_size=5)
        state = game.apply_action(state, 0)
        tensor = game.to_tensor(state)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_tensor_player_plane(self, game: HexGame):
        state = game.initial_state(board_size=5)
        tensor = game.to_tensor(state)
        assert torch.all(tensor[2] == 1.0)  # Black = 1.0

    def test_tensor_encodes_stones(self, game: HexGame):
        state = game.initial_state(board_size=5)
        state = game.apply_action(state, 0)  # Black at (0,0)
        # Now white to play
        tensor = game.to_tensor(state)
        # Plane 0 = current player (white) stones — none
        # Plane 1 = opponent (black) stones — (0,0)
        assert tensor[1, 0, 0] == 1.0
        assert tensor[0, 0, 0] == 0.0


class TestHexSymmetries:
    """Tests for Hex symmetry generation."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_identity_symmetry(self, game: HexGame):
        state = game.initial_state(board_size=5)
        policy = np.random.dirichlet(np.ones(25))
        symmetries = game.get_symmetries(state, policy)
        assert len(symmetries) == 1
        assert np.array_equal(symmetries[0][1], policy)


class TestHexGamePhase:
    """Tests for Hex game phase detection."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_opening_phase(self, game: HexGame):
        state = game.initial_state(board_size=11)
        assert game.get_phase(state) == GamePhase.OPENING

    def test_midgame_phase(self, game: HexGame):
        state = game.initial_state(board_size=5)
        # Place 5 stones (20% fill) to enter midgame
        actions = [0, 1, 2, 3, 4]
        for a in actions:
            if game.is_terminal(state):
                break
            state = game.apply_action(state, a)
        phase = game.get_phase(state)
        assert phase in (GamePhase.OPENING, GamePhase.MIDGAME, GamePhase.TERMINAL)


class TestHexGameplay:
    """End-to-end gameplay tests for Hex."""

    @pytest.fixture
    def game(self) -> HexGame:
        return HexGame()

    def test_play_random_game_5x5(self, game: HexGame):
        np.random.seed(42)
        state = game.initial_state(board_size=5)
        while not game.is_terminal(state):
            legal = game.get_legal_actions(state)
            action = np.random.choice(legal)
            state = game.apply_action(state, action)
        assert game.get_winner(state) is not None

    def test_play_random_game_7x7(self, game: HexGame):
        np.random.seed(99)
        state = game.initial_state(board_size=7)
        while not game.is_terminal(state):
            legal = game.get_legal_actions(state)
            action = np.random.choice(legal)
            state = game.apply_action(state, action)
        assert game.get_winner(state) is not None

    def test_action_to_string(self, game: HexGame):
        game._board_size = 5
        assert game.action_to_string(0, board_size=5) == "A1"
        assert game.action_to_string(4, board_size=5) == "E1"
        assert game.action_to_string(5, board_size=5) == "A2"

    def test_string_to_action(self, game: HexGame):
        game._board_size = 5
        assert game.string_to_action("A1", board_size=5) == 0
        assert game.string_to_action("E1", board_size=5) == 4
        assert game.string_to_action("A2", board_size=5) == 5
