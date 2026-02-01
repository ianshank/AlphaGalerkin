"""Tests for game manager."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.board import KOMI_BY_SIZE, SpaceConfig
from src.game_manager import GameManager, GameSession
from src.tools.gtp import SimpleGoGame


class TestGameSession:
    """Tests for GameSession dataclass."""

    def test_move_count_empty(self) -> None:
        """Test move count for empty history."""
        game = SimpleGoGame(9)
        session = GameSession(
            game=game,
            board_size=9,
            komi=5.5,
            move_history=[],
        )

        assert session.move_count == 0

    def test_move_count_with_moves(self) -> None:
        """Test move count with history."""
        game = SimpleGoGame(9)
        session = GameSession(
            game=game,
            board_size=9,
            komi=5.5,
            move_history=[(4, 4), (3, 3), "PASS"],
        )

        assert session.move_count == 3

    def test_current_player_name_black(self) -> None:
        """Test current player name when black to play."""
        game = SimpleGoGame(9)
        session = GameSession(game=game, board_size=9, komi=5.5)

        assert session.current_player_name == "Black"

    def test_current_player_name_white(self) -> None:
        """Test current player name when white to play."""
        game = SimpleGoGame(9)
        game.play(4, 4)  # Black plays
        session = GameSession(game=game, board_size=9, komi=5.5)

        assert session.current_player_name == "White"

    def test_is_terminal_false(self) -> None:
        """Test is_terminal for ongoing game."""
        game = SimpleGoGame(9)
        session = GameSession(game=game, board_size=9, komi=5.5)

        assert session.is_terminal is False

    def test_is_terminal_true(self) -> None:
        """Test is_terminal after two passes."""
        game = SimpleGoGame(9)
        game.play_pass()
        game.play_pass()
        session = GameSession(game=game, board_size=9, komi=5.5)

        assert session.is_terminal is True

    def test_is_zero_shot_9x9(self) -> None:
        """Test is_zero_shot for training size."""
        game = SimpleGoGame(9)
        session = GameSession(game=game, board_size=9, komi=5.5)

        assert session.is_zero_shot is False

    def test_is_zero_shot_13x13(self) -> None:
        """Test is_zero_shot for 13x13."""
        game = SimpleGoGame(13)
        session = GameSession(game=game, board_size=13, komi=6.5)

        assert session.is_zero_shot is True

    def test_is_zero_shot_19x19(self) -> None:
        """Test is_zero_shot for 19x19."""
        game = SimpleGoGame(19)
        session = GameSession(game=game, board_size=19, komi=7.5)

        assert session.is_zero_shot is True


class TestGameManager:
    """Tests for GameManager class."""

    def test_initialization_default(self) -> None:
        """Test initialization with default config."""
        manager = GameManager()
        assert manager.config is not None
        assert manager.evaluator is None

    def test_initialization_custom_config(
        self,
        space_config: SpaceConfig,
    ) -> None:
        """Test initialization with custom config."""
        manager = GameManager(config=space_config)
        assert manager.config == space_config

    def test_create_game_default_size(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating game with default size."""
        session = game_manager.create_game()

        assert session.board_size == 9
        assert session.komi == 5.5
        assert session.game.board_size == 9

    def test_create_game_9x9(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating 9x9 game."""
        session = game_manager.create_game(board_size=9)

        assert session.board_size == 9
        assert session.komi == KOMI_BY_SIZE[9]

    def test_create_game_13x13(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating 13x13 game."""
        session = game_manager.create_game(board_size=13)

        assert session.board_size == 13
        assert session.komi == KOMI_BY_SIZE[13]

    def test_create_game_19x19(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating 19x19 game."""
        session = game_manager.create_game(board_size=19)

        assert session.board_size == 19
        assert session.komi == KOMI_BY_SIZE[19]

    def test_create_game_unsupported_size(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating game with unsupported size raises error."""
        with pytest.raises(ValueError, match="not in supported sizes"):
            game_manager.create_game(board_size=7)

    def test_create_game_human_vs_ai(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating human vs AI game."""
        session = game_manager.create_game(is_human_vs_ai=True)

        assert session.is_human_vs_ai is True

    def test_create_game_ai_vs_ai(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test creating AI vs AI game."""
        session = game_manager.create_game(is_human_vs_ai=False)

        assert session.is_human_vs_ai is False

    def test_replay_history_empty(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test replaying empty history."""
        game = game_manager.replay_history([], 9)

        assert game.board_size == 9
        assert np.all(game.board == 0)

    def test_replay_history_with_moves(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test replaying history with moves."""
        history = [(4, 4), (3, 3), "PASS", (5, 5)]
        game = game_manager.replay_history(history, 9)

        assert game.board[4, 4] == SimpleGoGame.BLACK
        assert game.board[3, 3] == SimpleGoGame.WHITE
        assert game.board[5, 5] == SimpleGoGame.BLACK

    def test_replay_history_all_passes(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test replaying history of all passes."""
        history = ["PASS", "PASS"]
        game = game_manager.replay_history(history, 9)

        assert game.is_terminal()

    def test_get_score_display(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test score display formatting."""
        session = game_manager.create_game(9)
        display = game_manager.get_score_display(session)

        assert "Black captures" in display
        assert "White captures" in display
        assert "Move: 0" in display
        assert "Komi: 5.5" in display

    def test_get_score_display_zero_shot(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test score display shows zero-shot tag."""
        session = game_manager.create_game(19)
        display = game_manager.get_score_display(session)

        assert "[Zero-shot]" in display

    def test_calculate_final_score_black_wins(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test final score calculation when black wins."""
        session = game_manager.create_game(9)
        # Play many black stones
        for r in range(3):
            for c in range(3):
                if session.game.play(r, c):
                    session.game.play_pass()

        session.game.play_pass()
        session.game.play_pass()

        score = game_manager.calculate_final_score(session)

        assert "wins" in score or "Draw" in score

    def test_calculate_final_score_white_wins_with_komi(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test final score calculation with komi advantage."""
        session = game_manager.create_game(9)
        # End game immediately - white wins with komi
        session.game.play_pass()
        session.game.play_pass()

        score = game_manager.calculate_final_score(session)

        # White should win due to komi
        assert "White wins" in score

    def test_get_board_size_label_9x9(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test board size label for training size."""
        label = game_manager.get_board_size_label(9)

        assert "9×9" in label
        assert "Training" in label

    def test_get_board_size_label_13x13(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test board size label for 13x13."""
        label = game_manager.get_board_size_label(13)

        assert "13×13" in label
        assert "Zero-shot" in label

    def test_get_board_size_label_19x19(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test board size label for 19x19."""
        label = game_manager.get_board_size_label(19)

        assert "19×19" in label
        assert "Zero-shot" in label

    def test_get_board_size_choices(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test get_board_size_choices returns proper tuples."""
        choices = game_manager.get_board_size_choices()

        assert len(choices) == 3
        for label, value in choices:
            assert isinstance(label, str)
            assert isinstance(value, int)
            assert value in [9, 13, 19]

    def test_format_move(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test move formatting."""
        # Center of 9x9 (row 4, col 4) should be E5
        formatted = game_manager.format_move(4, 4, 9)
        assert "E" in formatted
        assert "5" in formatted

    def test_format_move_corner(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test move formatting for corner."""
        # Top-left corner (row 0, col 0) should be A9 on 9x9
        formatted = game_manager.format_move(0, 0, 9)
        assert "A" in formatted
        assert "9" in formatted

    def test_parse_move_valid(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing valid move input."""
        move = game_manager.parse_move("4,4", 9)
        assert move == (4, 4)

    def test_parse_move_pass(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing PASS move."""
        move = game_manager.parse_move("PASS", 9)
        assert move == "PASS"

    def test_parse_move_pass_lowercase(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing lowercase pass."""
        move = game_manager.parse_move("pass", 9)
        assert move == "PASS"

    def test_parse_move_with_spaces(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing move with extra spaces."""
        move = game_manager.parse_move("  4,4  ", 9)
        assert move == (4, 4)

    def test_parse_move_invalid_format(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid format"):
            game_manager.parse_move("invalid", 9)

    def test_parse_move_out_of_bounds(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing out-of-bounds move raises error."""
        with pytest.raises(ValueError, match="outside"):
            game_manager.parse_move("10,10", 9)

    def test_parse_move_non_numeric(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test parsing non-numeric coordinates raises error."""
        with pytest.raises(ValueError, match="numbers"):
            game_manager.parse_move("a,b", 9)


class TestGameManagerMCTSConfig:
    """Tests for MCTS configuration in GameManager."""

    def test_default_mcts_kwargs(
        self,
        game_manager: GameManager,
    ) -> None:
        """Test default MCTS kwargs are set."""
        assert "n_simulations" in game_manager.mcts_kwargs
        assert "c_puct" in game_manager.mcts_kwargs

    def test_custom_mcts_kwargs(
        self,
        space_config: SpaceConfig,
    ) -> None:
        """Test custom MCTS kwargs are used."""
        custom_kwargs = {
            "n_simulations": 100,
            "c_puct": 2.0,
        }
        manager = GameManager(config=space_config, mcts_kwargs=custom_kwargs)

        assert manager.mcts_kwargs["n_simulations"] == 100
        assert manager.mcts_kwargs["c_puct"] == 2.0
