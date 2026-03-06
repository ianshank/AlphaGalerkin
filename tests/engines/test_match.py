"""Tests for match orchestration framework.

Tests match lifecycle, game recording, PGN generation,
result aggregation, and the EngineMatch orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.config import MatchConfig
from src.engines.elo import EloEstimate
from src.engines.match import EngineMatch, GameRecord, MatchResult
from src.games.chess import ChessGame
from src.games.interface import GameResult
from src.games.state import GameState


class TestGameRecord:
    """Tests for GameRecord dataclass."""

    def test_default_values(self) -> None:
        record = GameRecord()
        assert record.moves == []
        assert record.result == ""
        assert record.move_count == 0

    def test_populated_record(self) -> None:
        record = GameRecord(
            moves=["e2e4", "e7e5", "g1f3"],
            result="1-0",
            result_reason="checkmate",
            model_color="white",
            move_count=3,
        )
        assert len(record.moves) == 3
        assert record.result == "1-0"


class TestMatchResult:
    """Tests for MatchResult aggregation."""

    def test_empty_result(self) -> None:
        result = MatchResult()
        assert result.total_games == 0
        assert result.win_rate == 0.0

    def test_win_rate_all_wins(self) -> None:
        result = MatchResult(wins=10, losses=0, draws=0)
        assert result.win_rate == pytest.approx(1.0)

    def test_win_rate_all_losses(self) -> None:
        result = MatchResult(wins=0, losses=10, draws=0)
        assert result.win_rate == pytest.approx(0.0)

    def test_win_rate_all_draws(self) -> None:
        result = MatchResult(wins=0, losses=0, draws=10)
        assert result.win_rate == pytest.approx(0.5)

    def test_win_rate_mixed(self) -> None:
        result = MatchResult(wins=6, losses=2, draws=4)
        # (6 + 0.5*4) / 12 = 8/12 ≈ 0.667
        assert result.win_rate == pytest.approx(8.0 / 12.0)

    def test_to_dict(self) -> None:
        result = MatchResult(
            wins=5,
            losses=3,
            draws=2,
            elo_estimate=EloEstimate(
                elo_difference=100.0,
                confidence_interval=(50.0, 150.0),
                likelihood_of_superiority=0.85,
                win_rate=0.6,
            ),
        )
        d = result.to_dict()
        assert d["wins"] == 5
        assert d["elo_difference"] == 100.0
        assert d["los"] == 0.85

    def test_to_dict_without_elo(self) -> None:
        result = MatchResult(wins=5, losses=3, draws=2)
        d = result.to_dict()
        assert "elo_difference" not in d


class TestMatchConfig:
    """Tests for match configuration edge cases."""

    def test_pgn_output_path(self, tmp_path: Path) -> None:
        config = MatchConfig(
            name="test",
            n_games=10,
            pgn_output_path=tmp_path / "output.pgn",
        )
        assert config.pgn_output_path is not None

    def test_custom_opening(self) -> None:
        config = MatchConfig(
            name="test",
            opening_fen="r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        )
        assert config.opening_fen is not None


class TestEngineMatchPlayMatch:
    """Tests for EngineMatch.play_match() orchestration.

    Uses mocked model, MCTS, and engine to test the full
    match loop without requiring real chess play.
    """

    @pytest.fixture
    def chess_game(self) -> ChessGame:
        return ChessGame()

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock AlphaGalerkinModel."""
        model = MagicMock()
        model.eval.return_value = model
        model.to.return_value = model
        return model

    @pytest.fixture
    def engine_config(self) -> MagicMock:
        """Create a minimal UCIConfig mock."""
        from src.engines.config import UCIConfig

        return UCIConfig(
            name="test_engine",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )

    @pytest.fixture
    def match_config_2_games(self) -> MatchConfig:
        return MatchConfig(
            name="test",
            n_games=2,
            max_moves=50,
            alternate_colors=True,
        )

    def _make_mock_game(
        self,
        terminal_after: int = 4,
        winner: int = 0,
    ) -> MagicMock:
        """Create a mock game that terminates after N moves."""
        game = MagicMock()
        game.action_space_size = 4672

        move_count = 0
        initial_state = MagicMock(spec=GameState)
        initial_state.current_player = 1
        initial_state.board = np.zeros((8, 8), dtype=np.int8)
        initial_state.metadata = {
            "castling_rights": {"K": True, "Q": True, "k": True, "q": True},
            "en_passant_square": None,
            "halfmove_clock": 0,
        }
        initial_state.move_number = 0

        def is_terminal(state: GameState) -> bool:
            return state.move_number >= terminal_after

        def get_legal_actions(state: GameState) -> list[int]:
            if state.move_number >= terminal_after:
                return []
            return [0, 1, 2, 3]

        def apply_action(state: GameState, action: int) -> GameState:
            new_state = MagicMock(spec=GameState)
            new_state.move_number = state.move_number + 1
            new_state.current_player = -state.current_player
            new_state.board = state.board
            new_state.metadata = state.metadata
            return new_state

        def get_result(state: GameState) -> GameResult:
            return GameResult(
                winner=winner,
                score_black=0.0,
                score_white=0.0,
                reason="adjudication",
                move_count=state.move_number,
            )

        def action_to_string(action: int, state: GameState | None = None) -> str:
            moves = ["e2e4", "e7e5", "d2d4", "d7d5"]
            return moves[action % len(moves)]

        def string_to_action(move_str: str, state: GameState) -> int | None:
            move_map = {"e2e4": 0, "e7e5": 1, "d2d4": 2, "d7d5": 3}
            return move_map.get(move_str)

        game.initial_state.return_value = initial_state
        game.is_terminal = is_terminal
        game.get_legal_actions = get_legal_actions
        game.apply_action = apply_action
        game.get_result = get_result
        game.action_to_string = action_to_string
        game.string_to_action = string_to_action

        return game

    def test_play_match_draws(
        self,
        mock_model: MagicMock,
        match_config_2_games: MatchConfig,
    ) -> None:
        """Test a match where all games end in draws."""
        from src.engines.config import UCIConfig

        game = self._make_mock_game(terminal_after=4, winner=0)
        engine_config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )

        match = EngineMatch(
            model=mock_model,
            engine_config=engine_config,
            match_config=match_config_2_games,
            game=game,
        )

        with patch.object(match, "_play_single_game") as mock_play:
            mock_play.return_value = (
                0.0,
                GameRecord(result="1/2-1/2", result_reason="draw", model_color="white"),
            )

            result = match.play_match()

        assert result.draws == 2
        assert result.wins == 0
        assert result.losses == 0
        assert result.elo_estimate is not None

    @patch("src.engines.match.fen_to_state")
    @patch("src.engines.match.state_to_fen")
    def test_play_match_color_alternation(
        self,
        mock_state_to_fen: MagicMock,
        mock_fen_to_state: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test that colors alternate between games."""
        from src.engines.config import UCIConfig

        game = self._make_mock_game(terminal_after=2, winner=0)
        engine_config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )
        match_config = MatchConfig(
            name="test",
            n_games=4,
            max_moves=50,
            alternate_colors=True,
        )

        match = EngineMatch(
            model=mock_model,
            engine_config=engine_config,
            match_config=match_config,
            game=game,
        )

        colors_played: list[str] = []

        def mock_play(model_is_white: bool, opening_fen: str) -> tuple[float, GameRecord]:
            color = "white" if model_is_white else "black"
            colors_played.append(color)
            return 0.0, GameRecord(
                result="1/2-1/2",
                result_reason="draw",
                model_color=color,
            )

        with patch.object(match, "_play_single_game", side_effect=mock_play):
            match.play_match()

        assert colors_played == ["white", "black", "white", "black"]

    @patch("src.engines.match.fen_to_state")
    @patch("src.engines.match.state_to_fen")
    def test_play_match_engine_crash_counts_as_win(
        self,
        mock_state_to_fen: MagicMock,
        mock_fen_to_state: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Engine crash during a game should count as a model win."""
        from src.engines.config import UCIConfig
        from src.engines.protocol import EngineCrashError

        game = self._make_mock_game(terminal_after=4, winner=0)
        engine_config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )
        match_config = MatchConfig(name="test", n_games=1, max_moves=50)

        match = EngineMatch(
            model=mock_model,
            engine_config=engine_config,
            match_config=match_config,
            game=game,
        )

        with patch.object(
            match,
            "_play_single_game",
            side_effect=EngineCrashError("Process died"),
        ):
            result = match.play_match()

        assert result.wins == 1
        assert result.losses == 0
        assert result.games[0].result == "engine_error"

    @patch("src.engines.match.fen_to_state")
    @patch("src.engines.match.state_to_fen")
    def test_play_match_win_loss_tracking(
        self,
        mock_state_to_fen: MagicMock,
        mock_fen_to_state: MagicMock,
        mock_model: MagicMock,
    ) -> None:
        """Test correct win/loss/draw counting across multiple games."""
        from src.engines.config import UCIConfig

        game = self._make_mock_game()
        engine_config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )
        match_config = MatchConfig(name="test", n_games=3, max_moves=50)

        match = EngineMatch(
            model=mock_model,
            engine_config=engine_config,
            match_config=match_config,
            game=game,
        )

        outcomes = [
            (1.0, GameRecord(result="1-0", result_reason="checkmate", model_color="white")),
            (-1.0, GameRecord(result="0-1", result_reason="checkmate", model_color="black")),
            (0.0, GameRecord(result="1/2-1/2", result_reason="stalemate", model_color="white")),
        ]

        with patch.object(match, "_play_single_game", side_effect=outcomes):
            result = match.play_match()

        assert result.wins == 1
        assert result.losses == 1
        assert result.draws == 1
        assert result.total_games == 3

    @patch("src.engines.match.fen_to_state")
    @patch("src.engines.match.state_to_fen")
    def test_play_match_pgn_output(
        self,
        mock_state_to_fen: MagicMock,
        mock_fen_to_state: MagicMock,
        mock_model: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test PGN file output when configured."""
        from src.engines.config import UCIConfig

        game = self._make_mock_game()
        engine_config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )
        pgn_path = tmp_path / "test_output.pgn"
        match_config = MatchConfig(
            name="test",
            n_games=1,
            max_moves=50,
            pgn_output_path=pgn_path,
        )

        match = EngineMatch(
            model=mock_model,
            engine_config=engine_config,
            match_config=match_config,
            game=game,
        )

        record = GameRecord(
            moves=["e2e4", "e7e5"],
            result="1/2-1/2",
            result_reason="draw",
            model_color="white",
            move_count=2,
        )

        with patch.object(match, "_play_single_game", return_value=(0.0, record)):
            result = match.play_match()

        assert pgn_path.exists()
        pgn_content = pgn_path.read_text()
        assert '[Event "AlphaGalerkin Match"]' in pgn_content
        assert "e2e4" in pgn_content
        assert "1/2-1/2" in pgn_content


class TestGameToPgn:
    """Tests for PGN generation."""

    @pytest.fixture
    def match(self) -> EngineMatch:
        from src.engines.config import UCIConfig

        model = MagicMock()
        game = MagicMock()
        engine_config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=5,
        )
        match_config = MatchConfig(name="test", n_games=1, max_moves=50)
        return EngineMatch(
            model=model,
            engine_config=engine_config,
            match_config=match_config,
            game=game,
        )

    def test_basic_pgn(self, match: EngineMatch) -> None:
        record = GameRecord(
            moves=["e2e4", "e7e5", "g1f3", "b8c6"],
            result="1-0",
            model_color="white",
            move_count=4,
        )
        pgn = match._game_to_pgn(record, 0)
        assert '[White "AlphaGalerkin"]' in pgn
        assert '[Black "Engine"]' in pgn
        assert '[Result "1-0"]' in pgn
        assert "1. e2e4 e7e5 2. g1f3 b8c6 1-0" in pgn

    def test_pgn_model_is_black(self, match: EngineMatch) -> None:
        record = GameRecord(
            moves=["e2e4", "e7e5"],
            result="0-1",
            model_color="black",
            move_count=2,
        )
        pgn = match._game_to_pgn(record, 0)
        assert '[White "Engine"]' in pgn
        assert '[Black "AlphaGalerkin"]' in pgn

    def test_pgn_custom_opening_fen(self, match: EngineMatch) -> None:
        custom_fen = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
        record = GameRecord(
            moves=["d2d4"],
            result="1/2-1/2",
            model_color="white",
            opening_fen=custom_fen,
            move_count=1,
        )
        pgn = match._game_to_pgn(record, 0)
        assert f'[FEN "{custom_fen}"]' in pgn
        assert '[SetUp "1"]' in pgn
