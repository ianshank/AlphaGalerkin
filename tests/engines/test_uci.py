"""Tests for UCI protocol implementation.

Tests command formatting, response parsing, lifecycle management,
and error handling. All tests use mocked subprocesses.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.engines.config import UCIConfig
from src.engines.protocol import (
    EngineCrashError,
    EngineStartupError,
)
from src.engines.uci import UCIEngine

from .conftest import MockUCIProcess


class TestUCIEngineStart:
    """Tests for engine startup and handshake."""

    def test_successful_start(self, mock_uci_engine: UCIEngine) -> None:
        assert mock_uci_engine._started is True
        assert mock_uci_engine.engine_name == "MockFish"

    def test_start_file_not_found(self, uci_config: UCIConfig) -> None:
        with patch("subprocess.Popen", side_effect=FileNotFoundError):
            engine = UCIEngine(uci_config)
            with pytest.raises(EngineStartupError, match="not found"):
                engine.start()

    def test_start_permission_denied(self, uci_config: UCIConfig) -> None:
        with patch("subprocess.Popen", side_effect=PermissionError):
            engine = UCIEngine(uci_config)
            with pytest.raises(EngineStartupError, match="Permission denied"):
                engine.start()

    def test_start_os_error(self, uci_config: UCIConfig) -> None:
        with patch("subprocess.Popen", side_effect=OSError("No such device")):
            engine = UCIEngine(uci_config)
            with pytest.raises(EngineStartupError, match="Failed to start"):
                engine.start()

    def test_double_start_is_noop(self, mock_uci_engine: UCIEngine) -> None:
        # Second start should do nothing
        mock_uci_engine.start()
        assert mock_uci_engine._started is True


class TestUCIEngineCommands:
    """Tests for UCI command sending and response parsing."""

    def test_set_position_fen(self, mock_uci_engine: UCIEngine) -> None:
        # Should not raise
        mock_uci_engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_set_position_with_moves(self, mock_uci_engine: UCIEngine) -> None:
        mock_uci_engine.set_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            moves=["e2e4", "e7e5"],
        )

    def test_go_returns_bestmove(self, mock_uci_engine: UCIEngine) -> None:
        mock_uci_engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        best_move, info = mock_uci_engine.go()
        assert best_move == "e2e4"

    def test_go_returns_info(self, mock_uci_engine: UCIEngine) -> None:
        mock_uci_engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        _, info = mock_uci_engine.go()
        assert info.get("depth") == 10
        assert info.get("score_cp") == 35
        assert info.get("nodes") == 12345

    def test_new_game(self, mock_uci_engine: UCIEngine) -> None:
        """Test ucinewgame command does not raise."""
        mock_uci_engine.new_game()

    def test_stop(self, mock_uci_engine: UCIEngine) -> None:
        """Test stop command does not raise."""
        mock_uci_engine.stop()


class TestUCIEngineInfoParsing:
    """Tests for UCI info line parsing."""

    def test_parse_depth_score(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 20 score cp -50 nodes 999999 nps 2000000",
            {},
        )
        assert info["depth"] == 20
        assert info["score_cp"] == -50
        assert info["nodes"] == 999999
        assert info["nps"] == 2000000

    def test_parse_mate_score(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 15 score mate 3",
            {},
        )
        assert info["score_mate"] == 3

    def test_parse_negative_mate(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 15 score mate -2",
            {},
        )
        assert info["score_mate"] == -2

    def test_parse_pv(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 10 pv e2e4 e7e5 g1f3",
            {},
        )
        assert info["pv"] == ["e2e4", "e7e5", "g1f3"]

    def test_parse_hashfull(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 10 hashfull 500",
            {},
        )
        assert info["hashfull"] == 500

    def test_parse_seldepth(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 20 seldepth 30",
            {},
        )
        assert info["seldepth"] == 30

    def test_parse_time(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 10 time 1500",
            {},
        )
        assert info["time_ms"] == 1500

    def test_parse_multipv(self, mock_uci_engine: UCIEngine) -> None:
        info = mock_uci_engine._parse_info_line(
            "info depth 10 multipv 2 score cp 30",
            {},
        )
        assert info["multipv"] == 2
        assert info["score_cp"] == 30

    def test_parse_unknown_tokens_skipped(self, mock_uci_engine: UCIEngine) -> None:
        """Unknown tokens should be skipped without error."""
        info = mock_uci_engine._parse_info_line(
            "info depth 10 string this is a test score cp 50",
            {},
        )
        assert info["depth"] == 10

    def test_parse_updates_existing_info(self, mock_uci_engine: UCIEngine) -> None:
        """Parsing should update existing info dict."""
        existing = {"depth": 5, "score_cp": 10}
        info = mock_uci_engine._parse_info_line(
            "info depth 20 score cp 50",
            existing,
        )
        assert info["depth"] == 20
        assert info["score_cp"] == 50


class TestUCIEngineBestmoveParsing:
    """Tests for bestmove line parsing."""

    def test_parse_simple_bestmove(self, mock_uci_engine: UCIEngine) -> None:
        move = mock_uci_engine._parse_bestmove("bestmove e2e4")
        assert move == "e2e4"

    def test_parse_bestmove_with_ponder(self, mock_uci_engine: UCIEngine) -> None:
        move = mock_uci_engine._parse_bestmove("bestmove e2e4 ponder d7d5")
        assert move == "e2e4"

    def test_parse_promotion(self, mock_uci_engine: UCIEngine) -> None:
        move = mock_uci_engine._parse_bestmove("bestmove e7e8q")
        assert move == "e7e8q"

    def test_malformed_bestmove(self, mock_uci_engine: UCIEngine) -> None:
        with pytest.raises(EngineCrashError, match="Malformed"):
            mock_uci_engine._parse_bestmove("garbage line")

    def test_empty_bestmove_line(self, mock_uci_engine: UCIEngine) -> None:
        with pytest.raises(EngineCrashError, match="Malformed"):
            mock_uci_engine._parse_bestmove("")

    def test_bestmove_only_keyword(self, mock_uci_engine: UCIEngine) -> None:
        with pytest.raises(EngineCrashError, match="Malformed"):
            mock_uci_engine._parse_bestmove("bestmove")


class TestUCIEngineLifecycle:
    """Tests for engine lifecycle management."""

    def test_quit(self, mock_uci_engine: UCIEngine) -> None:
        mock_uci_engine.quit()
        assert mock_uci_engine._started is False

    def test_quit_when_not_started(self, uci_config: UCIConfig) -> None:
        """Quit on unstarted engine should not raise."""
        engine = UCIEngine(uci_config)
        engine.quit()

    def test_context_manager(self, uci_config: UCIConfig) -> None:
        mock_proc = MockUCIProcess()
        with patch("subprocess.Popen", return_value=mock_proc):
            with UCIEngine(uci_config) as engine:
                assert engine._started is True
            # After exiting context, engine should be quit
            assert engine._started is False

    def test_is_ready_when_running(self, mock_uci_engine: UCIEngine) -> None:
        # is_ready sends isready and reads readyok
        assert mock_uci_engine.is_ready() is True

    def test_is_ready_when_not_started(self, uci_config: UCIConfig) -> None:
        engine = UCIEngine(uci_config)
        assert engine.is_ready() is False

    def test_is_ready_when_process_dead(self, uci_config: UCIConfig) -> None:
        mock_proc = MockUCIProcess()
        mock_proc.returncode = 1  # Dead
        engine = UCIEngine(uci_config)
        engine._process = mock_proc  # type: ignore[assignment]
        engine._started = True
        assert engine.is_ready() is False


class TestUCIEngineErrors:
    """Tests for error handling."""

    def test_send_to_dead_process(self, uci_config: UCIConfig) -> None:
        mock_proc = MockUCIProcess()
        mock_proc.returncode = 1  # Already dead
        with patch("subprocess.Popen", return_value=mock_proc):
            engine = UCIEngine(uci_config)
            engine._process = mock_proc  # type: ignore[assignment]
            engine._started = True
            with pytest.raises(EngineCrashError, match="exited"):
                engine._send("uci")

    def test_ensure_started_not_started(self, uci_config: UCIConfig) -> None:
        engine = UCIEngine(uci_config)
        with pytest.raises(EngineStartupError, match="not been started"):
            engine._ensure_started()

    def test_ensure_started_process_dead(self, uci_config: UCIConfig) -> None:
        """Engine process died after start should raise EngineCrashError."""
        mock_proc = MockUCIProcess()
        mock_proc.returncode = -11  # Segfault
        engine = UCIEngine(uci_config)
        engine._process = mock_proc  # type: ignore[assignment]
        engine._started = True
        with pytest.raises(EngineCrashError, match="exited"):
            engine._ensure_started()
        assert engine._started is False

    def test_send_to_none_process(self, uci_config: UCIConfig) -> None:
        """Sending to None process raises EngineCrashError."""
        engine = UCIEngine(uci_config)
        engine._process = None
        engine._started = True
        with pytest.raises(EngineCrashError, match="not running"):
            engine._send("uci")

    def test_set_position_not_started(self, uci_config: UCIConfig) -> None:
        engine = UCIEngine(uci_config)
        with pytest.raises(EngineStartupError, match="not been started"):
            engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def test_go_not_started(self, uci_config: UCIConfig) -> None:
        engine = UCIEngine(uci_config)
        with pytest.raises(EngineStartupError, match="not been started"):
            engine.go()


class TestUCIEngineConfig:
    """Tests for engine configuration during startup."""

    def test_custom_hash_and_threads(self, uci_config: UCIConfig) -> None:
        """Custom hash and threads should be sent during start."""
        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=10,
            hash_mb=256,
            threads=4,
        )
        mock_proc = MockUCIProcess()
        with patch("subprocess.Popen", return_value=mock_proc):
            engine = UCIEngine(config)
            engine.start()

        # Check that setoption commands were written to stdin
        stdin_content = mock_proc.stdin.getvalue()
        assert "setoption name Hash value 256" in stdin_content
        assert "setoption name Threads value 4" in stdin_content

    def test_custom_options(self) -> None:
        """Custom options dict should be sent during start."""
        config = UCIConfig(
            name="test",
            engine_path=Path("/fake/stockfish"),
            depth_limit=10,
            options={"Skill Level": 10, "UCI_Chess960": True},
        )
        mock_proc = MockUCIProcess()
        with patch("subprocess.Popen", return_value=mock_proc):
            engine = UCIEngine(config)
            engine.start()

        stdin_content = mock_proc.stdin.getvalue()
        assert "setoption name Skill Level value 10" in stdin_content
        assert "setoption name UCI_Chess960 value true" in stdin_content
