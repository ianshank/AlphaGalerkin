"""Tests for UCI protocol implementation.

Tests command formatting, response parsing, lifecycle management,
and error handling. All tests use mocked subprocesses.
"""

from __future__ import annotations

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


class TestUCIEngineLifecycle:
    """Tests for engine lifecycle management."""

    def test_quit(self, mock_uci_engine: UCIEngine) -> None:
        mock_uci_engine.quit()
        assert mock_uci_engine._started is False

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


class TestUCIEngineErrors:
    """Tests for error handling."""

    def test_send_to_dead_process(self, uci_config: UCIConfig) -> None:
        mock_proc = MockUCIProcess()
        mock_proc.returncode = 1  # Already dead
        with patch("subprocess.Popen", return_value=mock_proc):
            engine = UCIEngine(uci_config)
            engine._process = mock_proc
            engine._started = True
            with pytest.raises(EngineCrashError, match="exited"):
                engine._send("uci")

    def test_ensure_started_not_started(self, uci_config: UCIConfig) -> None:
        engine = UCIEngine(uci_config)
        with pytest.raises(EngineStartupError, match="not been started"):
            engine._ensure_started()
