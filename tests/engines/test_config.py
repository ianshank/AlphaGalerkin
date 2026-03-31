"""Tests for engine configuration schemas.

Tests Pydantic validation, cross-field validators, and edge cases
for all engine-related configs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.engines.config import (
    EloConfig,
    EngineConfig,
    MatchConfig,
    TimeControl,
    UCIConfig,
)
from src.engines.protocol import EngineProtocol


class TestTimeControl:
    """Tests for TimeControl configuration."""

    def test_default_values(self) -> None:
        tc = TimeControl(name="test")
        assert tc.initial_time_ms == 60000
        assert tc.increment_ms == 0
        assert tc.moves_per_period is None

    def test_custom_values(self) -> None:
        tc = TimeControl(
            name="blitz",
            initial_time_ms=180000,
            increment_ms=2000,
            moves_per_period=40,
        )
        assert tc.initial_time_ms == 180000
        assert tc.increment_ms == 2000
        assert tc.moves_per_period == 40

    def test_zero_time_allowed(self) -> None:
        tc = TimeControl(name="bullet", initial_time_ms=0, increment_ms=1000)
        assert tc.initial_time_ms == 0

    def test_negative_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeControl(name="bad", initial_time_ms=-1)


class TestEngineConfig:
    """Tests for base EngineConfig."""

    def test_valid_config(self) -> None:
        config = EngineConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
        )
        assert config.protocol == EngineProtocol.UCI
        assert config.startup_timeout_seconds == 10.0

    def test_custom_options(self) -> None:
        config = EngineConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            options={"Skill Level": 10, "Ponder": False},
        )
        assert config.options["Skill Level"] == 10
        assert config.options["Ponder"] is False


class TestUCIConfig:
    """Tests for UCI-specific configuration."""

    def test_valid_depth_config(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            depth_limit=20,
        )
        assert config.depth_limit == 20
        assert config.protocol == EngineProtocol.UCI

    def test_valid_movetime_config(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            movetime_ms=1000,
        )
        assert config.movetime_ms == 1000

    def test_valid_nodes_config(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            nodes_limit=100000,
        )
        assert config.nodes_limit == 100000

    def test_no_search_limit_rejected(self) -> None:
        with pytest.raises(ValidationError, match="search limit"):
            UCIConfig(
                name="test",
                engine_path=Path("/usr/bin/stockfish"),
            )

    def test_depth_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            UCIConfig(
                name="test",
                engine_path=Path("/usr/bin/stockfish"),
                depth_limit=0,
            )

    def test_depth_max(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            depth_limit=100,
        )
        assert config.depth_limit == 100

    def test_depth_above_max(self) -> None:
        with pytest.raises(ValidationError):
            UCIConfig(
                name="test",
                engine_path=Path("/usr/bin/stockfish"),
                depth_limit=101,
            )

    def test_movetime_too_low(self) -> None:
        with pytest.raises(ValidationError):
            UCIConfig(
                name="test",
                engine_path=Path("/usr/bin/stockfish"),
                movetime_ms=50,
            )

    def test_hash_range(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            depth_limit=10,
            hash_mb=1024,
        )
        assert config.hash_mb == 1024

    def test_threads_range(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            depth_limit=10,
            threads=8,
        )
        assert config.threads == 8

    def test_multiple_search_limits_allowed(self) -> None:
        config = UCIConfig(
            name="test",
            engine_path=Path("/usr/bin/stockfish"),
            depth_limit=20,
            movetime_ms=5000,
        )
        assert config.depth_limit == 20
        assert config.movetime_ms == 5000


class TestEloConfig:
    """Tests for Elo configuration."""

    def test_defaults(self) -> None:
        config = EloConfig(name="test")
        assert config.k_factor == 32.0
        assert config.initial_rating == 1500.0
        assert config.confidence_level == 0.95

    def test_custom_values(self) -> None:
        config = EloConfig(
            name="test",
            k_factor=16.0,
            initial_rating=2000.0,
            confidence_level=0.99,
            elo_divisor=400.0,
        )
        assert config.k_factor == 16.0
        assert config.initial_rating == 2000.0

    def test_invalid_k_factor(self) -> None:
        with pytest.raises(ValidationError):
            EloConfig(name="test", k_factor=0)

    def test_invalid_confidence(self) -> None:
        with pytest.raises(ValidationError):
            EloConfig(name="test", confidence_level=1.0)

        with pytest.raises(ValidationError):
            EloConfig(name="test", confidence_level=0.0)


class TestMatchConfig:
    """Tests for match configuration."""

    def test_defaults(self) -> None:
        config = MatchConfig(name="test")
        assert config.n_games == 10
        assert config.max_moves == 500
        assert config.alternate_colors is True
        assert config.opening_fen is None

    def test_custom_values(self) -> None:
        config = MatchConfig(
            name="test",
            n_games=100,
            max_moves=300,
            alternate_colors=False,
            opening_fen="8/8/8/8/8/8/8/4K2k w - - 0 1",
        )
        assert config.n_games == 100
        assert config.opening_fen is not None

    def test_invalid_n_games(self) -> None:
        with pytest.raises(ValidationError):
            MatchConfig(name="test", n_games=0)

    def test_invalid_max_moves(self) -> None:
        with pytest.raises(ValidationError):
            MatchConfig(name="test", max_moves=10)
