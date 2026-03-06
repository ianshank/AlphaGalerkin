"""Tests for match orchestration framework.

Tests match lifecycle, game recording, PGN generation,
and result aggregation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engines.config import MatchConfig
from src.engines.elo import EloEstimate
from src.engines.match import GameRecord, MatchResult


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
