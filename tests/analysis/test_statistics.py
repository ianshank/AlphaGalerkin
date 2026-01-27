"""Tests for game statistics."""

from __future__ import annotations

import pytest

from src.analysis.config import MoveClassification
from src.analysis.statistics import (
    AggregateStatistics,
    GameStatistics,
    StatisticsCollector,
)


class TestGameStatistics:
    """Tests for GameStatistics dataclass."""

    def test_initialization(self, game_statistics: GameStatistics) -> None:
        """Test statistics initialization."""
        assert game_statistics.game_id == "test_game_001"
        assert game_statistics.board_size == 19
        assert game_statistics.total_moves == 100

    def test_record_move_classification(
        self, game_statistics: GameStatistics
    ) -> None:
        """Test recording move classification."""
        game_statistics.record_move_classification(
            "B", MoveClassification.EXCELLENT
        )
        game_statistics.record_move_classification(
            "B", MoveClassification.GOOD
        )
        game_statistics.record_move_classification(
            "W", MoveClassification.MISTAKE
        )

        assert game_statistics.move_counts["B"]["excellent"] == 1
        assert game_statistics.move_counts["B"]["good"] == 1
        assert game_statistics.move_counts["W"]["mistake"] == 1

    def test_get_accuracy(self, game_statistics: GameStatistics) -> None:
        """Test accuracy calculation."""
        # Record some moves
        for _ in range(6):
            game_statistics.record_move_classification(
                "B", MoveClassification.EXCELLENT
            )
        for _ in range(3):
            game_statistics.record_move_classification(
                "B", MoveClassification.GOOD
            )
        for _ in range(1):
            game_statistics.record_move_classification(
                "B", MoveClassification.MISTAKE
            )

        # 9/10 good moves = 90%
        accuracy = game_statistics.get_accuracy("B")
        assert accuracy == pytest.approx(90.0)

    def test_get_accuracy_empty(self, game_statistics: GameStatistics) -> None:
        """Test accuracy with no moves."""
        accuracy = game_statistics.get_accuracy("B")
        assert accuracy == 0.0

    def test_get_mistake_rate(self, game_statistics: GameStatistics) -> None:
        """Test mistake rate calculation."""
        for _ in range(8):
            game_statistics.record_move_classification(
                "B", MoveClassification.GOOD
            )
        for _ in range(2):
            game_statistics.record_move_classification(
                "B", MoveClassification.MISTAKE
            )

        rate = game_statistics.get_mistake_rate("B")
        assert rate == pytest.approx(20.0)

    def test_to_dict(self, game_statistics: GameStatistics) -> None:
        """Test serialization to dict."""
        data = game_statistics.to_dict()

        assert data["game_id"] == "test_game_001"
        assert data["board_size"] == 19
        assert "black_accuracy" in data
        assert "white_accuracy" in data

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "game_id": "test_123",
            "board_size": 9,
            "total_moves": 50,
            "result": "W+5",
            "black_player": "Alice",
            "white_player": "Bob",
            "move_counts": {
                "B": {"excellent": 5, "good": 10},
                "W": {"excellent": 3, "good": 12},
            },
        }

        stats = GameStatistics.from_dict(data)

        assert stats.game_id == "test_123"
        assert stats.board_size == 9
        assert stats.move_counts["B"]["excellent"] == 5


class TestAggregateStatistics:
    """Tests for AggregateStatistics dataclass."""

    def test_initialization(self) -> None:
        """Test aggregate statistics initialization."""
        agg = AggregateStatistics()
        assert agg.total_games == 0
        assert agg.total_moves == 0

    def test_add_game(self) -> None:
        """Test adding game statistics."""
        agg = AggregateStatistics()
        stats = GameStatistics(
            game_id="test",
            total_moves=100,
            result="B+2.5",
        )
        stats.record_move_classification("B", MoveClassification.EXCELLENT)

        agg.add_game(stats)

        assert agg.total_games == 1
        assert agg.total_moves == 100
        assert agg.win_counts["B"] == 1

    def test_black_win_rate(self) -> None:
        """Test black win rate calculation."""
        agg = AggregateStatistics()

        # Add 3 black wins
        for i in range(3):
            agg.add_game(GameStatistics(
                game_id=f"b{i}",
                total_moves=100,
                result="B+R",
            ))

        # Add 2 white wins
        for i in range(2):
            agg.add_game(GameStatistics(
                game_id=f"w{i}",
                total_moves=100,
                result="W+5",
            ))

        assert agg.black_win_rate == pytest.approx(0.6)

    def test_average_game_length(self) -> None:
        """Test average game length calculation."""
        agg = AggregateStatistics()

        agg.add_game(GameStatistics(game_id="g1", total_moves=100))
        agg.add_game(GameStatistics(game_id="g2", total_moves=200))

        assert agg.average_game_length == 150.0

    def test_overall_accuracy(self) -> None:
        """Test overall accuracy calculation."""
        agg = AggregateStatistics()

        stats = GameStatistics(game_id="test")
        for _ in range(9):
            stats.record_move_classification("B", MoveClassification.GOOD)
        stats.record_move_classification("B", MoveClassification.MISTAKE)

        agg.add_game(stats)

        accuracy = agg.get_overall_accuracy("B")
        assert accuracy == pytest.approx(90.0)

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        agg = AggregateStatistics()
        agg.add_game(GameStatistics(
            game_id="test",
            total_moves=100,
            result="B+R",
        ))

        data = agg.to_dict()

        assert data["total_games"] == 1
        assert "black_win_rate" in data
        assert "average_game_length" in data


class TestStatisticsCollector:
    """Tests for StatisticsCollector."""

    def test_initialization(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test collector initialization."""
        assert statistics_collector.total_games == 0

    def test_add_game(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test adding a game."""
        stats = GameStatistics(game_id="test_1", total_moves=100)
        statistics_collector.add_game(stats)

        assert statistics_collector.total_games == 1

    def test_get_game(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test getting a specific game."""
        stats = GameStatistics(game_id="find_me", total_moves=100)
        statistics_collector.add_game(stats)

        found = statistics_collector.get_game("find_me")
        assert found is not None
        assert found.game_id == "find_me"

        assert statistics_collector.get_game("not_found") is None

    def test_get_recent(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test getting recent games."""
        for i in range(5):
            statistics_collector.add_game(
                GameStatistics(game_id=f"game_{i}", total_moves=100)
            )

        recent = statistics_collector.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].game_id == "game_4"

    def test_get_aggregate(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test getting aggregate statistics."""
        statistics_collector.add_game(
            GameStatistics(game_id="g1", total_moves=100)
        )

        agg = statistics_collector.get_aggregate()
        assert agg.total_games == 1

    def test_get_accuracy_trend(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test getting accuracy trend."""
        for i in range(5):
            stats = GameStatistics(game_id=f"game_{i}")
            stats.record_move_classification("B", MoveClassification.GOOD)
            statistics_collector.add_game(stats)

        trend = statistics_collector.get_accuracy_trend(3)
        assert len(trend) == 3

    def test_get_performance_by_board_size(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test performance breakdown by board size."""
        statistics_collector.add_game(
            GameStatistics(game_id="g1", board_size=9, total_moves=50)
        )
        statistics_collector.add_game(
            GameStatistics(game_id="g2", board_size=9, total_moves=60)
        )
        statistics_collector.add_game(
            GameStatistics(game_id="g3", board_size=19, total_moves=200)
        )

        by_size = statistics_collector.get_performance_by_board_size()

        assert 9 in by_size
        assert 19 in by_size
        assert by_size[9]["games"] == 2
        assert by_size[9]["average_length"] == 55.0

    def test_iter_games(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test iterating through games."""
        for i in range(3):
            statistics_collector.add_game(
                GameStatistics(game_id=f"game_{i}")
            )

        games = list(statistics_collector.iter_games())
        assert len(games) == 3

    def test_clear(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test clearing statistics."""
        statistics_collector.add_game(
            GameStatistics(game_id="test")
        )
        assert statistics_collector.total_games == 1

        statistics_collector.clear()
        assert statistics_collector.total_games == 0

    def test_export_to_dict(
        self, statistics_collector: StatisticsCollector
    ) -> None:
        """Test exporting to dictionary."""
        statistics_collector.add_game(
            GameStatistics(game_id="test", board_size=19)
        )

        data = statistics_collector.export_to_dict()

        assert "games" in data
        assert "aggregate" in data
        assert "by_board_size" in data
