"""Game statistics collection and analysis.

Provides:
- Per-game statistics
- Aggregate statistics across games
- Performance tracking
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.analysis.config import MoveClassification


@dataclass
class GameStatistics:
    """Statistics for a single game.

    Attributes:
        game_id: Unique game identifier.
        board_size: Board size.
        total_moves: Total number of moves.
        result: Game result string.
        black_player: Black player name.
        white_player: White player name.
        move_counts: Counts by move classification.
        win_rate_history: Win rate at each move.
        time_spent: Time spent per move (if available).

    """

    game_id: str = ""
    board_size: int = 19
    total_moves: int = 0
    result: str = ""
    black_player: str = "Black"
    white_player: str = "White"
    move_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    win_rate_history: list[float] = field(default_factory=list)
    time_spent: list[float] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Initialize move counts."""
        if not self.move_counts:
            self.move_counts = {
                "B": defaultdict(int),
                "W": defaultdict(int),
            }

    def record_move_classification(
        self,
        color: str,
        classification: MoveClassification,
    ) -> None:
        """Record a move classification.

        Args:
            color: Move color ("B" or "W").
            classification: Move classification.

        """
        if color not in self.move_counts:
            self.move_counts[color] = defaultdict(int)
        self.move_counts[color][classification.value] += 1

    def get_accuracy(self, color: str) -> float:
        """Calculate accuracy for a player.

        Args:
            color: Player color.

        Returns:
            Accuracy percentage (0-100).

        """
        if color not in self.move_counts:
            return 0.0

        counts = self.move_counts[color]
        total = sum(counts.values())

        if total == 0:
            return 0.0

        # Count excellent and good moves
        good_moves = counts.get(MoveClassification.EXCELLENT.value, 0)
        good_moves += counts.get(MoveClassification.GOOD.value, 0)
        good_moves += counts.get(MoveClassification.NEUTRAL.value, 0)

        return (good_moves / total) * 100

    def get_mistake_rate(self, color: str) -> float:
        """Calculate mistake rate for a player.

        Args:
            color: Player color.

        Returns:
            Mistake rate percentage (0-100).

        """
        if color not in self.move_counts:
            return 0.0

        counts = self.move_counts[color]
        total = sum(counts.values())

        if total == 0:
            return 0.0

        mistakes = counts.get(MoveClassification.MISTAKE.value, 0)
        mistakes += counts.get(MoveClassification.BLUNDER.value, 0)

        return (mistakes / total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "game_id": self.game_id,
            "board_size": self.board_size,
            "total_moves": self.total_moves,
            "result": self.result,
            "black_player": self.black_player,
            "white_player": self.white_player,
            "move_counts": {color: dict(counts) for color, counts in self.move_counts.items()},
            "black_accuracy": self.get_accuracy("B"),
            "white_accuracy": self.get_accuracy("W"),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameStatistics:
        """Create from dictionary."""
        stats = cls(
            game_id=data.get("game_id", ""),
            board_size=data.get("board_size", 19),
            total_moves=data.get("total_moves", 0),
            result=data.get("result", ""),
            black_player=data.get("black_player", "Black"),
            white_player=data.get("white_player", "White"),
        )

        if "move_counts" in data:
            for color, counts in data["move_counts"].items():
                stats.move_counts[color] = defaultdict(int, counts)

        return stats


@dataclass
class AggregateStatistics:
    """Aggregate statistics across multiple games.

    Attributes:
        total_games: Total number of games.
        total_moves: Total number of moves.
        win_counts: Win counts by result type.
        accuracy_history: Accuracy trend over games.

    """

    total_games: int = 0
    total_moves: int = 0
    win_counts: dict[str, int] = field(default_factory=dict)
    accuracy_history: list[float] = field(default_factory=list)
    classification_totals: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize counters."""
        if not self.win_counts:
            self.win_counts = {"B": 0, "W": 0, "Draw": 0}
        if not self.classification_totals:
            self.classification_totals = {"B": defaultdict(int), "W": defaultdict(int)}

    def add_game(self, stats: GameStatistics) -> None:
        """Add a game's statistics.

        Args:
            stats: Game statistics to add.

        """
        self.total_games += 1
        self.total_moves += stats.total_moves

        # Track result
        result = stats.result.upper()
        if result.startswith("B+"):
            self.win_counts["B"] += 1
        elif result.startswith("W+"):
            self.win_counts["W"] += 1
        elif result == "0" or result == "DRAW":
            self.win_counts["Draw"] += 1

        # Track classifications
        for color in ["B", "W"]:
            for classification, count in stats.move_counts.get(color, {}).items():
                self.classification_totals[color][classification] += count

        # Track accuracy trend
        avg_accuracy = (stats.get_accuracy("B") + stats.get_accuracy("W")) / 2
        self.accuracy_history.append(avg_accuracy)

    @property
    def black_win_rate(self) -> float:
        """Calculate black win rate."""
        total_decided = self.win_counts["B"] + self.win_counts["W"]
        if total_decided == 0:
            return 0.5
        return self.win_counts["B"] / total_decided

    @property
    def average_game_length(self) -> float:
        """Calculate average game length."""
        if self.total_games == 0:
            return 0.0
        return self.total_moves / self.total_games

    def get_overall_accuracy(self, color: str) -> float:
        """Get overall accuracy for a color.

        Args:
            color: Player color.

        Returns:
            Accuracy percentage.

        """
        if color not in self.classification_totals:
            return 0.0

        counts = self.classification_totals[color]
        total = sum(counts.values())

        if total == 0:
            return 0.0

        good = counts.get(MoveClassification.EXCELLENT.value, 0)
        good += counts.get(MoveClassification.GOOD.value, 0)
        good += counts.get(MoveClassification.NEUTRAL.value, 0)

        return (good / total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_games": self.total_games,
            "total_moves": self.total_moves,
            "average_game_length": self.average_game_length,
            "win_counts": self.win_counts,
            "black_win_rate": self.black_win_rate,
            "black_accuracy": self.get_overall_accuracy("B"),
            "white_accuracy": self.get_overall_accuracy("W"),
            "classification_totals": {
                color: dict(counts) for color, counts in self.classification_totals.items()
            },
        }


class StatisticsCollector:
    """Collects and manages game statistics.

    Features:
    - Per-game statistics
    - Aggregate tracking
    - Trend analysis
    - Export capabilities
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize statistics collector.

        Args:
            logger: Optional structured logger.

        """
        self._logger = logger or structlog.get_logger(__name__)
        self._games: list[GameStatistics] = []
        self._aggregate = AggregateStatistics()

    def add_game(self, stats: GameStatistics) -> None:
        """Add a game's statistics.

        Args:
            stats: Game statistics to add.

        """
        self._games.append(stats)
        self._aggregate.add_game(stats)

        self._logger.debug(
            "game_added",
            game_id=stats.game_id,
            total_games=self._aggregate.total_games,
        )

    def get_game(self, game_id: str) -> GameStatistics | None:
        """Get statistics for a specific game.

        Args:
            game_id: Game identifier.

        Returns:
            GameStatistics or None if not found.

        """
        for game in self._games:
            if game.game_id == game_id:
                return game
        return None

    def get_recent(self, n: int = 10) -> list[GameStatistics]:
        """Get most recent games.

        Args:
            n: Number of games to get.

        Returns:
            List of recent game statistics.

        """
        return self._games[-n:]

    def get_aggregate(self) -> AggregateStatistics:
        """Get aggregate statistics.

        Returns:
            AggregateStatistics.

        """
        return self._aggregate

    def get_accuracy_trend(self, window: int = 10) -> list[float]:
        """Get accuracy trend over recent games.

        Args:
            window: Number of games to consider.

        Returns:
            List of accuracy values.

        """
        return self._aggregate.accuracy_history[-window:]

    def get_performance_by_board_size(self) -> dict[int, dict[str, Any]]:
        """Get performance breakdown by board size.

        Returns:
            Dictionary of board size to performance metrics.

        """
        by_size: dict[int, list[GameStatistics]] = defaultdict(list)

        for game in self._games:
            by_size[game.board_size].append(game)

        result = {}
        for size, games in by_size.items():
            n_games = len(games)
            if n_games == 0:
                continue

            avg_accuracy = (
                sum((g.get_accuracy("B") + g.get_accuracy("W")) / 2 for g in games) / n_games
            )

            result[size] = {
                "games": n_games,
                "average_accuracy": avg_accuracy,
                "average_length": sum(g.total_moves for g in games) / n_games,
            }

        return result

    def iter_games(self) -> Iterator[GameStatistics]:
        """Iterate through all games.

        Yields:
            GameStatistics for each game.

        """
        yield from self._games

    def clear(self) -> None:
        """Clear all statistics."""
        self._games.clear()
        self._aggregate = AggregateStatistics()

    @property
    def total_games(self) -> int:
        """Get total number of games."""
        return len(self._games)

    def export_to_dict(self) -> dict[str, Any]:
        """Export all statistics to dictionary.

        Returns:
            Dictionary with all statistics.

        """
        return {
            "games": [g.to_dict() for g in self._games],
            "aggregate": self._aggregate.to_dict(),
            "by_board_size": self.get_performance_by_board_size(),
        }
