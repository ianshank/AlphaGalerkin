"""Elo rating system for tournaments.

Provides:
- Elo rating calculations
- Rating history tracking
- K-factor management
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.tournament.config import RatingConfig


@dataclass
class RatingChange:
    """Record of a rating change.

    Attributes:
        player_id: Player ID.
        old_rating: Rating before game.
        new_rating: Rating after game.
        change: Rating change amount.
        opponent_id: Opponent player ID.
        opponent_rating: Opponent's rating.
        result: Game result (1=win, 0.5=draw, 0=loss).
        expected: Expected score.
        k_factor: K-factor used.
        timestamp: When change occurred.
    """

    player_id: str
    old_rating: float
    new_rating: float
    change: float
    opponent_id: str
    opponent_rating: float
    result: float
    expected: float
    k_factor: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "old_rating": self.old_rating,
            "new_rating": self.new_rating,
            "change": self.change,
            "opponent_id": self.opponent_id,
            "opponent_rating": self.opponent_rating,
            "result": self.result,
            "expected": self.expected,
            "k_factor": self.k_factor,
            "timestamp": self.timestamp,
        }


@dataclass
class EloRating:
    """Elo rating with history.

    Tracks rating value and change history.
    """

    rating: float = 1500.0
    games_played: int = 0
    peak_rating: float = 1500.0
    lowest_rating: float = 1500.0
    history: list[RatingChange] = field(default_factory=list)

    def update(
        self,
        change: float,
        change_record: RatingChange | None = None,
    ) -> None:
        """Update rating with change.

        Args:
            change: Rating change amount.
            change_record: Optional change record for history.
        """
        self.rating += change
        self.games_played += 1

        # Track extremes
        if self.rating > self.peak_rating:
            self.peak_rating = self.rating
        if self.rating < self.lowest_rating:
            self.lowest_rating = self.rating

        # Add to history
        if change_record:
            self.history.append(change_record)

    def get_recent_history(self, n: int = 10) -> list[RatingChange]:
        """Get recent rating history.

        Args:
            n: Number of entries.

        Returns:
            List of recent RatingChange records.
        """
        return self.history[-n:]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rating": self.rating,
            "games_played": self.games_played,
            "peak_rating": self.peak_rating,
            "lowest_rating": self.lowest_rating,
            "history_length": len(self.history),
        }


class RatingSystem:
    """Elo rating system implementation.

    Calculates rating changes based on game results.
    """

    def __init__(
        self,
        config: RatingConfig | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize rating system.

        Args:
            config: Rating configuration.
            logger: Optional structured logger.
        """
        self.config = config or RatingConfig()
        self._logger = logger or structlog.get_logger(__name__)
        self._ratings: dict[str, EloRating] = {}

    def get_rating(self, player_id: str) -> float:
        """Get player's current rating.

        Args:
            player_id: Player ID.

        Returns:
            Current rating.
        """
        if player_id in self._ratings:
            return self._ratings[player_id].rating
        return self.config.initial_rating

    def get_elo_rating(self, player_id: str) -> EloRating:
        """Get player's EloRating object.

        Args:
            player_id: Player ID.

        Returns:
            EloRating object (creates if not exists).
        """
        if player_id not in self._ratings:
            self._ratings[player_id] = EloRating(
                rating=self.config.initial_rating
            )
        return self._ratings[player_id]

    def expected_score(
        self,
        player_rating: float,
        opponent_rating: float,
    ) -> float:
        """Calculate expected score for a player.

        Uses standard Elo formula:
        E = 1 / (1 + 10^((opponent - player) / 400))

        Args:
            player_rating: Player's rating.
            opponent_rating: Opponent's rating.

        Returns:
            Expected score (0.0 to 1.0).
        """
        diff = opponent_rating - player_rating
        return 1 / (1 + math.pow(10, diff / 400))

    def get_k_factor(self, player_id: str) -> float:
        """Get K-factor for a player.

        K-factor varies based on:
        - Number of games played (higher for new players)
        - Rating level (lower for high-rated players)

        Args:
            player_id: Player ID.

        Returns:
            K-factor value.
        """
        elo = self.get_elo_rating(player_id)

        # New player with few games
        if elo.games_played < self.config.new_player_games:
            return self.config.k_factor_new_player

        # High-rated player
        if elo.rating >= self.config.high_rating_threshold:
            return self.config.k_factor_high_rated

        # Standard player
        return self.config.k_factor

    def calculate_change(
        self,
        player_id: str,
        opponent_id: str,
        result: float,
        player_rating: float | None = None,
        opponent_rating: float | None = None,
    ) -> tuple[float, float]:
        """Calculate rating changes for both players.

        Args:
            player_id: Player 1 ID.
            opponent_id: Player 2 ID.
            result: Result from player's perspective (1=win, 0.5=draw, 0=loss).
            player_rating: Override player rating (for simulation).
            opponent_rating: Override opponent rating (for simulation).

        Returns:
            Tuple of (player_change, opponent_change).
        """
        if player_rating is None:
            player_rating = self.get_rating(player_id)
        if opponent_rating is None:
            opponent_rating = self.get_rating(opponent_id)

        # Get K-factors
        player_k = self.get_k_factor(player_id)
        opponent_k = self.get_k_factor(opponent_id)

        # Calculate expected scores
        player_expected = self.expected_score(player_rating, opponent_rating)
        opponent_expected = 1 - player_expected

        # Calculate changes
        player_change = player_k * (result - player_expected)
        opponent_change = opponent_k * ((1 - result) - opponent_expected)

        return player_change, opponent_change

    def record_game(
        self,
        player_id: str,
        opponent_id: str,
        result: float,
    ) -> tuple[RatingChange, RatingChange]:
        """Record a game result and update ratings.

        Args:
            player_id: Player ID.
            opponent_id: Opponent ID.
            result: Result from player's perspective (1=win, 0.5=draw, 0=loss).

        Returns:
            Tuple of RatingChange records for both players.
        """
        player_elo = self.get_elo_rating(player_id)
        opponent_elo = self.get_elo_rating(opponent_id)

        old_player_rating = player_elo.rating
        old_opponent_rating = opponent_elo.rating

        # Calculate changes
        player_change, opponent_change = self.calculate_change(
            player_id,
            opponent_id,
            result,
            old_player_rating,
            old_opponent_rating,
        )

        # Apply changes with bounds
        new_player_rating = self._clamp_rating(
            old_player_rating + player_change
        )
        new_opponent_rating = self._clamp_rating(
            old_opponent_rating + opponent_change
        )

        # Create change records
        player_record = RatingChange(
            player_id=player_id,
            old_rating=old_player_rating,
            new_rating=new_player_rating,
            change=new_player_rating - old_player_rating,
            opponent_id=opponent_id,
            opponent_rating=old_opponent_rating,
            result=result,
            expected=self.expected_score(old_player_rating, old_opponent_rating),
            k_factor=self.get_k_factor(player_id),
        )

        opponent_record = RatingChange(
            player_id=opponent_id,
            old_rating=old_opponent_rating,
            new_rating=new_opponent_rating,
            change=new_opponent_rating - old_opponent_rating,
            opponent_id=player_id,
            opponent_rating=old_player_rating,
            result=1 - result,
            expected=self.expected_score(old_opponent_rating, old_player_rating),
            k_factor=self.get_k_factor(opponent_id),
        )

        # Update ratings
        player_elo.rating = new_player_rating
        player_elo.games_played += 1
        player_elo.history.append(player_record)
        if new_player_rating > player_elo.peak_rating:
            player_elo.peak_rating = new_player_rating
        if new_player_rating < player_elo.lowest_rating:
            player_elo.lowest_rating = new_player_rating

        opponent_elo.rating = new_opponent_rating
        opponent_elo.games_played += 1
        opponent_elo.history.append(opponent_record)
        if new_opponent_rating > opponent_elo.peak_rating:
            opponent_elo.peak_rating = new_opponent_rating
        if new_opponent_rating < opponent_elo.lowest_rating:
            opponent_elo.lowest_rating = new_opponent_rating

        self._logger.debug(
            "rating_updated",
            player_id=player_id,
            old_rating=old_player_rating,
            new_rating=new_player_rating,
            change=player_record.change,
        )

        return player_record, opponent_record

    def _clamp_rating(self, rating: float) -> float:
        """Clamp rating to configured bounds.

        Args:
            rating: Raw rating value.

        Returns:
            Clamped rating.
        """
        return max(
            self.config.min_rating,
            min(self.config.max_rating, rating),
        )

    def set_rating(self, player_id: str, rating: float) -> None:
        """Set a player's rating directly.

        Args:
            player_id: Player ID.
            rating: New rating value.
        """
        elo = self.get_elo_rating(player_id)
        elo.rating = self._clamp_rating(rating)

    def get_leaderboard(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Get top players by rating.

        Args:
            top_n: Number of players to return.

        Returns:
            List of (player_id, rating) tuples.
        """
        sorted_players = sorted(
            self._ratings.items(),
            key=lambda x: x[1].rating,
            reverse=True,
        )
        return [
            (player_id, elo.rating)
            for player_id, elo in sorted_players[:top_n]
        ]

    def simulate_match_outcome(
        self,
        player1_id: str,
        player2_id: str,
        n_simulations: int = 1000,
    ) -> dict[str, float]:
        """Simulate match outcomes based on ratings.

        Args:
            player1_id: Player 1 ID.
            player2_id: Player 2 ID.
            n_simulations: Number of simulations.

        Returns:
            Dictionary with win/draw/loss probabilities.
        """
        import random

        p1_rating = self.get_rating(player1_id)
        p2_rating = self.get_rating(player2_id)

        expected = self.expected_score(p1_rating, p2_rating)

        p1_wins = 0
        draws = 0

        for _ in range(n_simulations):
            roll = random.random()
            if roll < expected - 0.1:  # Win margin
                p1_wins += 1
            elif roll < expected + 0.1:  # Draw zone
                draws += 1

        return {
            "player1_win": p1_wins / n_simulations,
            "draw": draws / n_simulations,
            "player2_win": (n_simulations - p1_wins - draws) / n_simulations,
            "expected_score": expected,
        }

    def to_dict(self) -> dict[str, Any]:
        """Export rating system state.

        Returns:
            Dictionary with all ratings.
        """
        return {
            "config": self.config.model_dump(),
            "ratings": {
                player_id: elo.to_dict()
                for player_id, elo in self._ratings.items()
            },
        }


def create_rating_system(
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    **kwargs: Any,
) -> RatingSystem:
    """Factory function to create rating system.

    Args:
        initial_rating: Initial rating for new players.
        k_factor: Default K-factor.
        **kwargs: Additional config options.

    Returns:
        RatingSystem instance.
    """
    config = RatingConfig(
        initial_rating=initial_rating,
        k_factor=k_factor,
        **kwargs,
    )
    return RatingSystem(config=config)
