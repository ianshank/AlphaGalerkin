"""Elo rating tracker for checkpoint evaluation.

Implements standard Elo rating system to track model strength
across training checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EloRating:
    """Elo rating for a checkpoint.

    Attributes:
        step: Training step when checkpoint was created.
        rating: Current Elo rating.
        games_played: Total games played by this checkpoint.
        wins: Number of wins.
        losses: Number of losses.
        draws: Number of draws.

    """

    step: int
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "rating": self.rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloRating:
        """Create from dictionary."""
        return cls(
            step=data["step"],
            rating=data.get("rating", 1500.0),
            games_played=data.get("games_played", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
        )


class EloTracker:
    """Tracks Elo ratings for training checkpoints.

    Uses standard Elo formula with configurable K-factor:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        R'_A = R_A + K * (S_A - E_A)

    Where:
        E_A: Expected score for player A
        R_A, R_B: Current ratings
        S_A: Actual score (1 for win, 0.5 for draw, 0 for loss)
        K: K-factor controlling rating volatility
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
    ) -> None:
        """Initialize Elo tracker.

        Args:
            k_factor: K-factor for rating updates (higher = more volatile).
            initial_rating: Starting Elo for new checkpoints.

        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self._ratings: dict[int, EloRating] = {}

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B.

        Args:
            rating_a: Rating of player A.
            rating_b: Rating of player B.

        Returns:
            Expected score for player A (between 0 and 1).

        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(
        self,
        step_a: int,
        step_b: int,
        score_a: float,
    ) -> tuple[float, float]:
        """Update Elo ratings after a match.

        Args:
            step_a: Training step of player A.
            step_b: Training step of player B.
            score_a: Score for player A (1.0=win, 0.5=draw, 0.0=loss).

        Returns:
            Tuple of (new_rating_a, new_rating_b).

        """
        # Initialize ratings if needed
        if step_a not in self._ratings:
            self._ratings[step_a] = EloRating(step=step_a, rating=self.initial_rating)
        if step_b not in self._ratings:
            self._ratings[step_b] = EloRating(step=step_b, rating=self.initial_rating)

        rating_a = self._ratings[step_a]
        rating_b = self._ratings[step_b]

        # Calculate expected scores
        expected_a = self.expected_score(rating_a.rating, rating_b.rating)
        expected_b = 1.0 - expected_a

        # Update ratings
        new_rating_a = rating_a.rating + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b.rating + self.k_factor * ((1.0 - score_a) - expected_b)

        # Update tracking
        rating_a.rating = new_rating_a
        rating_b.rating = new_rating_b
        rating_a.games_played += 1
        rating_b.games_played += 1

        # Track wins/losses/draws
        if score_a == 1.0:
            rating_a.wins += 1
            rating_b.losses += 1
        elif score_a == 0.0:
            rating_a.losses += 1
            rating_b.wins += 1
        else:
            rating_a.draws += 1
            rating_b.draws += 1

        logger.debug(
            "elo_updated",
            step_a=step_a,
            step_b=step_b,
            score_a=score_a,
            old_rating_a=rating_a.rating - self.k_factor * (score_a - expected_a),
            new_rating_a=new_rating_a,
            old_rating_b=rating_b.rating - self.k_factor * ((1.0 - score_a) - expected_b),
            new_rating_b=new_rating_b,
        )

        return new_rating_a, new_rating_b

    def get_rating(self, step: int) -> float:
        """Get Elo rating for a step.

        Args:
            step: Training step.

        Returns:
            Elo rating (initial_rating if not tracked).

        """
        if step in self._ratings:
            return self._ratings[step].rating
        return self.initial_rating

    def get_rating_info(self, step: int) -> EloRating | None:
        """Get full rating info for a step.

        Args:
            step: Training step.

        Returns:
            EloRating or None if not tracked.

        """
        return self._ratings.get(step)

    def get_history(self) -> list[tuple[int, float]]:
        """Get rating history sorted by step.

        Returns:
            List of (step, rating) tuples sorted by step.

        """
        return sorted([(s, r.rating) for s, r in self._ratings.items()])

    def get_all_ratings(self) -> dict[int, EloRating]:
        """Get all tracked ratings.

        Returns:
            Dictionary mapping step to EloRating.

        """
        return dict(self._ratings)

    def save(self, path: Path | str) -> None:
        """Save ratings to JSON file.

        Args:
            path: Path to save file.

        """
        path = Path(path)
        data = {
            "k_factor": self.k_factor,
            "initial_rating": self.initial_rating,
            "ratings": {str(k): v.to_dict() for k, v in self._ratings.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("elo_tracker_saved", path=str(path), n_ratings=len(self._ratings))

    def load(self, path: Path | str) -> None:
        """Load ratings from JSON file.

        Args:
            path: Path to load file.

        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self.k_factor = data.get("k_factor", self.k_factor)
        self.initial_rating = data.get("initial_rating", self.initial_rating)
        self._ratings = {int(k): EloRating.from_dict(v) for k, v in data.get("ratings", {}).items()}
        logger.info("elo_tracker_loaded", path=str(path), n_ratings=len(self._ratings))
