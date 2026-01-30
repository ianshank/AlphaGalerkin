"""Tests for evaluation utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.training.eval_utils import EloRating, EloTracker


class TestEloRating:
    """Tests for EloRating dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        rating = EloRating(step=100)
        assert rating.step == 100
        assert rating.rating == 1500.0
        assert rating.games_played == 0
        assert rating.wins == 0
        assert rating.losses == 0
        assert rating.draws == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        rating = EloRating(step=100, rating=1600.0, games_played=10, wins=6)
        d = rating.to_dict()
        assert d["step"] == 100
        assert d["rating"] == 1600.0
        assert d["games_played"] == 10
        assert d["wins"] == 6

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        d = {"step": 200, "rating": 1550.0, "games_played": 5}
        rating = EloRating.from_dict(d)
        assert rating.step == 200
        assert rating.rating == 1550.0
        assert rating.games_played == 5


class TestEloTracker:
    """Tests for EloTracker."""

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        tracker = EloTracker(k_factor=32.0, initial_rating=1500.0)
        assert tracker.k_factor == 32.0
        assert tracker.initial_rating == 1500.0

    def test_expected_score(self) -> None:
        """Test expected score calculation."""
        tracker = EloTracker()
        # Equal ratings should give 0.5
        score = tracker.expected_score(1500, 1500)
        assert abs(score - 0.5) < 0.001
        # Higher rating should expect to win
        score = tracker.expected_score(1600, 1400)
        assert score > 0.5

    def test_update_ratings_win(self) -> None:
        """Test rating update after win."""
        tracker = EloTracker()
        new_a, new_b = tracker.update_ratings(0, 100, 1.0)
        # Winner should gain rating
        assert new_a > 1500
        # Loser should lose rating
        assert new_b < 1500
        # Total rating should be conserved
        assert abs((new_a + new_b) - 3000) < 0.001

    def test_update_ratings_loss(self) -> None:
        """Test rating update after loss."""
        tracker = EloTracker()
        new_a, new_b = tracker.update_ratings(0, 100, 0.0)
        # Loser should lose rating
        assert new_a < 1500
        # Winner should gain rating
        assert new_b > 1500

    def test_update_ratings_draw(self) -> None:
        """Test rating update after draw."""
        tracker = EloTracker()
        new_a, new_b = tracker.update_ratings(0, 100, 0.5)
        # Draw between equal players should not change ratings much
        assert abs(new_a - 1500) < 1
        assert abs(new_b - 1500) < 1

    def test_get_rating_untracked(self) -> None:
        """Test getting rating for untracked step."""
        tracker = EloTracker()
        assert tracker.get_rating(999) == 1500.0

    def test_get_rating_tracked(self) -> None:
        """Test getting rating for tracked step."""
        tracker = EloTracker()
        tracker.update_ratings(0, 100, 1.0)
        rating = tracker.get_rating(0)
        assert rating > 1500

    def test_get_history(self) -> None:
        """Test getting rating history."""
        tracker = EloTracker()
        tracker.update_ratings(100, 200, 1.0)
        tracker.update_ratings(0, 100, 1.0)
        history = tracker.get_history()
        # Should be sorted by step
        steps = [h[0] for h in history]
        assert steps == sorted(steps)

    def test_save_load(self) -> None:
        """Test saving and loading tracker state."""
        tracker = EloTracker()
        tracker.update_ratings(0, 100, 1.0)
        tracker.update_ratings(100, 200, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "elo.json"
            tracker.save(path)

            new_tracker = EloTracker()
            new_tracker.load(path)

            assert new_tracker.get_rating(0) == tracker.get_rating(0)
            assert new_tracker.get_rating(100) == tracker.get_rating(100)
            assert new_tracker.get_rating(200) == tracker.get_rating(200)

    def test_game_tracking(self) -> None:
        """Test that games are properly tracked."""
        tracker = EloTracker()
        tracker.update_ratings(0, 100, 1.0)  # 0 wins
        tracker.update_ratings(0, 100, 0.0)  # 0 loses
        tracker.update_ratings(0, 100, 0.5)  # Draw

        info_0 = tracker.get_rating_info(0)
        assert info_0 is not None
        assert info_0.games_played == 3
        assert info_0.wins == 1
        assert info_0.losses == 1
        assert info_0.draws == 1
