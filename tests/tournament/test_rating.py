"""Tests for Elo rating system."""

from __future__ import annotations

import pytest

from src.tournament.rating import (
    EloRating,
    RatingChange,
    RatingSystem,
    create_rating_system,
)


class TestRatingChange:
    """Tests for RatingChange dataclass."""

    def test_initialization(self) -> None:
        """Test rating change initialization."""
        change = RatingChange(
            player_id="p1",
            old_rating=1500.0,
            new_rating=1516.0,
            change=16.0,
            opponent_id="p2",
            opponent_rating=1500.0,
            result=1.0,
            expected=0.5,
            k_factor=32.0,
        )

        assert change.player_id == "p1"
        assert change.old_rating == 1500.0
        assert change.new_rating == 1516.0
        assert change.change == 16.0
        assert change.result == 1.0

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        change = RatingChange(
            player_id="p1",
            old_rating=1500.0,
            new_rating=1516.0,
            change=16.0,
            opponent_id="p2",
            opponent_rating=1500.0,
            result=1.0,
            expected=0.5,
            k_factor=32.0,
        )

        data = change.to_dict()

        assert data["player_id"] == "p1"
        assert data["change"] == 16.0
        assert data["k_factor"] == 32.0


class TestEloRating:
    """Tests for EloRating dataclass."""

    def test_default_values(self, elo_rating: EloRating) -> None:
        """Test default Elo rating values."""
        assert elo_rating.rating == 1500.0
        assert elo_rating.games_played == 0
        assert elo_rating.peak_rating == 1500.0
        assert elo_rating.lowest_rating == 1500.0
        assert len(elo_rating.history) == 0

    def test_update_rating_increase(self, elo_rating: EloRating) -> None:
        """Test updating rating with increase."""
        elo_rating.update(20.0)

        assert elo_rating.rating == 1520.0
        assert elo_rating.games_played == 1
        assert elo_rating.peak_rating == 1520.0
        assert elo_rating.lowest_rating == 1500.0

    def test_update_rating_decrease(self, elo_rating: EloRating) -> None:
        """Test updating rating with decrease."""
        elo_rating.update(-30.0)

        assert elo_rating.rating == 1470.0
        assert elo_rating.games_played == 1
        assert elo_rating.peak_rating == 1500.0
        assert elo_rating.lowest_rating == 1470.0

    def test_update_with_record(self, elo_rating: EloRating) -> None:
        """Test updating with change record."""
        record = RatingChange(
            player_id="test",
            old_rating=1500.0,
            new_rating=1520.0,
            change=20.0,
            opponent_id="opp",
            opponent_rating=1500.0,
            result=1.0,
            expected=0.5,
            k_factor=32.0,
        )

        elo_rating.update(20.0, record)

        assert len(elo_rating.history) == 1
        assert elo_rating.history[0] == record

    def test_get_recent_history(self, elo_rating: EloRating) -> None:
        """Test getting recent history."""
        # Add some history
        for i in range(15):
            record = RatingChange(
                player_id="test",
                old_rating=1500.0 + i,
                new_rating=1500.0 + i + 1,
                change=1.0,
                opponent_id="opp",
                opponent_rating=1500.0,
                result=1.0,
                expected=0.5,
                k_factor=32.0,
            )
            elo_rating.history.append(record)

        recent = elo_rating.get_recent_history(5)
        assert len(recent) == 5
        assert recent[-1].old_rating == 1514.0

    def test_to_dict(self, elo_rating: EloRating) -> None:
        """Test serialization to dict."""
        elo_rating.update(50.0)
        elo_rating.update(-20.0)

        data = elo_rating.to_dict()

        assert data["rating"] == 1530.0
        assert data["games_played"] == 2
        assert data["peak_rating"] == 1550.0
        assert data["history_length"] == 0  # No records added


class TestRatingSystem:
    """Tests for RatingSystem."""

    def test_initialization(self, rating_system: RatingSystem) -> None:
        """Test rating system initialization."""
        assert rating_system.config is not None

    def test_get_rating_new_player(self, rating_system: RatingSystem) -> None:
        """Test getting rating for new player."""
        rating = rating_system.get_rating("new_player")
        assert rating == 1500.0

    def test_get_elo_rating(self, rating_system: RatingSystem) -> None:
        """Test getting EloRating object."""
        elo = rating_system.get_elo_rating("p1")
        assert isinstance(elo, EloRating)
        assert elo.rating == 1500.0

        # Getting again should return same object
        elo2 = rating_system.get_elo_rating("p1")
        assert elo is elo2

    def test_expected_score_equal_ratings(self, rating_system: RatingSystem) -> None:
        """Test expected score with equal ratings."""
        expected = rating_system.expected_score(1500.0, 1500.0)
        assert expected == 0.5

    def test_expected_score_higher_rated(self, rating_system: RatingSystem) -> None:
        """Test expected score when higher rated."""
        expected = rating_system.expected_score(1700.0, 1500.0)
        assert expected > 0.5
        assert expected < 1.0

    def test_expected_score_lower_rated(self, rating_system: RatingSystem) -> None:
        """Test expected score when lower rated."""
        expected = rating_system.expected_score(1300.0, 1500.0)
        assert expected < 0.5
        assert expected > 0.0

    def test_get_k_factor_new_player(self, rating_system: RatingSystem) -> None:
        """Test K-factor for new player."""
        k = rating_system.get_k_factor("new_player")
        assert k == 40.0  # New player K-factor

    def test_get_k_factor_established_player(self, rating_system: RatingSystem) -> None:
        """Test K-factor for established player."""
        elo = rating_system.get_elo_rating("established")
        elo.games_played = 50  # More than 30 games

        k = rating_system.get_k_factor("established")
        assert k == 32.0  # Standard K-factor

    def test_get_k_factor_high_rated(self, rating_system: RatingSystem) -> None:
        """Test K-factor for high-rated player."""
        elo = rating_system.get_elo_rating("master")
        elo.rating = 2500.0
        elo.games_played = 100

        k = rating_system.get_k_factor("master")
        assert k == 16.0  # High-rated K-factor

    def test_calculate_change_win(self, rating_system: RatingSystem) -> None:
        """Test calculating rating change for win."""
        p1_change, p2_change = rating_system.calculate_change("p1", "p2", result=1.0)

        assert p1_change > 0  # Winner gains
        assert p2_change < 0  # Loser loses
        assert abs(p1_change) == pytest.approx(abs(p2_change), rel=0.2)

    def test_calculate_change_loss(self, rating_system: RatingSystem) -> None:
        """Test calculating rating change for loss."""
        p1_change, p2_change = rating_system.calculate_change("p1", "p2", result=0.0)

        assert p1_change < 0  # Loser loses
        assert p2_change > 0  # Winner gains

    def test_calculate_change_draw(self, rating_system: RatingSystem) -> None:
        """Test calculating rating change for draw."""
        p1_change, p2_change = rating_system.calculate_change("p1", "p2", result=0.5)

        # Equal ratings means small or zero change on draw
        assert abs(p1_change) < 5
        assert abs(p2_change) < 5

    def test_record_game_win(self, rating_system: RatingSystem) -> None:
        """Test recording a game result."""
        initial_p1 = rating_system.get_rating("p1")
        initial_p2 = rating_system.get_rating("p2")

        p1_record, p2_record = rating_system.record_game("p1", "p2", 1.0)

        assert p1_record.old_rating == initial_p1
        assert p1_record.change > 0
        assert p2_record.change < 0

        # Check ratings updated
        assert rating_system.get_rating("p1") > initial_p1
        assert rating_system.get_rating("p2") < initial_p2

    def test_record_game_updates_history(self, rating_system: RatingSystem) -> None:
        """Test that recording game updates history."""
        rating_system.record_game("p1", "p2", 1.0)

        elo = rating_system.get_elo_rating("p1")
        assert len(elo.history) == 1
        assert elo.history[0].result == 1.0

    def test_rating_bounds(self, rating_system: RatingSystem) -> None:
        """Test rating stays within bounds."""
        # Force very low rating
        rating_system.set_rating("low", 50.0)
        rating = rating_system.get_rating("low")
        assert rating == 100.0  # Min rating

        # Force very high rating
        rating_system.set_rating("high", 5000.0)
        rating = rating_system.get_rating("high")
        assert rating == 4000.0  # Max rating

    def test_set_rating(self, rating_system: RatingSystem) -> None:
        """Test setting rating directly."""
        rating_system.set_rating("p1", 1800.0)
        assert rating_system.get_rating("p1") == 1800.0

    def test_get_leaderboard(self, rating_system: RatingSystem) -> None:
        """Test getting leaderboard."""
        # Create some players with different ratings
        rating_system.set_rating("p1", 1800.0)
        rating_system.set_rating("p2", 1600.0)
        rating_system.set_rating("p3", 2000.0)

        leaderboard = rating_system.get_leaderboard(2)
        assert len(leaderboard) == 2
        assert leaderboard[0][0] == "p3"  # Highest
        assert leaderboard[0][1] == 2000.0

    def test_simulate_match_outcome(self, rating_system: RatingSystem) -> None:
        """Test match outcome simulation."""
        rating_system.set_rating("p1", 1800.0)
        rating_system.set_rating("p2", 1400.0)

        outcome = rating_system.simulate_match_outcome("p1", "p2")

        assert "player1_win" in outcome
        assert "draw" in outcome
        assert "player2_win" in outcome
        assert "expected_score" in outcome

        # Higher rated player should have higher win probability
        assert outcome["player1_win"] > outcome["player2_win"]

    def test_to_dict(self, rating_system: RatingSystem) -> None:
        """Test serialization to dict."""
        rating_system.record_game("p1", "p2", 1.0)

        data = rating_system.to_dict()

        assert "config" in data
        assert "ratings" in data
        assert "p1" in data["ratings"]
        assert "p2" in data["ratings"]


class TestCreateRatingSystem:
    """Tests for create_rating_system factory."""

    def test_create_default(self) -> None:
        """Test creating default rating system."""
        system = create_rating_system()
        assert system.config.initial_rating == 1500.0
        assert system.config.k_factor == 32.0

    def test_create_with_custom_values(self) -> None:
        """Test creating with custom values."""
        system = create_rating_system(
            initial_rating=1600.0,
            k_factor=24.0,
        )
        assert system.config.initial_rating == 1600.0
        assert system.config.k_factor == 24.0

    def test_create_with_additional_kwargs(self) -> None:
        """Test creating with additional kwargs."""
        system = create_rating_system(
            k_factor_high_rated=10.0,
            high_rating_threshold=2500.0,
        )
        assert system.config.k_factor_high_rated == 10.0
        assert system.config.high_rating_threshold == 2500.0
