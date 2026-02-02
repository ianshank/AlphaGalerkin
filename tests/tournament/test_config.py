"""Tests for tournament configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.tournament.config import (
    MatchConfig,
    RatingConfig,
    TimeControl,
    TournamentConfig,
    TournamentFormat,
    create_tournament_config,
)


class TestTournamentFormat:
    """Tests for TournamentFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test all tournament formats exist."""
        assert TournamentFormat.ROUND_ROBIN.value == "round_robin"
        assert TournamentFormat.SWISS.value == "swiss"
        assert TournamentFormat.SINGLE_ELIMINATION.value == "single_elimination"
        assert TournamentFormat.DOUBLE_ELIMINATION.value == "double_elimination"
        assert TournamentFormat.MATCH.value == "match"


class TestTimeControl:
    """Tests for TimeControl enum."""

    def test_all_time_controls_exist(self) -> None:
        """Test all time controls exist."""
        assert TimeControl.BLITZ.value == "blitz"
        assert TimeControl.RAPID.value == "rapid"
        assert TimeControl.STANDARD.value == "standard"
        assert TimeControl.UNLIMITED.value == "unlimited"


class TestMatchConfig:
    """Tests for MatchConfig."""

    def test_default_values(self, default_match_config: MatchConfig) -> None:
        """Test default configuration values."""
        assert default_match_config.board_size == 19
        assert default_match_config.komi == 7.5
        assert default_match_config.handicap == 0
        assert default_match_config.games_per_match == 1
        assert default_match_config.alternating_colors is True
        assert default_match_config.time_control == TimeControl.STANDARD

    def test_board_size_validation(self) -> None:
        """Test board size validation."""
        # Valid
        MatchConfig(board_size=9)
        MatchConfig(board_size=19)
        MatchConfig(board_size=25)

        # Invalid
        with pytest.raises(ValidationError):
            MatchConfig(board_size=4)
        with pytest.raises(ValidationError):
            MatchConfig(board_size=26)

    def test_komi_validation(self) -> None:
        """Test komi validation."""
        # Valid
        MatchConfig(komi=0.0)
        MatchConfig(komi=6.5)

        # Invalid
        with pytest.raises(ValidationError):
            MatchConfig(komi=-1.0)
        with pytest.raises(ValidationError):
            MatchConfig(komi=101.0)

    def test_handicap_validation(self) -> None:
        """Test handicap validation."""
        # Valid
        MatchConfig(handicap=0)
        MatchConfig(handicap=9)

        # Invalid
        with pytest.raises(ValidationError):
            MatchConfig(handicap=-1)
        with pytest.raises(ValidationError):
            MatchConfig(handicap=10)

    def test_games_per_match_validation(self) -> None:
        """Test games per match validation."""
        # Valid
        MatchConfig(games_per_match=1)
        MatchConfig(games_per_match=7)

        # Invalid
        with pytest.raises(ValidationError):
            MatchConfig(games_per_match=0)
        with pytest.raises(ValidationError):
            MatchConfig(games_per_match=101)


class TestRatingConfig:
    """Tests for RatingConfig."""

    def test_default_values(self, default_rating_config: RatingConfig) -> None:
        """Test default configuration values."""
        assert default_rating_config.initial_rating == 1500.0
        assert default_rating_config.k_factor == 32.0
        assert default_rating_config.k_factor_new_player == 40.0
        assert default_rating_config.k_factor_high_rated == 16.0
        assert default_rating_config.high_rating_threshold == 2400.0
        assert default_rating_config.new_player_games == 30
        assert default_rating_config.min_rating == 100.0
        assert default_rating_config.max_rating == 4000.0

    def test_k_factor_validation(self) -> None:
        """Test K-factor validation."""
        # Valid
        RatingConfig(k_factor=16.0)
        RatingConfig(k_factor=64.0)

        # Invalid
        with pytest.raises(ValidationError):
            RatingConfig(k_factor=0)
        with pytest.raises(ValidationError):
            RatingConfig(k_factor=101)

    def test_rating_bounds_validation(self) -> None:
        """Test rating bounds validation."""
        # Valid
        RatingConfig(min_rating=0.0, max_rating=5000.0)

        # Invalid
        with pytest.raises(ValidationError):
            RatingConfig(min_rating=-1.0)


class TestTournamentConfig:
    """Tests for TournamentConfig."""

    def test_required_name(self) -> None:
        """Test that name is required."""
        with pytest.raises(ValidationError):
            TournamentConfig()  # type: ignore

    def test_default_values(self, round_robin_config: TournamentConfig) -> None:
        """Test default configuration values."""
        assert round_robin_config.format == TournamentFormat.ROUND_ROBIN
        assert round_robin_config.rounds == 1
        assert round_robin_config.allow_draws is True
        assert round_robin_config.tiebreak_method == "wins"

    def test_swiss_validation(self) -> None:
        """Test Swiss tournament validation."""
        # Valid
        config = TournamentConfig(
            name="Swiss",
            format=TournamentFormat.SWISS,
            rounds=5,
        )
        assert config.rounds == 5

    def test_tiebreak_method_validation(self) -> None:
        """Test tiebreak method validation."""
        # Valid
        TournamentConfig(name="Test", tiebreak_method="wins")
        TournamentConfig(name="Test", tiebreak_method="head_to_head")
        TournamentConfig(name="Test", tiebreak_method="rating")

        # Invalid
        with pytest.raises(ValidationError):
            TournamentConfig(name="Test", tiebreak_method="invalid")

    def test_compute_hash(self, round_robin_config: TournamentConfig) -> None:
        """Test configuration hash computation."""
        hash1 = round_robin_config.compute_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 16

        # Same config produces same hash
        hash2 = round_robin_config.compute_hash()
        assert hash1 == hash2

        # Different config produces different hash
        other = TournamentConfig(name="Other")
        assert round_robin_config.compute_hash() != other.compute_hash()

    def test_nested_config(self) -> None:
        """Test nested config objects."""
        config = TournamentConfig(
            name="Nested Test",
            match_config=MatchConfig(board_size=9, games_per_match=3),
            rating_config=RatingConfig(k_factor=24.0),
        )

        assert config.match_config.board_size == 9
        assert config.match_config.games_per_match == 3
        assert config.rating_config.k_factor == 24.0


class TestCreateTournamentConfig:
    """Tests for create_tournament_config factory."""

    def test_create_default(self) -> None:
        """Test creating default config."""
        config = create_tournament_config(name="Test")
        assert config.format == TournamentFormat.ROUND_ROBIN
        assert config.match_config.board_size == 19

    def test_create_with_format(self) -> None:
        """Test creating with specific format."""
        config = create_tournament_config(name="Swiss", format="swiss")
        assert config.format == TournamentFormat.SWISS

    def test_create_with_board_size(self) -> None:
        """Test creating with board size."""
        config = create_tournament_config(name="9x9", board_size=9)
        assert config.match_config.board_size == 9

    def test_create_with_games_per_match(self) -> None:
        """Test creating with games per match."""
        config = create_tournament_config(
            name="Best of 5",
            games_per_match=5,
        )
        assert config.match_config.games_per_match == 5

    def test_create_with_additional_kwargs(self) -> None:
        """Test creating with additional kwargs."""
        config = create_tournament_config(
            name="Custom",
            rounds=7,
            allow_draws=False,
        )
        assert config.rounds == 7
        assert config.allow_draws is False
