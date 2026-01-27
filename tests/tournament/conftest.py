"""Pytest fixtures for tournament tests."""

from __future__ import annotations

import pytest

from src.tournament.config import (
    MatchConfig,
    RatingConfig,
    TournamentConfig,
    TournamentFormat,
)
from src.tournament.match import GameRecord, Match, MatchResult, MatchStatus
from src.tournament.player import Player, PlayerRegistry, PlayerStats
from src.tournament.rating import EloRating, RatingSystem
from src.tournament.scheduler import TournamentScheduler


@pytest.fixture
def default_match_config() -> MatchConfig:
    """Create default match config."""
    return MatchConfig()


@pytest.fixture
def default_rating_config() -> RatingConfig:
    """Create default rating config."""
    return RatingConfig()


@pytest.fixture
def round_robin_config() -> TournamentConfig:
    """Create round-robin tournament config."""
    return TournamentConfig(
        name="Test Round Robin",
        format=TournamentFormat.ROUND_ROBIN,
    )


@pytest.fixture
def swiss_config() -> TournamentConfig:
    """Create Swiss tournament config."""
    return TournamentConfig(
        name="Test Swiss",
        format=TournamentFormat.SWISS,
        rounds=4,
    )


@pytest.fixture
def elimination_config() -> TournamentConfig:
    """Create single elimination config."""
    return TournamentConfig(
        name="Test Elimination",
        format=TournamentFormat.SINGLE_ELIMINATION,
    )


@pytest.fixture
def sample_player() -> Player:
    """Create a sample player."""
    return Player(
        name="TestPlayer",
        player_id="test123",
        rating=1500.0,
        is_ai=True,
    )


@pytest.fixture
def sample_players() -> list[Player]:
    """Create a list of sample players."""
    return [
        Player(name="Player1", player_id="p1", rating=1600.0),
        Player(name="Player2", player_id="p2", rating=1500.0),
        Player(name="Player3", player_id="p3", rating=1400.0),
        Player(name="Player4", player_id="p4", rating=1550.0),
    ]


@pytest.fixture
def player_registry() -> PlayerRegistry:
    """Create empty player registry."""
    return PlayerRegistry()


@pytest.fixture
def populated_registry(sample_players: list[Player]) -> PlayerRegistry:
    """Create player registry with sample players."""
    registry = PlayerRegistry()
    for player in sample_players:
        registry.register(player)
    return registry


@pytest.fixture
def sample_match() -> Match:
    """Create a sample match."""
    return Match(
        player1_id="p1",
        player2_id="p2",
        match_id="match123",
        board_size=19,
        games_to_play=3,
    )


@pytest.fixture
def completed_match() -> Match:
    """Create a completed match."""
    match = Match(
        player1_id="p1",
        player2_id="p2",
        match_id="complete123",
        board_size=19,
        games_to_play=1,
    )
    match.start()
    match.add_game(
        black_player_id="p1",
        white_player_id="p2",
        result="B+2.5",
        winner_id="p1",
    )
    return match


@pytest.fixture
def sample_game_record() -> GameRecord:
    """Create a sample game record."""
    return GameRecord(
        game_number=1,
        black_player_id="p1",
        white_player_id="p2",
        result="B+R",
        winner_id="p1",
        moves=156,
    )


@pytest.fixture
def sample_match_result() -> MatchResult:
    """Create a sample match result."""
    return MatchResult(
        winner_id="p1",
        loser_id="p2",
        is_draw=False,
        player1_score=2.0,
        player2_score=1.0,
    )


@pytest.fixture
def rating_system(default_rating_config: RatingConfig) -> RatingSystem:
    """Create a rating system."""
    return RatingSystem(config=default_rating_config)


@pytest.fixture
def elo_rating() -> EloRating:
    """Create an Elo rating object."""
    return EloRating(rating=1500.0)


@pytest.fixture
def scheduler(round_robin_config: TournamentConfig) -> TournamentScheduler:
    """Create a tournament scheduler."""
    return TournamentScheduler(config=round_robin_config)


@pytest.fixture
def swiss_scheduler(swiss_config: TournamentConfig) -> TournamentScheduler:
    """Create a Swiss tournament scheduler."""
    return TournamentScheduler(config=swiss_config)
