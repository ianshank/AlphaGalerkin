"""Configuration schemas for Tournament Play.

Provides Pydantic-validated configuration with:
- No hardcoded values
- Tournament format options
- Match and rating settings
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TournamentFormat(str, Enum):
    """Tournament format types."""

    ROUND_ROBIN = "round_robin"  # Everyone plays everyone
    SWISS = "swiss"  # Swiss pairing system
    SINGLE_ELIMINATION = "single_elimination"  # Knockout format
    DOUBLE_ELIMINATION = "double_elimination"  # Double knockout
    MATCH = "match"  # Single match between two players


class TimeControl(str, Enum):
    """Time control options."""

    BLITZ = "blitz"  # Fast games
    RAPID = "rapid"  # Medium speed
    STANDARD = "standard"  # Full time
    UNLIMITED = "unlimited"  # No time limit


class MatchConfig(BaseModel):
    """Configuration for individual matches.

    Controls game settings for matches.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    board_size: int = Field(
        default=19,
        ge=5,
        le=25,
        description="Board size for games",
    )
    komi: float = Field(
        default=7.5,
        ge=0.0,
        le=100.0,
        description="Komi (compensation for white)",
    )
    handicap: int = Field(
        default=0,
        ge=0,
        le=9,
        description="Handicap stones",
    )
    games_per_match: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of games per match",
    )
    alternating_colors: bool = Field(
        default=True,
        description="Alternate colors between games",
    )
    time_control: TimeControl = Field(
        default=TimeControl.STANDARD,
        description="Time control setting",
    )
    main_time_seconds: int = Field(
        default=600,
        ge=0,
        description="Main time in seconds",
    )
    byoyomi_seconds: int = Field(
        default=30,
        ge=0,
        description="Byoyomi time per period",
    )
    byoyomi_periods: int = Field(
        default=5,
        ge=0,
        description="Number of byoyomi periods",
    )
    mcts_simulations: int = Field(
        default=800,
        ge=1,
        description="MCTS simulations for AI players",
    )


class RatingConfig(BaseModel):
    """Configuration for Elo rating system.

    Controls rating calculations.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    initial_rating: float = Field(
        default=1500.0,
        ge=0.0,
        le=5000.0,
        description="Initial rating for new players",
    )
    k_factor: float = Field(
        default=32.0,
        gt=0.0,
        le=100.0,
        description="K-factor for rating changes",
    )
    k_factor_new_player: float = Field(
        default=40.0,
        gt=0.0,
        le=100.0,
        description="K-factor for players with < 30 games",
    )
    k_factor_high_rated: float = Field(
        default=16.0,
        gt=0.0,
        le=100.0,
        description="K-factor for players > 2400 rating",
    )
    high_rating_threshold: float = Field(
        default=2400.0,
        ge=1000.0,
        description="Rating threshold for reduced K-factor",
    )
    new_player_games: int = Field(
        default=30,
        ge=1,
        description="Games needed to be considered established",
    )
    min_rating: float = Field(
        default=100.0,
        ge=0.0,
        description="Minimum possible rating",
    )
    max_rating: float = Field(
        default=4000.0,
        ge=1000.0,
        description="Maximum possible rating",
    )


class TournamentConfig(BaseModel):
    """Complete tournament configuration.

    Defines tournament structure and settings.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Tournament name",
    )
    format: TournamentFormat = Field(
        default=TournamentFormat.ROUND_ROBIN,
        description="Tournament format",
    )
    rounds: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of rounds (for Swiss)",
    )
    match_config: MatchConfig = Field(
        default_factory=MatchConfig,
        description="Match configuration",
    )
    rating_config: RatingConfig = Field(
        default_factory=RatingConfig,
        description="Rating configuration",
    )

    # Tournament options
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for pairings",
    )
    allow_draws: bool = Field(
        default=True,
        description="Allow draw results",
    )
    tiebreak_method: str = Field(
        default="wins",
        pattern="^(wins|head_to_head|rating|random)$",
        description="Tiebreak method",
    )

    # Swiss-specific
    swiss_pair_down: bool = Field(
        default=True,
        description="Allow Swiss pairing down",
    )
    swiss_rating_cutoff: float = Field(
        default=200.0,
        ge=0.0,
        description="Rating difference cutoff for Swiss pairing",
    )

    # Elimination-specific
    third_place_match: bool = Field(
        default=True,
        description="Play third place match in elimination",
    )

    # Persistence
    save_games: bool = Field(
        default=True,
        description="Save game records (SGF)",
    )
    output_dir: str = Field(
        default="tournaments",
        description="Output directory for results",
    )

    @model_validator(mode="after")
    def validate_swiss_rounds(self) -> "TournamentConfig":
        """Validate Swiss rounds configuration."""
        if self.format == TournamentFormat.SWISS and self.rounds < 1:
            raise ValueError("Swiss tournament requires at least 1 round")
        return self

    def compute_hash(self) -> str:
        """Compute unique hash of configuration.

        Returns:
            Hexadecimal hash string.
        """
        data = self.model_dump(mode="json")
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def create_tournament_config(
    name: str,
    format: str = "round_robin",
    board_size: int = 19,
    games_per_match: int = 1,
    **kwargs: Any,
) -> TournamentConfig:
    """Factory function to create tournament config.

    Args:
        name: Tournament name.
        format: Tournament format.
        board_size: Board size for games.
        games_per_match: Games per match.
        **kwargs: Additional configuration.

    Returns:
        TournamentConfig instance.
    """
    match_config = MatchConfig(
        board_size=board_size,
        games_per_match=games_per_match,
    )

    return TournamentConfig(
        name=name,
        format=TournamentFormat(format),
        match_config=match_config,
        **kwargs,
    )
