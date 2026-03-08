"""Pydantic configuration schemas for engine integration.

Provides typed, validated configurations for UCI engine communication,
match orchestration, time controls, and Elo estimation. All parameters
have sensible defaults with no hardcoded values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from src.engines.protocol import EngineProtocol
from src.templates.config import BaseModuleConfig


class TimeControl(BaseModuleConfig):
    """Time control settings for engine matches.

    Supports classical, rapid, blitz, and bullet time controls
    with optional increments and period-based controls.
    """

    initial_time_ms: int = Field(
        default=60000,
        ge=0,
        description="Initial time per player in milliseconds",
    )
    increment_ms: int = Field(
        default=0,
        ge=0,
        description="Time increment per move in milliseconds",
    )
    moves_per_period: int | None = Field(
        default=None,
        ge=1,
        description="Moves per time period (None for sudden death)",
    )


class EngineConfig(BaseModuleConfig):
    """Base configuration for an external chess engine.

    Provides common settings shared across all engine protocols.
    """

    engine_path: Path = Field(
        description="Path to the engine binary executable",
    )
    protocol: EngineProtocol = Field(
        default=EngineProtocol.UCI,
        description="Communication protocol to use",
    )
    startup_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Maximum time to wait for engine startup",
    )
    move_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Maximum time to wait for a move response",
    )
    options: dict[str, str | int | bool] = Field(
        default_factory=dict,
        description="Engine-specific UCI options (e.g., {'Skill Level': 10})",
    )


class UCIConfig(EngineConfig):
    """UCI-specific engine configuration.

    Extends EngineConfig with UCI search parameters.
    At least one search limit (depth, nodes, or movetime) must be set.
    """

    protocol: Literal[EngineProtocol.UCI] = EngineProtocol.UCI
    depth_limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum search depth in plies",
    )
    nodes_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of nodes to search",
    )
    movetime_ms: int | None = Field(
        default=None,
        ge=100,
        description="Time to search per move in milliseconds",
    )
    hash_mb: int = Field(
        default=64,
        ge=1,
        le=32768,
        description="Hash table size in megabytes",
    )
    threads: int = Field(
        default=1,
        ge=1,
        le=512,
        description="Number of search threads",
    )

    @model_validator(mode="after")
    def validate_search_limits(self) -> UCIConfig:
        """Ensure at least one search limit is configured."""
        if self.depth_limit is None and self.nodes_limit is None and self.movetime_ms is None:
            raise ValueError(
                "At least one search limit must be set: depth_limit, nodes_limit, or movetime_ms"
            )
        return self


class EloConfig(BaseModuleConfig):
    """Configuration for Elo rating estimation."""

    k_factor: float = Field(
        default=32.0,
        gt=0,
        description="Elo K-factor for rating updates",
    )
    draw_elo: float = Field(
        default=0.0,
        description="Draw Elo advantage parameter",
    )
    initial_rating: float = Field(
        default=1500.0,
        gt=0,
        description="Default starting Elo rating",
    )
    confidence_level: float = Field(
        default=0.95,
        gt=0,
        lt=1,
        description="Confidence level for Elo interval estimation",
    )
    elo_base: float = Field(
        default=10.0,
        gt=1,
        description="Base for Elo expected score formula",
    )
    elo_divisor: float = Field(
        default=400.0,
        gt=0,
        description="Divisor for Elo expected score formula",
    )


class MatchConfig(BaseModuleConfig):
    """Configuration for engine match orchestration."""

    n_games: int = Field(
        default=10,
        ge=1,
        description="Number of games to play in the match",
    )
    time_control: TimeControl = Field(
        default_factory=lambda: TimeControl(name="default_tc"),
        description="Time control settings",
    )
    alternate_colors: bool = Field(
        default=True,
        description="Alternate colors between games",
    )
    max_moves: int = Field(
        default=500,
        ge=50,
        description="Maximum moves per game before adjudicating draw",
    )
    opening_fen: str | None = Field(
        default=None,
        description="Starting FEN position (None for standard opening)",
    )
    pgn_output_path: Path | None = Field(
        default=None,
        description="Path to write PGN output file",
    )
