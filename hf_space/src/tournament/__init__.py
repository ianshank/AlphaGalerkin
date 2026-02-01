"""Tournament Play module for AlphaGalerkin.

Provides:
- Tournament organization (round-robin, swiss, elimination)
- Match management and scheduling
- Elo rating system
- Results tracking and persistence
"""

from __future__ import annotations

from src.tournament.config import (
    TournamentConfig,
    TournamentFormat,
    MatchConfig,
    RatingConfig,
)
from src.tournament.match import Match, MatchResult, MatchStatus
from src.tournament.player import Player, PlayerRegistry
from src.tournament.rating import EloRating, RatingSystem
from src.tournament.scheduler import TournamentScheduler
from src.tournament.manager import TournamentManager, TournamentState

__all__ = [
    # Configuration
    "TournamentConfig",
    "TournamentFormat",
    "MatchConfig",
    "RatingConfig",
    # Match management
    "Match",
    "MatchResult",
    "MatchStatus",
    # Players
    "Player",
    "PlayerRegistry",
    # Rating system
    "EloRating",
    "RatingSystem",
    # Scheduler
    "TournamentScheduler",
    # Manager
    "TournamentManager",
    "TournamentState",
]
