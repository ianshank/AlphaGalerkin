"""Player management for tournaments.

Provides:
- Player representation
- Player registry
- Player statistics tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterator
import uuid

import structlog


@dataclass
class PlayerStats:
    """Statistics for a player.

    Tracks wins, losses, draws, and performance metrics.
    """

    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games_as_black: int = 0
    games_as_white: int = 0
    wins_as_black: int = 0
    wins_as_white: int = 0
    total_points: float = 0.0

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def draw_rate(self) -> float:
        """Calculate draw rate."""
        if self.games_played == 0:
            return 0.0
        return self.draws / self.games_played

    @property
    def score(self) -> float:
        """Calculate score (1 per win, 0.5 per draw)."""
        return self.wins + 0.5 * self.draws

    def record_result(
        self,
        won: bool,
        drawn: bool,
        as_black: bool,
        points: float = 1.0,
    ) -> None:
        """Record a game result.

        Args:
            won: Whether the player won.
            drawn: Whether the game was drawn.
            as_black: Whether player was black.
            points: Points for this game.
        """
        self.games_played += 1

        if as_black:
            self.games_as_black += 1
        else:
            self.games_as_white += 1

        if drawn:
            self.draws += 1
            self.total_points += 0.5
        elif won:
            self.wins += 1
            self.total_points += points
            if as_black:
                self.wins_as_black += 1
            else:
                self.wins_as_white += 1
        else:
            self.losses += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "score": self.score,
            "total_points": self.total_points,
        }


@dataclass
class Player:
    """Represents a tournament player.

    Can be human or AI with associated model.
    """

    name: str
    player_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    rating: float = 1500.0
    is_ai: bool = True
    model_path: str | None = None
    stats: PlayerStats = field(default_factory=PlayerStats)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    @property
    def games_played(self) -> int:
        """Get total games played."""
        return self.stats.games_played

    @property
    def win_rate(self) -> float:
        """Get win rate."""
        return self.stats.win_rate

    def record_result(
        self,
        won: bool,
        drawn: bool = False,
        as_black: bool = True,
        rating_change: float = 0.0,
    ) -> None:
        """Record a game result.

        Args:
            won: Whether the player won.
            drawn: Whether the game was drawn.
            as_black: Whether player was black.
            rating_change: Rating change from this game.
        """
        self.stats.record_result(won, drawn, as_black)
        self.rating += rating_change

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "player_id": self.player_id,
            "rating": self.rating,
            "is_ai": self.is_ai,
            "model_path": self.model_path,
            "stats": self.stats.to_dict(),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Player":
        """Create from dictionary."""
        stats_data = data.get("stats", {})
        stats = PlayerStats(
            games_played=stats_data.get("games_played", 0),
            wins=stats_data.get("wins", 0),
            losses=stats_data.get("losses", 0),
            draws=stats_data.get("draws", 0),
            total_points=stats_data.get("total_points", 0.0),
        )

        return cls(
            name=data["name"],
            player_id=data.get("player_id", str(uuid.uuid4())[:8]),
            rating=data.get("rating", 1500.0),
            is_ai=data.get("is_ai", True),
            model_path=data.get("model_path"),
            stats=stats,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )

    def __hash__(self) -> int:
        return hash(self.player_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Player):
            return self.player_id == other.player_id
        return False


class PlayerRegistry:
    """Registry for managing tournament players.

    Provides:
    - Player registration
    - Lookup by ID or name
    - Ranking and sorting
    """

    def __init__(
        self,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize player registry.

        Args:
            logger: Optional structured logger.
        """
        self._logger = logger or structlog.get_logger(__name__)
        self._players: dict[str, Player] = {}
        self._by_name: dict[str, str] = {}  # name -> player_id

    def register(self, player: Player) -> None:
        """Register a player.

        Args:
            player: Player to register.
        """
        if player.player_id in self._players:
            self._logger.warning(
                "player_already_registered",
                player_id=player.player_id,
            )
            return

        self._players[player.player_id] = player
        self._by_name[player.name.lower()] = player.player_id

        self._logger.debug(
            "player_registered",
            name=player.name,
            player_id=player.player_id,
        )

    def create_player(
        self,
        name: str,
        rating: float = 1500.0,
        is_ai: bool = True,
        model_path: str | None = None,
        **metadata: Any,
    ) -> Player:
        """Create and register a new player.

        Args:
            name: Player name.
            rating: Initial rating.
            is_ai: Whether this is an AI player.
            model_path: Path to model for AI players.
            **metadata: Additional metadata.

        Returns:
            Newly created Player.
        """
        player = Player(
            name=name,
            rating=rating,
            is_ai=is_ai,
            model_path=model_path,
            metadata=dict(metadata),
        )
        self.register(player)
        return player

    def get(self, player_id: str) -> Player | None:
        """Get player by ID.

        Args:
            player_id: Player ID.

        Returns:
            Player or None if not found.
        """
        return self._players.get(player_id)

    def get_by_name(self, name: str) -> Player | None:
        """Get player by name.

        Args:
            name: Player name (case-insensitive).

        Returns:
            Player or None if not found.
        """
        player_id = self._by_name.get(name.lower())
        if player_id:
            return self._players.get(player_id)
        return None

    def remove(self, player_id: str) -> bool:
        """Remove a player.

        Args:
            player_id: Player ID to remove.

        Returns:
            True if player was removed.
        """
        if player_id not in self._players:
            return False

        player = self._players[player_id]
        del self._players[player_id]
        if player.name.lower() in self._by_name:
            del self._by_name[player.name.lower()]

        return True

    def list_players(self) -> list[Player]:
        """List all registered players.

        Returns:
            List of players.
        """
        return list(self._players.values())

    def get_ranked(self, by: str = "rating") -> list[Player]:
        """Get players sorted by ranking criterion.

        Args:
            by: Ranking criterion ("rating", "wins", "win_rate", "score").

        Returns:
            Sorted list of players.
        """
        if by == "rating":
            return sorted(
                self._players.values(),
                key=lambda p: p.rating,
                reverse=True,
            )
        elif by == "wins":
            return sorted(
                self._players.values(),
                key=lambda p: p.stats.wins,
                reverse=True,
            )
        elif by == "win_rate":
            return sorted(
                self._players.values(),
                key=lambda p: p.win_rate,
                reverse=True,
            )
        elif by == "score":
            return sorted(
                self._players.values(),
                key=lambda p: p.stats.score,
                reverse=True,
            )
        return list(self._players.values())

    def iter_players(self) -> Iterator[Player]:
        """Iterate through all players.

        Yields:
            Player instances.
        """
        yield from self._players.values()

    def __len__(self) -> int:
        return len(self._players)

    def __contains__(self, player_id: str) -> bool:
        return player_id in self._players

    def to_dict(self) -> dict[str, Any]:
        """Export registry to dictionary.

        Returns:
            Dictionary with all players.
        """
        return {
            "players": [p.to_dict() for p in self._players.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerRegistry":
        """Create registry from dictionary.

        Args:
            data: Dictionary with players.

        Returns:
            PlayerRegistry instance.
        """
        registry = cls()
        for player_data in data.get("players", []):
            player = Player.from_dict(player_data)
            registry.register(player)
        return registry
