"""Match management for tournaments.

Provides:
- Match representation
- Match result tracking
- Game records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class MatchStatus(str, Enum):
    """Status of a match."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FORFEIT = "forfeit"

    def is_terminal(self) -> bool:
        """Check if status is terminal."""
        return self in (
            MatchStatus.COMPLETED,
            MatchStatus.CANCELLED,
            MatchStatus.FORFEIT,
        )


@dataclass
class GameRecord:
    """Record of a single game in a match.

    Attributes:
        game_number: Game number within match.
        black_player_id: ID of black player.
        white_player_id: ID of white player.
        result: Game result (e.g., "B+2.5", "W+R", "Draw").
        moves: Number of moves in game.
        sgf_path: Path to SGF file if saved.
        duration_seconds: Game duration.
    """

    game_number: int
    black_player_id: str
    white_player_id: str
    result: str = ""
    winner_id: str | None = None
    is_draw: bool = False
    moves: int = 0
    sgf_path: str | None = None
    duration_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_complete(self) -> bool:
        """Check if game is complete."""
        return bool(self.result)

    @property
    def black_won(self) -> bool:
        """Check if black won."""
        return self.winner_id == self.black_player_id

    @property
    def white_won(self) -> bool:
        """Check if white won."""
        return self.winner_id == self.white_player_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_number": self.game_number,
            "black_player_id": self.black_player_id,
            "white_player_id": self.white_player_id,
            "result": self.result,
            "winner_id": self.winner_id,
            "is_draw": self.is_draw,
            "moves": self.moves,
            "sgf_path": self.sgf_path,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        """Create from dictionary."""
        return cls(
            game_number=data["game_number"],
            black_player_id=data["black_player_id"],
            white_player_id=data["white_player_id"],
            result=data.get("result", ""),
            winner_id=data.get("winner_id"),
            is_draw=data.get("is_draw", False),
            moves=data.get("moves", 0),
            sgf_path=data.get("sgf_path"),
            duration_seconds=data.get("duration_seconds", 0.0),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class MatchResult:
    """Result of a match.

    Attributes:
        winner_id: ID of the match winner.
        is_draw: Whether match was drawn.
        player1_score: Score for player 1.
        player2_score: Score for player 2.
        player1_rating_change: Rating change for player 1.
        player2_rating_change: Rating change for player 2.
    """

    winner_id: str | None = None
    loser_id: str | None = None
    is_draw: bool = False
    player1_score: float = 0.0
    player2_score: float = 0.0
    player1_rating_change: float = 0.0
    player2_rating_change: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "winner_id": self.winner_id,
            "loser_id": self.loser_id,
            "is_draw": self.is_draw,
            "player1_score": self.player1_score,
            "player2_score": self.player2_score,
            "player1_rating_change": self.player1_rating_change,
            "player2_rating_change": self.player2_rating_change,
        }


@dataclass
class Match:
    """Represents a match between two players.

    A match consists of one or more games.
    """

    player1_id: str
    player2_id: str
    match_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    round_number: int = 1
    board_size: int = 19
    games_to_play: int = 1
    status: MatchStatus = MatchStatus.SCHEDULED
    games: list[GameRecord] = field(default_factory=list)
    result: MatchResult | None = None
    scheduled_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if match is complete."""
        return self.status.is_terminal()

    @property
    def games_played(self) -> int:
        """Get number of games played."""
        return len(self.games)

    @property
    def player1_score(self) -> float:
        """Calculate player 1 score."""
        score = 0.0
        for game in self.games:
            if game.winner_id == self.player1_id:
                score += 1.0
            elif game.is_draw:
                score += 0.5
        return score

    @property
    def player2_score(self) -> float:
        """Calculate player 2 score."""
        score = 0.0
        for game in self.games:
            if game.winner_id == self.player2_id:
                score += 1.0
            elif game.is_draw:
                score += 0.5
        return score

    @property
    def current_leader(self) -> str | None:
        """Get current match leader."""
        if self.player1_score > self.player2_score:
            return self.player1_id
        elif self.player2_score > self.player1_score:
            return self.player2_id
        return None

    def start(self) -> None:
        """Start the match."""
        self.status = MatchStatus.IN_PROGRESS
        self.start_time = datetime.now(timezone.utc).isoformat()

    def add_game(
        self,
        black_player_id: str,
        white_player_id: str,
        result: str,
        winner_id: str | None = None,
        is_draw: bool = False,
        moves: int = 0,
        sgf_path: str | None = None,
        duration_seconds: float = 0.0,
    ) -> GameRecord:
        """Add a game result to the match.

        Args:
            black_player_id: ID of black player.
            white_player_id: ID of white player.
            result: Game result string.
            winner_id: ID of winner (None for draw).
            is_draw: Whether game was drawn.
            moves: Number of moves.
            sgf_path: Path to SGF file.
            duration_seconds: Game duration.

        Returns:
            Created GameRecord.
        """
        game = GameRecord(
            game_number=len(self.games) + 1,
            black_player_id=black_player_id,
            white_player_id=white_player_id,
            result=result,
            winner_id=winner_id,
            is_draw=is_draw,
            moves=moves,
            sgf_path=sgf_path,
            duration_seconds=duration_seconds,
        )
        self.games.append(game)

        # Check if match should complete
        if len(self.games) >= self.games_to_play:
            self._complete_match()

        return game

    def _complete_match(self) -> None:
        """Complete the match and determine result."""
        self.status = MatchStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc).isoformat()

        p1_score = self.player1_score
        p2_score = self.player2_score

        if p1_score > p2_score:
            winner_id = self.player1_id
            loser_id = self.player2_id
            is_draw = False
        elif p2_score > p1_score:
            winner_id = self.player2_id
            loser_id = self.player1_id
            is_draw = False
        else:
            winner_id = None
            loser_id = None
            is_draw = True

        self.result = MatchResult(
            winner_id=winner_id,
            loser_id=loser_id,
            is_draw=is_draw,
            player1_score=p1_score,
            player2_score=p2_score,
        )

    def cancel(self, reason: str = "") -> None:
        """Cancel the match.

        Args:
            reason: Reason for cancellation.
        """
        self.status = MatchStatus.CANCELLED
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.metadata["cancel_reason"] = reason

    def forfeit(self, forfeiter_id: str) -> None:
        """Record a forfeit.

        Args:
            forfeiter_id: ID of the player who forfeited.
        """
        self.status = MatchStatus.FORFEIT
        self.end_time = datetime.now(timezone.utc).isoformat()

        winner_id = (
            self.player2_id if forfeiter_id == self.player1_id
            else self.player1_id
        )

        self.result = MatchResult(
            winner_id=winner_id,
            loser_id=forfeiter_id,
            is_draw=False,
        )

    def get_next_colors(self) -> tuple[str, str]:
        """Get player IDs for next game's colors.

        Returns:
            Tuple of (black_player_id, white_player_id).
        """
        # Alternate colors
        if len(self.games) % 2 == 0:
            return self.player1_id, self.player2_id
        else:
            return self.player2_id, self.player1_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "player1_id": self.player1_id,
            "player2_id": self.player2_id,
            "round_number": self.round_number,
            "board_size": self.board_size,
            "games_to_play": self.games_to_play,
            "status": self.status.value,
            "games": [g.to_dict() for g in self.games],
            "result": self.result.to_dict() if self.result else None,
            "scheduled_time": self.scheduled_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Match":
        """Create from dictionary."""
        games = [
            GameRecord.from_dict(g) for g in data.get("games", [])
        ]

        result_data = data.get("result")
        result = None
        if result_data:
            result = MatchResult(**result_data)

        return cls(
            match_id=data.get("match_id", str(uuid.uuid4())[:8]),
            player1_id=data["player1_id"],
            player2_id=data["player2_id"],
            round_number=data.get("round_number", 1),
            board_size=data.get("board_size", 19),
            games_to_play=data.get("games_to_play", 1),
            status=MatchStatus(data.get("status", "scheduled")),
            games=games,
            result=result,
            scheduled_time=data.get("scheduled_time"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            metadata=data.get("metadata", {}),
        )
