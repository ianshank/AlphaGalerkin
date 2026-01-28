"""Tournament management and orchestration.

Provides:
- Tournament lifecycle management
- Match orchestration
- Results persistence
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import structlog

from src.tournament.config import TournamentConfig, TournamentFormat
from src.tournament.match import Match, MatchResult, MatchStatus
from src.tournament.player import Player, PlayerRegistry
from src.tournament.rating import RatingSystem
from src.tournament.scheduler import TournamentScheduler


class TournamentState(str, Enum):
    """State of a tournament."""

    CREATED = "created"
    REGISTRATION = "registration"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        return self in (TournamentState.COMPLETED, TournamentState.CANCELLED)


@dataclass
class TournamentStandings:
    """Current tournament standings.

    Tracks scores, rankings, and tiebreakers.
    """

    scores: dict[str, float] = field(default_factory=dict)
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    draws: dict[str, int] = field(default_factory=dict)
    head_to_head: dict[str, dict[str, float]] = field(default_factory=dict)

    def update_from_match(self, match: Match) -> None:
        """Update standings from a completed match.

        Args:
            match: Completed match.
        """
        if not match.result:
            return

        p1 = match.player1_id
        p2 = match.player2_id

        # Initialize if needed
        for pid in [p1, p2]:
            if pid not in self.scores:
                self.scores[pid] = 0.0
                self.wins[pid] = 0
                self.losses[pid] = 0
                self.draws[pid] = 0
                self.head_to_head[pid] = {}

        # Update scores
        self.scores[p1] += match.result.player1_score
        self.scores[p2] += match.result.player2_score

        # Update W/L/D
        if match.result.is_draw:
            self.draws[p1] += 1
            self.draws[p2] += 1
        elif match.result.winner_id == p1:
            self.wins[p1] += 1
            self.losses[p2] += 1
        else:
            self.wins[p2] += 1
            self.losses[p1] += 1

        # Update head-to-head
        if p2 not in self.head_to_head[p1]:
            self.head_to_head[p1][p2] = 0.0
            self.head_to_head[p2][p1] = 0.0

        self.head_to_head[p1][p2] += match.result.player1_score
        self.head_to_head[p2][p1] += match.result.player2_score

    def get_ranked(self, tiebreak: str = "wins") -> list[tuple[str, float]]:
        """Get ranked standings.

        Args:
            tiebreak: Tiebreak method ("wins", "head_to_head", "rating").

        Returns:
            List of (player_id, score) sorted by rank.
        """
        if tiebreak == "wins":
            return sorted(
                self.scores.items(),
                key=lambda x: (x[1], self.wins.get(x[0], 0)),
                reverse=True,
            )
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scores": self.scores,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
        }


class TournamentManager:
    """Manages tournament lifecycle and operations.

    Coordinates:
    - Player registration
    - Match scheduling
    - Result tracking
    - Rating updates
    """

    def __init__(
        self,
        config: TournamentConfig,
        player_registry: PlayerRegistry | None = None,
        rating_system: RatingSystem | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize tournament manager.

        Args:
            config: Tournament configuration.
            player_registry: Optional existing player registry.
            rating_system: Optional existing rating system.
            logger: Optional structured logger.
        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__).bind(
            tournament=config.name,
        )

        self._registry = player_registry or PlayerRegistry()
        self._rating_system = rating_system or RatingSystem(config.rating_config)
        self._scheduler = TournamentScheduler(config)

        # Tournament state
        self._state = TournamentState.CREATED
        self._participants: list[str] = []
        self._matches: list[Match] = []
        self._standings = TournamentStandings()
        self._current_round = 0

        # Timing
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._started_at: str | None = None
        self._completed_at: str | None = None

        # Callbacks
        self._on_match_complete: list[Callable[[Match], None]] = []
        self._on_round_complete: list[Callable[[int], None]] = []
        self._on_tournament_complete: list[Callable[[], None]] = []

    @property
    def state(self) -> TournamentState:
        """Get tournament state."""
        return self._state

    @property
    def is_complete(self) -> bool:
        """Check if tournament is complete."""
        return self._state.is_terminal()

    @property
    def current_round(self) -> int:
        """Get current round number."""
        return self._current_round

    @property
    def participants(self) -> list[Player]:
        """Get list of participants."""
        return [
            self._registry.get(pid)
            for pid in self._participants
            if self._registry.get(pid)
        ]

    @property
    def standings(self) -> TournamentStandings:
        """Get current standings."""
        return self._standings

    def open_registration(self) -> None:
        """Open tournament for registration."""
        if self._state != TournamentState.CREATED:
            raise RuntimeError("Cannot open registration - tournament already started")

        self._state = TournamentState.REGISTRATION
        self._logger.info("registration_opened")

    def register_player(self, player: Player) -> bool:
        """Register a player for the tournament.

        Args:
            player: Player to register.

        Returns:
            True if registration successful.
        """
        if self._state not in (TournamentState.CREATED, TournamentState.REGISTRATION):
            self._logger.warning(
                "registration_rejected",
                reason="registration_closed",
                player=player.name,
            )
            return False

        if player.player_id in self._participants:
            return False

        # Ensure player is in registry
        if not self._registry.get(player.player_id):
            self._registry.register(player)

        self._participants.append(player.player_id)

        self._logger.info(
            "player_registered",
            player=player.name,
            total_participants=len(self._participants),
        )

        return True

    def start(self) -> None:
        """Start the tournament."""
        if self._state.is_terminal():
            raise RuntimeError("Tournament already completed")

        if len(self._participants) < 2:
            raise RuntimeError("Need at least 2 participants")

        self._state = TournamentState.IN_PROGRESS
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._current_round = 1

        # Generate initial pairings
        self._generate_round_pairings()

        self._logger.info(
            "tournament_started",
            participants=len(self._participants),
            format=self.config.format.value,
        )

    def _generate_round_pairings(self) -> None:
        """Generate pairings for current round."""
        players = self.participants

        if self._current_round == 1:
            pairings = self._scheduler.generate_pairings(
                players,
                self._standings.scores,
            )
        else:
            pairings = self._scheduler.get_next_round(
                players,
                self._matches,
                self._standings.scores,
            )

        # Create matches
        for pairing in pairings:
            match = pairing.to_match(
                board_size=self.config.match_config.board_size,
                games_to_play=self.config.match_config.games_per_match,
            )
            match.round_number = self._current_round
            self._matches.append(match)

    def get_pending_matches(self) -> list[Match]:
        """Get matches that haven't been played.

        Returns:
            List of pending matches.
        """
        return [
            m for m in self._matches
            if m.status == MatchStatus.SCHEDULED
            and m.round_number == self._current_round
        ]

    def get_in_progress_matches(self) -> list[Match]:
        """Get matches currently in progress.

        Returns:
            List of in-progress matches.
        """
        return [
            m for m in self._matches
            if m.status == MatchStatus.IN_PROGRESS
        ]

    def get_completed_matches(self) -> list[Match]:
        """Get all completed matches.

        Returns:
            List of completed matches.
        """
        return [
            m for m in self._matches
            if m.status == MatchStatus.COMPLETED
        ]

    def get_match(self, match_id: str) -> Match | None:
        """Get a specific match.

        Args:
            match_id: Match ID.

        Returns:
            Match or None if not found.
        """
        for match in self._matches:
            if match.match_id == match_id:
                return match
        return None

    def start_match(self, match_id: str) -> Match | None:
        """Start a specific match.

        Args:
            match_id: Match ID to start.

        Returns:
            Started match or None if not found.
        """
        match = self.get_match(match_id)
        if match and match.status == MatchStatus.SCHEDULED:
            match.start()
            self._logger.info(
                "match_started",
                match_id=match_id,
                player1=match.player1_id,
                player2=match.player2_id,
            )
            return match
        return None

    def record_match_result(
        self,
        match_id: str,
        result: MatchResult,
    ) -> bool:
        """Record result for a completed match.

        Args:
            match_id: Match ID.
            result: Match result.

        Returns:
            True if recorded successfully.
        """
        match = self.get_match(match_id)
        if not match:
            return False

        match.result = result
        match.status = MatchStatus.COMPLETED
        match.end_time = datetime.now(timezone.utc).isoformat()

        # Update standings
        self._standings.update_from_match(match)

        # Update ratings
        if result.winner_id:
            player_result = 1.0 if result.winner_id == match.player1_id else 0.0
        else:
            player_result = 0.5

        p1_change, p2_change = self._rating_system.record_game(
            match.player1_id,
            match.player2_id,
            player_result,
        )

        result.player1_rating_change = p1_change.change
        result.player2_rating_change = p2_change.change

        # Update player objects
        p1 = self._registry.get(match.player1_id)
        p2 = self._registry.get(match.player2_id)

        if p1:
            p1.rating = self._rating_system.get_rating(match.player1_id)
        if p2:
            p2.rating = self._rating_system.get_rating(match.player2_id)

        self._logger.info(
            "match_completed",
            match_id=match_id,
            winner=result.winner_id or "draw",
            score=f"{result.player1_score}-{result.player2_score}",
        )

        # Fire callbacks
        for callback in self._on_match_complete:
            callback(match)

        # Check if round is complete
        self._check_round_complete()

        return True

    def _check_round_complete(self) -> None:
        """Check if current round is complete."""
        pending = [
            m for m in self._matches
            if m.round_number == self._current_round
            and not m.is_complete
        ]

        if not pending:
            self._logger.info("round_completed", round=self._current_round)

            # Fire callbacks
            for callback in self._on_round_complete:
                callback(self._current_round)

            # Check if tournament is complete
            if self._should_complete():
                self._complete_tournament()
            else:
                # Start next round
                self._current_round += 1
                self._generate_round_pairings()

    def _should_complete(self) -> bool:
        """Check if tournament should complete.

        Returns:
            True if tournament should end.
        """
        format = self.config.format

        if format == TournamentFormat.ROUND_ROBIN:
            # Complete after all pairings played
            expected_matches = len(self._participants) * (len(self._participants) - 1) // 2
            completed = len(self.get_completed_matches())
            return completed >= expected_matches

        elif format == TournamentFormat.SWISS:
            return self._current_round >= self.config.rounds

        elif format in (
            TournamentFormat.SINGLE_ELIMINATION,
            TournamentFormat.DOUBLE_ELIMINATION,
        ):
            # Complete when only one undefeated player
            # Simplified check
            return self._current_round > len(self._participants).bit_length()

        elif format == TournamentFormat.MATCH:
            return len(self.get_completed_matches()) >= 1

        return False

    def _complete_tournament(self) -> None:
        """Complete the tournament."""
        self._state = TournamentState.COMPLETED
        self._completed_at = datetime.now(timezone.utc).isoformat()

        self._logger.info(
            "tournament_completed",
            rounds=self._current_round,
            matches=len(self._matches),
        )

        # Fire callbacks
        for callback in self._on_tournament_complete:
            callback()

    def get_results(self) -> dict[str, Any]:
        """Get tournament results.

        Returns:
            Dictionary with complete results.
        """
        ranked = self._standings.get_ranked(self.config.tiebreak_method)

        return {
            "tournament_name": self.config.name,
            "format": self.config.format.value,
            "state": self._state.value,
            "participants": len(self._participants),
            "rounds_played": self._current_round,
            "total_matches": len(self._matches),
            "standings": [
                {
                    "rank": i + 1,
                    "player_id": pid,
                    "player_name": (
                        self._registry.get(pid).name
                        if self._registry.get(pid)
                        else "Unknown"
                    ),
                    "score": score,
                    "wins": self._standings.wins.get(pid, 0),
                    "losses": self._standings.losses.get(pid, 0),
                    "draws": self._standings.draws.get(pid, 0),
                }
                for i, (pid, score) in enumerate(ranked)
            ],
            "winner": ranked[0][0] if ranked else None,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
        }

    def on_match_complete(self, callback: Callable[[Match], None]) -> None:
        """Register callback for match completion.

        Args:
            callback: Function to call when match completes.
        """
        self._on_match_complete.append(callback)

    def on_round_complete(self, callback: Callable[[int], None]) -> None:
        """Register callback for round completion.

        Args:
            callback: Function to call when round completes.
        """
        self._on_round_complete.append(callback)

    def on_tournament_complete(self, callback: Callable[[], None]) -> None:
        """Register callback for tournament completion.

        Args:
            callback: Function to call when tournament completes.
        """
        self._on_tournament_complete.append(callback)

    def save_state(self, path: Path | str) -> None:
        """Save tournament state to file.

        Args:
            path: Path to save state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config.model_dump(mode="json"),
            "state": self._state.value,
            "participants": self._participants,
            "matches": [m.to_dict() for m in self._matches],
            "standings": self._standings.to_dict(),
            "current_round": self._current_round,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self._logger.debug("state_saved", path=str(path))

    def get_summary(self) -> dict[str, Any]:
        """Get tournament summary.

        Returns:
            Summary dictionary.
        """
        return {
            "name": self.config.name,
            "format": self.config.format.value,
            "state": self._state.value,
            "participants": len(self._participants),
            "current_round": self._current_round,
            "total_matches": len(self._matches),
            "completed_matches": len(self.get_completed_matches()),
            "pending_matches": len(self.get_pending_matches()),
        }


def create_tournament(
    name: str,
    format: str = "round_robin",
    board_size: int = 19,
    **kwargs: Any,
) -> TournamentManager:
    """Factory function to create tournament.

    Args:
        name: Tournament name.
        format: Tournament format.
        board_size: Board size for games.
        **kwargs: Additional configuration.

    Returns:
        TournamentManager instance.
    """
    from src.tournament.config import create_tournament_config

    config = create_tournament_config(
        name=name,
        format=format,
        board_size=board_size,
        **kwargs,
    )

    return TournamentManager(config=config)
