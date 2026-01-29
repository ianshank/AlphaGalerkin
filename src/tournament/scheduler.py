"""Tournament scheduling and pairing.

Provides:
- Round-robin pairing
- Swiss pairing
- Elimination brackets
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import structlog

from src.tournament.config import TournamentConfig, TournamentFormat
from src.tournament.match import Match
from src.tournament.player import Player


@dataclass
class Pairing:
    """A pairing between two players."""

    player1: Player
    player2: Player
    round_number: int = 1

    def to_match(self, board_size: int = 19, games_to_play: int = 1) -> Match:
        """Convert pairing to a Match.

        Args:
            board_size: Board size for games.
            games_to_play: Number of games.

        Returns:
            Match instance.

        """
        return Match(
            player1_id=self.player1.player_id,
            player2_id=self.player2.player_id,
            round_number=self.round_number,
            board_size=board_size,
            games_to_play=games_to_play,
        )


class TournamentScheduler:
    """Schedules matches for tournaments.

    Supports multiple tournament formats:
    - Round-robin: Everyone plays everyone
    - Swiss: Pair similar-rated players
    - Single/Double elimination: Bracket format
    """

    def __init__(
        self,
        config: TournamentConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize scheduler.

        Args:
            config: Tournament configuration.
            logger: Optional structured logger.

        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__)
        self._rng = random.Random(config.seed)

        # Track pairings history (for Swiss)
        self._past_pairings: set[frozenset[str]] = set()
        self._current_round = 0

    def generate_pairings(
        self,
        players: list[Player],
        standings: dict[str, float] | None = None,
    ) -> list[Pairing]:
        """Generate pairings based on tournament format.

        Args:
            players: List of players.
            standings: Optional current standings (score by player_id).

        Returns:
            List of Pairings.

        """
        if len(players) < 2:
            return []

        self._current_round += 1

        if self.config.format == TournamentFormat.ROUND_ROBIN:
            return self._round_robin_pairings(players)
        elif self.config.format == TournamentFormat.SWISS:
            return self._swiss_pairings(players, standings or {})
        elif self.config.format == TournamentFormat.SINGLE_ELIMINATION:
            return self._elimination_pairings(players, double=False)
        elif self.config.format == TournamentFormat.DOUBLE_ELIMINATION:
            return self._elimination_pairings(players, double=True)
        else:
            # Single match
            return [Pairing(players[0], players[1], self._current_round)]

    def _round_robin_pairings(
        self,
        players: list[Player],
    ) -> list[Pairing]:
        """Generate round-robin pairings.

        Uses circle method for balanced scheduling.

        Args:
            players: List of players.

        Returns:
            All pairings for round-robin.

        """
        n = len(players)
        player_list = list(players)

        # Add bye player if odd number
        if n % 2 == 1:
            player_list.append(None)  # type: ignore
            n += 1

        pairings = []
        round_num = 1

        # Generate all rounds
        for _ in range(n - 1):
            round_pairings = []

            for i in range(n // 2):
                p1 = player_list[i]
                p2 = player_list[n - 1 - i]

                if p1 is not None and p2 is not None:
                    round_pairings.append(Pairing(p1, p2, round_num))

            pairings.extend(round_pairings)
            round_num += 1

            # Rotate players (keep first player fixed)
            player_list = [player_list[0]] + [player_list[-1]] + player_list[1:-1]

        return pairings

    def _swiss_pairings(
        self,
        players: list[Player],
        standings: dict[str, float],
    ) -> list[Pairing]:
        """Generate Swiss pairings.

        Pairs players with similar scores who haven't played.

        Args:
            players: List of players.
            standings: Current standings.

        Returns:
            Pairings for current round.

        """
        # Sort by score, then by rating
        sorted_players = sorted(
            players,
            key=lambda p: (standings.get(p.player_id, 0), p.rating),
            reverse=True,
        )

        pairings = []
        available = set(p.player_id for p in sorted_players)
        player_map = {p.player_id: p for p in sorted_players}

        for player in sorted_players:
            if player.player_id not in available:
                continue

            available.remove(player.player_id)

            # Find best opponent
            best_opponent = None
            best_score_diff = float("inf")

            for opp_id in list(available):
                # Check if already played
                pair_key = frozenset([player.player_id, opp_id])
                if pair_key in self._past_pairings:
                    continue

                opponent = player_map[opp_id]

                # Check rating cutoff (skip if too large a rating gap)
                rating_diff = abs(player.rating - opponent.rating)
                if rating_diff > self.config.swiss_rating_cutoff:
                    continue

                score_diff = abs(
                    standings.get(player.player_id, 0)
                    - standings.get(opp_id, 0)
                )

                if score_diff < best_score_diff:
                    best_score_diff = score_diff
                    best_opponent = opponent

            if best_opponent:
                available.remove(best_opponent.player_id)
                pairings.append(Pairing(player, best_opponent, self._current_round))

                # Record pairing
                pair_key = frozenset([player.player_id, best_opponent.player_id])
                self._past_pairings.add(pair_key)

        return pairings

    def _elimination_pairings(
        self,
        players: list[Player],
        double: bool = False,
    ) -> list[Pairing]:
        """Generate elimination bracket pairings.

        Args:
            players: List of players.
            double: Whether double elimination.

        Returns:
            First round pairings.

        """
        # Seed players by rating
        seeded = sorted(players, key=lambda p: p.rating, reverse=True)

        # Pad to power of 2
        n = len(seeded)
        bracket_size = 1
        while bracket_size < n:
            bracket_size *= 2

        # Add byes
        byes_needed = bracket_size - n

        pairings = []

        # Pair 1 vs last, 2 vs second-last, etc.
        for i in range(bracket_size // 2):
            p1_idx = i
            p2_idx = bracket_size - 1 - i

            if p1_idx < n and p2_idx < n:
                pairings.append(Pairing(
                    seeded[p1_idx],
                    seeded[p2_idx],
                    self._current_round,
                ))
            elif p1_idx < n:
                # p1 gets a bye
                pass

        return pairings

    def get_next_round(
        self,
        players: list[Player],
        results: list[Match],
        standings: dict[str, float] | None = None,
    ) -> list[Pairing]:
        """Get pairings for next round based on results.

        Args:
            players: List of active players.
            results: Previous match results.
            standings: Optional standings override.

        Returns:
            Pairings for next round.

        """
        if self.config.format == TournamentFormat.SWISS:
            # Update standings from results
            if standings is None:
                standings = self._calculate_standings(results)
            return self._swiss_pairings(players, standings)

        elif self.config.format in (
            TournamentFormat.SINGLE_ELIMINATION,
            TournamentFormat.DOUBLE_ELIMINATION,
        ):
            # Filter to winners only
            winners = []
            for match in results:
                if match.result and match.result.winner_id:
                    winner = next(
                        (p for p in players if p.player_id == match.result.winner_id),
                        None,
                    )
                    if winner:
                        winners.append(winner)
            return self._elimination_pairings(winners, double=self.config.format == TournamentFormat.DOUBLE_ELIMINATION)

        return []

    def _calculate_standings(
        self,
        matches: list[Match],
    ) -> dict[str, float]:
        """Calculate standings from match results.

        Args:
            matches: List of completed matches.

        Returns:
            Dictionary of player_id to score.

        """
        standings: dict[str, float] = {}

        for match in matches:
            if not match.result:
                continue

            p1_score = match.result.player1_score
            p2_score = match.result.player2_score

            standings[match.player1_id] = standings.get(match.player1_id, 0) + p1_score
            standings[match.player2_id] = standings.get(match.player2_id, 0) + p2_score

        return standings

    def reset(self) -> None:
        """Reset scheduler state."""
        self._past_pairings.clear()
        self._current_round = 0
        self._rng = random.Random(self.config.seed)


def create_round_robin_schedule(
    players: list[Player],
    board_size: int = 19,
    games_per_match: int = 1,
) -> list[Match]:
    """Create complete round-robin schedule.

    Args:
        players: List of players.
        board_size: Board size for games.
        games_per_match: Games per match.

    Returns:
        List of all matches.

    """
    from src.tournament.config import TournamentConfig

    config = TournamentConfig(
        name="round_robin",
        format=TournamentFormat.ROUND_ROBIN,
    )

    scheduler = TournamentScheduler(config)
    pairings = scheduler.generate_pairings(players)

    return [
        pairing.to_match(board_size, games_per_match)
        for pairing in pairings
    ]
