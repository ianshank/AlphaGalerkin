"""Tests for tournament scheduling."""

from __future__ import annotations

import pytest

from src.tournament.config import TournamentConfig, TournamentFormat
from src.tournament.match import Match, MatchResult, MatchStatus
from src.tournament.player import Player
from src.tournament.scheduler import (
    Pairing,
    TournamentScheduler,
    create_round_robin_schedule,
)


class TestPairing:
    """Tests for Pairing dataclass."""

    def test_initialization(self, sample_players: list[Player]) -> None:
        """Test pairing initialization."""
        pairing = Pairing(
            player1=sample_players[0],
            player2=sample_players[1],
            round_number=1,
        )

        assert pairing.player1 == sample_players[0]
        assert pairing.player2 == sample_players[1]
        assert pairing.round_number == 1

    def test_to_match(self, sample_players: list[Player]) -> None:
        """Test converting pairing to match."""
        pairing = Pairing(
            player1=sample_players[0],
            player2=sample_players[1],
            round_number=2,
        )

        match = pairing.to_match(board_size=9, games_to_play=3)

        assert match.player1_id == sample_players[0].player_id
        assert match.player2_id == sample_players[1].player_id
        assert match.round_number == 2
        assert match.board_size == 9
        assert match.games_to_play == 3


class TestTournamentScheduler:
    """Tests for TournamentScheduler."""

    def test_initialization(self, scheduler: TournamentScheduler) -> None:
        """Test scheduler initialization."""
        assert scheduler.config is not None
        assert scheduler._current_round == 0

    def test_generate_pairings_empty(
        self, scheduler: TournamentScheduler
    ) -> None:
        """Test generating pairings with no players."""
        pairings = scheduler.generate_pairings([])
        assert len(pairings) == 0

    def test_generate_pairings_one_player(
        self, scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test generating pairings with one player."""
        pairings = scheduler.generate_pairings([sample_players[0]])
        assert len(pairings) == 0


class TestRoundRobinScheduler:
    """Tests for round-robin scheduling."""

    def test_generate_round_robin_pairings(
        self, scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test generating round-robin pairings."""
        pairings = scheduler.generate_pairings(sample_players)

        # 4 players: n*(n-1)/2 = 6 matches
        assert len(pairings) == 6

    def test_round_robin_all_play_each_other(
        self, scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test that everyone plays everyone in round-robin."""
        pairings = scheduler.generate_pairings(sample_players)

        # Create set of all pairings
        pairing_set = set()
        for p in pairings:
            pair = frozenset([p.player1.player_id, p.player2.player_id])
            pairing_set.add(pair)

        # Verify all pairs exist
        for i, p1 in enumerate(sample_players):
            for p2 in sample_players[i + 1:]:
                pair = frozenset([p1.player_id, p2.player_id])
                assert pair in pairing_set

    def test_round_robin_odd_players(
        self, scheduler: TournamentScheduler
    ) -> None:
        """Test round-robin with odd number of players."""
        players = [
            Player(name=f"P{i}", player_id=f"p{i}")
            for i in range(5)
        ]

        pairings = scheduler.generate_pairings(players)

        # 5 players: 5*4/2 = 10 matches
        assert len(pairings) == 10


class TestSwissScheduler:
    """Tests for Swiss scheduling."""

    def test_swiss_pairings_initial(
        self, swiss_scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test initial Swiss pairings."""
        pairings = swiss_scheduler.generate_pairings(sample_players)

        # First round: pair by rating
        assert len(pairings) == 2

    def test_swiss_pairings_with_standings(
        self, swiss_scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test Swiss pairings with standings."""
        standings = {
            "p1": 2.0,
            "p2": 1.5,
            "p3": 1.0,
            "p4": 0.5,
        }

        pairings = swiss_scheduler.generate_pairings(sample_players, standings)

        assert len(pairings) == 2

    def test_swiss_avoids_repeat_pairings(
        self, swiss_scheduler: TournamentScheduler
    ) -> None:
        """Test that Swiss avoids repeat pairings."""
        players = [
            Player(name="P1", player_id="p1", rating=1600.0),
            Player(name="P2", player_id="p2", rating=1500.0),
            Player(name="P3", player_id="p3", rating=1400.0),
            Player(name="P4", player_id="p4", rating=1300.0),
        ]

        # First round
        round1 = swiss_scheduler.generate_pairings(players)
        round1_pairs = set()
        for p in round1:
            pair = frozenset([p.player1.player_id, p.player2.player_id])
            round1_pairs.add(pair)

        # Second round with same standings - should get different pairings
        standings = {"p1": 1.0, "p2": 1.0, "p3": 0.0, "p4": 0.0}
        round2 = swiss_scheduler.generate_pairings(players, standings)

        for p in round2:
            pair = frozenset([p.player1.player_id, p.player2.player_id])
            assert pair not in round1_pairs


class TestEliminationScheduler:
    """Tests for elimination scheduling."""

    def test_single_elimination_pairings(
        self, elimination_config: TournamentConfig
    ) -> None:
        """Test single elimination pairings."""
        scheduler = TournamentScheduler(config=elimination_config)
        players = [
            Player(name=f"P{i}", player_id=f"p{i}", rating=1500.0 + i * 100)
            for i in range(4)
        ]

        pairings = scheduler.generate_pairings(players)

        # 4 players: 2 first round matches
        assert len(pairings) == 2

    def test_elimination_seeds_by_rating(
        self, elimination_config: TournamentConfig
    ) -> None:
        """Test elimination seeds by rating."""
        scheduler = TournamentScheduler(config=elimination_config)
        players = [
            Player(name="Weak", player_id="p4", rating=1200.0),
            Player(name="Strong", player_id="p1", rating=1800.0),
            Player(name="Medium", player_id="p2", rating=1500.0),
            Player(name="MedLow", player_id="p3", rating=1400.0),
        ]

        pairings = scheduler.generate_pairings(players)

        # Top seed should play lowest seed
        pair1 = {pairings[0].player1.player_id, pairings[0].player2.player_id}
        assert "p1" in pair1  # Highest rated
        assert "p4" in pair1  # Lowest rated

    def test_elimination_byes(
        self, elimination_config: TournamentConfig
    ) -> None:
        """Test elimination with byes for non-power-of-2."""
        scheduler = TournamentScheduler(config=elimination_config)
        players = [
            Player(name=f"P{i}", player_id=f"p{i}", rating=1500.0)
            for i in range(3)
        ]

        pairings = scheduler.generate_pairings(players)

        # 3 players: only 1 match (one bye)
        assert len(pairings) == 1


class TestGetNextRound:
    """Tests for getting next round pairings."""

    def test_swiss_next_round(
        self, swiss_scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test getting next round for Swiss."""
        # First round
        round1 = swiss_scheduler.generate_pairings(sample_players)

        # Create fake match results
        matches = []
        for pairing in round1:
            match = pairing.to_match()
            match.status = MatchStatus.COMPLETED
            match.result = MatchResult(
                winner_id=pairing.player1.player_id,
                player1_score=1.0,
                player2_score=0.0,
            )
            matches.append(match)

        # Get next round
        round2 = swiss_scheduler.get_next_round(
            sample_players, matches
        )

        assert len(round2) == 2

    def test_elimination_next_round(
        self, elimination_config: TournamentConfig
    ) -> None:
        """Test getting next round for elimination."""
        scheduler = TournamentScheduler(config=elimination_config)
        players = [
            Player(name=f"P{i}", player_id=f"p{i}")
            for i in range(4)
        ]

        # First round
        round1 = scheduler.generate_pairings(players)

        # Create match results (p0 and p2 win)
        matches = []
        for pairing in round1:
            match = pairing.to_match()
            match.status = MatchStatus.COMPLETED
            match.result = MatchResult(
                winner_id=pairing.player1.player_id,
            )
            matches.append(match)

        # Get next round (finals)
        round2 = scheduler.get_next_round(players, matches)

        # Should have 1 match (finals)
        assert len(round2) == 1


class TestReset:
    """Tests for scheduler reset."""

    def test_reset(
        self, swiss_scheduler: TournamentScheduler, sample_players: list[Player]
    ) -> None:
        """Test resetting scheduler."""
        # Generate some pairings
        swiss_scheduler.generate_pairings(sample_players)
        assert swiss_scheduler._current_round == 1
        assert len(swiss_scheduler._past_pairings) > 0

        # Reset
        swiss_scheduler.reset()

        assert swiss_scheduler._current_round == 0
        assert len(swiss_scheduler._past_pairings) == 0


class TestCreateRoundRobinSchedule:
    """Tests for create_round_robin_schedule factory."""

    def test_create_schedule(self) -> None:
        """Test creating round-robin schedule."""
        players = [
            Player(name=f"P{i}", player_id=f"p{i}")
            for i in range(4)
        ]

        matches = create_round_robin_schedule(
            players,
            board_size=9,
            games_per_match=2,
        )

        assert len(matches) == 6  # 4*3/2
        assert all(m.board_size == 9 for m in matches)
        assert all(m.games_to_play == 2 for m in matches)

    def test_empty_schedule(self) -> None:
        """Test creating schedule with no players."""
        matches = create_round_robin_schedule([])
        assert len(matches) == 0
