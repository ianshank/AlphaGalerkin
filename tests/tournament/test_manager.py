"""Tests for tournament manager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.tournament.config import TournamentConfig, TournamentFormat
from src.tournament.manager import (
    TournamentManager,
    TournamentStandings,
    TournamentState,
    create_tournament,
)
from src.tournament.match import Match, MatchResult, MatchStatus
from src.tournament.player import Player, PlayerRegistry


class TestTournamentState:
    """Tests for TournamentState enum."""

    def test_all_states_exist(self) -> None:
        """Test all tournament states exist."""
        assert TournamentState.CREATED.value == "created"
        assert TournamentState.REGISTRATION.value == "registration"
        assert TournamentState.IN_PROGRESS.value == "in_progress"
        assert TournamentState.COMPLETED.value == "completed"
        assert TournamentState.CANCELLED.value == "cancelled"

    def test_is_terminal(self) -> None:
        """Test is_terminal method."""
        assert not TournamentState.CREATED.is_terminal()
        assert not TournamentState.REGISTRATION.is_terminal()
        assert not TournamentState.IN_PROGRESS.is_terminal()
        assert TournamentState.COMPLETED.is_terminal()
        assert TournamentState.CANCELLED.is_terminal()


class TestTournamentStandings:
    """Tests for TournamentStandings dataclass."""

    def test_default_values(self) -> None:
        """Test default standings values."""
        standings = TournamentStandings()
        assert len(standings.scores) == 0
        assert len(standings.wins) == 0
        assert len(standings.losses) == 0

    def test_update_from_match_win(self) -> None:
        """Test updating standings from a win."""
        standings = TournamentStandings()

        match = Match(player1_id="p1", player2_id="p2")
        match.result = MatchResult(
            winner_id="p1",
            player1_score=1.0,
            player2_score=0.0,
        )

        standings.update_from_match(match)

        assert standings.scores["p1"] == 1.0
        assert standings.scores["p2"] == 0.0
        assert standings.wins["p1"] == 1
        assert standings.losses["p2"] == 1

    def test_update_from_match_draw(self) -> None:
        """Test updating standings from a draw."""
        standings = TournamentStandings()

        match = Match(player1_id="p1", player2_id="p2")
        match.result = MatchResult(
            is_draw=True,
            player1_score=0.5,
            player2_score=0.5,
        )

        standings.update_from_match(match)

        assert standings.scores["p1"] == 0.5
        assert standings.scores["p2"] == 0.5
        assert standings.draws["p1"] == 1
        assert standings.draws["p2"] == 1

    def test_update_head_to_head(self) -> None:
        """Test head-to-head tracking."""
        standings = TournamentStandings()

        match = Match(player1_id="p1", player2_id="p2")
        match.result = MatchResult(
            winner_id="p1",
            player1_score=1.0,
            player2_score=0.0,
        )

        standings.update_from_match(match)

        assert standings.head_to_head["p1"]["p2"] == 1.0
        assert standings.head_to_head["p2"]["p1"] == 0.0

    def test_get_ranked(self) -> None:
        """Test getting ranked standings."""
        standings = TournamentStandings()
        standings.scores = {"p1": 3.0, "p2": 2.0, "p3": 2.5}
        standings.wins = {"p1": 3, "p2": 2, "p3": 2}

        ranked = standings.get_ranked()

        assert ranked[0][0] == "p1"
        assert ranked[1][0] == "p3"
        assert ranked[2][0] == "p2"

    def test_get_ranked_with_tiebreak(self) -> None:
        """Test ranked standings with wins tiebreak."""
        standings = TournamentStandings()
        standings.scores = {"p1": 2.5, "p2": 2.5, "p3": 1.0}
        standings.wins = {"p1": 2, "p2": 1, "p3": 1}

        ranked = standings.get_ranked(tiebreak="wins")

        # p1 and p2 have same score but p1 has more wins
        assert ranked[0][0] == "p1"
        assert ranked[1][0] == "p2"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        standings = TournamentStandings()
        standings.scores = {"p1": 2.0}
        standings.wins = {"p1": 2}

        data = standings.to_dict()

        assert data["scores"]["p1"] == 2.0
        assert data["wins"]["p1"] == 2


class TestTournamentManager:
    """Tests for TournamentManager."""

    def test_initialization(self, round_robin_config: TournamentConfig) -> None:
        """Test manager initialization."""
        manager = TournamentManager(config=round_robin_config)

        assert manager.state == TournamentState.CREATED
        assert manager.current_round == 0
        assert len(manager.participants) == 0

    def test_with_existing_registry(
        self,
        round_robin_config: TournamentConfig,
        populated_registry: PlayerRegistry,
    ) -> None:
        """Test manager with existing registry."""
        manager = TournamentManager(
            config=round_robin_config,
            player_registry=populated_registry,
        )

        assert len(manager._registry) == 4

    def test_open_registration(self, round_robin_config: TournamentConfig) -> None:
        """Test opening registration."""
        manager = TournamentManager(config=round_robin_config)
        manager.open_registration()

        assert manager.state == TournamentState.REGISTRATION

    def test_open_registration_already_started(self, round_robin_config: TournamentConfig) -> None:
        """Test cannot open registration after start."""
        manager = TournamentManager(config=round_robin_config)
        manager.open_registration()
        manager.register_player(Player(name="P1"))
        manager.register_player(Player(name="P2"))
        manager.start()

        with pytest.raises(RuntimeError):
            manager.open_registration()

    def test_register_player(self, round_robin_config: TournamentConfig) -> None:
        """Test registering a player."""
        manager = TournamentManager(config=round_robin_config)
        player = Player(name="TestPlayer")

        result = manager.register_player(player)

        assert result is True
        assert len(manager.participants) == 1

    def test_register_player_duplicate(self, round_robin_config: TournamentConfig) -> None:
        """Test registering duplicate player."""
        manager = TournamentManager(config=round_robin_config)
        player = Player(name="TestPlayer", player_id="same")

        manager.register_player(player)
        result = manager.register_player(player)

        assert result is False
        assert len(manager.participants) == 1

    def test_register_player_closed(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test cannot register after tournament starts."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players[:2]:
            manager.register_player(p)
        manager.start()

        result = manager.register_player(sample_players[2])
        assert result is False

    def test_start_tournament(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test starting tournament."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)

        manager.start()

        assert manager.state == TournamentState.IN_PROGRESS
        assert manager.current_round == 1
        assert len(manager.get_pending_matches()) > 0

    def test_start_requires_players(self, round_robin_config: TournamentConfig) -> None:
        """Test start requires at least 2 players."""
        manager = TournamentManager(config=round_robin_config)
        manager.register_player(Player(name="Only"))

        with pytest.raises(RuntimeError):
            manager.start()

    def test_get_pending_matches(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test getting pending matches."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)
        manager.start()

        pending = manager.get_pending_matches()
        assert len(pending) > 0
        assert all(m.status == MatchStatus.SCHEDULED for m in pending)

    def test_start_match(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test starting a specific match."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)
        manager.start()

        pending = manager.get_pending_matches()
        match_id = pending[0].match_id

        match = manager.start_match(match_id)

        assert match is not None
        assert match.status == MatchStatus.IN_PROGRESS

    def test_record_match_result(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test recording match result."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)
        manager.start()

        pending = manager.get_pending_matches()
        match = pending[0]
        match_id = match.match_id
        manager.start_match(match_id)

        result = MatchResult(
            winner_id=match.player1_id,
            player1_score=1.0,
            player2_score=0.0,
        )

        success = manager.record_match_result(match_id, result)

        assert success is True
        assert manager.standings.scores[match.player1_id] == 1.0

    def test_callbacks_on_match_complete(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test callbacks fire on match completion."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players[:2]:
            manager.register_player(p)
        manager.start()

        callback_data = []

        def on_complete(match: Match) -> None:
            callback_data.append(match.match_id)

        manager.on_match_complete(on_complete)

        pending = manager.get_pending_matches()
        match = pending[0]
        manager.start_match(match.match_id)

        result = MatchResult(
            winner_id=match.player1_id,
            player1_score=1.0,
            player2_score=0.0,
        )
        manager.record_match_result(match.match_id, result)

        assert len(callback_data) == 1
        assert callback_data[0] == match.match_id

    def test_get_match(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test getting a specific match."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)
        manager.start()

        pending = manager.get_pending_matches()
        match_id = pending[0].match_id

        match = manager.get_match(match_id)
        assert match is not None
        assert match.match_id == match_id

        assert manager.get_match("nonexistent") is None

    def test_get_results(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test getting tournament results."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players[:2]:
            manager.register_player(p)
        manager.start()

        results = manager.get_results()

        assert results["tournament_name"] == "Test Round Robin"
        assert results["participants"] == 2
        assert "standings" in results

    def test_get_summary(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test getting tournament summary."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)
        manager.start()

        summary = manager.get_summary()

        assert summary["name"] == "Test Round Robin"
        assert summary["participants"] == 4
        assert summary["current_round"] == 1
        assert "total_matches" in summary

    def test_save_state(
        self, round_robin_config: TournamentConfig, sample_players: list[Player]
    ) -> None:
        """Test saving tournament state."""
        manager = TournamentManager(config=round_robin_config)
        for p in sample_players:
            manager.register_player(p)
        manager.start()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tournament.json"
            manager.save_state(path)

            assert path.exists()


class TestTournamentCompletion:
    """Tests for tournament completion logic."""

    def test_match_format_completes(self) -> None:
        """Test match format tournament completes after one match."""
        config = TournamentConfig(
            name="Single Match",
            format=TournamentFormat.MATCH,
        )
        manager = TournamentManager(config=config)

        p1 = Player(name="P1", player_id="p1")
        p2 = Player(name="P2", player_id="p2")
        manager.register_player(p1)
        manager.register_player(p2)
        manager.start()

        # Complete the single match
        pending = manager.get_pending_matches()
        match = pending[0]
        manager.start_match(match.match_id)

        result = MatchResult(winner_id="p1", player1_score=1.0, player2_score=0.0)
        manager.record_match_result(match.match_id, result)

        assert manager.state == TournamentState.COMPLETED
        assert manager.is_complete

    def test_swiss_rounds_complete(self) -> None:
        """Test Swiss tournament completes after configured rounds."""
        config = TournamentConfig(
            name="Swiss",
            format=TournamentFormat.SWISS,
            rounds=2,
        )
        manager = TournamentManager(config=config)

        players = [Player(name=f"P{i}", player_id=f"p{i}") for i in range(4)]
        for p in players:
            manager.register_player(p)
        manager.start()

        # Play all matches for each round
        for _ in range(2):  # 2 rounds
            pending = manager.get_pending_matches()
            for match in pending:
                manager.start_match(match.match_id)
                result = MatchResult(
                    winner_id=match.player1_id,
                    player1_score=1.0,
                    player2_score=0.0,
                )
                manager.record_match_result(match.match_id, result)

        assert manager.state == TournamentState.COMPLETED


class TestCreateTournament:
    """Tests for create_tournament factory."""

    def test_create_default(self) -> None:
        """Test creating default tournament."""
        manager = create_tournament(name="Test")

        assert manager.config.name == "Test"
        assert manager.config.format == TournamentFormat.ROUND_ROBIN

    def test_create_with_format(self) -> None:
        """Test creating with specific format."""
        manager = create_tournament(name="Swiss", format="swiss")

        assert manager.config.format == TournamentFormat.SWISS

    def test_create_with_board_size(self) -> None:
        """Test creating with board size."""
        manager = create_tournament(name="9x9", board_size=9)

        assert manager.config.match_config.board_size == 9

    def test_create_with_kwargs(self) -> None:
        """Test creating with additional kwargs."""
        manager = create_tournament(
            name="Custom",
            rounds=5,
            allow_draws=False,
        )

        assert manager.config.rounds == 5
        assert manager.config.allow_draws is False
