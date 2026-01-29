"""Tests for match and game record serialization.

Verifies that all tournament data structures can be serialized
and deserialized correctly (round-trip testing).
"""

from __future__ import annotations

from src.tournament.match import GameRecord, Match, MatchResult, MatchStatus


class TestMatchResultSerialization:
    """Tests for MatchResult serialization."""

    def test_match_result_to_dict(self) -> None:
        """Verify MatchResult converts to dict correctly."""
        result = MatchResult(
            winner_id="player1",
            loser_id="player2",
            is_draw=False,
            player1_score=1.0,
            player2_score=0.0,
            player1_rating_change=16.0,
            player2_rating_change=-16.0,
        )
        data = result.to_dict()

        assert data["winner_id"] == "player1"
        assert data["loser_id"] == "player2"
        assert data["is_draw"] is False
        assert data["player1_score"] == 1.0
        assert data["player2_score"] == 0.0
        assert data["player1_rating_change"] == 16.0
        assert data["player2_rating_change"] == -16.0

    def test_match_result_from_dict(self) -> None:
        """Verify MatchResult can be created from dict."""
        data = {
            "winner_id": "player1",
            "loser_id": "player2",
            "is_draw": False,
            "player1_score": 1.0,
            "player2_score": 0.0,
            "player1_rating_change": 16.0,
            "player2_rating_change": -16.0,
        }
        result = MatchResult.from_dict(data)

        assert result.winner_id == "player1"
        assert result.loser_id == "player2"
        assert result.is_draw is False
        assert result.player1_score == 1.0
        assert result.player2_score == 0.0
        assert result.player1_rating_change == 16.0
        assert result.player2_rating_change == -16.0

    def test_match_result_round_trip(self) -> None:
        """Verify MatchResult survives round-trip serialization."""
        original = MatchResult(
            winner_id="winner",
            loser_id="loser",
            is_draw=False,
            player1_score=2.5,
            player2_score=0.5,
            player1_rating_change=12.5,
            player2_rating_change=-12.5,
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = MatchResult.from_dict(data)

        # Verify all fields match
        assert restored.winner_id == original.winner_id
        assert restored.loser_id == original.loser_id
        assert restored.is_draw == original.is_draw
        assert restored.player1_score == original.player1_score
        assert restored.player2_score == original.player2_score
        assert restored.player1_rating_change == original.player1_rating_change
        assert restored.player2_rating_change == original.player2_rating_change

    def test_match_result_from_dict_with_defaults(self) -> None:
        """Verify MatchResult handles missing optional fields."""
        # Minimal dict with only required fields
        data = {"winner_id": "player1"}

        result = MatchResult.from_dict(data)

        assert result.winner_id == "player1"
        assert result.loser_id is None
        assert result.is_draw is False
        assert result.player1_score == 0.0
        assert result.player2_score == 0.0
        assert result.player1_rating_change == 0.0
        assert result.player2_rating_change == 0.0

    def test_match_result_draw(self) -> None:
        """Verify draw MatchResult serialization."""
        result = MatchResult(
            winner_id=None,
            loser_id=None,
            is_draw=True,
            player1_score=0.5,
            player2_score=0.5,
        )

        data = result.to_dict()
        restored = MatchResult.from_dict(data)

        assert restored.is_draw is True
        assert restored.winner_id is None
        assert restored.player1_score == 0.5
        assert restored.player2_score == 0.5


class TestGameRecordSerialization:
    """Tests for GameRecord serialization."""

    def test_game_record_to_dict(self) -> None:
        """Verify GameRecord converts to dict correctly."""
        record = GameRecord(
            game_number=1,
            black_player_id="player1",
            white_player_id="player2",
            result="B+2.5",
            winner_id="player1",
            is_draw=False,
            moves=250,
            sgf_path="/path/to/game.sgf",
            duration_seconds=1800.0,
        )
        data = record.to_dict()

        assert data["game_number"] == 1
        assert data["black_player_id"] == "player1"
        assert data["white_player_id"] == "player2"
        assert data["result"] == "B+2.5"
        assert data["winner_id"] == "player1"
        assert data["is_draw"] is False
        assert data["moves"] == 250
        assert data["sgf_path"] == "/path/to/game.sgf"
        assert data["duration_seconds"] == 1800.0

    def test_game_record_from_dict(self) -> None:
        """Verify GameRecord can be created from dict."""
        data = {
            "game_number": 2,
            "black_player_id": "player2",
            "white_player_id": "player1",
            "result": "W+R",
            "winner_id": "player1",
            "is_draw": False,
            "moves": 180,
            "sgf_path": None,
            "duration_seconds": 1200.0,
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        record = GameRecord.from_dict(data)

        assert record.game_number == 2
        assert record.black_player_id == "player2"
        assert record.white_player_id == "player1"
        assert record.result == "W+R"
        assert record.winner_id == "player1"
        assert record.is_draw is False
        assert record.moves == 180
        assert record.sgf_path is None
        assert record.duration_seconds == 1200.0

    def test_game_record_round_trip(self) -> None:
        """Verify GameRecord survives round-trip serialization."""
        original = GameRecord(
            game_number=3,
            black_player_id="alice",
            white_player_id="bob",
            result="Draw",
            winner_id=None,
            is_draw=True,
            moves=300,
        )

        data = original.to_dict()
        restored = GameRecord.from_dict(data)

        assert restored.game_number == original.game_number
        assert restored.black_player_id == original.black_player_id
        assert restored.white_player_id == original.white_player_id
        assert restored.result == original.result
        assert restored.is_draw == original.is_draw


class TestMatchSerialization:
    """Tests for Match serialization."""

    def test_match_to_dict(self) -> None:
        """Verify Match converts to dict correctly."""
        match = Match(
            player1_id="player1",
            player2_id="player2",
            round_number=1,
            board_size=19,
            games_to_play=3,
        )
        data = match.to_dict()

        assert data["player1_id"] == "player1"
        assert data["player2_id"] == "player2"
        assert data["round_number"] == 1
        assert data["board_size"] == 19
        assert data["games_to_play"] == 3
        assert data["status"] == MatchStatus.SCHEDULED.value

    def test_match_from_dict(self) -> None:
        """Verify Match can be created from dict."""
        data = {
            "match_id": "test-match-id",
            "player1_id": "alice",
            "player2_id": "bob",
            "round_number": 2,
            "board_size": 13,
            "games_to_play": 1,
            "status": "scheduled",
            "games": [],
            "result": None,
        }
        match = Match.from_dict(data)

        assert match.player1_id == "alice"
        assert match.player2_id == "bob"
        assert match.round_number == 2
        assert match.board_size == 13
        assert match.games_to_play == 1
        assert match.status == MatchStatus.SCHEDULED

    def test_match_round_trip(self) -> None:
        """Verify Match survives round-trip serialization."""
        original = Match(
            player1_id="player1",
            player2_id="player2",
            round_number=1,
            board_size=9,
            games_to_play=5,
        )

        data = original.to_dict()
        restored = Match.from_dict(data)

        assert restored.player1_id == original.player1_id
        assert restored.player2_id == original.player2_id
        assert restored.round_number == original.round_number
        assert restored.board_size == original.board_size
        assert restored.games_to_play == original.games_to_play
