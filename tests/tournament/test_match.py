"""Tests for match management."""

from __future__ import annotations


from src.tournament.match import (
    GameRecord,
    Match,
    MatchResult,
    MatchStatus,
)


class TestMatchStatus:
    """Tests for MatchStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Test all match statuses exist."""
        assert MatchStatus.SCHEDULED.value == "scheduled"
        assert MatchStatus.IN_PROGRESS.value == "in_progress"
        assert MatchStatus.COMPLETED.value == "completed"
        assert MatchStatus.CANCELLED.value == "cancelled"
        assert MatchStatus.FORFEIT.value == "forfeit"

    def test_is_terminal(self) -> None:
        """Test is_terminal method."""
        assert not MatchStatus.SCHEDULED.is_terminal()
        assert not MatchStatus.IN_PROGRESS.is_terminal()
        assert MatchStatus.COMPLETED.is_terminal()
        assert MatchStatus.CANCELLED.is_terminal()
        assert MatchStatus.FORFEIT.is_terminal()


class TestGameRecord:
    """Tests for GameRecord dataclass."""

    def test_initialization(self, sample_game_record: GameRecord) -> None:
        """Test game record initialization."""
        assert sample_game_record.game_number == 1
        assert sample_game_record.black_player_id == "p1"
        assert sample_game_record.white_player_id == "p2"
        assert sample_game_record.result == "B+R"
        assert sample_game_record.winner_id == "p1"
        assert sample_game_record.moves == 156

    def test_is_complete(self) -> None:
        """Test is_complete property."""
        complete = GameRecord(
            game_number=1,
            black_player_id="p1",
            white_player_id="p2",
            result="B+5",
        )
        incomplete = GameRecord(
            game_number=1,
            black_player_id="p1",
            white_player_id="p2",
        )

        assert complete.is_complete
        assert not incomplete.is_complete

    def test_black_won(self, sample_game_record: GameRecord) -> None:
        """Test black_won property."""
        assert sample_game_record.black_won
        assert not sample_game_record.white_won

    def test_white_won(self) -> None:
        """Test white_won property."""
        game = GameRecord(
            game_number=1,
            black_player_id="p1",
            white_player_id="p2",
            result="W+R",
            winner_id="p2",
        )

        assert game.white_won
        assert not game.black_won

    def test_to_dict(self, sample_game_record: GameRecord) -> None:
        """Test serialization to dict."""
        data = sample_game_record.to_dict()

        assert data["game_number"] == 1
        assert data["black_player_id"] == "p1"
        assert data["white_player_id"] == "p2"
        assert data["result"] == "B+R"
        assert data["moves"] == 156

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "game_number": 2,
            "black_player_id": "p2",
            "white_player_id": "p1",
            "result": "W+3.5",
            "winner_id": "p1",
            "moves": 200,
        }

        game = GameRecord.from_dict(data)

        assert game.game_number == 2
        assert game.winner_id == "p1"
        assert game.moves == 200


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_initialization(self, sample_match_result: MatchResult) -> None:
        """Test match result initialization."""
        assert sample_match_result.winner_id == "p1"
        assert sample_match_result.loser_id == "p2"
        assert sample_match_result.is_draw is False
        assert sample_match_result.player1_score == 2.0
        assert sample_match_result.player2_score == 1.0

    def test_draw_result(self) -> None:
        """Test draw result."""
        result = MatchResult(
            is_draw=True,
            player1_score=1.5,
            player2_score=1.5,
        )

        assert result.winner_id is None
        assert result.is_draw

    def test_to_dict(self, sample_match_result: MatchResult) -> None:
        """Test serialization to dict."""
        data = sample_match_result.to_dict()

        assert data["winner_id"] == "p1"
        assert data["is_draw"] is False
        assert data["player1_score"] == 2.0


class TestMatch:
    """Tests for Match dataclass."""

    def test_initialization(self, sample_match: Match) -> None:
        """Test match initialization."""
        assert sample_match.player1_id == "p1"
        assert sample_match.player2_id == "p2"
        assert sample_match.match_id == "match123"
        assert sample_match.board_size == 19
        assert sample_match.games_to_play == 3
        assert sample_match.status == MatchStatus.SCHEDULED

    def test_is_complete(self, sample_match: Match, completed_match: Match) -> None:
        """Test is_complete property."""
        assert not sample_match.is_complete
        assert completed_match.is_complete

    def test_games_played(self, sample_match: Match) -> None:
        """Test games_played property."""
        assert sample_match.games_played == 0

        sample_match.add_game("p1", "p2", "B+R", winner_id="p1")
        assert sample_match.games_played == 1

    def test_player_scores(self, sample_match: Match) -> None:
        """Test player score calculations."""
        assert sample_match.player1_score == 0.0
        assert sample_match.player2_score == 0.0

        sample_match.add_game("p1", "p2", "B+R", winner_id="p1")
        assert sample_match.player1_score == 1.0
        assert sample_match.player2_score == 0.0

        sample_match.add_game("p2", "p1", "B+R", winner_id="p2")
        assert sample_match.player1_score == 1.0
        assert sample_match.player2_score == 1.0

    def test_player_score_with_draw(self, sample_match: Match) -> None:
        """Test player scores with draws."""
        sample_match.add_game("p1", "p2", "Draw", is_draw=True)
        assert sample_match.player1_score == 0.5
        assert sample_match.player2_score == 0.5

    def test_current_leader(self, sample_match: Match) -> None:
        """Test current_leader property."""
        assert sample_match.current_leader is None

        sample_match.add_game("p1", "p2", "B+R", winner_id="p1")
        assert sample_match.current_leader == "p1"

        sample_match.add_game("p2", "p1", "B+R", winner_id="p2")
        assert sample_match.current_leader is None  # Tied

    def test_start_match(self, sample_match: Match) -> None:
        """Test starting a match."""
        sample_match.start()

        assert sample_match.status == MatchStatus.IN_PROGRESS
        assert sample_match.start_time is not None

    def test_add_game(self, sample_match: Match) -> None:
        """Test adding a game."""
        game = sample_match.add_game(
            black_player_id="p1",
            white_player_id="p2",
            result="B+2.5",
            winner_id="p1",
            moves=150,
            sgf_path="/path/to/game.sgf",
        )

        assert game.game_number == 1
        assert game.winner_id == "p1"
        assert len(sample_match.games) == 1

    def test_add_game_completes_match(self, sample_match: Match) -> None:
        """Test that adding enough games completes the match."""
        sample_match.games_to_play = 1
        sample_match.add_game("p1", "p2", "B+R", winner_id="p1")

        assert sample_match.status == MatchStatus.COMPLETED
        assert sample_match.result is not None
        assert sample_match.result.winner_id == "p1"

    def test_match_result_draw(self) -> None:
        """Test match resulting in draw."""
        match = Match(
            player1_id="p1",
            player2_id="p2",
            games_to_play=2,
        )

        match.add_game("p1", "p2", "B+R", winner_id="p1")
        match.add_game("p2", "p1", "B+R", winner_id="p2")

        assert match.status == MatchStatus.COMPLETED
        assert match.result.is_draw

    def test_cancel_match(self, sample_match: Match) -> None:
        """Test cancelling a match."""
        sample_match.cancel(reason="Weather")

        assert sample_match.status == MatchStatus.CANCELLED
        assert sample_match.metadata["cancel_reason"] == "Weather"
        assert sample_match.end_time is not None

    def test_forfeit(self, sample_match: Match) -> None:
        """Test forfeit."""
        sample_match.forfeit("p1")

        assert sample_match.status == MatchStatus.FORFEIT
        assert sample_match.result.winner_id == "p2"
        assert sample_match.result.loser_id == "p1"

    def test_get_next_colors(self, sample_match: Match) -> None:
        """Test getting next game colors."""
        black, white = sample_match.get_next_colors()
        assert black == "p1"
        assert white == "p2"

        sample_match.add_game("p1", "p2", "B+R", winner_id="p1")
        black, white = sample_match.get_next_colors()
        assert black == "p2"
        assert white == "p1"

    def test_to_dict(self, sample_match: Match) -> None:
        """Test serialization to dict."""
        sample_match.add_game("p1", "p2", "B+R", winner_id="p1")
        data = sample_match.to_dict()

        assert data["match_id"] == "match123"
        assert data["player1_id"] == "p1"
        assert data["player2_id"] == "p2"
        assert data["status"] == "scheduled"
        assert len(data["games"]) == 1

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "match_id": "loaded123",
            "player1_id": "p1",
            "player2_id": "p2",
            "round_number": 2,
            "board_size": 9,
            "games_to_play": 1,
            "status": "completed",
            "games": [
                {
                    "game_number": 1,
                    "black_player_id": "p1",
                    "white_player_id": "p2",
                    "result": "B+R",
                    "winner_id": "p1",
                }
            ],
            "result": {
                "winner_id": "p1",
                "loser_id": "p2",
                "is_draw": False,
                "player1_score": 1.0,
                "player2_score": 0.0,
            },
        }

        match = Match.from_dict(data)

        assert match.match_id == "loaded123"
        assert match.round_number == 2
        assert match.board_size == 9
        assert match.status == MatchStatus.COMPLETED
        assert len(match.games) == 1
        assert match.result.winner_id == "p1"
