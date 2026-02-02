"""Tests for game reviewer."""

from __future__ import annotations

from src.analysis.config import AnnotationLevel, MoveClassification
from src.analysis.reviewer import (
    GameAnalysis,
    GameReviewer,
    MoveAnalysis,
    create_game_reviewer,
)


class TestMoveAnalysis:
    """Tests for MoveAnalysis dataclass."""

    def test_is_pass(self) -> None:
        """Test is_pass property."""
        pass_move = MoveAnalysis(move_number=1, color="B", move=None)
        normal_move = MoveAnalysis(move_number=1, color="B", move=(3, 3))

        assert pass_move.is_pass
        assert not normal_move.is_pass

    def test_is_mistake(self) -> None:
        """Test is_mistake property."""
        mistake = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.MISTAKE,
        )
        blunder = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.BLUNDER,
        )
        good = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.GOOD,
        )

        assert mistake.is_mistake
        assert blunder.is_mistake
        assert not good.is_mistake

    def test_is_inaccuracy(self) -> None:
        """Test is_inaccuracy property."""
        inaccuracy = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.INACCURACY,
        )
        assert inaccuracy.is_inaccuracy

    def test_is_good(self) -> None:
        """Test is_good property."""
        excellent = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.EXCELLENT,
        )
        good = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.GOOD,
        )
        neutral = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.NEUTRAL,
        )

        assert excellent.is_good
        assert good.is_good
        assert not neutral.is_good

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        analysis = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.GOOD,
            win_rate_change=0.02,
        )

        data = analysis.to_dict()

        assert data["move_number"] == 1
        assert data["color"] == "B"
        assert data["move"] == [3, 3]
        assert data["classification"] == "good"


class TestGameAnalysis:
    """Tests for GameAnalysis dataclass."""

    def test_total_moves(self) -> None:
        """Test total_moves property."""
        analysis = GameAnalysis()
        assert analysis.total_moves == 0

        analysis.move_analyses.append(MoveAnalysis(move_number=1, color="B", move=(3, 3)))
        assert analysis.total_moves == 1

    def test_black_mistakes(self) -> None:
        """Test black_mistakes property."""
        analysis = GameAnalysis(
            move_analyses=[
                MoveAnalysis(
                    move_number=1,
                    color="B",
                    move=(3, 3),
                    classification=MoveClassification.MISTAKE,
                ),
                MoveAnalysis(
                    move_number=2,
                    color="W",
                    move=(15, 3),
                    classification=MoveClassification.MISTAKE,
                ),
                MoveAnalysis(
                    move_number=3,
                    color="B",
                    move=(3, 15),
                    classification=MoveClassification.GOOD,
                ),
            ]
        )

        mistakes = analysis.black_mistakes
        assert len(mistakes) == 1
        assert mistakes[0].move_number == 1

    def test_white_mistakes(self) -> None:
        """Test white_mistakes property."""
        analysis = GameAnalysis(
            move_analyses=[
                MoveAnalysis(
                    move_number=1,
                    color="B",
                    move=(3, 3),
                    classification=MoveClassification.GOOD,
                ),
                MoveAnalysis(
                    move_number=2,
                    color="W",
                    move=(15, 3),
                    classification=MoveClassification.BLUNDER,
                ),
            ]
        )

        mistakes = analysis.white_mistakes
        assert len(mistakes) == 1

    def test_get_move_at(self) -> None:
        """Test get_move_at method."""
        analysis = GameAnalysis(
            move_analyses=[
                MoveAnalysis(move_number=1, color="B", move=(3, 3)),
                MoveAnalysis(move_number=2, color="W", move=(15, 3)),
            ]
        )

        move = analysis.get_move_at(1)
        assert move is not None
        assert move.color == "B"

        assert analysis.get_move_at(99) is None

    def test_get_moves_by_classification(self) -> None:
        """Test get_moves_by_classification method."""
        analysis = GameAnalysis(
            move_analyses=[
                MoveAnalysis(
                    move_number=1,
                    color="B",
                    move=(3, 3),
                    classification=MoveClassification.EXCELLENT,
                ),
                MoveAnalysis(
                    move_number=2,
                    color="W",
                    move=(15, 3),
                    classification=MoveClassification.GOOD,
                ),
                MoveAnalysis(
                    move_number=3,
                    color="B",
                    move=(3, 15),
                    classification=MoveClassification.EXCELLENT,
                ),
            ]
        )

        excellent = analysis.get_moves_by_classification(MoveClassification.EXCELLENT)
        assert len(excellent) == 2

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        analysis = GameAnalysis(
            black_stats={"accuracy": 0.85},
            white_stats={"accuracy": 0.80},
            turning_points=[10, 25],
        )

        data = analysis.to_dict()

        assert "total_moves" in data
        assert "black_stats" in data
        assert "turning_points" in data


class TestGameReviewer:
    """Tests for GameReviewer."""

    def test_initialization(self, game_reviewer: GameReviewer) -> None:
        """Test reviewer initialization."""
        assert game_reviewer.config is not None
        assert game_reviewer._evaluator is not None

    def test_review_game(
        self,
        game_reviewer: GameReviewer,
        sample_game_moves: list[tuple[str, int, int]],
    ) -> None:
        """Test reviewing a game."""
        analysis = game_reviewer.review_game(
            moves=sample_game_moves,
            board_size=19,
        )

        assert isinstance(analysis, GameAnalysis)
        assert analysis.total_moves == len(sample_game_moves)
        assert "black_stats" in analysis.to_dict()
        assert "white_stats" in analysis.to_dict()

    def test_review_game_statistics(
        self,
        game_reviewer: GameReviewer,
        sample_game_moves: list[tuple[str, int, int]],
    ) -> None:
        """Test that review generates statistics."""
        analysis = game_reviewer.review_game(
            moves=sample_game_moves,
            board_size=19,
        )

        assert "total_moves" in analysis.black_stats
        assert "total_moves" in analysis.white_stats

    def test_review_empty_game(self, game_reviewer: GameReviewer) -> None:
        """Test reviewing empty game."""
        analysis = game_reviewer.review_game(moves=[], board_size=19)

        assert analysis.total_moves == 0
        assert len(analysis.turning_points) == 0


class TestCreateGameReviewer:
    """Tests for create_game_reviewer factory."""

    def test_create_default(self) -> None:
        """Test creating default reviewer."""
        reviewer = create_game_reviewer()
        assert reviewer.config is not None

    def test_create_with_mode(self) -> None:
        """Test creating with specific mode."""
        reviewer = create_game_reviewer(mode="quick")
        assert reviewer.config.mode.value == "quick"

    def test_create_with_options(self) -> None:
        """Test creating with additional options."""
        reviewer = create_game_reviewer(
            mode="deep",
            annotation_level=AnnotationLevel.DETAILED,
        )
        assert reviewer.config.annotation_level == AnnotationLevel.DETAILED
