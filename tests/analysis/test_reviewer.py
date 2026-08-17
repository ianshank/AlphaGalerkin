"""Tests for game reviewer."""

from __future__ import annotations

import numpy as np

from src.analysis.config import AnalysisConfig, AnnotationLevel, MoveClassification
from src.analysis.go_adapter import EMPTY_MARK
from src.analysis.reviewer import (
    GameAnalysis,
    GameReviewer,
    MoveAnalysis,
    create_game_reviewer,
)


def _constant_model_evaluator(value: float):
    """Build a model evaluator returning a fixed value + uniform legal policy."""

    def _evaluate(board_state: list[list[int]]) -> tuple[float, np.ndarray]:
        board_size = len(board_state)
        n = board_size * board_size
        policy = np.full(n, 1.0 / n, dtype=np.float32)
        return value, policy

    return _evaluate


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


class TestGameReviewerModelWiring:
    """The reviewer must produce real model signal when an evaluator is wired."""

    def test_no_model_falls_back_to_dummy(
        self,
        game_reviewer: GameReviewer,
        sample_game_moves: list[tuple[str, int, int]],
    ) -> None:
        # Without a model the evaluator yields the uniform dummy (win_rate=0.5).
        assert game_reviewer._evaluator._model_evaluator is None
        analysis = game_reviewer.review_game(moves=sample_game_moves, board_size=19)
        win_rates = {
            m.evaluation_before.win_rate
            for m in analysis.move_analyses
            if m.evaluation_before is not None
        }
        assert win_rates == {0.5}

    def test_injected_model_evaluator_produces_real_signal(
        self,
        sample_game_moves: list[tuple[str, int, int]],
    ) -> None:
        reviewer = GameReviewer(
            config=AnalysisConfig(),
            model_evaluator=_constant_model_evaluator(value=0.6),
        )
        assert reviewer._evaluator._model_evaluator is not None

        analysis = reviewer.review_game(moves=sample_game_moves, board_size=19)
        # value=0.6 -> win_rate=(0.6+1)/2=0.8, distinct from the 0.5 dummy.
        win_rates = {
            m.evaluation_before.win_rate
            for m in analysis.move_analyses
            if m.evaluation_before is not None
        }
        assert win_rates == {0.8}

    def test_factory_accepts_model_evaluator(self) -> None:
        reviewer = create_game_reviewer(
            mode="standard",
            model_evaluator=_constant_model_evaluator(value=0.0),
        )
        assert reviewer._evaluator._model_evaluator is not None

    def test_capture_reflected_in_review(self) -> None:
        # White centre captured by surrounding black; the reconstructed board the
        # reviewer feeds the evaluator must show the empty (captured) point.
        captured_states: list[list[list[int]]] = []
        reviewer = GameReviewer(
            config=AnalysisConfig(),
            model_evaluator=lambda bs: (captured_states.append(bs), (0.0, np.full(9, 1 / 9)))[1],
        )
        moves = [
            ("W", 1, 1),
            ("B", 0, 1),
            ("B", 2, 1),
            ("B", 1, 0),
            ("B", 1, 2),
            ("W", 0, 0),  # triggers an evaluation on the post-capture board
        ]
        reviewer.review_game(moves=moves, board_size=3)
        # The last evaluated board (before W plays at 0,0) has the white centre gone.
        last_board = captured_states[-1]
        assert last_board[1][1] == EMPTY_MARK


class TestGameReviewerCheckpointWiring:
    """Checkpoint-backed wiring builds a model, with a graceful failure fallback."""

    def test_checkpoint_load_failure_falls_back(self, tmp_path) -> None:
        config = AnalysisConfig(model_checkpoint_path=str(tmp_path / "missing.pt"))
        reviewer = GameReviewer(config=config)
        # Load fails -> warning logged, dummy fallback retained.
        assert reviewer._evaluator._model_evaluator is None

    def test_checkpoint_path_wires_model(self, tmp_path, monkeypatch) -> None:
        import src.analysis.go_adapter as go_adapter

        def fake_builder(checkpoint_path, *, device="cpu", temperature=1.0):
            return _constant_model_evaluator(value=0.2)

        monkeypatch.setattr(go_adapter, "build_checkpoint_model_evaluator", fake_builder)

        config = AnalysisConfig(model_checkpoint_path=str(tmp_path / "model.pt"))
        reviewer = GameReviewer(config=config)
        assert reviewer._evaluator._model_evaluator is not None


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
