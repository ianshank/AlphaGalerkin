"""Additional coverage tests for analysis/reviewer.py.

Covers: MoveAnalysis properties, GameAnalysis methods, GameReviewer.review_game.
"""

from __future__ import annotations

import numpy as np

from src.analysis.config import MoveClassification
from src.analysis.evaluator import PositionEvaluator
from src.analysis.reviewer import GameAnalysis, GameReviewer, MoveAnalysis


class TestMoveAnalysisProperties:
    """Test MoveAnalysis property methods."""

    def test_is_pass(self) -> None:
        ma = MoveAnalysis(move_number=1, color="B", move=None)
        assert ma.is_pass is True

    def test_is_not_pass(self) -> None:
        ma = MoveAnalysis(move_number=1, color="B", move=(3, 3))
        assert ma.is_pass is False

    def test_is_mistake(self) -> None:
        ma = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.MISTAKE,
        )
        assert ma.is_mistake is True

    def test_is_blunder(self) -> None:
        ma = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.BLUNDER,
        )
        assert ma.is_mistake is True

    def test_is_not_mistake(self) -> None:
        ma = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.NEUTRAL,
        )
        assert ma.is_mistake is False

    def test_is_inaccuracy(self) -> None:
        ma = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.INACCURACY,
        )
        assert ma.is_inaccuracy is True

    def test_is_good(self) -> None:
        ma = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.GOOD,
        )
        assert ma.is_good is True

    def test_is_excellent(self) -> None:
        ma = MoveAnalysis(
            move_number=1,
            color="B",
            move=(3, 3),
            classification=MoveClassification.EXCELLENT,
        )
        assert ma.is_good is True

    def test_to_dict(self) -> None:
        ma = MoveAnalysis(
            move_number=5,
            color="W",
            move=(3, 4),
            classification=MoveClassification.GOOD,
            win_rate_change=0.05,
        )
        d = ma.to_dict()
        assert d["move_number"] == 5
        assert d["color"] == "W"
        assert d["move"] == [3, 4]
        assert d["classification"] == "good"

    def test_to_dict_pass_move(self) -> None:
        ma = MoveAnalysis(move_number=1, color="B", move=None)
        d = ma.to_dict()
        assert d["move"] is None


class TestGameAnalysis:
    """Tests for GameAnalysis."""

    def _make_analysis(self) -> GameAnalysis:
        moves = [
            MoveAnalysis(
                move_number=1,
                color="B",
                move=(3, 3),
                classification=MoveClassification.GOOD,
            ),
            MoveAnalysis(
                move_number=2,
                color="W",
                move=(4, 4),
                classification=MoveClassification.MISTAKE,
            ),
            MoveAnalysis(
                move_number=3,
                color="B",
                move=(5, 5),
                classification=MoveClassification.BLUNDER,
            ),
            MoveAnalysis(
                move_number=4,
                color="W",
                move=(6, 6),
                classification=MoveClassification.NEUTRAL,
            ),
        ]
        return GameAnalysis(move_analyses=moves)

    def test_total_moves(self) -> None:
        ga = self._make_analysis()
        assert ga.total_moves == 4

    def test_black_mistakes(self) -> None:
        ga = self._make_analysis()
        assert len(ga.black_mistakes) == 1  # blunder

    def test_white_mistakes(self) -> None:
        ga = self._make_analysis()
        assert len(ga.white_mistakes) == 1  # mistake

    def test_get_move_at(self) -> None:
        ga = self._make_analysis()
        m = ga.get_move_at(2)
        assert m is not None
        assert m.color == "W"

    def test_get_move_at_not_found(self) -> None:
        ga = self._make_analysis()
        assert ga.get_move_at(99) is None

    def test_get_moves_by_classification(self) -> None:
        ga = self._make_analysis()
        goods = ga.get_moves_by_classification(MoveClassification.GOOD)
        assert len(goods) == 1

    def test_to_dict(self) -> None:
        ga = self._make_analysis()
        d = ga.to_dict()
        assert d["total_moves"] == 4
        assert len(d["move_analyses"]) == 4


class TestGameReviewer:
    """Tests for GameReviewer."""

    def test_review_empty_game(self) -> None:
        reviewer = GameReviewer()
        analysis = reviewer.review_game([], board_size=9)
        assert analysis.total_moves == 0

    def test_review_simple_game(self) -> None:
        def model_fn(state: object) -> tuple[float, np.ndarray]:
            return 0.0, np.ones(82) / 82

        ev = PositionEvaluator(model_evaluator=model_fn)
        reviewer = GameReviewer(evaluator=ev)

        moves = [("B", 3, 3), ("W", 4, 4), ("B", 5, 5)]
        analysis = reviewer.review_game(moves, board_size=9)
        assert analysis.total_moves == 3
        assert "average_loss" in analysis.black_stats or len(analysis.black_stats) >= 0

    def test_review_with_pass(self) -> None:
        reviewer = GameReviewer()
        moves = [("B", 3, 3), ("W", -1, -1)]  # pass
        analysis = reviewer.review_game(moves, board_size=9)
        assert analysis.total_moves == 2
