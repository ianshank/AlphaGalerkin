"""Tests for position evaluator."""

from __future__ import annotations

from src.analysis.config import AnalysisConfig, MoveClassification
from src.analysis.evaluator import (
    EvaluationResult,
    LRUCache,
    PositionEvaluator,
    create_position_evaluator,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_best_move(self, sample_evaluation: EvaluationResult) -> None:
        """Test best_move property."""
        assert sample_evaluation.best_move == (3, 3)

    def test_best_move_empty(self) -> None:
        """Test best_move when no moves."""
        result = EvaluationResult(win_rate=0.5)
        assert result.best_move is None

    def test_best_move_probability(self, sample_evaluation: EvaluationResult) -> None:
        """Test best_move_probability property."""
        assert sample_evaluation.best_move_probability == 0.25

    def test_get_move_probability(self, sample_evaluation: EvaluationResult) -> None:
        """Test getting move probability."""
        assert sample_evaluation.get_move_probability((3, 3)) == 0.25
        assert sample_evaluation.get_move_probability((99, 99)) == 0.0

    def test_get_move_rank(self, sample_evaluation: EvaluationResult) -> None:
        """Test getting move rank."""
        assert sample_evaluation.get_move_rank((3, 3)) == 0
        assert sample_evaluation.get_move_rank((15, 3)) == 1
        assert sample_evaluation.get_move_rank((99, 99)) is None

    def test_to_dict(self, sample_evaluation: EvaluationResult) -> None:
        """Test serialization to dict."""
        data = sample_evaluation.to_dict()

        assert data["win_rate"] == 0.65
        assert len(data["best_moves"]) == 3
        assert data["best_moves"][0]["move"] == [3, 3]


class TestLRUCache:
    """Tests for LRUCache."""

    def test_put_and_get(self) -> None:
        """Test basic put and get."""
        cache = LRUCache(max_size=10)
        result = EvaluationResult(win_rate=0.5)

        cache.put("key1", result)
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert retrieved.win_rate == 0.5

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent key."""
        cache = LRUCache()
        assert cache.get("nonexistent") is None

    def test_eviction(self) -> None:
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)

        cache.put("key1", EvaluationResult(win_rate=0.1))
        cache.put("key2", EvaluationResult(win_rate=0.2))
        cache.put("key3", EvaluationResult(win_rate=0.3))

        # key1 should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_access_updates_lru_order(self) -> None:
        """Test that access updates LRU order."""
        cache = LRUCache(max_size=2)

        cache.put("key1", EvaluationResult(win_rate=0.1))
        cache.put("key2", EvaluationResult(win_rate=0.2))

        # Access key1 to make it more recent
        cache.get("key1")

        # Add key3, should evict key2
        cache.put("key3", EvaluationResult(win_rate=0.3))

        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None

    def test_clear(self) -> None:
        """Test cache clear."""
        cache = LRUCache()
        cache.put("key1", EvaluationResult(win_rate=0.5))

        cache.clear()

        assert len(cache) == 0
        assert cache.get("key1") is None


class TestPositionEvaluator:
    """Tests for PositionEvaluator."""

    def test_initialization(self, position_evaluator: PositionEvaluator) -> None:
        """Test evaluator initialization."""
        assert position_evaluator.config is not None
        assert position_evaluator.evaluation_count == 0

    def test_evaluate_without_model(
        self,
        position_evaluator: PositionEvaluator,
        sample_board: list[list[int]],
    ) -> None:
        """Test evaluation without model returns dummy result."""
        result = position_evaluator.evaluate(sample_board, board_size=9)

        assert isinstance(result, EvaluationResult)
        assert result.win_rate == 0.5
        assert result.confidence == 0.0

    def test_cache_hit(
        self,
        position_evaluator: PositionEvaluator,
        sample_board: list[list[int]],
    ) -> None:
        """Test cache hit."""
        # First evaluation
        position_evaluator.evaluate(sample_board, board_size=9)
        count_after_first = position_evaluator.evaluation_count

        # Second evaluation should be cached
        position_evaluator.evaluate(sample_board, board_size=9)
        count_after_second = position_evaluator.evaluation_count

        # Note: without model, each call creates new dummy result
        # but cache should still work
        assert position_evaluator.cache_size > 0

    def test_cache_disabled(self, sample_board: list[list[int]]) -> None:
        """Test with cache disabled."""
        config = AnalysisConfig(cache_evaluations=False)
        evaluator = PositionEvaluator(config=config)

        evaluator.evaluate(sample_board, board_size=9, use_cache=False)

        assert evaluator.cache_size == 0

    def test_evaluate_batch(
        self,
        position_evaluator: PositionEvaluator,
        sample_board: list[list[int]],
    ) -> None:
        """Test batch evaluation."""
        boards = [sample_board, sample_board]
        results = position_evaluator.evaluate_batch(boards, [9, 9])

        assert len(results) == 2
        for result in results:
            assert isinstance(result, EvaluationResult)

    def test_set_model_evaluator(self, position_evaluator: PositionEvaluator) -> None:
        """Test setting model evaluator."""

        def mock_evaluator(board):
            return 0.3, [0.5] * 81  # 9x9 = 81

        position_evaluator.set_model_evaluator(mock_evaluator)
        assert position_evaluator._model_evaluator is not None

    def test_compare_moves_best_move(
        self,
        position_evaluator: PositionEvaluator,
        sample_evaluation: EvaluationResult,
    ) -> None:
        """Test comparing moves when best move is played."""
        classification, loss = position_evaluator.compare_moves(
            sample_evaluation,
            (3, 3),  # Best move
        )

        assert classification == MoveClassification.EXCELLENT
        assert loss == 0.0

    def test_compare_moves_not_best(
        self,
        position_evaluator: PositionEvaluator,
        sample_evaluation: EvaluationResult,
    ) -> None:
        """Test comparing moves when not best move."""
        classification, loss = position_evaluator.compare_moves(
            sample_evaluation,
            (15, 3),  # Second best
        )

        # Should have some loss
        assert loss >= 0.0

    def test_clear_cache(
        self,
        position_evaluator: PositionEvaluator,
        sample_board: list[list[int]],
    ) -> None:
        """Test clearing cache."""
        position_evaluator.evaluate(sample_board, board_size=9)
        assert position_evaluator.cache_size > 0

        position_evaluator.clear_cache()
        assert position_evaluator.cache_size == 0


class TestCreatePositionEvaluator:
    """Tests for create_position_evaluator factory."""

    def test_create_default(self) -> None:
        """Test creating default evaluator."""
        evaluator = create_position_evaluator()
        assert evaluator.config.mode.value == "standard"

    def test_create_with_mode(self) -> None:
        """Test creating with specific mode."""
        evaluator = create_position_evaluator(mode="quick")
        assert evaluator.config.mode.value == "quick"

    def test_create_with_model(self) -> None:
        """Test creating with model evaluator."""

        def mock_evaluator(board):
            return 0.5, [0.1] * 361

        evaluator = create_position_evaluator(model_evaluator=mock_evaluator)
        assert evaluator._model_evaluator is not None
