"""Additional coverage tests for analysis/evaluator.py.

Covers: _process_model_output, evaluate with model, cache, batch evaluation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analysis.config import AnalysisConfig
from src.analysis.evaluator import EvaluationResult, LRUCache, PositionEvaluator


class TestLRUCacheEdge:
    """Edge cases for LRU cache."""

    def test_eviction_on_overflow(self) -> None:
        cache = LRUCache(max_size=2)
        r1 = EvaluationResult(win_rate=0.5)
        r2 = EvaluationResult(win_rate=0.6)
        r3 = EvaluationResult(win_rate=0.7)
        cache.put("a", r1)
        cache.put("b", r2)
        cache.put("c", r3)
        assert cache.get("a") is None  # evicted
        assert cache.get("c") is not None

    def test_update_existing_key(self) -> None:
        cache = LRUCache(max_size=2)
        r1 = EvaluationResult(win_rate=0.5)
        r2 = EvaluationResult(win_rate=0.9)
        cache.put("a", r1)
        cache.put("a", r2)
        assert len(cache) == 1
        assert cache.get("a") is not None
        assert cache.get("a").win_rate == 0.9  # type: ignore[union-attr]


class TestPositionEvaluatorWithModel:
    """Tests for evaluate with a model evaluator."""

    def test_evaluate_with_numpy_policy(self) -> None:
        def model_fn(state: object) -> tuple[float, np.ndarray]:
            policy = np.ones(82) / 82  # 9x9 + 1 pass
            return 0.5, policy

        ev = PositionEvaluator(model_evaluator=model_fn)
        result = ev.evaluate(board_state="state", board_size=9)
        assert result.win_rate == pytest.approx(0.75)  # (0.5+1)/2
        assert len(result.best_moves) > 0

    def test_evaluate_with_torch_policy(self) -> None:
        def model_fn(state: object) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor(0.0), torch.ones(82) / 82

        ev = PositionEvaluator(model_evaluator=model_fn)
        result = ev.evaluate(board_state="state", board_size=9)
        assert 0.0 <= result.win_rate <= 1.0
        assert result.depth == 1

    def test_evaluate_with_legal_moves(self) -> None:
        def model_fn(state: object) -> tuple[float, np.ndarray]:
            policy = np.zeros(82)
            policy[0] = 0.9  # (0, 0)
            policy[1] = 0.1  # (1, 0)
            return 0.3, policy

        legal = [(0, 0), (1, 0)]
        ev = PositionEvaluator(model_evaluator=model_fn)
        result = ev.evaluate(board_state="s", board_size=9, legal_moves=legal)
        assert (0, 0) in result.policy

    def test_evaluate_model_failure_fallback(self) -> None:
        def model_fn(state: object) -> tuple[float, np.ndarray]:
            raise RuntimeError("model error")

        ev = PositionEvaluator(model_evaluator=model_fn)
        result = ev.evaluate(board_state="s", board_size=9)
        assert result.win_rate == 0.5
        assert result.confidence == 0.0

    def test_evaluate_cache_hit(self) -> None:
        call_count = 0

        def model_fn(state: object) -> tuple[float, np.ndarray]:
            nonlocal call_count
            call_count += 1
            return 0.5, np.ones(82) / 82

        config = AnalysisConfig(cache_evaluations=True, max_cache_size=100)
        ev = PositionEvaluator(config=config, model_evaluator=model_fn)
        # Two evaluations of same state should only call model once
        ev.evaluate(board_state=np.array([1, 2, 3]), board_size=9)
        ev.evaluate(board_state=np.array([1, 2, 3]), board_size=9)
        assert call_count == 1

    def test_evaluate_batch(self) -> None:
        def model_fn(state: object) -> tuple[float, np.ndarray]:
            return 0.0, np.ones(82) / 82

        ev = PositionEvaluator(model_evaluator=model_fn)
        results = ev.evaluate_batch(["s1", "s2"], board_sizes=[9, 9])
        assert len(results) == 2

    def test_evaluate_batch_default_sizes(self) -> None:
        def model_fn(state: object) -> tuple[float, np.ndarray]:
            return 0.0, np.ones(362) / 362

        ev = PositionEvaluator(model_evaluator=model_fn)
        results = ev.evaluate_batch(["s1"])
        assert len(results) == 1

    def test_cache_key_list(self) -> None:
        ev = PositionEvaluator()
        key = ev._compute_cache_key([[1, 2], [3, 4]])
        assert isinstance(key, str)

    def test_cache_key_unhashable(self) -> None:
        ev = PositionEvaluator()
        key = ev._compute_cache_key({"a": 1})
        assert isinstance(key, str)


class TestEvaluationResultMethods:
    """Tests for EvaluationResult methods."""

    def test_best_move_probability_empty(self) -> None:
        r = EvaluationResult(win_rate=0.5)
        assert r.best_move_probability == 0.0

    def test_get_move_rank_not_found(self) -> None:
        r = EvaluationResult(
            win_rate=0.5,
            best_moves=[((0, 0), 0.5), ((1, 1), 0.3)],
        )
        assert r.get_move_rank((9, 9)) is None
        assert r.get_move_rank((0, 0)) == 0
        assert r.get_move_rank((1, 1)) == 1
