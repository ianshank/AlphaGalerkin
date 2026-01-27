"""Tests for quick evaluator."""

from __future__ import annotations

from typing import Any

import pytest

from src.prototyping.config import QuickEvalConfig
from src.prototyping.builder import PrototypeModel
from src.prototyping.evaluator import (
    QuickEvaluator,
    EvalResult,
    MetricResult,
    create_quick_evaluator,
)


class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_initialization(self) -> None:
        """Test metric result initialization."""
        metric = MetricResult(
            name="mse",
            value=0.01,
            ci_lower=0.005,
            ci_upper=0.015,
            std=0.002,
        )

        assert metric.name == "mse"
        assert metric.value == 0.01
        assert metric.ci_lower == 0.005

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        metric = MetricResult(name="mse", value=0.01)
        data = metric.to_dict()

        assert data["name"] == "mse"
        assert data["value"] == 0.01


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_initialization(self, eval_result: EvalResult) -> None:
        """Test result initialization."""
        assert eval_result.result_id == "eval123"
        assert eval_result.model_id == "model123"
        assert eval_result.n_samples == 100

    def test_get_metric(self, eval_result: EvalResult) -> None:
        """Test getting metric value."""
        assert eval_result.get_metric("mse") == 0.01
        assert eval_result.get_metric("mae") == 0.05
        assert eval_result.get_metric("unknown") is None

    def test_to_dict(self, eval_result: EvalResult) -> None:
        """Test serialization to dict."""
        data = eval_result.to_dict()

        assert data["result_id"] == "eval123"
        assert "mse" in data["metrics"]

    def test_summary(self, eval_result: EvalResult) -> None:
        """Test summary generation."""
        summary = eval_result.summary()

        assert "eval123" in summary
        assert "mse" in summary


class TestQuickEvaluator:
    """Tests for QuickEvaluator."""

    def test_initialization(self, quick_evaluator: QuickEvaluator) -> None:
        """Test evaluator initialization."""
        assert quick_evaluator.config is not None
        assert len(quick_evaluator.results) == 0

    def test_evaluate_basic(self, quick_evaluator: QuickEvaluator) -> None:
        """Test basic evaluation."""
        class MockModel:
            pass

        model = MockModel()

        def predict_fn(m: Any, inp: Any) -> float:
            return inp[0] + 0.1  # Add small error

        data = [([1.0], [1.0]), ([2.0], [2.0]), ([3.0], [3.0])]

        result = quick_evaluator.evaluate(
            model=model,
            predict_fn=predict_fn,
            data=data,
        )

        assert result is not None
        assert result.n_samples == 3
        assert "mse" in result.metrics
        assert len(quick_evaluator.results) == 1

    def test_evaluate_with_prototype_model(
        self,
        quick_evaluator: QuickEvaluator,
        prototype_model: PrototypeModel,
    ) -> None:
        """Test evaluation with PrototypeModel."""
        def predict_fn(m: Any, inp: Any) -> float:
            return 0.0

        data = [([1.0], [1.0])]

        result = quick_evaluator.evaluate(
            model=prototype_model,
            predict_fn=predict_fn,
            data=data,
        )

        assert result.model_id == prototype_model.model_id

    def test_register_metric(self, quick_evaluator: QuickEvaluator) -> None:
        """Test registering custom metric."""
        def custom_metric(preds: list[float], targets: list[float]) -> float:
            return sum(abs(p - t) ** 3 for p, t in zip(preds, targets)) / len(preds)

        quick_evaluator.register_metric("custom", custom_metric)

        def predict_fn(m: Any, inp: Any) -> float:
            return inp[0] + 0.1

        data = [([1.0], [1.0])]

        result = quick_evaluator.evaluate(
            model=None,
            predict_fn=predict_fn,
            data=data,
            metrics=["custom"],
        )

        assert "custom" in result.metrics

    def test_compare(self, quick_evaluator: QuickEvaluator) -> None:
        """Test comparing results."""
        # Create results
        result1 = EvalResult(
            result_id="r1",
            model_id="model1",
            n_samples=10,
            metrics={"mse": MetricResult(name="mse", value=0.1)},
        )
        result2 = EvalResult(
            result_id="r2",
            model_id="model2",
            n_samples=10,
            metrics={"mse": MetricResult(name="mse", value=0.05)},
        )

        comparison = quick_evaluator.compare([result1, result2], "mse")

        assert comparison["metric"] == "mse"
        assert comparison["n_models"] == 2
        assert comparison["best"]["model_id"] == "model2"  # Lower MSE is better

    def test_clear(self, quick_evaluator: QuickEvaluator) -> None:
        """Test clearing results."""
        def predict_fn(m: Any, inp: Any) -> float:
            return 0.0

        quick_evaluator.evaluate(
            model=None,
            predict_fn=predict_fn,
            data=[([1.0], [1.0])],
        )

        assert len(quick_evaluator.results) == 1
        quick_evaluator.clear()
        assert len(quick_evaluator.results) == 0

    def test_built_in_metrics(self) -> None:
        """Test built-in metric functions."""
        evaluator = QuickEvaluator()

        # Test MSE
        mse = evaluator._compute_mse([1, 2, 3], [1, 2, 3])
        assert mse == 0.0

        mse = evaluator._compute_mse([0, 0, 0], [1, 1, 1])
        assert mse == 1.0

        # Test MAE
        mae = evaluator._compute_mae([0, 0], [1, 1])
        assert mae == 1.0

        # Test RMSE
        rmse = evaluator._compute_rmse([0], [1])
        assert rmse == 1.0

        # Test accuracy
        acc = evaluator._compute_accuracy([0, 1, 1], [0, 1, 0])
        assert acc == pytest.approx(2/3)

        # Test R2
        r2 = evaluator._compute_r2([1, 2, 3], [1, 2, 3])
        assert r2 == 1.0


class TestCreateQuickEvaluator:
    """Tests for create_quick_evaluator factory."""

    def test_create_default(self) -> None:
        """Test creating default evaluator."""
        evaluator = create_quick_evaluator()
        assert evaluator.config.n_samples == 1000

    def test_create_with_metrics(self) -> None:
        """Test creating with custom metrics."""
        evaluator = create_quick_evaluator(
            metrics=["mse", "rmse"],
        )
        assert "mse" in evaluator.config.metrics
        assert "rmse" in evaluator.config.metrics

    def test_create_without_confidence(self) -> None:
        """Test creating without confidence intervals."""
        evaluator = create_quick_evaluator(
            compute_confidence=False,
        )
        assert not evaluator.config.compute_confidence
