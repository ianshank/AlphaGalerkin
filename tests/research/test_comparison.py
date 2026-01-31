"""Tests for model comparison."""

from __future__ import annotations

import pytest

from src.research.comparison import (
    ComparisonResult,
    ModelComparison,
    ModelMetrics,
    create_comparison,
)


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_initialization(self, model_metrics: ModelMetrics) -> None:
        """Test metrics initialization."""
        assert model_metrics.model_name == "test_model"
        assert len(model_metrics.metrics_by_size) == 2

    def test_add_metrics(self) -> None:
        """Test adding metrics."""
        metrics = ModelMetrics(model_name="test")
        metrics.add_metrics(9, {"mse": 0.01})
        metrics.add_metrics(19, {"mse": 0.02})

        assert len(metrics.metrics_by_size) == 2
        assert metrics.metrics_by_size[9]["mse"] == 0.01

    def test_compute_aggregates(self, model_metrics: ModelMetrics) -> None:
        """Test computing aggregates."""
        assert "mse_mean" in model_metrics.aggregate_metrics
        assert "mse_min" in model_metrics.aggregate_metrics
        assert "mse_max" in model_metrics.aggregate_metrics

        # Mean of 0.01 and 0.02
        assert model_metrics.aggregate_metrics["mse_mean"] == pytest.approx(
            0.015, rel=0.01
        )

    def test_to_dict(self, model_metrics: ModelMetrics) -> None:
        """Test serialization to dict."""
        data = model_metrics.to_dict()

        assert data["model_name"] == "test_model"
        assert 9 in data["metrics_by_size"]


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_initialization(self, comparison_result: ComparisonResult) -> None:
        """Test result initialization."""
        assert comparison_result.comparison_id == "compare123"
        assert len(comparison_result.model_metrics) == 2

    def test_n_models(self, comparison_result: ComparisonResult) -> None:
        """Test n_models property."""
        assert comparison_result.n_models == 2

    def test_model_names(self, comparison_result: ComparisonResult) -> None:
        """Test model_names property."""
        names = comparison_result.model_names
        assert "model_a" in names
        assert "model_b" in names

    def test_get_ranking(self, comparison_result: ComparisonResult) -> None:
        """Test getting ranking."""
        ranking = comparison_result.get_ranking("mse", minimize=True)
        assert ranking[0] == "model_a"  # Lower MSE is better

    def test_get_best_model(self, comparison_result: ComparisonResult) -> None:
        """Test getting best model."""
        best = comparison_result.get_best_model("mse", minimize=True)
        assert best == "model_a"

    def test_to_dict(self, comparison_result: ComparisonResult) -> None:
        """Test serialization to dict."""
        data = comparison_result.to_dict()

        assert data["comparison_id"] == "compare123"
        assert "model_a" in data["model_metrics"]

    def test_summary(self, comparison_result: ComparisonResult) -> None:
        """Test summary generation."""
        summary = comparison_result.summary()

        assert "compare123" in summary
        assert "model_a" in summary


class TestModelComparison:
    """Tests for ModelComparison."""

    def test_initialization(
        self, model_comparison: ModelComparison
    ) -> None:
        """Test comparison initialization."""
        assert model_comparison.config.n_bootstrap == 1000
        assert len(model_comparison.results) == 0

    def test_compare(
        self, model_comparison: ModelComparison
    ) -> None:
        """Test compare method."""

        class MockModel:
            def __init__(self, name: str, error: float) -> None:
                self.name = name
                self.error = error

        models = {
            "model_a": MockModel("a", 0.01),
            "model_b": MockModel("b", 0.02),
        }

        def data_generator(
            size: int, n_samples: int
        ) -> tuple[list[list[float]], list[list[float]]]:
            return [[0.0] * size], [[0.0] * size]

        def evaluate_fn(
            model: MockModel,
            inputs: list[list[float]],
            targets: list[list[float]],
        ) -> dict[str, float]:
            return {"mse": model.error, "mae": model.error * 2}

        result = model_comparison.compare(
            models=models,
            data_generator=data_generator,
            evaluate_fn=evaluate_fn,
        )

        assert result.n_models == 2
        assert "model_a" in result.model_metrics
        assert len(model_comparison.results) == 1

    def test_get_summary_table(
        self, model_comparison: ModelComparison,
        comparison_result: ComparisonResult,
    ) -> None:
        """Test getting summary table."""
        model_comparison._results.append(comparison_result)

        table = model_comparison.get_summary_table(
            metrics=["mse", "mae"]
        )

        assert len(table) == 2
        assert any(r["model"] == "model_a" for r in table)


class TestCreateComparison:
    """Tests for create_comparison factory."""

    def test_create_default(self) -> None:
        """Test creating default comparison."""
        comparison = create_comparison()
        assert 19 in comparison.config.eval_sizes

    def test_create_with_custom_config(self) -> None:
        """Test creating with custom config."""
        comparison = create_comparison(
            eval_sizes=[5, 9],
            n_bootstrap=2000,
        )
        assert comparison.config.eval_sizes == [5, 9]
        assert comparison.config.n_bootstrap == 2000
