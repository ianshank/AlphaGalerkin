"""Model comparison utilities.

Provides:
- Multi-model comparison
- Statistical significance testing
- Comparative metrics computation
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.research.config import ComparisonConfig


@dataclass
class ModelMetrics:
    """Metrics for a single model.

    Contains evaluation results at multiple sizes.
    """

    model_name: str
    model_path: str | None = None
    metrics_by_size: dict[int, dict[str, float]] = field(default_factory=dict)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metrics(self, size: int, metrics: dict[str, float]) -> None:
        """Add metrics for a size.

        Args:
            size: Evaluation size.
            metrics: Metrics dictionary.

        """
        self.metrics_by_size[size] = metrics

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics across sizes."""
        if not self.metrics_by_size:
            return

        all_metrics: dict[str, list[float]] = {}
        for size_metrics in self.metrics_by_size.values():
            for name, value in size_metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        for name, values in all_metrics.items():
            self.aggregate_metrics[f"{name}_mean"] = sum(values) / len(values)
            self.aggregate_metrics[f"{name}_min"] = min(values)
            self.aggregate_metrics[f"{name}_max"] = max(values)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "metrics_by_size": self.metrics_by_size,
            "aggregate_metrics": self.aggregate_metrics,
            "metadata": self.metadata,
        }


@dataclass
class ComparisonResult:
    """Result of a model comparison.

    Contains metrics for all compared models and statistical tests.
    """

    comparison_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_metrics: dict[str, ModelMetrics] = field(default_factory=dict)
    rankings: dict[str, list[str]] = field(default_factory=dict)  # metric -> ranked models
    pairwise_tests: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_time: str | None = None
    end_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_models(self) -> int:
        """Get number of models compared."""
        return len(self.model_metrics)

    @property
    def model_names(self) -> list[str]:
        """Get model names."""
        return list(self.model_metrics.keys())

    def get_ranking(self, metric: str, minimize: bool = True) -> list[str]:
        """Get models ranked by metric.

        Args:
            metric: Metric name.
            minimize: Whether lower is better.

        Returns:
            List of model names in rank order.

        """
        key = (
            f"{metric}_mean"
            if f"{metric}_mean" in next(iter(self.model_metrics.values())).aggregate_metrics
            else metric
        )

        model_values = []
        for name, metrics in self.model_metrics.items():
            value = metrics.aggregate_metrics.get(key, float("inf") if minimize else float("-inf"))
            model_values.append((name, value))

        sorted_models = sorted(model_values, key=lambda x: x[1], reverse=not minimize)
        return [name for name, _ in sorted_models]

    def get_best_model(self, metric: str, minimize: bool = True) -> str | None:
        """Get best model by metric.

        Args:
            metric: Metric name.
            minimize: Whether lower is better.

        Returns:
            Best model name or None.

        """
        ranking = self.get_ranking(metric, minimize)
        return ranking[0] if ranking else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "comparison_id": self.comparison_id,
            "model_metrics": {k: v.to_dict() for k, v in self.model_metrics.items()},
            "rankings": self.rankings,
            "pairwise_tests": self.pairwise_tests,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Model Comparison [{self.comparison_id}]",
            f"  Models: {', '.join(self.model_names)}",
            "",
        ]

        # Show rankings for key metrics
        for metric in ["mse", "mae", "throughput"]:
            if any(f"{metric}_mean" in m.aggregate_metrics for m in self.model_metrics.values()):
                ranking = self.get_ranking(metric, minimize=metric != "throughput")
                lines.append(f"  {metric} ranking: {' > '.join(ranking)}")

        return "\n".join(lines)


class ModelComparison:
    """Compares multiple models.

    Evaluates models and performs statistical comparison.
    """

    def __init__(
        self,
        config: ComparisonConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize comparison.

        Args:
            config: Comparison configuration.
            logger: Optional structured logger.

        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__).bind(
            comparison=config.name,
        )
        self._results: list[ComparisonResult] = []

    @property
    def results(self) -> list[ComparisonResult]:
        """Get all comparison results."""
        return self._results

    def compare(
        self,
        models: dict[str, Any],
        data_generator: Callable[[int, int], tuple[Any, Any]],
        evaluate_fn: Callable[[Any, Any, Any], dict[str, float]],
    ) -> ComparisonResult:
        """Compare multiple models.

        Args:
            models: Dict of model_name -> model.
            data_generator: Function(size, n_samples) -> (inputs, targets).
            evaluate_fn: Function(model, inputs, targets) -> metrics.

        Returns:
            ComparisonResult with all metrics.

        """
        result = ComparisonResult(
            start_time=datetime.now(timezone.utc).isoformat(),
        )

        model_metrics = {}

        for model_name, model in models.items():
            self._logger.info("evaluating_model", model=model_name)

            metrics = ModelMetrics(model_name=model_name)

            for size in self.config.eval_sizes:
                inputs, targets = data_generator(size, self.config.n_eval_samples)
                size_metrics = evaluate_fn(model, inputs, targets)
                metrics.add_metrics(size, size_metrics)

                self._logger.debug(
                    "size_evaluated",
                    model=model_name,
                    size=size,
                    mse=size_metrics.get("mse"),
                )

            metrics.compute_aggregates()
            model_metrics[model_name] = metrics

        result.model_metrics = model_metrics
        result.end_time = datetime.now(timezone.utc).isoformat()

        # Compute rankings
        for metric in self.config.metrics:
            minimize = metric not in ["throughput", "accuracy"]
            result.rankings[metric] = result.get_ranking(metric, minimize)

        # Statistical tests
        if len(models) >= 2 and self.config.n_bootstrap > 0:
            result.pairwise_tests = self._run_pairwise_tests(model_metrics)

        self._results.append(result)

        self._logger.info(
            "comparison_complete",
            n_models=len(models),
            comparison_id=result.comparison_id,
        )

        return result

    def _run_pairwise_tests(
        self,
        model_metrics: dict[str, ModelMetrics],
    ) -> dict[str, dict[str, Any]]:
        """Run pairwise statistical tests.

        Args:
            model_metrics: Metrics for all models.

        Returns:
            Pairwise test results.

        """
        from src.poc.statistics.significance import SignificanceTest, StatisticalAnalyzer

        analyzer = StatisticalAnalyzer(
            SignificanceTest(
                n_bootstrap=self.config.n_bootstrap,
                alpha=self.config.alpha,
            )
        )

        pairwise: dict[str, dict[str, dict[str, float | bool]]] = {}
        model_names = list(model_metrics.keys())

        for i, name1 in enumerate(model_names):
            for name2 in model_names[i + 1 :]:
                pair_key = f"{name1}_vs_{name2}"
                pairwise[pair_key] = {}

                for metric in self.config.metrics:
                    # Get metric values across sizes
                    values1 = [
                        model_metrics[name1].metrics_by_size.get(s, {}).get(metric, 0)
                        for s in self.config.eval_sizes
                    ]
                    values2 = [
                        model_metrics[name2].metrics_by_size.get(s, {}).get(metric, 0)
                        for s in self.config.eval_sizes
                    ]

                    if values1 and values2:
                        try:
                            comparison = analyzer.compare_runs(values1, values2)
                            pairwise[pair_key][metric] = {
                                "p_value": comparison.p_value,
                                "is_significant": comparison.is_significant,
                                "effect_size": comparison.effect_size,
                                "mean_diff": comparison.mean_difference,
                            }
                        except Exception as e:
                            self._logger.warning(
                                "pairwise_test_failed",
                                pair=pair_key,
                                metric=metric,
                                error=str(e),
                            )

        return pairwise

    def get_summary_table(
        self,
        result: ComparisonResult | None = None,
        metrics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate summary table data.

        Args:
            result: Comparison result (uses latest if None).
            metrics: Metrics to include (uses config if None).

        Returns:
            List of row dictionaries.

        """
        result = result or (self._results[-1] if self._results else None)
        if not result:
            return []

        metrics = metrics or self.config.metrics
        rows = []

        for model_name, model_metrics in result.model_metrics.items():
            row: dict[str, Any] = {"model": model_name}
            for metric in metrics:
                key = f"{metric}_mean"
                if key in model_metrics.aggregate_metrics:
                    row[metric] = model_metrics.aggregate_metrics[key]
            rows.append(row)

        return rows


def create_comparison(
    model_paths: list[str] | None = None,
    model_names: list[str] | None = None,
    eval_sizes: list[int] | None = None,
    **kwargs: Any,
) -> ModelComparison:
    """Factory function to create a model comparison.

    Args:
        model_paths: Paths to models.
        model_names: Names for models.
        eval_sizes: Sizes to evaluate.
        **kwargs: Additional configuration.

    Returns:
        ModelComparison instance.

    """
    config = ComparisonConfig(
        model_paths=model_paths or [],
        model_names=model_names or [],
        eval_sizes=eval_sizes or [9, 13, 19],
        **kwargs,
    )
    return ModelComparison(config=config)
