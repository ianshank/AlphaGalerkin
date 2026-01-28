"""Quick evaluator for rapid prototyping.

Provides simplified evaluation with automatic
metrics computation and confidence intervals.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.prototyping.builder import PrototypeModel
from src.prototyping.config import QuickEvalConfig

logger = structlog.get_logger(__name__)


@dataclass
class MetricResult:
    """Result for a single metric.

    Attributes:
        name: Metric name.
        value: Metric value.
        ci_lower: Lower confidence interval.
        ci_upper: Upper confidence interval.
        std: Standard deviation.

    """

    name: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "std": self.std,
        }


@dataclass
class EvalResult:
    """Result of an evaluation.

    Attributes:
        result_id: Unique identifier.
        model_id: Model identifier.
        n_samples: Number of samples evaluated.
        metrics: Computed metrics.
        duration_seconds: Evaluation duration.
        metadata: Additional metadata.

    """

    result_id: str
    model_id: str
    n_samples: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    duration_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str) -> float | None:
        """Get a metric value by name."""
        if name in self.metrics:
            return self.metrics[name].value
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "model_id": self.model_id,
            "n_samples": self.n_samples,
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate result summary."""
        lines = [
            f"Evaluation Result: {self.result_id}",
            f"Model: {self.model_id}",
            f"Samples: {self.n_samples}",
            f"Duration: {self.duration_seconds:.2f}s",
            "",
            "Metrics:",
        ]
        for name, metric in self.metrics.items():
            if metric.ci_lower is not None and metric.ci_upper is not None:
                lines.append(
                    f"  {name}: {metric.value:.6f} "
                    f"[{metric.ci_lower:.6f}, {metric.ci_upper:.6f}]"
                )
            else:
                lines.append(f"  {name}: {metric.value:.6f}")
        return "\n".join(lines)


class QuickEvaluator:
    """Quick evaluator for prototype models.

    Provides fast evaluation with:
    - Multiple metric computation
    - Bootstrap confidence intervals
    - Automatic batching

    Attributes:
        config: Evaluation configuration.

    """

    def __init__(
        self,
        config: QuickEvalConfig | None = None,
    ) -> None:
        """Initialize evaluator.

        Args:
            config: Evaluation configuration.

        """
        self.config = config or QuickEvalConfig()
        self._results: list[EvalResult] = []
        self._metric_fns: dict[str, Callable[[list[float], list[float]], float]] = {
            "mse": self._compute_mse,
            "mae": self._compute_mae,
            "rmse": self._compute_rmse,
            "accuracy": self._compute_accuracy,
            "r2": self._compute_r2,
        }
        self._logger = logger.bind(evaluator="QuickEvaluator")

    @property
    def results(self) -> list[EvalResult]:
        """Get all evaluation results."""
        return self._results

    def register_metric(
        self,
        name: str,
        fn: Callable[[list[float], list[float]], float],
    ) -> None:
        """Register a custom metric function.

        Args:
            name: Metric name.
            fn: Function(predictions, targets) -> value.

        """
        self._metric_fns[name] = fn
        self._logger.info("registered_metric", name=name)

    def evaluate(
        self,
        model: PrototypeModel | Any,
        predict_fn: Callable[[Any, Any], Any],
        data: list[tuple[Any, Any]],
        metrics: list[str] | None = None,
    ) -> EvalResult:
        """Evaluate a model.

        Args:
            model: Model to evaluate (PrototypeModel or raw model).
            predict_fn: Function(model, input) -> prediction.
            data: List of (input, target) tuples.
            metrics: Optional list of metrics to compute.

        Returns:
            Evaluation result.

        """
        # Extract model if wrapped
        if isinstance(model, PrototypeModel):
            model_id = model.model_id
            raw_model = model.model
        else:
            model_id = str(uuid.uuid4())[:8]
            raw_model = model

        metrics_to_compute = metrics or self.config.metrics

        self._logger.info(
            "evaluation_start",
            model_id=model_id,
            n_samples=len(data),
            metrics=metrics_to_compute,
        )

        start_time = time.time()

        # Collect predictions
        predictions: list[float] = []
        targets: list[float] = []

        for inp, target in data:
            pred = predict_fn(raw_model, inp)
            # Handle various prediction types
            if isinstance(pred, (list, tuple)):
                predictions.extend(pred)
            else:
                predictions.append(float(pred))
            if isinstance(target, (list, tuple)):
                targets.extend(target)
            else:
                targets.append(float(target))

        # Compute metrics
        metric_results: dict[str, MetricResult] = {}

        for metric_name in metrics_to_compute:
            if metric_name not in self._metric_fns:
                self._logger.warning("unknown_metric", metric=metric_name)
                continue

            metric_fn = self._metric_fns[metric_name]
            value = metric_fn(predictions, targets)

            # Compute confidence intervals if requested
            ci_lower = None
            ci_upper = None
            std = None

            if self.config.compute_confidence:
                ci_lower, ci_upper, std = self._bootstrap_ci(
                    predictions, targets, metric_fn
                )

            metric_results[metric_name] = MetricResult(
                name=metric_name,
                value=value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                std=std,
            )

        duration = time.time() - start_time

        result = EvalResult(
            result_id=str(uuid.uuid4())[:8],
            model_id=model_id,
            n_samples=len(data),
            metrics=metric_results,
            duration_seconds=duration,
            metadata={
                "config_hash": self.config.compute_hash(),
                "metrics_computed": list(metric_results.keys()),
            },
        )

        self._results.append(result)

        self._logger.info(
            "evaluation_complete",
            result_id=result.result_id,
            duration=duration,
            metrics={k: v.value for k, v in metric_results.items()},
        )

        return result

    def compare(
        self,
        results: list[EvalResult],
        metric: str,
    ) -> dict[str, Any]:
        """Compare multiple evaluation results.

        Args:
            results: List of evaluation results.
            metric: Metric to compare.

        Returns:
            Comparison summary.

        """
        values: list[dict[str, str | float | None]] = []
        for result in results:
            if metric in result.metrics:
                values.append({
                    "model_id": result.model_id,
                    "value": result.metrics[metric].value,
                    "ci_lower": result.metrics[metric].ci_lower,
                    "ci_upper": result.metrics[metric].ci_upper,
                })

        # Sort by value (ascending for error metrics)
        values.sort(key=lambda x: float(x["value"]) if x["value"] is not None else float("inf"))

        return {
            "metric": metric,
            "n_models": len(values),
            "best": values[0] if values else None,
            "worst": values[-1] if values else None,
            "ranking": [v["model_id"] for v in values],
            "values": values,
        }

    def _bootstrap_ci(
        self,
        predictions: list[float],
        targets: list[float],
        metric_fn: Callable[[list[float], list[float]], float],
        alpha: float = 0.05,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute bootstrap confidence intervals.

        Args:
            predictions: Predictions.
            targets: Targets.
            metric_fn: Metric function.
            alpha: Significance level (must be between 0 and 1).

        Returns:
            (ci_lower, ci_upper, std) or (None, None, None) if insufficient data.

        """
        n = len(predictions)

        # Validate inputs
        if n < 2:
            self._logger.debug(
                "bootstrap_insufficient_data",
                n_samples=n,
                required=2,
            )
            return None, None, None

        if not 0 < alpha < 1:
            self._logger.warning(
                "bootstrap_invalid_alpha",
                alpha=alpha,
                using_default=0.05,
            )
            alpha = 0.05

        bootstrap_values = []

        for _ in range(self.config.n_bootstrap):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            boot_preds = [predictions[i] for i in indices]
            boot_targets = [targets[i] for i in indices]
            value = metric_fn(boot_preds, boot_targets)
            bootstrap_values.append(value)

        bootstrap_values.sort()
        lower_idx = int(alpha / 2 * len(bootstrap_values))
        upper_idx = int((1 - alpha / 2) * len(bootstrap_values))

        # Clamp indices to valid range
        lower_idx = max(0, min(lower_idx, len(bootstrap_values) - 1))
        upper_idx = max(0, min(upper_idx, len(bootstrap_values) - 1))

        ci_lower = bootstrap_values[lower_idx]
        ci_upper = bootstrap_values[upper_idx]
        std = self._std(bootstrap_values)

        return ci_lower, ci_upper, std

    @staticmethod
    def _compute_mse(predictions: list[float], targets: list[float]) -> float:
        """Compute mean squared error."""
        if not predictions:
            return 0.0
        return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

    @staticmethod
    def _compute_mae(predictions: list[float], targets: list[float]) -> float:
        """Compute mean absolute error."""
        if not predictions:
            return 0.0
        return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)

    @staticmethod
    def _compute_rmse(predictions: list[float], targets: list[float]) -> float:
        """Compute root mean squared error."""
        mse = QuickEvaluator._compute_mse(predictions, targets)
        return math.sqrt(mse)

    @staticmethod
    def _compute_accuracy(predictions: list[float], targets: list[float]) -> float:
        """Compute accuracy (for classification)."""
        if not predictions:
            return 0.0
        # Round to nearest integer for classification
        correct = sum(
            1 for p, t in zip(predictions, targets)
            if round(p) == round(t)
        )
        return correct / len(predictions)

    @staticmethod
    def _compute_r2(predictions: list[float], targets: list[float]) -> float:
        """Compute R-squared score."""
        if not predictions:
            return 0.0
        mean_target = sum(targets) / len(targets)
        ss_tot = sum((t - mean_target) ** 2 for t in targets)
        ss_res = sum((t - p) ** 2 for p, t in zip(predictions, targets))
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)

    @staticmethod
    def _std(values: list[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()


def create_quick_evaluator(
    n_samples: int = 1000,
    metrics: list[str] | None = None,
    compute_confidence: bool = True,
    **kwargs: Any,
) -> QuickEvaluator:
    """Create a quick evaluator.

    Args:
        n_samples: Number of evaluation samples.
        metrics: Metrics to compute.
        compute_confidence: Whether to compute CIs.
        **kwargs: Additional configuration.

    Returns:
        Configured QuickEvaluator.

    """
    config = QuickEvalConfig(
        n_samples=n_samples,
        metrics=metrics or ["mse", "mae", "accuracy"],
        compute_confidence=compute_confidence,
        **kwargs,
    )
    return QuickEvaluator(config=config)
