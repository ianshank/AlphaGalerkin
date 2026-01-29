"""Transfer learning validation utilities.

Provides:
- Zero-shot transfer validation
- Cross-resolution evaluation
- Transfer metrics computation
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.research.config import TransferConfig


@dataclass
class TransferMetrics:
    """Metrics for a single transfer evaluation.

    Contains error metrics at a specific target size.
    """

    target_size: int
    source_size: int
    n_samples: int

    # Error metrics
    mse: float
    mae: float
    rmse: float
    max_error: float

    # Additional metrics
    r_squared: float = 0.0
    correlation: float = 0.0

    # Pass/fail
    passed: bool = False
    threshold: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_size": self.target_size,
            "source_size": self.source_size,
            "n_samples": self.n_samples,
            "mse": self.mse,
            "mae": self.mae,
            "rmse": self.rmse,
            "max_error": self.max_error,
            "r_squared": self.r_squared,
            "correlation": self.correlation,
            "passed": self.passed,
            "threshold": self.threshold,
        }


@dataclass
class TransferResult:
    """Result from a complete transfer validation.

    Contains metrics for all target sizes.
    """

    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_size: int = 9
    target_metrics: dict[int, TransferMetrics] = field(default_factory=dict)
    primary_target: int = 19
    config_hash: str = ""

    # Overall status
    passed: bool = False
    all_passed: bool = False

    # Training info
    train_loss: float = 0.0
    train_epochs: int = 0
    train_duration_seconds: float = 0.0

    # Timestamps
    start_time: str | None = None
    end_time: str | None = None

    # Artifacts
    model_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_metrics(self) -> TransferMetrics | None:
        """Get metrics for primary target."""
        return self.target_metrics.get(self.primary_target)

    @property
    def primary_mse(self) -> float | None:
        """Get MSE for primary target."""
        metrics = self.primary_metrics
        return metrics.mse if metrics else None

    @property
    def duration_seconds(self) -> float | None:
        """Get total duration."""
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "source_size": self.source_size,
            "target_metrics": {
                k: v.to_dict() for k, v in self.target_metrics.items()
            },
            "primary_target": self.primary_target,
            "config_hash": self.config_hash,
            "passed": self.passed,
            "all_passed": self.all_passed,
            "train_loss": self.train_loss,
            "train_epochs": self.train_epochs,
            "train_duration_seconds": self.train_duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "model_path": self.model_path,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Transfer Validation Result [{self.result_id}]",
            f"  Source: {self.source_size}x{self.source_size}",
            f"  Status: {'PASS' if self.passed else 'FAIL'}",
            "",
            "  Target Results:",
        ]

        for size, metrics in sorted(self.target_metrics.items()):
            status = "PASS" if metrics.passed else "FAIL"
            primary = " (primary)" if size == self.primary_target else ""
            lines.append(
                f"    {size}x{size}{primary}: MSE={metrics.mse:.6f} [{status}]"
            )

        return "\n".join(lines)


class TransferValidator:
    """Validates zero-shot transfer learning.

    Tests model generalization to unseen resolutions.
    """

    def __init__(
        self,
        config: TransferConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            config: Transfer configuration.
            logger: Optional structured logger.

        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__).bind(
            validator="transfer",
        )
        self._results: list[TransferResult] = []

    @property
    def results(self) -> list[TransferResult]:
        """Get all validation results."""
        return self._results

    def validate(
        self,
        model: Any,
        data_generator: Callable[[int, int], tuple[Any, Any]],
        evaluate_fn: Callable[[Any, Any, Any], dict[str, float]],
    ) -> TransferResult:
        """Validate transfer to all target sizes.

        Args:
            model: Trained model.
            data_generator: Function(size, n_samples) -> (inputs, targets).
            evaluate_fn: Function(model, inputs, targets) -> metrics dict.

        Returns:
            TransferResult with all metrics.

        """
        result = TransferResult(
            source_size=self.config.source_size,
            primary_target=self.config.primary_target,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

        target_metrics = {}

        for target_size in self.config.target_sizes:
            self._logger.info(
                "evaluating_transfer",
                source=self.config.source_size,
                target=target_size,
            )

            # Generate evaluation data
            inputs, targets = data_generator(
                target_size,
                self.config.n_eval_samples,
            )

            # Evaluate
            raw_metrics = evaluate_fn(model, inputs, targets)

            # Create transfer metrics
            metrics = TransferMetrics(
                target_size=target_size,
                source_size=self.config.source_size,
                n_samples=self.config.n_eval_samples,
                mse=raw_metrics.get("mse", 0.0),
                mae=raw_metrics.get("mae", 0.0),
                rmse=raw_metrics.get("rmse", 0.0),
                max_error=raw_metrics.get("max_error", 0.0),
                r_squared=raw_metrics.get("r_squared", 0.0),
                correlation=raw_metrics.get("correlation", 0.0),
                threshold=self.config.mse_threshold,
                passed=raw_metrics.get("mse", float("inf")) < self.config.mse_threshold,
            )

            target_metrics[target_size] = metrics

            self._logger.info(
                "transfer_evaluated",
                target=target_size,
                mse=metrics.mse,
                passed=metrics.passed,
            )

        result.target_metrics = target_metrics
        result.end_time = datetime.now(timezone.utc).isoformat()

        # Determine overall pass/fail
        all_passed = all(m.passed for m in target_metrics.values())
        primary_passed = target_metrics.get(self.config.primary_target, TransferMetrics(
            target_size=0, source_size=0, n_samples=0, mse=1.0, mae=0, rmse=0, max_error=0
        )).passed

        if self.config.require_all_targets:
            result.passed = all_passed
        else:
            result.passed = primary_passed

        result.all_passed = all_passed

        self._results.append(result)

        self._logger.info(
            "validation_complete",
            passed=result.passed,
            all_passed=result.all_passed,
        )

        return result

    def validate_with_training(
        self,
        model_factory: Callable[[], Any],
        train_fn: Callable[[Any, Any, Any, int], tuple[Any, float]],
        data_generator: Callable[[int, int], tuple[Any, Any]],
        evaluate_fn: Callable[[Any, Any, Any], dict[str, float]],
    ) -> TransferResult:
        """Full validation including training.

        Args:
            model_factory: Function() -> model.
            train_fn: Function(model, inputs, targets, epochs) -> (model, loss).
            data_generator: Function(size, n_samples) -> (inputs, targets).
            evaluate_fn: Function(model, inputs, targets) -> metrics.

        Returns:
            TransferResult with training and evaluation metrics.

        """
        result = TransferResult(
            source_size=self.config.source_size,
            primary_target=self.config.primary_target,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

        # Create model
        model = model_factory()

        # Generate training data
        train_inputs, train_targets = data_generator(
            self.config.source_size,
            self.config.n_train_samples,
        )

        # Train
        self._logger.info(
            "training_start",
            source_size=self.config.source_size,
            n_samples=self.config.n_train_samples,
            n_epochs=self.config.n_epochs,
        )

        import time
        train_start = time.perf_counter()

        model, train_loss = train_fn(
            model,
            train_inputs,
            train_targets,
            self.config.n_epochs,
        )

        result.train_duration_seconds = time.perf_counter() - train_start
        result.train_loss = train_loss
        result.train_epochs = self.config.n_epochs

        self._logger.info(
            "training_complete",
            loss=train_loss,
            duration=result.train_duration_seconds,
        )

        # Evaluate on all targets
        target_metrics = {}

        for target_size in self.config.target_sizes:
            inputs, targets = data_generator(
                target_size,
                self.config.n_eval_samples,
            )

            raw_metrics = evaluate_fn(model, inputs, targets)

            metrics = TransferMetrics(
                target_size=target_size,
                source_size=self.config.source_size,
                n_samples=self.config.n_eval_samples,
                mse=raw_metrics.get("mse", 0.0),
                mae=raw_metrics.get("mae", 0.0),
                rmse=raw_metrics.get("rmse", 0.0),
                max_error=raw_metrics.get("max_error", 0.0),
                threshold=self.config.mse_threshold,
                passed=raw_metrics.get("mse", float("inf")) < self.config.mse_threshold,
            )

            target_metrics[target_size] = metrics

        result.target_metrics = target_metrics
        result.end_time = datetime.now(timezone.utc).isoformat()

        # Determine pass/fail
        all_passed = all(m.passed for m in target_metrics.values())
        primary_passed = target_metrics.get(self.config.primary_target, TransferMetrics(
            target_size=0, source_size=0, n_samples=0, mse=1.0, mae=0, rmse=0, max_error=0
        )).passed

        result.passed = all_passed if self.config.require_all_targets else primary_passed
        result.all_passed = all_passed

        self._results.append(result)

        return result

    def get_best_result(self) -> TransferResult | None:
        """Get best validation result.

        Returns:
            Best result by primary MSE.

        """
        passed_results = [r for r in self._results if r.passed]
        if not passed_results:
            # Return best failing result
            if not self._results:
                return None
            return min(
                self._results,
                key=lambda r: r.primary_mse or float("inf"),
            )

        return min(
            passed_results,
            key=lambda r: r.primary_mse or float("inf"),
        )

    def compare_results(
        self,
        result1: TransferResult,
        result2: TransferResult,
    ) -> dict[str, Any]:
        """Compare two validation results.

        Args:
            result1: First result.
            result2: Second result.

        Returns:
            Comparison data.

        """
        targets: dict[int, dict[str, Any]] = {}
        comparison: dict[str, Any] = {
            "result1_id": result1.result_id,
            "result2_id": result2.result_id,
            "targets": targets,
        }

        all_targets = set(result1.target_metrics.keys()) | set(result2.target_metrics.keys())

        for target in sorted(all_targets):
            m1 = result1.target_metrics.get(target)
            m2 = result2.target_metrics.get(target)

            target_comparison = {}
            if m1:
                target_comparison["result1_mse"] = m1.mse
            if m2:
                target_comparison["result2_mse"] = m2.mse
            if m1 and m2:
                target_comparison["mse_diff"] = m2.mse - m1.mse
                target_comparison["improvement"] = (m1.mse - m2.mse) / m1.mse if m1.mse > 0 else 0.0

            comparison["targets"][target] = target_comparison

        return comparison


def create_transfer_validator(
    source_size: int = 9,
    target_sizes: list[int] | None = None,
    mse_threshold: float = 0.05,
    **kwargs: Any,
) -> TransferValidator:
    """Factory function to create a transfer validator.

    Args:
        source_size: Source training size.
        target_sizes: Target evaluation sizes.
        mse_threshold: MSE threshold for passing.
        **kwargs: Additional configuration.

    Returns:
        TransferValidator instance.

    """
    config = TransferConfig(
        source_size=source_size,
        target_sizes=target_sizes or [9, 13, 19],
        mse_threshold=mse_threshold,
        **kwargs,
    )
    return TransferValidator(config=config)
