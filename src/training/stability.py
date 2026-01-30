"""Training stability monitoring and early stopping.

Provides:
- EarlyStopping: Stop training when metric stops improving
- PlateauDetector: Reduce learning rate on plateaus
- GradientMonitor: Track gradient health
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import structlog
import torch
from torch.optim import Optimizer

logger = structlog.get_logger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping.

    Attributes:
        patience: Number of evaluations without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        metric: Metric to monitor (e.g., "eval/win_rate").
        mode: "max" if higher is better, "min" if lower is better.

    """

    patience: int = 10
    min_delta: float = 0.001
    metric: str = "eval/win_rate"
    mode: Literal["max", "min"] = "max"


class EarlyStopping:
    """Early stopping monitor.

    Stops training if monitored metric doesn't improve for `patience` evaluations.

    Example:
        early_stopping = EarlyStopping(EarlyStoppingConfig(patience=5))

        for epoch in range(max_epochs):
            train_one_epoch()
            val_metric = evaluate()

            if early_stopping.step(val_metric):
                print("Early stopping triggered!")
                break

    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        """Initialize early stopping.

        Args:
            config: Early stopping configuration.

        """
        self.config = config
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update with new metric value.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.

        """
        if self.best_value is None:
            self.best_value = value
            logger.debug("early_stopping_initialized", initial_value=value)
            return False

        improved = self._check_improvement(value)

        if improved:
            self.best_value = value
            self.counter = 0
            logger.debug(
                "early_stopping_improvement",
                new_best=value,
            )
        else:
            self.counter += 1
            logger.debug(
                "early_stopping_no_improvement",
                counter=self.counter,
                patience=self.config.patience,
                current=value,
                best=self.best_value,
            )

            if self.counter >= self.config.patience:
                self.should_stop = True
                logger.info(
                    "early_stopping_triggered",
                    best_value=self.best_value,
                    patience=self.config.patience,
                )

        return self.should_stop

    def _check_improvement(self, value: float) -> bool:
        """Check if value represents improvement.

        Args:
            value: Current metric value.

        Returns:
            True if improved.

        """
        if self.best_value is None:
            return True

        if self.config.mode == "max":
            return value > self.best_value + self.config.min_delta
        return value < self.best_value - self.config.min_delta

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_value = None
        self.counter = 0
        self.should_stop = False


@dataclass
class PlateauConfig:
    """Configuration for learning rate plateau detection.

    Attributes:
        patience: Steps without improvement before reducing LR.
        factor: Factor to reduce LR by (new_lr = old_lr * factor).
        min_lr: Minimum learning rate.
        metric: Metric to monitor.
        mode: "max" if higher is better, "min" if lower is better.
        threshold: Relative threshold for measuring improvement.

    """

    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6
    metric: str = "train/loss/total"
    mode: Literal["max", "min"] = "min"
    threshold: float = 0.01


class PlateauDetector:
    """Detects training plateaus and reduces learning rate.

    Similar to PyTorch's ReduceLROnPlateau scheduler.

    Example:
        detector = PlateauDetector(PlateauConfig(), optimizer)

        for step in range(total_steps):
            loss = train_step()
            if detector.step(loss):
                print("Learning rate reduced!")

    """

    def __init__(self, config: PlateauConfig, optimizer: Optimizer) -> None:
        """Initialize plateau detector.

        Args:
            config: Plateau detection configuration.
            optimizer: Optimizer to adjust learning rate.

        """
        self.config = config
        self.optimizer = optimizer
        self.best_value: float | None = None
        self.counter = 0
        self.num_reductions = 0

    def step(self, value: float) -> bool:
        """Update with new metric value.

        Args:
            value: Current metric value.

        Returns:
            True if learning rate was reduced.

        """
        if self.best_value is None:
            self.best_value = value
            return False

        improved = self._check_improvement(value)

        if improved:
            self.best_value = value
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.config.patience:
            reduced = self._reduce_lr()
            self.counter = 0
            return reduced

        return False

    def _check_improvement(self, value: float) -> bool:
        """Check if value represents improvement.

        Args:
            value: Current metric value.

        Returns:
            True if improved beyond threshold.

        """
        if self.best_value is None:
            return True

        if self.config.mode == "min":
            return value < self.best_value * (1 - self.config.threshold)
        return value > self.best_value * (1 + self.config.threshold)

    def _reduce_lr(self) -> bool:
        """Reduce learning rate by factor.

        Returns:
            True if LR was actually reduced (not at min already).

        """
        reduced = False
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.config.factor, self.config.min_lr)

            if new_lr < old_lr:
                param_group["lr"] = new_lr
                reduced = True
                logger.info(
                    "learning_rate_reduced",
                    old_lr=old_lr,
                    new_lr=new_lr,
                    num_reductions=self.num_reductions + 1,
                )

        if reduced:
            self.num_reductions += 1

        return reduced

    def get_current_lr(self) -> float:
        """Get current learning rate.

        Returns:
            Current learning rate from first param group.

        """
        return self.optimizer.param_groups[0]["lr"]


@dataclass
class GradientStatus:
    """Status of gradient health check.

    Attributes:
        gradient_norm: Current gradient norm.
        is_exploding: True if gradient is too large.
        is_vanishing: True if gradient is too small.
        is_nan: True if gradient contains NaN.

    """

    gradient_norm: float
    is_exploding: bool = False
    is_vanishing: bool = False
    is_nan: bool = False

    @property
    def is_healthy(self) -> bool:
        """Check if gradients are healthy."""
        return not (self.is_exploding or self.is_vanishing or self.is_nan)


class GradientMonitor:
    """Monitors gradient health during training.

    Tracks gradient norm history and detects anomalies.

    Example:
        monitor = GradientMonitor()

        for step in range(total_steps):
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm)

            status = monitor.check(grad_norm)
            if not status.is_healthy:
                print(f"Warning: {status}")

    """

    def __init__(
        self,
        exploding_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
        history_size: int = 100,
    ) -> None:
        """Initialize gradient monitor.

        Args:
            exploding_threshold: Norm above this is considered exploding.
            vanishing_threshold: Norm below this is considered vanishing.
            history_size: Size of gradient history to keep.

        """
        self.exploding_threshold = exploding_threshold
        self.vanishing_threshold = vanishing_threshold
        self.history: deque[float] = deque(maxlen=history_size)

    def check(self, grad_norm: float | torch.Tensor) -> GradientStatus:
        """Check gradient norm and return status.

        Args:
            grad_norm: Current gradient norm.

        Returns:
            Gradient health status.

        """
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()

        # Check for NaN
        is_nan = grad_norm != grad_norm  # NaN check

        status = GradientStatus(
            gradient_norm=grad_norm,
            is_exploding=grad_norm > self.exploding_threshold,
            is_vanishing=grad_norm < self.vanishing_threshold and not is_nan,
            is_nan=is_nan,
        )

        if not is_nan:
            self.history.append(grad_norm)

        # Log warnings
        if status.is_nan:
            logger.warning("gradient_nan_detected")
        elif status.is_exploding:
            logger.warning(
                "gradient_explosion_detected",
                grad_norm=grad_norm,
                threshold=self.exploding_threshold,
            )
        elif status.is_vanishing:
            logger.warning(
                "gradient_vanishing_detected",
                grad_norm=grad_norm,
                threshold=self.vanishing_threshold,
            )

        return status

    def get_statistics(self) -> dict[str, float]:
        """Get gradient statistics from history.

        Returns:
            Dictionary with mean, std, min, max of gradient norms.

        """
        if not self.history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        history_list = list(self.history)
        import statistics

        return {
            "mean": statistics.mean(history_list),
            "std": statistics.stdev(history_list) if len(history_list) > 1 else 0.0,
            "min": min(history_list),
            "max": max(history_list),
        }


class TrainingStabilityMonitor:
    """Combined stability monitoring for training.

    Integrates early stopping, plateau detection, and gradient monitoring.
    """

    def __init__(
        self,
        early_stopping: EarlyStopping | None = None,
        plateau_detector: PlateauDetector | None = None,
        gradient_monitor: GradientMonitor | None = None,
    ) -> None:
        """Initialize stability monitor.

        Args:
            early_stopping: Early stopping monitor.
            plateau_detector: Plateau detector.
            gradient_monitor: Gradient monitor.

        """
        self.early_stopping = early_stopping
        self.plateau_detector = plateau_detector
        self.gradient_monitor = gradient_monitor or GradientMonitor()

    def check_gradient(self, grad_norm: float) -> GradientStatus:
        """Check gradient health.

        Args:
            grad_norm: Current gradient norm.

        Returns:
            Gradient status.

        """
        return self.gradient_monitor.check(grad_norm)

    def check_early_stopping(self, metric_value: float) -> bool:
        """Check if training should stop early.

        Args:
            metric_value: Current metric value.

        Returns:
            True if should stop.

        """
        if self.early_stopping is None:
            return False
        return self.early_stopping.step(metric_value)

    def check_plateau(self, metric_value: float) -> bool:
        """Check for plateau and reduce LR if needed.

        Args:
            metric_value: Current metric value.

        Returns:
            True if LR was reduced.

        """
        if self.plateau_detector is None:
            return False
        return self.plateau_detector.step(metric_value)

    @property
    def should_stop(self) -> bool:
        """Check if training should stop."""
        if self.early_stopping is None:
            return False
        return self.early_stopping.should_stop
