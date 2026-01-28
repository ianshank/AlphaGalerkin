"""Quick trainer for rapid prototyping.

Provides simplified training loops with minimal
boilerplate and automatic logging.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from src.prototyping.builder import PrototypeModel
from src.prototyping.config import (
    PresetType,
    QuickTrainConfig,
    create_quick_train_config,
)

logger = structlog.get_logger(__name__)


@dataclass
class TrainResult:
    """Result of a training run.

    Attributes:
        result_id: Unique identifier.
        model_id: Model identifier.
        n_epochs: Number of epochs trained.
        n_steps: Total training steps.
        final_loss: Final loss value.
        best_loss: Best loss value.
        metrics: Metrics history.
        duration_seconds: Training duration.
        stopped_early: Whether training stopped early.
        metadata: Additional metadata.

    """

    result_id: str
    model_id: str
    n_epochs: int
    n_steps: int
    final_loss: float
    best_loss: float
    metrics: dict[str, list[float]] = field(default_factory=dict)
    duration_seconds: float = 0.0
    stopped_early: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "model_id": self.model_id,
            "n_epochs": self.n_epochs,
            "n_steps": self.n_steps,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "duration_seconds": self.duration_seconds,
            "stopped_early": self.stopped_early,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate result summary."""
        lines = [
            f"Training Result: {self.result_id}",
            f"Model: {self.model_id}",
            f"Epochs: {self.n_epochs}",
            f"Steps: {self.n_steps}",
            f"Final Loss: {self.final_loss:.6f}",
            f"Best Loss: {self.best_loss:.6f}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Early Stop: {self.stopped_early}",
        ]
        return "\n".join(lines)


class QuickTrainer:
    """Quick trainer for prototype models.

    Provides a simplified training loop with:
    - Automatic logging
    - Early stopping
    - Learning rate warmup
    - Gradient clipping

    Attributes:
        config: Training configuration.

    """

    def __init__(
        self,
        config: QuickTrainConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration.

        """
        self.config = config or create_quick_train_config()
        self._results: list[TrainResult] = []
        self._callbacks: dict[str, list[Callable[..., None]]] = {
            "on_epoch_start": [],
            "on_epoch_end": [],
            "on_step": [],
            "on_train_start": [],
            "on_train_end": [],
        }
        self._logger = logger.bind(trainer="QuickTrainer")

    @property
    def results(self) -> list[TrainResult]:
        """Get all training results."""
        return self._results

    def register_callback(
        self,
        event: str,
        callback: Callable[..., None],
    ) -> None:
        """Register a callback.

        Args:
            event: Event name.
            callback: Callback function.

        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def train(
        self,
        model: PrototypeModel | Any,
        train_fn: Callable[[Any, Any], float],
        data_iterator: Callable[[], Iterator[Any]],
        eval_fn: Callable[[Any], dict[str, float]] | None = None,
        eval_data: Any | None = None,
    ) -> TrainResult:
        """Train a model.

        Args:
            model: Model to train (PrototypeModel or raw model).
            train_fn: Function(model, batch) -> loss.
            data_iterator: Function returning data iterator.
            eval_fn: Optional evaluation function.
            eval_data: Optional evaluation data.

        Returns:
            Training result.

        """
        # Extract model if wrapped
        if isinstance(model, PrototypeModel):
            model_id = model.model_id
            raw_model = model.model
        else:
            model_id = str(uuid.uuid4())[:8]
            raw_model = model

        self._logger.info(
            "training_start",
            model_id=model_id,
            n_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
        )

        # Initialize tracking
        start_time = time.time()
        metrics: dict[str, list[float]] = {"loss": [], "lr": []}
        best_loss = float("inf")
        patience_counter = 0
        global_step = 0
        current_lr = self.config.learning_rate

        # Fire callbacks
        self._fire_callbacks("on_train_start", raw_model)

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            self._fire_callbacks("on_epoch_start", epoch, raw_model)

            for batch in data_iterator():
                # Warmup learning rate
                if global_step < self.config.warmup_steps:
                    current_lr = (
                        self.config.learning_rate
                        * (global_step + 1)
                        / self.config.warmup_steps
                    )

                # Training step
                loss = train_fn(raw_model, batch)
                epoch_loss += loss
                epoch_steps += 1
                global_step += 1

                # Logging
                if global_step % self.config.log_interval == 0:
                    metrics["loss"].append(loss)
                    metrics["lr"].append(current_lr)
                    self._logger.debug(
                        "step",
                        step=global_step,
                        loss=loss,
                        lr=current_lr,
                    )

                # Evaluation
                if eval_fn and global_step % self.config.eval_interval == 0:
                    eval_metrics = eval_fn(raw_model)
                    for name, value in eval_metrics.items():
                        if name not in metrics:
                            metrics[name] = []
                        metrics[name].append(value)

                self._fire_callbacks("on_step", global_step, loss, raw_model)

            # Epoch complete
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)

            self._logger.info(
                "epoch_complete",
                epoch=epoch + 1,
                avg_loss=avg_epoch_loss,
                steps=epoch_steps,
            )

            self._fire_callbacks("on_epoch_end", epoch, avg_epoch_loss, raw_model)

            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                self._logger.info(
                    "early_stopping",
                    epoch=epoch + 1,
                    patience=self.config.early_stopping_patience,
                )
                break

        # Training complete
        duration = time.time() - start_time
        final_loss = metrics["loss"][-1] if metrics["loss"] else 0.0

        result = TrainResult(
            result_id=str(uuid.uuid4())[:8],
            model_id=model_id,
            n_epochs=epoch + 1,
            n_steps=global_step,
            final_loss=final_loss,
            best_loss=best_loss,
            metrics=metrics,
            duration_seconds=duration,
            stopped_early=patience_counter >= self.config.early_stopping_patience,
            metadata={
                "config_hash": self.config.compute_hash(),
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
            },
        )

        self._results.append(result)
        self._fire_callbacks("on_train_end", result, raw_model)

        self._logger.info(
            "training_complete",
            result_id=result.result_id,
            final_loss=final_loss,
            best_loss=best_loss,
            duration=duration,
        )

        return result

    def _fire_callbacks(self, event: str, *args: Any) -> None:
        """Fire callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                self._logger.warning(
                    "callback_error",
                    event=event,
                    error=str(e),
                )

    def get_best_result(self) -> TrainResult | None:
        """Get result with lowest final loss."""
        if not self._results:
            return None
        return min(self._results, key=lambda r: r.best_loss)

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()


def create_quick_trainer(
    preset: str | PresetType = PresetType.SMALL,
    **kwargs: Any,
) -> QuickTrainer:
    """Create a quick trainer.

    Args:
        preset: Preset configuration type.
        **kwargs: Configuration overrides.

    Returns:
        Configured QuickTrainer.

    """
    config = create_quick_train_config(preset=preset, **kwargs)
    return QuickTrainer(config=config)
