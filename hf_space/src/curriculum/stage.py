"""Curriculum stage state management.

Provides:
- StageStatus enum for tracking stage state
- CurriculumStage class for managing individual stage state
- Metrics collection and progression evaluation
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.curriculum.config import StageConfig


class StageStatus(str, Enum):
    """Status of a curriculum stage."""

    PENDING = "pending"
    ACTIVE = "active"
    WARMUP = "warmup"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    SKIPPED = "skipped"

    def is_terminal(self) -> bool:
        """Check if status is terminal."""
        return self in (StageStatus.COMPLETED, StageStatus.SKIPPED)

    def is_active(self) -> bool:
        """Check if status is active (including warmup/evaluating)."""
        return self in (
            StageStatus.ACTIVE,
            StageStatus.WARMUP,
            StageStatus.EVALUATING,
        )


@dataclass
class StageMetrics:
    """Collected metrics for a curriculum stage."""

    games_played: int = 0
    games_won: int = 0
    games_lost: int = 0
    games_drawn: int = 0
    training_steps: int = 0
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    recent_win_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def win_rate(self) -> float:
        """Calculate overall win rate."""
        total = self.games_played
        if total == 0:
            return 0.0
        return self.games_won / total

    @property
    def recent_win_rate(self) -> float:
        """Calculate win rate over recent window."""
        if not self.recent_win_rates:
            return 0.0
        return sum(self.recent_win_rates) / len(self.recent_win_rates)

    @property
    def average_loss(self) -> float:
        """Calculate average loss over recent window."""
        if not self.recent_losses:
            return float("inf")
        return sum(self.recent_losses) / len(self.recent_losses)

    def record_game(self, won: bool, drawn: bool = False) -> None:
        """Record a game result.

        Args:
            won: Whether the game was won.
            drawn: Whether the game was drawn.
        """
        self.games_played += 1
        if drawn:
            self.games_drawn += 1
            self.recent_win_rates.append(0.5)
        elif won:
            self.games_won += 1
            self.recent_win_rates.append(1.0)
        else:
            self.games_lost += 1
            self.recent_win_rates.append(0.0)

    def record_training_step(
        self,
        total_loss: float,
        policy_loss: float | None = None,
        value_loss: float | None = None,
    ) -> None:
        """Record a training step.

        Args:
            total_loss: Total loss value.
            policy_loss: Policy loss component.
            value_loss: Value loss component.
        """
        self.training_steps += 1
        self.total_loss += total_loss
        self.recent_losses.append(total_loss)

        if policy_loss is not None:
            self.policy_loss += policy_loss
        if value_loss is not None:
            self.value_loss += value_loss

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "games_played": self.games_played,
            "games_won": self.games_won,
            "games_lost": self.games_lost,
            "games_drawn": self.games_drawn,
            "training_steps": self.training_steps,
            "win_rate": self.win_rate,
            "recent_win_rate": self.recent_win_rate,
            "average_loss": self.average_loss,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageMetrics":
        """Create from dictionary."""
        metrics = cls()
        metrics.games_played = data.get("games_played", 0)
        metrics.games_won = data.get("games_won", 0)
        metrics.games_lost = data.get("games_lost", 0)
        metrics.games_drawn = data.get("games_drawn", 0)
        metrics.training_steps = data.get("training_steps", 0)
        return metrics


@dataclass
class CurriculumStage:
    """Manages state for a single curriculum stage.

    Tracks metrics, evaluates progression criteria,
    and handles stage lifecycle.
    """

    config: StageConfig
    status: StageStatus = StageStatus.PENDING
    metrics: StageMetrics = field(default_factory=StageMetrics)
    start_time: datetime | None = None
    end_time: datetime | None = None
    warmup_games_remaining: int = 0

    def activate(self, warmup_games: int = 0) -> None:
        """Activate this stage.

        Args:
            warmup_games: Number of warmup games before evaluation.
        """
        self.status = StageStatus.WARMUP if warmup_games > 0 else StageStatus.ACTIVE
        self.warmup_games_remaining = warmup_games
        self.start_time = datetime.now(timezone.utc)
        self.metrics = StageMetrics()

    def complete(self) -> None:
        """Mark stage as completed."""
        self.status = StageStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)

    def skip(self) -> None:
        """Skip this stage."""
        self.status = StageStatus.SKIPPED
        self.end_time = datetime.now(timezone.utc)

    @property
    def board_size(self) -> int:
        """Get board size for this stage."""
        return self.config.board_size

    @property
    def duration_seconds(self) -> float | None:
        """Get stage duration in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    def record_game(self, won: bool, drawn: bool = False) -> None:
        """Record a game result.

        Args:
            won: Whether the game was won.
            drawn: Whether the game was drawn.
        """
        self.metrics.record_game(won, drawn)

        # Update warmup status
        if self.status == StageStatus.WARMUP:
            self.warmup_games_remaining -= 1
            if self.warmup_games_remaining <= 0:
                self.status = StageStatus.ACTIVE

    def record_training_step(
        self,
        total_loss: float,
        policy_loss: float | None = None,
        value_loss: float | None = None,
    ) -> None:
        """Record a training step.

        Args:
            total_loss: Total loss value.
            policy_loss: Policy loss component.
            value_loss: Value loss component.
        """
        self.metrics.record_training_step(total_loss, policy_loss, value_loss)

    def check_min_requirements(self) -> bool:
        """Check if minimum requirements are met.

        Returns:
            True if minimum games and steps are reached.
        """
        return (
            self.metrics.games_played >= self.config.min_games
            and self.metrics.training_steps >= self.config.min_steps
        )

    def check_max_limits(self) -> bool:
        """Check if maximum limits are exceeded.

        Returns:
            True if max games or steps are exceeded.
        """
        if (
            self.config.max_games is not None
            and self.metrics.games_played >= self.config.max_games
        ):
            return True
        if (
            self.config.max_steps is not None
            and self.metrics.training_steps >= self.config.max_steps
        ):
            return True
        return False

    def check_progression_criteria(self) -> bool:
        """Check if progression criteria are satisfied.

        Returns:
            True if ready to advance to next stage.
        """
        # Skip evaluation during warmup
        if self.status == StageStatus.WARMUP:
            return False

        # Check minimum requirements
        if not self.check_min_requirements():
            return False

        # Check forced progression
        if self.check_max_limits():
            return True

        # No criteria means stay in stage forever (terminal stage)
        if not self.config.progression_criteria:
            return False

        # Get metric values
        metric_values = self._get_metric_values()

        # Evaluate criteria
        results = []
        for criterion in self.config.progression_criteria:
            value = metric_values.get(criterion.metric)
            if value is None:
                results.append(False)
                continue
            results.append(
                criterion.is_satisfied(value, self.metrics.games_played)
            )

        # Apply criteria mode
        if self.config.criteria_mode == "all":
            return all(results)
        else:  # "any"
            return any(results)

    def _get_metric_values(self) -> dict[str, float]:
        """Get current metric values for criterion evaluation.

        Returns:
            Dictionary of metric name to value.
        """
        return {
            "win_rate": self.metrics.win_rate,
            "recent_win_rate": self.metrics.recent_win_rate,
            "loss": self.metrics.average_loss,
            "games_played": float(self.metrics.games_played),
            "training_steps": float(self.metrics.training_steps),
        }

    def get_effective_learning_rate(self, base_lr: float) -> float:
        """Get effective learning rate for this stage.

        Args:
            base_lr: Base learning rate.

        Returns:
            Adjusted learning rate.
        """
        return base_lr * self.config.learning_rate_multiplier

    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """Get effective batch size for this stage.

        Args:
            base_batch_size: Base batch size.

        Returns:
            Adjusted batch size.
        """
        return max(1, int(base_batch_size * self.config.batch_size_multiplier))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_name": self.config.name,
            "board_size": self.config.board_size,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "warmup_games_remaining": self.warmup_games_remaining,
            "duration_seconds": self.duration_seconds,
        }
