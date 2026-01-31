"""Curriculum scheduler for managing stage transitions.

Provides:
- Stage progression logic
- Regression detection and handling
- Training parameter adjustment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import structlog

from src.curriculum.config import CurriculumConfig
from src.curriculum.stage import CurriculumStage, StageMetrics, StageStatus


@dataclass
class SchedulerState:
    """Serializable state of the scheduler."""

    current_stage_index: int
    stages_data: list[dict[str, Any]]
    total_games: int
    total_steps: int
    progression_history: list[dict[str, Any]]
    config_hash: str


class CurriculumScheduler:
    """Schedules and manages curriculum stage transitions.

    Handles:
    - Stage activation and completion
    - Progression evaluation
    - Regression detection
    - State persistence
    """

    def __init__(
        self,
        config: CurriculumConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize curriculum scheduler.

        Args:
            config: Curriculum configuration.
            logger: Optional structured logger.
        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__).bind(
            curriculum=config.name,
        )

        # Initialize stages
        self._stages: list[CurriculumStage] = [
            CurriculumStage(config=stage_config)
            for stage_config in config.stages
        ]
        self._current_stage_index = 0
        self._games_since_evaluation = 0

        # Tracking
        self._total_games = 0
        self._total_steps = 0
        self._progression_history: list[dict[str, Any]] = []

        # Callbacks
        self._on_stage_complete: list[Callable[[CurriculumStage], None]] = []
        self._on_stage_start: list[Callable[[CurriculumStage], None]] = []
        self._on_regression: list[Callable[[int, int], None]] = []

    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self._stages[self._current_stage_index]

    @property
    def current_board_size(self) -> int:
        """Get current board size."""
        return self.current_stage.board_size

    @property
    def current_stage_name(self) -> str:
        """Get current stage name."""
        return self.current_stage.config.name

    @property
    def is_final_stage(self) -> bool:
        """Check if currently in final stage."""
        return self._current_stage_index == len(self._stages) - 1

    @property
    def total_stages(self) -> int:
        """Get total number of stages."""
        return len(self._stages)

    @property
    def progress_percentage(self) -> float:
        """Get overall curriculum progress as percentage."""
        if self.is_final_stage and self.current_stage.status.is_terminal():
            return 100.0

        completed = sum(
            1 for s in self._stages if s.status.is_terminal()
        )
        return (completed / self.total_stages) * 100

    def start(self) -> None:
        """Start the curriculum from the first stage."""
        self._logger.info(
            "curriculum_started",
            n_stages=self.total_stages,
            first_stage=self._stages[0].config.name,
        )
        self._activate_stage(0)

    def _activate_stage(self, index: int) -> None:
        """Activate a specific stage.

        Args:
            index: Stage index to activate.
        """
        if index < 0 or index >= len(self._stages):
            return

        self._current_stage_index = index
        stage = self._stages[index]

        stage.activate(warmup_games=self.config.warmup_games_per_stage)
        self._games_since_evaluation = 0

        self._logger.info(
            "stage_activated",
            stage=stage.config.name,
            board_size=stage.board_size,
            warmup_games=self.config.warmup_games_per_stage,
        )

        # Fire callbacks
        for callback in self._on_stage_start:
            callback(stage)

    def record_game(
        self,
        won: bool,
        drawn: bool = False,
    ) -> bool:
        """Record a game result and check for progression.

        Args:
            won: Whether the game was won.
            drawn: Whether the game was drawn.

        Returns:
            True if stage transition occurred.
        """
        self._total_games += 1
        self._games_since_evaluation += 1

        stage = self.current_stage
        stage.record_game(won, drawn)

        # Check for progression at evaluation interval
        if self._games_since_evaluation >= self.config.evaluation_interval:
            self._games_since_evaluation = 0
            return self._evaluate_progression()

        return False

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
        self._total_steps += 1
        self.current_stage.record_training_step(total_loss, policy_loss, value_loss)

    def _evaluate_progression(self) -> bool:
        """Evaluate progression criteria for current stage.

        Returns:
            True if stage transition occurred.
        """
        stage = self.current_stage

        if stage.status == StageStatus.WARMUP:
            return False

        # Check if ready to progress
        if stage.check_progression_criteria():
            return self._advance_to_next_stage()

        # Check for regression (if enabled)
        if self.config.allow_regression and self._current_stage_index > 0:
            if self._should_regress():
                return self._regress_to_previous_stage()

        return False

    def _advance_to_next_stage(self) -> bool:
        """Advance to the next curriculum stage.

        Returns:
            True if advancement occurred.
        """
        current_stage = self.current_stage
        current_stage.complete()

        self._logger.info(
            "stage_completed",
            stage=current_stage.config.name,
            games_played=current_stage.metrics.games_played,
            win_rate=current_stage.metrics.win_rate,
            training_steps=current_stage.metrics.training_steps,
        )

        # Record progression
        self._progression_history.append({
            "type": "advance",
            "from_stage": current_stage.config.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": current_stage.metrics.to_dict(),
        })

        # Fire callbacks
        for callback in self._on_stage_complete:
            callback(current_stage)

        # Activate next stage
        if not self.is_final_stage:
            self._activate_stage(self._current_stage_index + 1)
            return True

        self._logger.info(
            "curriculum_completed",
            total_games=self._total_games,
            total_steps=self._total_steps,
        )
        return False

    def _should_regress(self) -> bool:
        """Check if should regress to previous stage.

        Returns:
            True if regression should occur.
        """
        stage = self.current_stage
        win_rate = stage.metrics.recent_win_rate

        # Check if win rate dropped significantly
        if win_rate < self.config.regression_threshold:
            # Ensure we have enough samples
            if stage.metrics.games_played >= 50:
                return True

        return False

    def _regress_to_previous_stage(self) -> bool:
        """Regress to previous curriculum stage.

        Returns:
            True if regression occurred.
        """
        if self._current_stage_index <= 0:
            return False

        current_stage = self.current_stage
        previous_index = self._current_stage_index - 1

        self._logger.warning(
            "stage_regression",
            from_stage=current_stage.config.name,
            to_stage=self._stages[previous_index].config.name,
            win_rate=current_stage.metrics.recent_win_rate,
            threshold=self.config.regression_threshold,
        )

        # Record regression
        self._progression_history.append({
            "type": "regress",
            "from_stage": current_stage.config.name,
            "to_stage": self._stages[previous_index].config.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": current_stage.metrics.to_dict(),
        })

        # Fire callbacks
        for callback in self._on_regression:
            callback(self._current_stage_index, previous_index)

        # Activate previous stage
        self._activate_stage(previous_index)
        return True

    def force_advance(self) -> bool:
        """Force advancement to next stage.

        Returns:
            True if advancement occurred.
        """
        if self.is_final_stage:
            return False

        self._logger.info(
            "forced_advance",
            from_stage=self.current_stage.config.name,
        )
        return self._advance_to_next_stage()

    def skip_to_stage(self, stage_name: str) -> bool:
        """Skip to a specific stage.

        Args:
            stage_name: Name of stage to skip to.

        Returns:
            True if skip was successful.
        """
        index = self.config.get_stage_index(stage_name)
        if index < 0:
            return False

        # Mark skipped stages
        for i in range(self._current_stage_index, index):
            self._stages[i].skip()

        self._logger.info(
            "skipped_to_stage",
            stage=stage_name,
            skipped_count=index - self._current_stage_index,
        )

        self._activate_stage(index)
        return True

    def get_training_params(
        self,
        base_learning_rate: float,
        base_batch_size: int,
        base_mcts_simulations: int,
    ) -> dict[str, Any]:
        """Get adjusted training parameters for current stage.

        Args:
            base_learning_rate: Base learning rate.
            base_batch_size: Base batch size.
            base_mcts_simulations: Base MCTS simulations.

        Returns:
            Dictionary of adjusted parameters.
        """
        stage = self.current_stage
        config = stage.config

        return {
            "learning_rate": stage.get_effective_learning_rate(base_learning_rate),
            "batch_size": stage.get_effective_batch_size(base_batch_size),
            "mcts_simulations": config.mcts_simulations or base_mcts_simulations,
            "board_size": config.board_size,
        }

    def on_stage_complete(self, callback: Callable[[CurriculumStage], None]) -> None:
        """Register callback for stage completion.

        Args:
            callback: Function to call when stage completes.
        """
        self._on_stage_complete.append(callback)

    def on_stage_start(self, callback: Callable[[CurriculumStage], None]) -> None:
        """Register callback for stage start.

        Args:
            callback: Function to call when stage starts.
        """
        self._on_stage_start.append(callback)

    def on_regression(self, callback: Callable[[int, int], None]) -> None:
        """Register callback for stage regression.

        Args:
            callback: Function called with (from_index, to_index).
        """
        self._on_regression.append(callback)

    def get_state(self) -> SchedulerState:
        """Get serializable scheduler state.

        Returns:
            SchedulerState object.
        """
        return SchedulerState(
            current_stage_index=self._current_stage_index,
            stages_data=[s.to_dict() for s in self._stages],
            total_games=self._total_games,
            total_steps=self._total_steps,
            progression_history=self._progression_history,
            config_hash=self.config.compute_hash(),
        )

    def save_state(self, path: Path | str) -> None:
        """Save scheduler state to file.

        Args:
            path: Path to save state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = self.get_state()
        data = {
            "current_stage_index": state.current_stage_index,
            "stages_data": state.stages_data,
            "total_games": state.total_games,
            "total_steps": state.total_steps,
            "progression_history": state.progression_history,
            "config_hash": state.config_hash,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self._logger.debug("state_saved", path=str(path))

    def load_state(self, path: Path | str) -> None:
        """Load scheduler state from file.

        Args:
            path: Path to load state from.

        Raises:
            ValueError: If config hash doesn't match.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Verify config hash
        if data.get("config_hash") != self.config.compute_hash():
            raise ValueError("Config hash mismatch - curriculum configuration changed")

        # Restore state
        self._current_stage_index = data["current_stage_index"]
        self._total_games = data["total_games"]
        self._total_steps = data["total_steps"]
        self._progression_history = data["progression_history"]

        # Restore stage metrics
        for i, stage_data in enumerate(data["stages_data"]):
            if i < len(self._stages):
                self._stages[i].status = StageStatus(stage_data["status"])
                self._stages[i].metrics = StageMetrics.from_dict(stage_data["metrics"])

        self._logger.info(
            "state_loaded",
            path=str(path),
            current_stage=self.current_stage_name,
            total_games=self._total_games,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get curriculum summary.

        Returns:
            Dictionary with curriculum status summary.
        """
        return {
            "curriculum_name": self.config.name,
            "current_stage": self.current_stage_name,
            "current_stage_index": self._current_stage_index,
            "total_stages": self.total_stages,
            "progress_percentage": self.progress_percentage,
            "current_board_size": self.current_board_size,
            "total_games": self._total_games,
            "total_steps": self._total_steps,
            "stages": [
                {
                    "name": s.config.name,
                    "board_size": s.board_size,
                    "status": s.status.value,
                    "games_played": s.metrics.games_played,
                    "win_rate": s.metrics.win_rate,
                }
                for s in self._stages
            ],
        }
