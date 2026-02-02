"""High-level curriculum learning manager.

Provides:
- Integration with training loop
- Model zoo integration for opponent selection
- Complete curriculum orchestration
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from src.curriculum.config import CurriculumConfig, create_default_curriculum
from src.curriculum.scheduler import CurriculumScheduler
from src.curriculum.stage import CurriculumStage, StageStatus

if TYPE_CHECKING:
    from src.distributed.model_zoo import ModelZoo


@dataclass
class CurriculumMetrics:
    """Aggregated curriculum learning metrics."""

    curriculum_name: str
    current_stage: str
    current_board_size: int
    stage_progress: float
    curriculum_progress: float
    total_games: int
    total_steps: int
    current_win_rate: float
    stages_completed: int
    total_stages: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "curriculum_name": self.curriculum_name,
            "current_stage": self.current_stage,
            "current_board_size": self.current_board_size,
            "stage_progress": self.stage_progress,
            "curriculum_progress": self.curriculum_progress,
            "total_games": self.total_games,
            "total_steps": self.total_steps,
            "current_win_rate": self.current_win_rate,
            "stages_completed": self.stages_completed,
            "total_stages": self.total_stages,
        }


class CurriculumManager:
    """Manages curriculum learning lifecycle.

    Integrates:
    - Curriculum scheduler for stage management
    - Model zoo for opponent selection
    - Training infrastructure hooks
    - Checkpoint management
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        model_zoo: ModelZoo | None = None,
        checkpoint_dir: Path | str | None = None,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize curriculum manager.

        Args:
            config: Curriculum configuration (default curriculum if None).
            model_zoo: Optional model zoo for opponent selection.
            checkpoint_dir: Directory for curriculum checkpoints.
            logger: Optional structured logger.

        """
        self.config = config or create_default_curriculum()
        self._model_zoo = model_zoo
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self._logger = logger or structlog.get_logger(__name__).bind(
            curriculum=self.config.name,
        )

        # Initialize scheduler
        self._scheduler = CurriculumScheduler(self.config, self._logger)

        # Track state
        self._started = False
        self._completed = False
        self._start_time: datetime | None = None

        # Callbacks
        self._on_curriculum_complete: list[Callable[[], None]] = []

        # Register scheduler callbacks
        self._scheduler.on_stage_complete(self._handle_stage_complete)
        self._scheduler.on_stage_start(self._handle_stage_start)

    @property
    def scheduler(self) -> CurriculumScheduler:
        """Access the underlying scheduler."""
        return self._scheduler

    @property
    def is_started(self) -> bool:
        """Check if curriculum has started."""
        return self._started

    @property
    def is_completed(self) -> bool:
        """Check if curriculum is completed."""
        return self._completed

    @property
    def current_board_size(self) -> int:
        """Get current board size."""
        return self._scheduler.current_board_size

    @property
    def current_stage_name(self) -> str:
        """Get current stage name."""
        return self._scheduler.current_stage_name

    def start(self) -> None:
        """Start curriculum learning."""
        if self._started:
            self._logger.warning("curriculum_already_started")
            return

        self._started = True
        self._start_time = datetime.now(timezone.utc)
        self._scheduler.start()

        self._logger.info(
            "curriculum_manager_started",
            stages=self._scheduler.total_stages,
            first_board_size=self.current_board_size,
        )

    def step(
        self,
        game_result: dict[str, Any] | None = None,
        training_metrics: dict[str, float] | None = None,
    ) -> bool:
        """Process a curriculum step.

        Can record game results and/or training metrics.

        Args:
            game_result: Optional game result with 'won', 'drawn' keys.
            training_metrics: Optional training metrics with loss values.

        Returns:
            True if stage transition occurred.

        """
        if not self._started:
            raise RuntimeError("Curriculum not started. Call start() first.")

        if self._completed:
            return False

        transition_occurred = False

        # Record game result
        if game_result is not None:
            won = game_result.get("won", False)
            drawn = game_result.get("drawn", False)
            transition_occurred = self._scheduler.record_game(won, drawn)

        # Record training metrics
        if training_metrics is not None:
            self._scheduler.record_training_step(
                total_loss=training_metrics.get("total_loss", 0.0),
                policy_loss=training_metrics.get("policy_loss"),
                value_loss=training_metrics.get("value_loss"),
            )

        # Check if curriculum is complete
        if self._scheduler.is_final_stage:
            stage = self._scheduler.current_stage
            if stage.status == StageStatus.COMPLETED:
                self._complete()

        return transition_occurred

    def _complete(self) -> None:
        """Mark curriculum as complete."""
        if self._completed:
            return

        self._completed = True

        duration = None
        if self._start_time:
            duration = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        self._logger.info(
            "curriculum_completed",
            duration_seconds=duration,
            total_games=self._scheduler._total_games,
            total_steps=self._scheduler._total_steps,
        )

        # Fire callbacks
        for callback in self._on_curriculum_complete:
            callback()

    def _handle_stage_complete(self, stage: CurriculumStage) -> None:
        """Handle stage completion.

        Args:
            stage: Completed stage.

        """
        # Save checkpoint if configured
        if self.config.checkpoint_on_stage_complete and self._checkpoint_dir:
            checkpoint_path = self._checkpoint_dir / f"curriculum_{stage.config.name}.json"
            self._scheduler.save_state(checkpoint_path)
            self._logger.info("stage_checkpoint_saved", path=str(checkpoint_path))

        # Update model zoo if available
        if self._model_zoo is not None:
            self._model_zoo.update_metrics(
                version=self._model_zoo.get_latest_version() or 0,
                metrics={
                    f"curriculum_stage_{stage.config.name}_win_rate": stage.metrics.win_rate,
                    f"curriculum_stage_{stage.config.name}_games": float(
                        stage.metrics.games_played
                    ),
                },
            )

    def _handle_stage_start(self, stage: CurriculumStage) -> None:
        """Handle stage start.

        Args:
            stage: Started stage.

        """
        self._logger.info(
            "new_stage_started",
            stage=stage.config.name,
            board_size=stage.board_size,
        )

    def get_opponent_model(self) -> tuple[Any, Any] | None:
        """Get opponent model from model zoo.

        Returns:
            Tuple of (state_dict, metadata) or None.

        """
        if self._model_zoo is None:
            return None
        return self._model_zoo.get_curriculum_opponent()

    def get_training_params(
        self,
        base_learning_rate: float = 1e-4,
        base_batch_size: int = 64,
        base_mcts_simulations: int = 800,
    ) -> dict[str, Any]:
        """Get adjusted training parameters for current stage.

        Args:
            base_learning_rate: Base learning rate.
            base_batch_size: Base batch size.
            base_mcts_simulations: Base MCTS simulations.

        Returns:
            Dictionary of adjusted parameters.

        """
        return self._scheduler.get_training_params(
            base_learning_rate,
            base_batch_size,
            base_mcts_simulations,
        )

    def get_metrics(self) -> CurriculumMetrics:
        """Get current curriculum metrics.

        Returns:
            CurriculumMetrics object.

        """
        stage = self._scheduler.current_stage
        summary = self._scheduler.get_summary()

        # Calculate stage progress
        if stage.config.min_games > 0:
            stage_progress = min(
                100.0,
                (stage.metrics.games_played / stage.config.min_games) * 100,
            )
        else:
            stage_progress = 100.0

        stages_completed = sum(
            1 for s in summary["stages"] if s["status"] in ("completed", "skipped")
        )

        return CurriculumMetrics(
            curriculum_name=self.config.name,
            current_stage=self._scheduler.current_stage_name,
            current_board_size=self.current_board_size,
            stage_progress=stage_progress,
            curriculum_progress=summary["progress_percentage"],
            total_games=summary["total_games"],
            total_steps=summary["total_steps"],
            current_win_rate=stage.metrics.recent_win_rate,
            stages_completed=stages_completed,
            total_stages=self._scheduler.total_stages,
        )

    def save_checkpoint(self, path: Path | str | None = None) -> Path:
        """Save curriculum checkpoint.

        Args:
            path: Optional specific path.

        Returns:
            Path to saved checkpoint.

        """
        if path is None:
            if self._checkpoint_dir is None:
                raise ValueError("No checkpoint directory configured")
            path = self._checkpoint_dir / "curriculum_state.json"

        path = Path(path)
        self._scheduler.save_state(path)
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load curriculum checkpoint.

        Args:
            path: Path to checkpoint file.

        """
        self._scheduler.load_state(path)
        self._started = True

        # Check if already completed
        if self._scheduler.is_final_stage:
            stage = self._scheduler.current_stage
            if stage.status == StageStatus.COMPLETED:
                self._completed = True

    def on_curriculum_complete(self, callback: Callable[[], None]) -> None:
        """Register callback for curriculum completion.

        Args:
            callback: Function to call when curriculum completes.

        """
        self._on_curriculum_complete.append(callback)

    def force_advance(self) -> bool:
        """Force advancement to next stage.

        Returns:
            True if advancement occurred.

        """
        return self._scheduler.force_advance()

    def skip_to_board_size(self, board_size: int) -> bool:
        """Skip to stage with specified board size.

        Args:
            board_size: Target board size.

        Returns:
            True if skip was successful.

        """
        for stage_config in self.config.stages:
            if stage_config.board_size == board_size:
                return self._scheduler.skip_to_stage(stage_config.name)
        return False

    def get_summary(self) -> dict[str, Any]:
        """Get complete curriculum summary.

        Returns:
            Dictionary with full curriculum status.

        """
        summary = self._scheduler.get_summary()
        summary["is_started"] = self._started
        summary["is_completed"] = self._completed
        summary["start_time"] = self._start_time.isoformat() if self._start_time else None
        return summary


def create_curriculum_manager(
    board_sizes: list[int] | None = None,
    name: str = "default",
    win_rate_threshold: float = 0.55,
    min_games_per_stage: int = 1000,
    model_zoo: ModelZoo | None = None,
    checkpoint_dir: Path | str | None = None,
) -> CurriculumManager:
    """Factory function to create curriculum manager.

    Args:
        board_sizes: Board sizes for stages (default: [9, 13, 19]).
        name: Curriculum name.
        win_rate_threshold: Win rate threshold for progression.
        min_games_per_stage: Minimum games per stage.
        model_zoo: Optional model zoo for opponent selection.
        checkpoint_dir: Optional checkpoint directory.

    Returns:
        Configured CurriculumManager.

    """
    config = create_default_curriculum(
        name=name,
        board_sizes=board_sizes,
        win_rate_threshold=win_rate_threshold,
        min_games_per_stage=min_games_per_stage,
    )

    return CurriculumManager(
        config=config,
        model_zoo=model_zoo,
        checkpoint_dir=checkpoint_dir,
    )
