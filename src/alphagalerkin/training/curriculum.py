"""Progressive difficulty scheduling for training."""

from __future__ import annotations

from typing import Any

import structlog

from src.alphagalerkin.core.config import CurriculumConfig

logger = structlog.get_logger("training.curriculum")


class CurriculumManager:
    """Manages progressive difficulty for self-play training.

    The curriculum consists of an ordered list of *stages*, each
    represented as a dict of environment-config overrides.  The
    manager advances to the next stage once the agent's recent
    win rate exceeds ``advance_threshold`` over a sliding window
    of ``evaluation_window`` episodes.

    Parameters
    ----------
    config:
        Curriculum configuration specifying stages, thresholds,
        and evaluation window size.

    """

    def __init__(self, config: CurriculumConfig) -> None:
        self._config = config
        self._current_stage_idx: int = 0
        self._win_rate_history: list[float] = []

    # ---------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------

    @property
    def current_stage_index(self) -> int:
        """Zero-based index of the current curriculum stage."""
        return self._current_stage_idx

    @property
    def is_at_final_stage(self) -> bool:
        """Whether the curriculum has reached the last stage."""
        if not self._config.stages:
            return True
        return self._current_stage_idx >= len(self._config.stages) - 1

    @property
    def num_stages(self) -> int:
        """Total number of curriculum stages."""
        return max(1, len(self._config.stages))

    # ---------------------------------------------------------------
    # Update logic
    # ---------------------------------------------------------------

    def update(self, win_rate: float) -> bool:
        """Update curriculum based on recent performance.

        Records the win rate, and if the running average over
        the evaluation window exceeds the advancement threshold,
        advances to the next stage.

        Parameters
        ----------
        win_rate:
            Win-rate metric from the latest evaluation
            (in ``[0, 1]``).

        Returns
        -------
        bool
            ``True`` if the curriculum advanced to a new stage.

        """
        if not self._config.enabled:
            return False

        self._win_rate_history.append(win_rate)

        # Compute windowed average
        window = self._config.evaluation_window
        recent = self._win_rate_history[-window:]
        avg_win_rate = sum(recent) / len(recent)

        if avg_win_rate >= self._config.advance_threshold and not self.is_at_final_stage:
            old_stage = self._current_stage_idx
            self._current_stage_idx += 1
            logger.info(
                "curriculum.advanced",
                old_stage=old_stage,
                new_stage=self._current_stage_idx,
                avg_win_rate=round(avg_win_rate, 4),
                threshold=self._config.advance_threshold,
            )
            return True

        return False

    # ---------------------------------------------------------------
    # Stage queries
    # ---------------------------------------------------------------

    def get_stage_overrides(self) -> dict[str, Any]:
        """Return environment-config overrides for the current stage.

        Returns an empty dict when curriculum is disabled or
        when no stages are defined.
        """
        if not self._config.enabled or not self._config.stages:
            return {}

        idx = min(
            self._current_stage_idx,
            len(self._config.stages) - 1,
        )
        return dict(self._config.stages[idx])

    def reset(self) -> None:
        """Reset the curriculum to the first stage."""
        self._current_stage_idx = 0
        self._win_rate_history.clear()
        logger.info("curriculum.reset")
