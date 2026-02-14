"""Versioned checkpoint system with migration support.

Implements checkpoint save/load with:
- Monotonically increasing version numbers for schema evolution.
- A migration registry that upgrades old checkpoints on load.
- Rotation of old checkpoints to bound disk usage.
- Best-model tracking based on a configurable metric.

Section reference (system prompt): Section 14.1 -- Checkpointing.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
import torch

from src.alphagalerkin.core.config import CheckpointConfig

logger = structlog.get_logger("training.checkpointing")

# -------------------------------------------------------------------
# Versioning
# -------------------------------------------------------------------

CURRENT_VERSION: int = 1
"""Current checkpoint schema version.

Increment this when the checkpoint payload changes in a way that
is not backward-compatible.  Add a corresponding migration function
to ``MIGRATIONS`` so that older checkpoints can be upgraded
transparently on load.
"""

MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    # Example migration from v0 -> v1:
    # 0: _migrate_v0_to_v1,
}
"""Registry of version migration functions.

Each key is the *source* version; the callable transforms the
checkpoint dict **in place** and returns it.  Migrations are
applied sequentially until ``CURRENT_VERSION`` is reached.
"""


def _apply_migrations(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Upgrade *checkpoint* to ``CURRENT_VERSION`` via migrations.

    Parameters
    ----------
    checkpoint:
        Raw checkpoint dict loaded from disk.

    Returns
    -------
    dict[str, Any]
        The same dict, mutated in place, at ``CURRENT_VERSION``.

    Raises
    ------
    ValueError
        If no migration path exists from the stored version.

    """
    version = checkpoint.get("version", 0)

    while version < CURRENT_VERSION:
        if version not in MIGRATIONS:
            msg = (
                f"No migration from checkpoint version "
                f"{version} to {version + 1}. "
                f"Current version is {CURRENT_VERSION}."
            )
            raise ValueError(msg)

        logger.info(
            "checkpoint.migrating",
            from_version=version,
            to_version=version + 1,
        )
        checkpoint = MIGRATIONS[version](checkpoint)
        version += 1
        checkpoint["version"] = version

    return checkpoint


# -------------------------------------------------------------------
# Checkpoint file naming
# -------------------------------------------------------------------

_CHECKPOINT_PATTERN = re.compile(
    r"^checkpoint_(\d+)\.pt$",
)
"""Regex matching ``checkpoint_<iteration>.pt`` filenames."""

_BEST_FILENAME = "best_model.pt"


# -------------------------------------------------------------------
# CheckpointManager
# -------------------------------------------------------------------

class CheckpointManager:
    """Manages saving, loading, and rotating checkpoints.

    Parameters
    ----------
    config:
        Checkpoint configuration controlling directory paths,
        rotation policy, and best-model tracking.

    """

    def __init__(self, config: CheckpointConfig) -> None:
        self._config = config
        self._dir = Path(config.checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._best_metric_value: float | None = None

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    def save(
        self,
        iteration: int,
        network_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        replay_buffer_state: dict[str, Any] | None = None,
        training_metrics: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a training checkpoint to disk.

        Parameters
        ----------
        iteration:
            Current training iteration (used in the filename).
        network_state:
            ``model.state_dict()`` payload.
        optimizer_state:
            ``optimizer.state_dict()`` payload.
        replay_buffer_state:
            Optional replay buffer serialization.
        training_metrics:
            Optional metrics history dict.
        extra:
            Any additional data to persist.

        Returns
        -------
        Path
            Absolute path to the saved checkpoint file.

        """
        payload: dict[str, Any] = {
            "version": CURRENT_VERSION,
            "iteration": iteration,
            "network_state_dict": network_state,
            "optimizer_state_dict": optimizer_state,
        }

        if replay_buffer_state is not None:
            payload["replay_buffer_state"] = (
                replay_buffer_state
            )
        if training_metrics is not None:
            payload["training_metrics"] = training_metrics
        if extra is not None:
            payload["extra"] = extra

        filename = f"checkpoint_{iteration:08d}.pt"
        path = self._dir / filename
        torch.save(payload, path)

        logger.info(
            "checkpoint.saved",
            path=str(path),
            iteration=iteration,
            version=CURRENT_VERSION,
        )

        self._cleanup_old_checkpoints()
        return path

    # ---------------------------------------------------------------
    # Best model tracking
    # ---------------------------------------------------------------

    def save_best(
        self,
        metric_value: float,
        network_state: dict[str, Any],
        iteration: int,
    ) -> Path | None:
        """Conditionally save the best model checkpoint.

        Compares *metric_value* against the running best using
        the configured ``best_metric_mode`` (``"min"`` or
        ``"max"``).  Saves only when improvement is detected.

        Parameters
        ----------
        metric_value:
            The metric value for the current checkpoint.
        network_state:
            ``model.state_dict()`` payload.
        iteration:
            Current training iteration.

        Returns
        -------
        Path | None
            Path to the saved best model, or ``None`` if the
            current model is not the best.

        """
        if not self._config.save_best:
            return None

        is_better = self._is_improvement(metric_value)
        if not is_better:
            return None

        self._best_metric_value = metric_value
        path = self._dir / _BEST_FILENAME

        payload: dict[str, Any] = {
            "version": CURRENT_VERSION,
            "iteration": iteration,
            "network_state_dict": network_state,
            "best_metric": self._config.best_metric,
            "best_metric_value": metric_value,
        }
        torch.save(payload, path)

        logger.info(
            "checkpoint.best_saved",
            path=str(path),
            metric=self._config.best_metric,
            value=metric_value,
            iteration=iteration,
        )
        return path

    def _is_improvement(self, value: float) -> bool:
        """Check whether *value* improves on the running best."""
        if self._best_metric_value is None:
            return True
        if self._config.best_metric_mode == "min":
            return value < self._best_metric_value
        return value > self._best_metric_value

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    def load(
        self,
        path: Path | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint from disk, applying migrations.

        Parameters
        ----------
        path:
            Explicit checkpoint path.  When ``None``, loads the
            latest checkpoint from ``checkpoint_dir``.

        Returns
        -------
        dict[str, Any]
            Migrated checkpoint payload at ``CURRENT_VERSION``.

        Raises
        ------
        FileNotFoundError
            If no checkpoint file is found.

        """
        if path is None:
            path = self._latest_checkpoint()
            if path is None:
                msg = (
                    f"No checkpoints found in "
                    f"{self._dir}"
                )
                raise FileNotFoundError(msg)

        checkpoint: dict[str, Any] = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        stored_version = checkpoint.get("version", 0)
        if stored_version < CURRENT_VERSION:
            checkpoint = _apply_migrations(checkpoint)

        logger.info(
            "checkpoint.loaded",
            path=str(path),
            iteration=checkpoint.get("iteration", -1),
            version=checkpoint.get("version", 0),
        )
        return checkpoint

    # ---------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------

    def _latest_checkpoint(self) -> Path | None:
        """Find the most recent checkpoint by iteration number.

        Returns
        -------
        Path | None
            Path to the latest checkpoint, or ``None`` if the
            checkpoint directory contains no matching files.

        """
        best_iter = -1
        best_path: Path | None = None

        for candidate in self._dir.iterdir():
            match = _CHECKPOINT_PATTERN.match(
                candidate.name,
            )
            if match:
                iteration = int(match.group(1))
                if iteration > best_iter:
                    best_iter = iteration
                    best_path = candidate

        return best_path

    def list_checkpoints(self) -> list[Path]:
        """Return all checkpoint paths sorted by iteration."""
        checkpoints: list[tuple[int, Path]] = []
        for candidate in self._dir.iterdir():
            match = _CHECKPOINT_PATTERN.match(
                candidate.name,
            )
            if match:
                iteration = int(match.group(1))
                checkpoints.append((iteration, candidate))

        checkpoints.sort(key=lambda t: t[0])
        return [p for _, p in checkpoints]

    # ---------------------------------------------------------------
    # Rotation
    # ---------------------------------------------------------------

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping ``keep_last_n``.

        The best-model checkpoint is never deleted.
        """
        all_ckpts = self.list_checkpoints()
        keep_n = self._config.keep_last_n

        if len(all_ckpts) <= keep_n:
            return

        to_remove = all_ckpts[: len(all_ckpts) - keep_n]
        for old_path in to_remove:
            if old_path.name == _BEST_FILENAME:
                continue
            old_path.unlink(missing_ok=True)
            logger.debug(
                "checkpoint.removed",
                path=str(old_path),
            )
