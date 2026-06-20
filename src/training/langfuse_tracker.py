"""Langfuse experiment tracking for AlphaGalerkin training.

Replaces the former Weights & Biases logger. ``LangfuseTracker`` exposes the
same method surface the trainer already calls (``log_training_step``,
``log_evaluation``, ``log_buffer_stats``, ``log_metrics``, ``log_model_artifact``,
``log_summary``, ``log_config_update``, ``log_histogram``, ``log_table``,
``define_metric``, ``alert``, ``watch_model``, ``set_step_offset``, ``finish``)
so call sites change only at construction.

Langfuse is an LLM-observability platform (traces / spans / generations /
scores), not a training-metrics tracker, so the mapping is deliberate and
lossy:

- Scalar metrics → Langfuse ``score`` entries on a run-level trace, plus a
  per-call ``event`` carrying the full metric dict + step in metadata.
- Hyperparameters / config → trace metadata.
- ``log_model_artifact`` → an ``event`` recording the checkpoint path and
  metadata (Langfuse has **no artifact store**; nothing is uploaded).
- ``watch_model`` / ``log_histogram`` / ``log_table`` / ``define_metric`` →
  best-effort metadata events or documented no-ops (Langfuse has no
  gradient/histogram/table UI).

The tracker is a **graceful no-op when ``LANGFUSE_PUBLIC_KEY`` /
``LANGFUSE_SECRET_KEY`` are absent** (or ``enabled=False``), so CI and key-less
local runs work without a Langfuse server. The ``langfuse`` SDK is imported
lazily so unit tests can mock it and a misconfigured environment degrades
instead of crashing.

Shutdown safety: the ``atexit`` handler only flushes the client and **never
emits structlog events** — this is the explicit fix for the old W&B logger,
whose ``atexit``→``logger.info`` flooded CI logs with "I/O operation on closed
file" once stdout/stderr were closed.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from src.training.evaluation import EvaluationResult
    from src.training.trainer import TrainingMetrics

logger = structlog.get_logger(__name__)

# Default configuration values (centralized, no hardcoding).
DEFAULT_PROJECT = "alphagalerkin"
DEFAULT_LOG_INTERVAL = 10
DEFAULT_HOST = "https://cloud.langfuse.com"

# Environment variables that carry Langfuse credentials. Keys are never stored
# in YAML config — only sourced from the environment.
ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"
ENV_HOST = "LANGFUSE_HOST"
ENV_BASE_URL = "LANGFUSE_BASE_URL"  # alternate spelling used elsewhere in the repo


class LogLevel(str, Enum):
    """Alert levels (kept for API compatibility with the former W&B logger)."""

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class LangfuseTracker:
    """Langfuse-backed experiment tracker for AlphaGalerkin training.

    Thread-safe with graceful degradation: when credentials are missing or the
    ``langfuse`` SDK is unavailable, every method becomes a no-op and training
    continues unaffected.

    Example:
        >>> from config.schemas import LangfuseConfig
        >>> tracker = create_tracker(LangfuseConfig(enabled=True).model_dump())
        >>> tracker.log_metrics({"loss": 0.5}, step=1)
        >>> tracker.finish()

    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            config: Configuration dict with fields matching ``LangfuseConfig``.
            training_config: Full training configuration, recorded as trace metadata.

        """
        self._config = config or {}
        self._training_config = training_config or {}
        self._client: Any = None
        self._trace: Any = None
        self._langfuse: Any = None
        self._initialized = False
        self._step_offset = 0
        self._finished = False
        self._atexit_registered = False
        self._lock = threading.RLock()

        self._enabled = self._config.get("enabled", True)
        self._project = self._config.get("project", DEFAULT_PROJECT)
        self._run_name = self._config.get("run_name") or self._config.get("name")
        self._session_name = self._config.get("session_name")
        self._tags = self._safe_list(self._config.get("tags"))
        self._host = (
            self._config.get("host") or os.environ.get(ENV_HOST) or os.environ.get(ENV_BASE_URL)
        )
        self._log_model = self._config.get("log_model", True)
        self._log_interval = self._config.get("log_interval", DEFAULT_LOG_INTERVAL)

        if not self._enabled:
            logger.info("langfuse_disabled")
            return

        self._initialize()

    @staticmethod
    def _safe_list(value: Any) -> list[str]:
        """Coerce a value into a list of strings (None/invalid → empty list)."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        return []

    def _initialize(self) -> None:
        """Construct the Langfuse client + run-level trace, degrading gracefully."""
        public_key = os.environ.get(ENV_PUBLIC_KEY)
        secret_key = os.environ.get(ENV_SECRET_KEY)
        if not public_key or not secret_key:
            logger.info(
                "langfuse_disabled_no_credentials",
                message=f"Set {ENV_PUBLIC_KEY} and {ENV_SECRET_KEY} to enable Langfuse tracking.",
            )
            self._enabled = False
            return

        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning(
                "langfuse_import_failed",
                message="Install langfuse with: pip install 'langfuse>=2,<3'",
            )
            self._enabled = False
            return

        try:
            client_kwargs: dict[str, Any] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            if self._host:
                client_kwargs["host"] = self._host
            self._client = Langfuse(**client_kwargs)
            self._trace = self._client.trace(
                name=self._run_name or self._project,
                session_id=self._session_name,
                tags=self._tags or None,
                metadata={"project": self._project, "training_config": self._training_config},
            )
        except Exception as e:  # noqa: BLE001 - any SDK/transport failure degrades to no-op
            logger.warning(
                "langfuse_init_failed",
                error=str(e),
                error_type=type(e).__name__,
                message="Langfuse tracking disabled, training will continue without it",
            )
            self._enabled = False
            return

        self._initialized = True
        logger.info("langfuse_initialized", project=self._project, run_name=self._run_name)

        with self._lock:
            if not self._atexit_registered:
                atexit.register(self._atexit_flush)
                self._atexit_registered = True

    def _atexit_flush(self) -> None:
        """Flush the client at interpreter shutdown.

        Deliberately performs **no logging** — emitting structlog events here is
        what flooded CI logs with "I/O operation on closed file" under the old
        W&B logger once stdout/stderr were closed. All exceptions are swallowed.
        """
        client = self._client
        if client is None:
            return
        # No logging here — emitting structlog at interpreter shutdown is exactly
        # what flooded CI under the old W&B logger. Swallow everything.
        with contextlib.suppress(Exception):
            client.flush()

    @property
    def is_enabled(self) -> bool:
        """True when Langfuse is enabled, initialized, and not finished."""
        with self._lock:
            return self._enabled and self._initialized and not self._finished

    @property
    def run(self) -> Any:
        """The run-level Langfuse trace object, or None."""
        with self._lock:
            return self._trace

    @property
    def run_id(self) -> str | None:
        """The Langfuse trace id, or None."""
        with self._lock:
            return getattr(self._trace, "id", None) if self._trace else None

    @property
    def run_name(self) -> str | None:
        """The configured run name, or None."""
        return self._run_name

    def set_step_offset(self, offset: int) -> None:
        """Set the step offset for resumed training."""
        with self._lock:
            self._step_offset = offset
            logger.debug("langfuse_step_offset_set", offset=offset)

    def watch_model(self, model: Any) -> None:
        """No-op: Langfuse has no gradient/parameter watching (W&B-only feature)."""
        return

    def _apply_step_offset(self, step: int | None) -> int | None:
        if step is None:
            return None
        return step + self._step_offset

    def _record(
        self,
        name: str,
        data: dict[str, Any],
        step: int | None = None,
    ) -> None:
        """Record a metric dict: one trace event + a score per numeric value."""
        if not self.is_enabled or self._trace is None:
            return
        metadata = dict(data)
        if step is not None:
            metadata["step"] = step
        try:
            self._trace.event(name=name, metadata=metadata)
            for key, value in data.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    self._trace.score(name=key, value=float(value))
        except Exception as e:  # noqa: BLE001 - tracking is best-effort
            logger.debug("langfuse_record_failed", error=str(e), name=name)

    def log_training_step(self, metrics: TrainingMetrics, commit: bool = True) -> None:
        """Record training-step metrics (loss components, LR, buffer, perf)."""
        if not self.is_enabled:
            return
        step = metrics.step + self._step_offset
        if self._log_interval > 0 and step % self._log_interval != 0:
            return
        self._record(
            "train_step",
            {
                "train/loss/total": metrics.total_loss,
                "train/loss/policy": metrics.policy_loss,
                "train/loss/value": metrics.value_loss,
                "train/loss/lbb": metrics.lbb_loss,
                "train/lbb_constant": metrics.lbb_constant,
                "train/gradient_norm": metrics.gradient_norm,
                "train/learning_rate": metrics.learning_rate,
                "data/buffer_size": metrics.buffer_size,
                "data/games_generated": metrics.games_generated,
                "perf/step_time_ms": metrics.step_time_ms,
            },
            step=step,
        )

    def log_evaluation(
        self,
        result: EvaluationResult,
        prefix: str = "eval",
        step: int | None = None,
    ) -> None:
        """Record evaluation results."""
        if not self.is_enabled:
            return
        log_dict: dict[str, Any] = {
            f"{prefix}/win_rate": result.win_rate,
            f"{prefix}/n_games": result.n_games,
            f"{prefix}/wins": result.wins,
            f"{prefix}/losses": result.losses,
            f"{prefix}/draws": result.draws,
            f"{prefix}/avg_game_length": result.avg_game_length,
            f"{prefix}/avg_value_error": result.avg_value_error,
            f"{prefix}/policy_agreement": result.policy_agreement,
        }
        for key, value in result.metadata.items():
            if isinstance(value, int | float | str | bool):
                log_dict[f"{prefix}/meta/{key}"] = value
        self._record(prefix, log_dict, step=self._apply_step_offset(step))

    def log_buffer_stats(
        self,
        buffer_size: int,
        capacity: int,
        value_mean: float | None = None,
        value_std: float | None = None,
        board_size_distribution: dict[int, int] | None = None,
        step: int | None = None,
    ) -> None:
        """Record replay-buffer statistics."""
        if not self.is_enabled:
            return
        log_dict: dict[str, Any] = {
            "data/buffer_size": buffer_size,
            "data/buffer_fill_ratio": buffer_size / capacity if capacity > 0 else 0.0,
        }
        if value_mean is not None:
            log_dict["data/value_mean"] = value_mean
        if value_std is not None:
            log_dict["data/value_std"] = value_std
        if board_size_distribution:
            for size, count in board_size_distribution.items():
                log_dict[f"data/board_size_{size}x{size}"] = count
        self._record("buffer_stats", log_dict, step=self._apply_step_offset(step))

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Record arbitrary metrics."""
        if not self.is_enabled:
            return
        self._record("metrics", metrics, step=self._apply_step_offset(step))

    def log_histogram(self, key: str, values: Any, step: int | None = None) -> None:
        """No-op: Langfuse has no histogram UI (W&B-only feature)."""
        return

    def log_model_artifact(
        self,
        checkpoint_path: Path | str,
        name: str = "model",
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Record a checkpoint reference (path + metadata) — no upload.

        Langfuse has no artifact store, so the checkpoint is *not* uploaded; the
        path and metadata are recorded as a trace event for provenance.
        """
        if not self.is_enabled or not self._log_model or self._trace is None:
            return
        try:
            self._trace.event(
                name="model_artifact",
                metadata={
                    "name": name,
                    "path": str(checkpoint_path),
                    "aliases": list(aliases or []),
                    **(metadata or {}),
                },
            )
            logger.info("langfuse_artifact_recorded", name=name, path=str(checkpoint_path))
        except Exception as e:  # noqa: BLE001 - best-effort
            logger.debug("langfuse_artifact_failed", error=str(e))

    def log_config_update(self, config_updates: dict[str, Any]) -> None:
        """Merge configuration updates into the trace metadata."""
        if not self.is_enabled or self._trace is None:
            return
        try:
            self._trace.update(metadata={"config_update": config_updates})
        except Exception as e:  # noqa: BLE001 - best-effort
            logger.debug("langfuse_config_update_failed", error=str(e))

    def log_summary(self, summary: dict[str, Any]) -> None:
        """Record final summary metrics on the trace output."""
        if not self.is_enabled or self._trace is None:
            return
        try:
            self._trace.update(output=summary)
        except Exception as e:  # noqa: BLE001 - best-effort
            logger.debug("langfuse_summary_failed", error=str(e))

    def log_table(
        self,
        key: str,
        columns: list[str],
        data: list[list[Any]],
        step: int | None = None,
    ) -> None:
        """Record a table as a trace-event metadata payload (no table UI)."""
        if not self.is_enabled or self._trace is None:
            return
        try:
            self._trace.event(name=key, metadata={"columns": columns, "data": data})
        except Exception as e:  # noqa: BLE001 - best-effort
            logger.debug("langfuse_table_failed", error=str(e), key=key)

    def define_metric(
        self,
        name: str,
        step_metric: str = "trainer/global_step",
        summary: str | None = None,
        goal: str | None = None,
    ) -> None:
        """No-op: Langfuse has no custom-metric definition (W&B-only feature)."""
        return

    def alert(self, title: str, text: str, level: str = "INFO") -> None:
        """Record an alert as a trace event (Langfuse has no native alerts)."""
        if not self.is_enabled or self._trace is None:
            return
        try:
            self._trace.event(name="alert", metadata={"title": title, "text": text, "level": level})
        except Exception as e:  # noqa: BLE001 - best-effort
            logger.debug("langfuse_alert_failed", error=str(e), title=title)

    def finish(self) -> None:
        """Flush and finalize the run. Thread-safe and idempotent.

        Safe to call explicitly during training teardown (streams open). The
        ``atexit`` path uses :meth:`_atexit_flush`, which never logs.
        """
        with self._lock:
            if self._finished:
                return
            if self._client is not None:
                try:
                    self._client.flush()
                except Exception as e:  # noqa: BLE001 - best-effort
                    logger.debug("langfuse_flush_error", error=str(e))
                self._trace = None
                self._initialized = False
                logger.info("langfuse_finished")
            self._finished = True


def create_tracker(
    langfuse_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> LangfuseTracker:
    """Create a configured :class:`LangfuseTracker`.

    Args:
        langfuse_config: Configuration dict (a ``LangfuseConfig`` model dump).
        training_config: Full training configuration, recorded as trace metadata.

    Returns:
        A configured :class:`LangfuseTracker`.

    """
    return LangfuseTracker(config=langfuse_config, training_config=training_config)
