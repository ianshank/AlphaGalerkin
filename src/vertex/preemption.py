"""Spot instance preemption handling for Vertex AI.

This module provides utilities for handling preemption events when
using Vertex AI spot instances, enabling graceful checkpoint saving
and job resumption.

Vertex AI sends SIGTERM when preempting spot instances, giving the
container a brief window to save state before termination.

Example:
    from src.vertex.preemption import PreemptionHandler

    handler = PreemptionHandler(
        checkpoint_callback=save_checkpoint,
        checkpoint_interval=100,
    )

    for step in range(total_steps):
        if handler.is_preempted:
            break

        train_step()

        if handler.should_save_checkpoint(step):
            save_checkpoint(step)

"""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Default checkpoint intervals
DEFAULT_CHECKPOINT_INTERVAL = 500  # Steps between checkpoints
SPOT_CHECKPOINT_INTERVAL = 100  # More aggressive for spot instances

# Grace period after preemption signal (seconds)
PREEMPTION_GRACE_PERIOD = 30.0


@dataclass
class PreemptionEvent:
    """Record of a preemption event.

    Attributes:
        timestamp: When preemption was detected.
        signal: Signal that triggered preemption.
        checkpoint_saved: Whether emergency checkpoint was saved.
        step_at_preemption: Training step when preempted.

    """

    timestamp: str
    signal: str
    checkpoint_saved: bool = False
    step_at_preemption: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "signal": self.signal,
            "checkpoint_saved": self.checkpoint_saved,
            "step_at_preemption": self.step_at_preemption,
        }


class PreemptionHandler:
    """Handle Vertex AI spot instance preemption.

    This handler monitors for preemption signals (SIGTERM) and
    triggers emergency checkpoint saving when preemption is detected.

    Features:
    - Signal-based preemption detection
    - Emergency checkpoint callback
    - Configurable checkpoint intervals
    - Preemption event logging

    Example:
        handler = PreemptionHandler(
            checkpoint_callback=lambda: save_checkpoint(current_step),
            checkpoint_interval=100,
        )

        # Check periodically
        if handler.is_preempted:
            logger.warning("Preemption detected, shutting down")
            break

    """

    def __init__(
        self,
        checkpoint_callback: Callable[[], None] | None = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        enable_spot_mode: bool = True,
    ) -> None:
        """Initialize preemption handler.

        Args:
            checkpoint_callback: Function to call for emergency checkpoint.
            checkpoint_interval: Steps between regular checkpoints.
            enable_spot_mode: Use more aggressive checkpointing for spot.

        """
        self._checkpoint_callback = checkpoint_callback
        self._base_checkpoint_interval = checkpoint_interval
        self._enable_spot_mode = enable_spot_mode

        # Preemption state
        self._preempted = threading.Event()
        self._preemption_event: PreemptionEvent | None = None
        self._current_step = 0
        self._lock = threading.Lock()

        # Calculate effective interval
        if enable_spot_mode:
            self._checkpoint_interval = min(
                checkpoint_interval,
                SPOT_CHECKPOINT_INTERVAL,
            )
        else:
            self._checkpoint_interval = checkpoint_interval

        # Register signal handlers
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._setup_signal_handlers()

        logger.info(
            "preemption_handler_initialized",
            checkpoint_interval=self._checkpoint_interval,
            spot_mode=enable_spot_mode,
        )

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for preemption detection."""
        signal.signal(signal.SIGTERM, self._handle_preemption)
        # Also handle SIGINT for graceful Ctrl+C
        signal.signal(signal.SIGINT, self._handle_preemption)

    def _handle_preemption(self, signum: int, frame: Any) -> None:
        """Handle preemption signal.

        Args:
            signum: Signal number.
            frame: Current stack frame.

        """
        signal_name = signal.Signals(signum).name

        with self._lock:
            if self._preempted.is_set():
                # Already handled
                return

            logger.warning(
                "preemption_signal_received",
                signal=signal_name,
                step=self._current_step,
            )

            self._preemption_event = PreemptionEvent(
                timestamp=datetime.now().isoformat(),
                signal=signal_name,
                step_at_preemption=self._current_step,
            )

            self._preempted.set()
            self._on_preemption()

    def _on_preemption(self) -> None:
        """Handle preemption event."""
        if self._checkpoint_callback is not None:
            logger.info("saving_emergency_checkpoint", step=self._current_step)
            try:
                self._checkpoint_callback()
                if self._preemption_event:
                    self._preemption_event.checkpoint_saved = True
                logger.info("emergency_checkpoint_saved")
            except Exception as e:
                logger.error("emergency_checkpoint_failed", error=str(e))

    @property
    def is_preempted(self) -> bool:
        """Check if preemption signal was received.

        Returns:
            True if preemption was detected.

        """
        return self._preempted.is_set()

    @property
    def preemption_event(self) -> PreemptionEvent | None:
        """Get the preemption event if one occurred.

        Returns:
            PreemptionEvent or None.

        """
        return self._preemption_event

    def update_step(self, step: int) -> None:
        """Update current training step.

        Args:
            step: Current training step.

        """
        with self._lock:
            self._current_step = step

    def should_save_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step.

        This uses more aggressive checkpointing when in spot mode.

        Args:
            step: Current training step.

        Returns:
            True if checkpoint should be saved.

        """
        self.update_step(step)
        return step > 0 and step % self._checkpoint_interval == 0

    def reset(self) -> None:
        """Reset handler state.

        This clears the preemption flag and event, allowing the
        handler to be reused (e.g., after restart).
        """
        with self._lock:
            self._preempted.clear()
            self._preemption_event = None
            self._current_step = 0

    def cleanup(self) -> None:
        """Restore original signal handlers."""
        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)
        logger.debug("signal_handlers_restored")


class PreemptionMonitor:
    """Background thread for monitoring preemption status.

    This monitor can be used to periodically check for preemption
    conditions that may not be signaled (e.g., resource quotas).

    Example:
        monitor = PreemptionMonitor(check_interval=10.0)
        monitor.start()

        try:
            while not monitor.is_preempted:
                train_step()
        finally:
            monitor.stop()

    """

    def __init__(
        self,
        check_interval: float = 10.0,
        preemption_file: str | None = None,
    ) -> None:
        """Initialize preemption monitor.

        Args:
            check_interval: Seconds between checks.
            preemption_file: Optional file to check for preemption.

        """
        self._check_interval = check_interval
        self._preemption_file = preemption_file
        self._preempted = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start background monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="preemption-monitor",
        )
        self._thread.start()
        logger.debug("preemption_monitor_started")

    def stop(self) -> None:
        """Stop monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.debug("preemption_monitor_stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                if self._check_preemption():
                    self._preempted.set()
                    logger.warning("preemption_detected_by_monitor")
                    break
            except Exception as e:
                logger.debug("monitor_check_error", error=str(e))

            self._stop_event.wait(timeout=self._check_interval)

    def _check_preemption(self) -> bool:
        """Check for preemption conditions.

        Returns:
            True if preemption detected.

        """
        # Check for preemption file
        if self._preemption_file and os.path.exists(self._preemption_file):
            return True

        # Check for Vertex AI preemption marker
        # Vertex AI may set environment variables or files
        return os.environ.get("VERTEX_PREEMPTION") == "true"

    @property
    def is_preempted(self) -> bool:
        """Check if preemption was detected.

        Returns:
            True if preempted.

        """
        return self._preempted.is_set()


def create_preemption_handler(
    checkpoint_callback: Callable[[], None] | None = None,
    enable_spot: bool = True,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
) -> PreemptionHandler:
    """Factory function to create preemption handler.

    Args:
        checkpoint_callback: Emergency checkpoint function.
        enable_spot: Enable spot instance mode.
        checkpoint_interval: Steps between checkpoints.

    Returns:
        Configured PreemptionHandler.

    """
    return PreemptionHandler(
        checkpoint_callback=checkpoint_callback,
        checkpoint_interval=checkpoint_interval,
        enable_spot_mode=enable_spot,
    )
