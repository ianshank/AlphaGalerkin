"""Tests for preemption handling."""

from __future__ import annotations

import signal
import time
from unittest.mock import MagicMock

import pytest

from src.vertex.preemption import (
    SPOT_CHECKPOINT_INTERVAL,
    PreemptionEvent,
    PreemptionHandler,
    PreemptionMonitor,
    create_preemption_handler,
)


class TestPreemptionEvent:
    """Tests for PreemptionEvent."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        event = PreemptionEvent(
            timestamp="2026-01-01T00:00:00",
            signal="SIGTERM",
            checkpoint_saved=True,
            step_at_preemption=1000,
        )
        d = event.to_dict()
        assert d["timestamp"] == "2026-01-01T00:00:00"
        assert d["signal"] == "SIGTERM"
        assert d["checkpoint_saved"] is True
        assert d["step_at_preemption"] == 1000


class TestPreemptionHandler:
    """Tests for PreemptionHandler."""

    @pytest.fixture
    def handler(self) -> PreemptionHandler:
        """Create handler instance."""
        handler = PreemptionHandler(
            checkpoint_callback=None,
            checkpoint_interval=100,
            enable_spot_mode=False,
        )
        yield handler
        handler.cleanup()

    @pytest.fixture
    def spot_handler(self) -> PreemptionHandler:
        """Create handler in spot mode."""
        handler = PreemptionHandler(
            checkpoint_callback=None,
            checkpoint_interval=500,
            enable_spot_mode=True,
        )
        yield handler
        handler.cleanup()

    def test_initialization(self, handler: PreemptionHandler) -> None:
        """Test handler initialization."""
        assert handler.is_preempted is False
        assert handler.preemption_event is None

    def test_initialization_spot_mode(self, spot_handler: PreemptionHandler) -> None:
        """Test handler initialization in spot mode."""
        assert spot_handler._checkpoint_interval == SPOT_CHECKPOINT_INTERVAL

    def test_should_save_checkpoint(self, handler: PreemptionHandler) -> None:
        """Test checkpoint interval checking."""
        assert handler.should_save_checkpoint(0) is False
        assert handler.should_save_checkpoint(50) is False
        assert handler.should_save_checkpoint(100) is True
        assert handler.should_save_checkpoint(200) is True

    def test_update_step(self, handler: PreemptionHandler) -> None:
        """Test step update."""
        handler.update_step(500)
        assert handler._current_step == 500

    def test_reset(self, handler: PreemptionHandler) -> None:
        """Test handler reset."""
        handler.update_step(500)
        handler._preempted.set()
        handler._preemption_event = PreemptionEvent(
            timestamp="test",
            signal="SIGTERM",
        )

        handler.reset()

        assert handler.is_preempted is False
        assert handler.preemption_event is None
        assert handler._current_step == 0

    def test_checkpoint_callback_called(self) -> None:
        """Test checkpoint callback is called on preemption."""
        callback = MagicMock()
        handler = PreemptionHandler(
            checkpoint_callback=callback,
            checkpoint_interval=100,
        )

        try:
            # Simulate preemption signal (directly call handler)
            handler._handle_preemption(signal.SIGTERM, None)

            assert handler.is_preempted is True
            callback.assert_called_once()
            assert handler.preemption_event is not None
            assert handler.preemption_event.checkpoint_saved is True
        finally:
            handler.cleanup()

    def test_checkpoint_callback_failure(self) -> None:
        """Test handling of checkpoint callback failure."""
        callback = MagicMock(side_effect=Exception("Save failed"))
        handler = PreemptionHandler(
            checkpoint_callback=callback,
            checkpoint_interval=100,
        )

        try:
            handler._handle_preemption(signal.SIGTERM, None)

            assert handler.is_preempted is True
            assert handler.preemption_event is not None
            assert handler.preemption_event.checkpoint_saved is False
        finally:
            handler.cleanup()

    def test_cleanup_restores_signals(self, handler: PreemptionHandler) -> None:
        """Test cleanup restores original signal handlers."""
        original_sigterm = handler._original_sigterm
        handler.cleanup()
        assert signal.getsignal(signal.SIGTERM) == original_sigterm


class TestPreemptionMonitor:
    """Tests for PreemptionMonitor."""

    def test_initialization(self) -> None:
        """Test monitor initialization."""
        monitor = PreemptionMonitor(check_interval=1.0)
        assert monitor.is_preempted is False

    def test_start_stop(self) -> None:
        """Test starting and stopping monitor."""
        monitor = PreemptionMonitor(check_interval=0.1)
        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        monitor.stop()
        time.sleep(0.2)
        assert not monitor._stop_event.is_set() or not monitor._thread.is_alive()

    def test_does_not_double_start(self) -> None:
        """Test monitor doesn't create multiple threads."""
        monitor = PreemptionMonitor(check_interval=0.1)
        monitor.start()
        thread1 = monitor._thread
        monitor.start()  # Should not create new thread
        assert monitor._thread is thread1
        monitor.stop()


class TestCreatePreemptionHandler:
    """Tests for create_preemption_handler factory."""

    def test_creates_handler(self) -> None:
        """Test factory creates handler."""
        handler = create_preemption_handler(
            checkpoint_callback=None,
            enable_spot=True,
            checkpoint_interval=200,
        )
        try:
            assert isinstance(handler, PreemptionHandler)
            assert handler._enable_spot_mode is True
        finally:
            handler.cleanup()

    def test_spot_mode_adjusts_interval(self) -> None:
        """Test spot mode uses more aggressive interval."""
        handler = create_preemption_handler(
            enable_spot=True,
            checkpoint_interval=1000,  # Large interval
        )
        try:
            # Should be capped at SPOT_CHECKPOINT_INTERVAL
            assert handler._checkpoint_interval == SPOT_CHECKPOINT_INTERVAL
        finally:
            handler.cleanup()
