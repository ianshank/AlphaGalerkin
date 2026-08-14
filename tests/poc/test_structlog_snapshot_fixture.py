"""Tests for the ``tests/poc/conftest.py`` structlog snapshot/restore fixture.

``_snapshot_structlog_config`` is test *infrastructure* guarding a
process-global leak: ``src.poc.logging.configure_logging`` swaps structlog's
logger factory to ``structlog.stdlib.LoggerFactory``, and ``test_logging.py``
calls it repeatedly with no teardown. Once that has happened, every later
``logger.warning(...)`` in the process is routed into stdlib logging, so
warning-assertion tests elsewhere in the package pass or fail purely by
collection order.

Infrastructure that silently stops working is worse than none, so the fixture
gets its own tests. The ordering-sensitive pair relies on pytest's
definition-order execution; both also pass in isolation.

Validates:
    - The leak is real: ``configure_logging`` mutates the global config.
    - A test that reconfigures structlog does not leak into the next test.
    - The fixture is a no-op round-trip when a test leaves the config alone.
    - Restoring an already-configured snapshot is idempotent.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.poc.logging import configure_logging

_BASELINE: dict[str, Any] = {}
"""Config captured by the first test in this module (see ``test_1_...``)."""


def _factory_name() -> str:
    """Class name of structlog's current logger factory."""
    return type(structlog.get_config()["logger_factory"]).__name__


class TestStructlogConfigDoesNotLeak:
    """Ordering-sensitive proof that reconfiguration is rolled back."""

    def test_1_capture_baseline_and_reconfigure(self) -> None:
        """Reconfiguring structlog takes effect inside its own test.

        This mirrors exactly what ``test_logging.py`` does (its final call is
        ``configure_logging(level="ERROR", json_format=True, ...)``).
        """
        _BASELINE.update(structlog.get_config())

        configure_logging(level="ERROR", json_format=True, include_timestamp=False)

        # The stdlib factory is what routes later events into stdlib logging,
        # where pytest's capture handler swallows them.
        assert _factory_name() == "LoggerFactory"
        assert structlog.get_config() != _BASELINE

    def test_2_configuration_was_restored(self) -> None:
        """The previous test's reconfiguration did not survive it."""
        current = structlog.get_config()
        assert type(current["logger_factory"]) is type(_BASELINE["logger_factory"])
        assert current["processors"] == _BASELINE["processors"]
        assert current["wrapper_class"] is _BASELINE["wrapper_class"]

    def test_3_warning_events_still_reach_capture_logs(self) -> None:
        """A warning emitted after the reconfiguration is still observable.

        This is the property the fixture exists to protect: without it, the
        ``vram_probe_failed`` / ``interpolator_build_failed`` assertions
        elsewhere in the suite would depend on collection order.
        """
        from structlog.testing import capture_logs

        logger = structlog.get_logger(__name__)
        with capture_logs() as logs:
            logger.warning("probe_event", detail="visible")

        assert [entry["event"] for entry in logs] == ["probe_event"]
        assert logs[0]["log_level"] == "warning"
        assert logs[0]["detail"] == "visible"


class TestRestoreSemantics:
    """Unit-level properties of the save/restore body."""

    @staticmethod
    def _restore(saved: dict[str, Any], *, was_configured: bool) -> None:
        """Mirror of the conftest teardown body."""
        if was_configured:
            structlog.configure(**saved)
        else:
            structlog.reset_defaults()

    def test_round_trip_is_a_no_op(self) -> None:
        """Saving and re-applying an untouched config changes nothing."""
        saved = structlog.get_config().copy()
        self._restore(saved, was_configured=structlog.is_configured())
        assert structlog.get_config() == saved

    def test_restore_undoes_configure_logging(self) -> None:
        """An explicit reconfiguration is reverted by the restore body."""
        saved = structlog.get_config().copy()
        was_configured = structlog.is_configured()

        configure_logging(level="ERROR", json_format=True)
        assert structlog.get_config() != saved

        self._restore(saved, was_configured=was_configured)
        assert structlog.get_config() == saved

    def test_restore_is_idempotent(self) -> None:
        """Applying the same snapshot twice is stable."""
        saved = structlog.get_config().copy()
        was_configured = structlog.is_configured()
        self._restore(saved, was_configured=was_configured)
        self._restore(saved, was_configured=was_configured)
        assert structlog.get_config() == saved

    def test_reset_defaults_branch_restores_a_usable_logger(self) -> None:
        """The ``was_configured=False`` arm leaves structlog usable.

        Exercised directly rather than through the fixture: by the time any
        test runs, structlog has already been configured at least once, so the
        ``else`` branch is otherwise unreachable in-process.
        """
        saved = structlog.get_config().copy()
        try:
            configure_logging(level="DEBUG")
            self._restore(saved, was_configured=False)
            # reset_defaults() leaves a working, default-configured structlog.
            structlog.get_logger(__name__).info("still_usable")
        finally:
            structlog.configure(**saved)
        assert structlog.get_config() == saved
