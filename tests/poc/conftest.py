"""Shared fixtures for ``tests/poc/``.

Both fixtures here guard the same failure shape: a process-wide singleton that
a test in this package mutates without restoring, silently changing behaviour
for whichever test file pytest collects next.

``ScenarioRegistry`` (``src/poc/registry.py``) is cleared by local ``autouse``
fixtures in several modules to get a clean slate per test. None of them restore
what they cleared, so registrations made by one file (including the
``@scenario``-decorated built-ins registered at import time) can leak into — or
vanish from — an unrelated file depending on collection order.
``test_charter_alignment.py`` documents this exact hazard and works around it by
reading ``ScenarioRegistry().list_scenarios()`` in a subprocess.

``structlog``'s global configuration is the same problem one layer down:
``test_logging.py`` calls ``configure_logging()`` repeatedly (ending at
``level="ERROR", json_format=True``) with no teardown. That swaps the logger
factory to ``structlog.stdlib.LoggerFactory``, routing every subsequent
``logger.warning(...)`` in the process into stdlib logging, where pytest's
capture handler swallows it. The observable effect is that warnings such as
``vram_probe_failed`` / ``interpolator_build_failed`` render fine when their
test file runs alone and vanish in a full run — so a ``caplog`` assertion on
them would pass or fail purely by collection order.

Neither fixture changes any test's own behaviour; each wraps the package so
that whatever a test does to the singleton, the next file starts where this
one did.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog

from src.poc.registry import ScenarioRegistry


@pytest.fixture(autouse=True)
def _snapshot_scenario_registry() -> Iterator[None]:
    """Snapshot the scenario registry before each test, restore it after.

    Conftest-level autouse fixtures set up before same-scope fixtures defined
    in the test module and tear down after them, so this wraps every local
    ``clean_registry``/``_isolate_registry`` fixture in ``tests/poc/`` without
    requiring changes to them.
    """
    registry = ScenarioRegistry()
    snapshot = registry.get_all()
    yield
    registry.clear()
    for name, scenario_cls in snapshot.items():
        registry.register(name, scenario_cls)


@pytest.fixture(autouse=True)
def _snapshot_structlog_config() -> Iterator[None]:
    """Restore structlog's global configuration after each test.

    ``structlog.get_config()`` returns the live config dict; re-applying it
    through ``structlog.configure(**config)`` restores the exact processor
    chain, wrapper class, logger factory and cache setting. When a test never
    touches the configuration this is a no-op round-trip.
    """
    saved = structlog.get_config().copy()
    was_configured = structlog.is_configured()
    yield
    if was_configured:
        structlog.configure(**saved)
    else:
        structlog.reset_defaults()
