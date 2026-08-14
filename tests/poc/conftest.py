"""Shared fixtures for ``tests/poc/``.

``ScenarioRegistry`` (``src/poc/registry.py``) is a process-wide singleton, and
several modules in this package clear it via local ``autouse`` fixtures to get a
clean slate per test. None of those local fixtures restore what they cleared,
so registrations made by one test file (including the ``@scenario``-decorated
built-ins registered at import time) can leak into — or vanish from — an
unrelated test file depending on collection order. ``test_charter_alignment.py``
documents this exact hazard and works around it by reading
``ScenarioRegistry().list_scenarios()`` in a subprocess instead of in-process.

This fixture does not change any test file's local clear/re-register
behaviour; it wraps the whole package in a snapshot/restore so that whatever a
test does to the registry, the *next* test file starts from the same state
this one did.
"""

from __future__ import annotations

import pytest

from src.poc.registry import ScenarioRegistry


@pytest.fixture(autouse=True)
def _snapshot_scenario_registry() -> None:
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
