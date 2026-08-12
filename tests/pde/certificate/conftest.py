"""Shared fixtures for the certificate foundation surface.

The stability registry is a *process-global* singleton — tests that mutate it
(e.g. register-duplicate raises, replace() escape hatch) must reset it between
runs. This ``autouse`` fixture reruns the built-in registration block after
each test, so mutation tests do not leak state into siblings.
"""

from __future__ import annotations

import pytest

from src.pde.certificate.stability import (
    StabilityConstantRegistry,
    _register_builtin_stability_entries,
)


@pytest.fixture(autouse=True)
def _reset_stability_registry() -> None:
    """Drop the singleton, then re-populate the built-in entries.

    Reset happens *before* the test so a test can inspect a fresh registry;
    tests that need an empty registry can call ``_reset_for_tests()`` again.
    """
    StabilityConstantRegistry._reset_for_tests()
    _register_builtin_stability_entries()
