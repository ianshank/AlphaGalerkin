"""Shared fixtures for the ``tests/alphagalerkin/`` package."""

from __future__ import annotations

import pytest

from src.alphagalerkin.solver import _resolve_device_cached


@pytest.fixture(autouse=True)
def _clear_device_cache() -> None:
    """Reset the module-level ``_resolve_device_cached`` LRU between tests.

    The cache is process-wide and would otherwise leak state across tests
    that toggle ``torch.cuda.is_available`` (notably the device-resolution
    suite). ``autouse=True`` so every test in the package starts from a
    clean cache; promoting from a single test file to package conftest
    avoids the known flake hazard where a test in another module sees a
    pre-populated entry and silently observes no warning.
    """
    _resolve_device_cached.cache_clear()
