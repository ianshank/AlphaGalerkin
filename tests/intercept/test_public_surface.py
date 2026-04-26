"""Smoke test for :mod:`src.intercept` public surface.

PR #53 review (Copilot, ``src/intercept/__init__.py:55``) flagged that
``__all__`` declared 27 symbols that the package neither imported nor
exposed via ``__getattr__``. This test imports every name listed in
``__all__`` and asserts that the lazy ``__getattr__`` resolves it
to the same object that the submodule defines.
"""

from __future__ import annotations

import importlib

import pytest

import src.intercept as intercept_pkg


@pytest.mark.parametrize("name", intercept_pkg.__all__)
def test_all_symbols_resolve(name: str) -> None:
    """Every entry in ``__all__`` must resolve via ``__getattr__``."""
    obj = getattr(intercept_pkg, name)
    assert obj is not None, f"src.intercept.{name} resolved to None"


def test_lazy_attr_matches_submodule() -> None:
    """The lazy resolver must return the canonical submodule object."""
    from src.intercept import (
        AeroModel,
        ProportionalNavigation,
        RigidBody6DOF,
    )

    aero = importlib.import_module("src.intercept.aero")
    guidance = importlib.import_module("src.intercept.guidance")
    dynamics = importlib.import_module("src.intercept.dynamics")
    assert AeroModel is aero.AeroModel
    assert ProportionalNavigation is guidance.ProportionalNavigation
    assert RigidBody6DOF is dynamics.RigidBody6DOF


def test_unknown_attr_raises_attribute_error() -> None:
    """Lazy ``__getattr__`` must raise AttributeError for unknown names."""
    with pytest.raises(AttributeError, match="DoesNotExist"):
        _ = intercept_pkg.DoesNotExist  # type: ignore[attr-defined]


def test_dir_includes_all_symbols() -> None:
    """``dir(src.intercept)`` should expose every ``__all__`` entry."""
    available = set(dir(intercept_pkg))
    missing = set(intercept_pkg.__all__) - available
    assert not missing, f"missing from dir(): {missing}"
