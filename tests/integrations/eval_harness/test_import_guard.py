"""The lazy harness importer fails with a helpful message when absent (CPU)."""

from __future__ import annotations

import importlib

import pytest

from src.integrations.eval_harness import _import_harness


def test_import_eval_harness_returns_module_when_present() -> None:
    module = _import_harness.import_eval_harness()
    assert module.__name__ == "eval_harness"


def test_import_eval_harness_raises_helpful_error_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(name: str) -> object:
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(importlib, "import_module", _raise)
    with pytest.raises(ImportError, match=r"alphagalerkin\[eval-harness\]"):
        _import_harness.import_eval_harness()
