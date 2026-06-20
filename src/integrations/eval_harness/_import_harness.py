"""Lazy importer for the optional ``eval_harness`` package.

Keeps the rest of the codebase importable on a base install that does not have
the ``[eval-harness]`` extra. Mirrors ``lm_studio.client._import_openai``: the
import is funnelled through one function so a missing dependency fails with a
clear, actionable message exactly when the integration is used — and so tests
can monkeypatch this single seam.
"""

from __future__ import annotations

import importlib
from typing import Any

_INSTALL_HINT = (
    "The 'langfuse-eval-harness' package is required for the eval-harness "
    "integration. Install with: pip install 'alphagalerkin[eval-harness]'"
)


def import_eval_harness() -> Any:
    """Import and return the ``eval_harness`` package, or raise a helpful error.

    Returns:
        The imported ``eval_harness`` module.

    Raises:
        ImportError: If the optional ``langfuse-eval-harness`` dependency is not
            installed; the message points at the ``[eval-harness]`` extra.

    """
    try:
        return importlib.import_module("eval_harness")
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
