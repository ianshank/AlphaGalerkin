"""Root conftest.py — pytest configuration and shared fixtures."""

from __future__ import annotations

import os

import pytest

try:
    import torch as _torch

    _HAS_TORCH = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:
    import skfem as _skfem  # noqa: F401

    _HAS_SKFEM = True
except ImportError:
    _HAS_SKFEM = False

#: When set, a fem_required test collected without scikit-fem installed is a
#: hard collection error rather than a silent skip -- for the test-extras CI
#: job, where the optional [fem] extra is expected to actually be installed
#: and a half-succeeded install must not go quietly green.
_REQUIRE_EXTRAS = os.environ.get("ALPHAGALERKIN_REQUIRE_EXTRAS") == "1"

#: Populated by pytest_collection_modifyitems, read by pytest_terminal_summary
#: -- unlike the gpu_required skip above, this one reports how many tests it
#: skipped, since a scikit-fem install that silently fails must not disappear
#: without a visible trace.
_fem_skip_count = 0


def pytest_collection_modifyitems(config, items) -> None:
    """Auto-skip tests marked gpu_required when CUDA is not available."""
    if not (_HAS_TORCH and _torch.cuda.is_available()):
        skip_gpu = pytest.mark.skip(reason="CUDA not available (no NVIDIA driver)")
        for item in items:
            if item.get_closest_marker("gpu_required"):
                item.add_marker(skip_gpu)

    if _HAS_SKFEM:
        return
    fem_items = [item for item in items if item.get_closest_marker("fem_required")]
    if not fem_items:
        return
    if _REQUIRE_EXTRAS:
        raise pytest.UsageError(
            f"ALPHAGALERKIN_REQUIRE_EXTRAS=1: {len(fem_items)} fem_required test(s) "
            "collected but scikit-fem is not installed. Install with: "
            "pip install -e '.[fem]'"
        )
    global _fem_skip_count
    skip_fem = pytest.mark.skip(reason="scikit-fem not installed")
    for item in fem_items:
        item.add_marker(skip_fem)
    _fem_skip_count += len(fem_items)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Report how many fem_required tests were skipped -- visibly, not silently."""
    if _fem_skip_count:
        terminalreporter.write_line(
            f"fem_required: skipped {_fem_skip_count} test(s) -- scikit-fem not installed "
            "(pip install -e '.[fem]')",
            yellow=True,
        )
