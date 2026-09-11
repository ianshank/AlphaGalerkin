"""Root conftest.py — pytest configuration and shared fixtures."""

from __future__ import annotations

import importlib.util
import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.terminal import TerminalReporter

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

#: The optional [eval-harness] git extra (langfuse-eval-harness). Probed with
#: ``find_spec`` rather than imported: nothing in the base install needs it, and
#: importing it here would load its Langfuse client into every test session.
_EVAL_HARNESS_MODULE = "eval_harness"
_HAS_EVAL_HARNESS = importlib.util.find_spec(_EVAL_HARNESS_MODULE) is not None

#: The install command named in the eval_harness_required error and skip
#: reasons, so a reader never has to look it up.
_EVAL_HARNESS_INSTALL_HINT = "pip install -e '.[eval-harness]'"

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

#: Same treatment for the gpu_required skip. Every workflow in this repo runs on
#: `ubuntu-latest`, so all ~22 gpu_required sites have skipped on every CI run
#: ever -- and until now, silently. A count is not a gate, but it is the
#: difference between "the GPU suite is skipped here" being visible and being
#: assumed; a GPU host whose driver disappeared otherwise looks identical to a
#: green run.
_gpu_skip_count = 0

#: Same treatment for eval_harness_required (R-13). Until 2026-09-11 the eight
#: files under tests/integrations/eval_harness/ used a module-level
#: ``pytest.importorskip("eval_harness")``, which yields ZERO items -- so no hook
#: could count them and no gate could fail them: the test-extras job installed
#: the extra "specifically to un-skip these modules" and then selected none of
#: them, and nothing noticed. A marker is countable; an importorskip is not.
_eval_harness_skip_count = 0


def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests marked gpu_required when CUDA is not available."""
    if not (_HAS_TORCH and _torch.cuda.is_available()):
        skip_gpu = pytest.mark.skip(reason="CUDA not available (no NVIDIA driver)")
        gpu_items = [item for item in items if item.get_closest_marker("gpu_required")]
        for item in gpu_items:
            item.add_marker(skip_gpu)
        global _gpu_skip_count
        _gpu_skip_count += len(gpu_items)

    _gate_optional_extra(
        items,
        marker="fem_required",
        installed=_HAS_SKFEM,
        package_label="scikit-fem",
        install_hint="pip install -e '.[fem]'",
    )
    _gate_optional_extra(
        items,
        marker="eval_harness_required",
        installed=_HAS_EVAL_HARNESS,
        package_label="langfuse-eval-harness",
        install_hint=_EVAL_HARNESS_INSTALL_HINT,
    )


def _gate_optional_extra(
    items: list[pytest.Item],
    *,
    marker: str,
    installed: bool,
    package_label: str,
    install_hint: str,
) -> None:
    """Skip-with-a-count, or hard-fail under ALPHAGALERKIN_REQUIRE_EXTRAS=1.

    One body for both optional extras, so the two cannot drift: the fem hook
    was inlined and the eval-harness one would otherwise have been a copy.

    Args:
        items: The collected items, mutated in place (skip markers added).
        marker: The marker naming the extra (``fem_required`` /
            ``eval_harness_required``).
        installed: Whether the extra's import is resolvable.
        package_label: Human name used in messages.
        install_hint: The exact ``pip`` command that installs the extra.

    Raises:
        pytest.UsageError: Marked tests were collected, the extra is absent,
            and ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` -- a half-succeeded install
            in the test-extras job must not go quietly green.

    """
    if installed:
        return
    marked = [item for item in items if item.get_closest_marker(marker)]
    if not marked:
        return
    if _REQUIRE_EXTRAS:
        raise pytest.UsageError(
            f"ALPHAGALERKIN_REQUIRE_EXTRAS=1: {len(marked)} {marker} test(s) "
            f"collected but {package_label} is not installed. Install with: "
            f"{install_hint}"
        )
    skip = pytest.mark.skip(reason=f"{package_label} not installed")
    for item in marked:
        item.add_marker(skip)
    _record_optional_skip(marker, len(marked))


def _record_optional_skip(marker: str, count: int) -> None:
    """Bump the per-marker skip counter read by ``pytest_terminal_summary``."""
    global _fem_skip_count, _eval_harness_skip_count
    if marker == "fem_required":
        _fem_skip_count += count
    elif marker == "eval_harness_required":
        _eval_harness_skip_count += count
    else:  # pragma: no cover - a new extra must add its counter explicitly
        raise ValueError(f"no skip counter registered for marker {marker!r}")


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int, config: Config
) -> None:
    """Report how many optional-dependency tests were skipped -- visibly."""
    if _fem_skip_count:
        terminalreporter.write_line(
            f"fem_required: skipped {_fem_skip_count} test(s) -- scikit-fem not installed "
            "(pip install -e '.[fem]')",
            yellow=True,
        )
    if _eval_harness_skip_count:
        terminalreporter.write_line(
            f"eval_harness_required: skipped {_eval_harness_skip_count} test(s) -- "
            f"langfuse-eval-harness not installed ({_EVAL_HARNESS_INSTALL_HINT})",
            yellow=True,
        )
    if _gpu_skip_count:
        terminalreporter.write_line(
            f"gpu_required: skipped {_gpu_skip_count} test(s) -- CUDA not available",
            yellow=True,
        )
