"""Enforce the Mouse-Droid-AGI fusion-head integration ADR contract in CI.

The ADR ``docs/adr/0002-mouse-droid-fusion-integration.md``
freezes 18 classes in ``src.modeling`` for the cross-repo integration
window. Stability rule §1 says constructor signatures must stay
backwards-compatible; §2 says ``forward`` signatures must stay stable.
These tests turn those rules into a mechanical CI check.

Class resolution is deliberately lazy (via ``resolve_class``): a removed
or renamed export becomes an ``AssertionError`` with the remediation
message below, rather than an ``ImportError`` that derails pytest
collection before any contract test can even run.

Failure here is a signal to the author: either revert the signature
change, or — if the change is an intentional, backwards-compatible
addition — regenerate the golden with ``python
tests/modeling/gen_public_surface_golden.py`` and update the ADR's
"§Key signatures (frozen)" block in the same PR.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

import src.modeling as modeling_pkg
from tests.modeling._public_surface_adr import PUBLIC_SURFACE, SurfaceEntry
from tests.modeling._signature_utils import EMPTY as _EMPTY
from tests.modeling._signature_utils import (
    forward_entries as _current_forward_entries,
)
from tests.modeling._signature_utils import (
    init_entries as _current_init_entries,
)
from tests.modeling._signature_utils import (
    resolve_class as _resolve_class,
)

_GOLDEN_PATH = Path(__file__).parent / "_public_surface_golden.json"
_REMEDIATION = (
    "If this change is intentional and backwards-compatible "
    "(new parameter appended at the end with a default that preserves "
    "prior behaviour — positional-or-keyword or keyword-only, both are "
    "accepted by the additive rule), regenerate the golden via\n"
    "    python tests/modeling/gen_public_surface_golden.py\n"
    "and update docs/adr/0002-mouse-droid-fusion-integration.md "
    "'§Key signatures (frozen)' in the same PR. Non-additive changes "
    "(renames, reorders, default-value changes, forward-signature changes) "
    "require superseding the ADR."
)


@cache
def _golden() -> dict[str, dict[str, Any]]:
    """Lazily load the golden snapshot with actionable errors.

    Loading is deferred from module import to first test execution so a
    missing or corrupted snapshot surfaces as an ``AssertionError`` with
    regeneration guidance, not a ``FileNotFoundError``/``JSONDecodeError``
    that derails pytest collection before any remediation can be shown.
    """
    if not _GOLDEN_PATH.exists():
        raise AssertionError(
            f"Golden snapshot missing at {_GOLDEN_PATH}. "
            f"Restore it from git (shallow/partial checkouts can miss "
            f"it) or regenerate via\n"
            f"    python tests/modeling/gen_public_surface_golden.py"
        )
    try:
        with _GOLDEN_PATH.open(encoding="utf-8") as f:
            data: dict[str, dict[str, Any]] = json.load(f)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Golden snapshot at {_GOLDEN_PATH} is corrupted "
            f"(JSON decode error: {exc}). Restore from git or "
            f"regenerate via\n"
            f"    python tests/modeling/gen_public_surface_golden.py"
        ) from exc
    return data


def _entry_id(entry: SurfaceEntry) -> str:
    return entry.class_name


def _require_class(entry: SurfaceEntry) -> type:
    """Resolve *entry*'s class or fail the test with ADR remediation."""
    cls = _resolve_class(entry.class_name)
    assert cls is not None, (
        f"{entry.class_name} is not reachable from src.modeling "
        f"(ADR stable surface). Add it to src/modeling/__init__.py "
        f"imports + __all__, or — if this removal is intentional — "
        f"supersede the ADR.\n\n"
        f"{_REMEDIATION}"
    )
    return cls


def _require_golden_record(name: str) -> dict[str, Any]:
    """Fetch the golden record for *name* or fail with actionable guidance.

    Guards against the out-of-sync case where ``PUBLIC_SURFACE`` lists a
    class that the golden doesn't record. ``test_golden_has_every_adr_class``
    also catches this, but pytest doesn't guarantee execution order, so
    the per-class tests must produce their own actionable message instead
    of a cryptic ``KeyError``.
    """
    record = _golden().get(name)
    assert record is not None, (
        f"{name} is listed in PUBLIC_SURFACE but missing from the golden "
        f"snapshot. Regenerate via\n"
        f"    python tests/modeling/gen_public_surface_golden.py\n"
        f"or remove {name!r} from PUBLIC_SURFACE if the class is being "
        f"dropped (which also requires superseding the ADR)."
    )
    return record


@pytest.mark.parametrize("entry", PUBLIC_SURFACE, ids=_entry_id)
def test_class_importable_from_toplevel(entry: SurfaceEntry) -> None:
    """Each ADR class is reachable via ``from src.modeling import <Class>``.

    Also asserts ``__module__`` matches the submodule declared in the
    ADR table, so an accidental move (e.g. copying ``FNetBlock`` into a
    new file) is caught. The golden's own ``module`` field is
    cross-checked against the ADR table to catch a drifted golden
    without the regenerator being re-run.
    """
    cls = _require_class(entry)
    assert cls.__module__ == entry.expected_module, (
        f"{entry.class_name}.__module__ is {cls.__module__!r}, "
        f"ADR declares {entry.expected_module!r}. Did the class move?"
    )
    record = _require_golden_record(entry.class_name)
    golden_module = record.get("module")
    assert golden_module == entry.expected_module, (
        f"Golden's module for {entry.class_name!r} is {golden_module!r} "
        f"but the ADR table declares {entry.expected_module!r}. "
        f"Regenerate via\n"
        f"    python tests/modeling/gen_public_surface_golden.py\n"
        f"or reconcile PUBLIC_SURFACE with the ADR."
    )


@pytest.mark.parametrize("entry", PUBLIC_SURFACE, ids=_entry_id)
def test_class_is_in_all(entry: SurfaceEntry) -> None:
    """Every ADR class appears in ``src.modeling.__all__``.

    Redundant with ``test_class_is_in_public_all`` in
    ``test_modeling_gap_coverage.py``, but kept here so this file is
    self-contained — the gap-coverage file may be refactored for other
    reasons.
    """
    assert entry.class_name in modeling_pkg.__all__, (
        f"{entry.class_name} missing from src.modeling.__all__. "
        f"The ADR requires all 18 stable-surface classes to be listed."
    )


@pytest.mark.parametrize("entry", PUBLIC_SURFACE, ids=_entry_id)
def test_constructor_signature_is_additive_only(entry: SurfaceEntry) -> None:
    """Constructor signature obeys stability rule §1 (additive-only).

    Rules enforced:

    1. Every parameter in the golden must appear at the **same index**
       in the current signature, with the same name, kind, and default.
    2. Extra parameters (beyond the golden) are allowed **only at the
       end** and must have a non-empty default.
    """
    cls = _require_class(entry)
    name = entry.class_name
    golden_entries: list[dict[str, str]] = _require_golden_record(name)["init"]
    current_entries = _current_init_entries(cls)

    assert len(current_entries) >= len(golden_entries), (
        f"{name}.__init__ dropped parameters — had "
        f"{len(golden_entries)} in the ADR golden, now has "
        f"{len(current_entries)}. Removing a parameter is not "
        f"backwards-compatible.\n\n"
        f"{_REMEDIATION}"
    )

    for i, golden_param in enumerate(golden_entries):
        current = current_entries[i]
        assert current == golden_param, (
            f"{name}.__init__ parameter #{i} drifted from the ADR "
            f"golden.\n"
            f"  golden:  {golden_param}\n"
            f"  current: {current}\n\n"
            f"{_REMEDIATION}"
        )

    # Any extra parameters (i.e., new kwargs appended) must have defaults.
    for extra in current_entries[len(golden_entries) :]:
        assert extra["default"] != _EMPTY, (
            f"{name}.__init__ added a parameter {extra['name']!r} "
            f"without a default, which is a breaking change.\n\n"
            f"{_REMEDIATION}"
        )


@pytest.mark.parametrize("entry", PUBLIC_SURFACE, ids=_entry_id)
def test_forward_signature_matches_golden(entry: SurfaceEntry) -> None:
    """``forward`` signature is frozen exactly (stability rule §2).

    Input/output shapes are not introspectable from a bare signature,
    but parameter names, kinds, and defaults are — and those are what
    Mouse-Droid-AGI's ``sensing/galerkin_fusion.py`` binds against.
    Additive tolerance is intentionally **not** granted here: a new
    positional argument on ``forward`` silently breaks every existing
    caller.
    """
    cls = _require_class(entry)
    name = entry.class_name
    golden_record = _require_golden_record(name)
    golden_forward = golden_record.get("forward")
    current_forward = _current_forward_entries(cls)

    if golden_forward is None:
        assert current_forward is None, (
            f"{name} gained a forward() method; the ADR golden does not "
            f"expect one. Either revert, or extend the golden.\n\n"
            f"{_REMEDIATION}"
        )
        return

    assert current_forward is not None, (
        f"{name} lost its forward() method since the ADR was frozen. "
        f"This is a breaking change for any downstream caller.\n\n"
        f"{_REMEDIATION}"
    )
    assert current_forward == golden_forward, (
        f"{name}.forward signature drifted from the ADR golden.\n"
        f"  golden:  {golden_forward}\n"
        f"  current: {current_forward}\n\n"
        f"{_REMEDIATION}"
    )


def test_golden_has_every_adr_class() -> None:
    """Detect a golden that fell out of sync with the ADR class list.

    If a new class is added to :data:`PUBLIC_SURFACE` but the golden is
    not regenerated, the per-class lookups would raise ``KeyError``.
    This whole-surface test gives a clearer error and also catches the
    reverse (classes lingering in the golden after removal from the
    ADR tuple).
    """
    adr_names = {entry.class_name for entry in PUBLIC_SURFACE}
    golden_names = set(_golden().keys())
    missing_from_golden = adr_names - golden_names
    extra_in_golden = golden_names - adr_names
    assert not missing_from_golden, (
        f"Classes in PUBLIC_SURFACE missing from the golden: "
        f"{sorted(missing_from_golden)}. "
        f"Run `python tests/modeling/gen_public_surface_golden.py` "
        f"to regenerate."
    )
    assert not extra_in_golden, (
        f"Classes in the golden no longer in PUBLIC_SURFACE: "
        f"{sorted(extra_in_golden)}. "
        f"Run `python tests/modeling/gen_public_surface_golden.py` "
        f"to regenerate."
    )
