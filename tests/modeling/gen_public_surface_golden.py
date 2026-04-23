"""Regenerate ``tests/modeling/_public_surface_golden.json``.

Run locally when you intentionally add a backwards-compatible parameter
to one of the 18 ADR-frozen classes **and** have updated
``docs/architecture/ADR-mouse-droid-fusion-integration.md`` to reflect
the new signature.

Usage::

    python tests/modeling/gen_public_surface_golden.py

The generated file is deterministic (sorted keys, trailing newline) so
``git diff`` shows exactly what moved. CI never runs this script; it
only compares the in-repo snapshot against the introspected signatures.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

# Make ``tests.modeling._public_surface_adr`` importable when running as a
# script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import BaseModel  # noqa: E402
from pydantic_core import PydanticUndefined  # noqa: E402

from tests.modeling._public_surface_adr import PUBLIC_SURFACE  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "_public_surface_golden.json"

EMPTY = "<empty>"
FACTORY = "<factory>"


def _param_entries_from_signature(sig: inspect.Signature) -> list[dict[str, str]]:
    """Extract a stable list of parameter entries from ``inspect.Signature``.

    ``self`` is dropped; defaults are serialised via ``repr`` so JSON can
    carry tuples / ``None`` / bools uniformly. Missing defaults become
    ``<empty>``.
    """
    entries: list[dict[str, str]] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            default_repr = EMPTY
        else:
            default_repr = repr(param.default)
        entries.append(
            {
                "name": name,
                "kind": param.kind.name,
                "default": default_repr,
            }
        )
    return entries


def _pydantic_init_entries(cls: type[BaseModel]) -> list[dict[str, str]]:
    """Synthesise init-surface entries for Pydantic configs from ``model_fields``.

    Pydantic v2 rewrites ``__init__`` to ``(**data)``; the true surface is
    the field set (including inherited). Emitting entries in
    ``model_fields`` iteration order (Pydantic guarantees declaration
    order, MRO-aware) keeps the golden stable across re-runs.
    """
    entries: list[dict[str, str]] = []
    for field_name, field_info in cls.model_fields.items():
        if field_info.default is not PydanticUndefined:
            default_repr = repr(field_info.default)
        elif field_info.default_factory is not None:
            default_repr = FACTORY
        else:
            default_repr = EMPTY
        entries.append(
            {
                "name": field_name,
                "kind": "PYDANTIC_FIELD",
                "default": default_repr,
            }
        )
    return entries


def _init_entries(cls: type) -> list[dict[str, str]]:
    if isinstance(cls, type) and issubclass(cls, BaseModel):
        return _pydantic_init_entries(cls)
    return _param_entries_from_signature(inspect.signature(cls.__init__))


def _forward_entries(cls: type) -> list[dict[str, str]] | None:
    forward = getattr(cls, "forward", None)
    if forward is None:
        return None
    return _param_entries_from_signature(inspect.signature(forward))


def build_golden() -> dict[str, Any]:
    golden: dict[str, Any] = {}
    for entry in PUBLIC_SURFACE:
        cls = entry.cls
        record: dict[str, Any] = {
            "module": entry.expected_module,
            "init": _init_entries(cls),
        }
        forward = _forward_entries(cls)
        if forward is not None:
            record["forward"] = forward
        golden[cls.__name__] = record
    return golden


def main() -> None:
    golden = build_golden()
    GOLDEN_PATH.write_text(
        json.dumps(golden, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {GOLDEN_PATH} ({len(golden)} classes)")


if __name__ == "__main__":
    main()
