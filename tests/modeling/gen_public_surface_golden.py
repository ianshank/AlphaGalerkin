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

Structured logging via ``structlog`` is emitted per-class as the
generator walks the surface, so intentional regenerations leave a
reproducible audit trail (which classes were inspected, what factory
identifiers were emitted, total duration).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Make ``tests.modeling._public_surface_adr`` importable when running as a
# script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import structlog  # noqa: E402

from tests.modeling._public_surface_adr import PUBLIC_SURFACE  # noqa: E402
from tests.modeling._signature_utils import (  # noqa: E402
    forward_entries,
    init_entries,
    resolve_class,
)

GOLDEN_PATH = Path(__file__).parent / "_public_surface_golden.json"

_logger = structlog.get_logger(__name__)


def build_golden() -> dict[str, Any]:
    """Introspect every ADR class and collect its init + forward surface.

    Raises:
        RuntimeError: if any class listed in :data:`PUBLIC_SURFACE` is
            not reachable from ``src.modeling`` — the re-export surface
            is broken and must be fixed before regenerating.

    """
    golden: dict[str, Any] = {}
    for entry in PUBLIC_SURFACE:
        cls = resolve_class(entry.class_name)
        if cls is None:
            _logger.error(
                "unresolved_class",
                class_name=entry.class_name,
                expected_module=entry.expected_module,
            )
            raise RuntimeError(
                f"Cannot generate golden: {entry.class_name!r} is not "
                f"reachable from src.modeling. The ADR re-export surface "
                f"is broken; fix src/modeling/__init__.py before "
                f"regenerating the golden."
            )
        init_surface = init_entries(cls)
        forward_surface = forward_entries(cls)
        record: dict[str, Any] = {
            "module": entry.expected_module,
            "init": init_surface,
        }
        if forward_surface is not None:
            record["forward"] = forward_surface
        golden[entry.class_name] = record
        _logger.debug(
            "inspected_class",
            class_name=entry.class_name,
            actual_module=cls.__module__,
            init_params=len(init_surface),
            has_forward=forward_surface is not None,
            forward_params=len(forward_surface) if forward_surface else 0,
        )
    return golden


def main() -> None:
    """CLI entry point: regenerate the golden JSON and log a summary."""
    started_at = time.monotonic()
    _logger.info(
        "golden_regeneration_started",
        adr_classes=len(PUBLIC_SURFACE),
        golden_path=str(GOLDEN_PATH),
    )
    golden = build_golden()
    GOLDEN_PATH.write_text(
        json.dumps(golden, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    elapsed_ms = round((time.monotonic() - started_at) * 1000, 1)
    _logger.info(
        "golden_regeneration_completed",
        classes=len(golden),
        golden_path=str(GOLDEN_PATH),
        elapsed_ms=elapsed_ms,
    )
    print(f"wrote {GOLDEN_PATH} ({len(golden)} classes, {elapsed_ms} ms)")


if __name__ == "__main__":
    main()
