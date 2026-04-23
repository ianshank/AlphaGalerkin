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

import json
import sys
from pathlib import Path
from typing import Any

# Make ``tests.modeling._public_surface_adr`` importable when running as a
# script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.modeling._public_surface_adr import PUBLIC_SURFACE  # noqa: E402
from tests.modeling._signature_utils import forward_entries, init_entries  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "_public_surface_golden.json"


def build_golden() -> dict[str, Any]:
    golden: dict[str, Any] = {}
    for entry in PUBLIC_SURFACE:
        cls = entry.cls
        record: dict[str, Any] = {
            "module": entry.expected_module,
            "init": init_entries(cls),
        }
        fwd = forward_entries(cls)
        if fwd is not None:
            record["forward"] = fwd
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
