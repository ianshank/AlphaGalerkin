"""Anti-drift guard: ARCHITECTURE.md must enumerate exactly the src/ packages.

The old ``CLAUDE.md`` "Directory Structure" block drifted — it silently lost
``src/refinement/`` and ``src/alphagalerkin/``. This test makes the canonical
repository map (``ARCHITECTURE.md``) executable: add or remove a top-level
``src/`` package and CI fails until the package-map table is updated to match.

The table is delimited by ``<!-- package-map:start -->`` /
``<!-- package-map:end -->`` markers, and each package row names the package as
an inline-code ``src/<name>/`` token.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = REPO_ROOT / "ARCHITECTURE.md"
SRC = REPO_ROOT / "src"

START_MARKER = "<!-- package-map:start -->"
END_MARKER = "<!-- package-map:end -->"

# Matches a package-map row's leading ``| `src/<name>/` |`` cell.
_ROW_PACKAGE = re.compile(r"\|\s*`src/([a-z0-9_]+)/`\s*\|")


def _documented_packages() -> set[str]:
    text = ARCHITECTURE.read_text(encoding="utf-8")
    assert START_MARKER in text, f"missing {START_MARKER} in {ARCHITECTURE}"
    assert END_MARKER in text, f"missing {END_MARKER} in {ARCHITECTURE}"
    region = text.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0]
    return set(_ROW_PACKAGE.findall(region))


def _filesystem_packages() -> set[str]:
    return {p.parent.name for p in SRC.glob("*/__init__.py")}


def test_architecture_map_matches_src_packages() -> None:
    documented = _documented_packages()
    actual = _filesystem_packages()

    missing_from_doc = actual - documented
    stale_in_doc = documented - actual

    assert not missing_from_doc, (
        "ARCHITECTURE.md package map is missing src/ packages that exist on "
        f"disk: {sorted(missing_from_doc)}. Add a row for each in the "
        "package-map table."
    )
    assert not stale_in_doc, (
        "ARCHITECTURE.md package map lists packages that no longer exist under "
        f"src/: {sorted(stale_in_doc)}. Remove their rows."
    )


def test_package_map_is_non_empty() -> None:
    assert _documented_packages(), "the package-map region parsed to zero packages"
