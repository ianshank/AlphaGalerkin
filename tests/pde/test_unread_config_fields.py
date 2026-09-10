"""Self-expiring guards for reserved / unread PDE config fields.

``PDEGameConfig.success_metrics`` has no production reader. The default list
is kept so ``compute_hash()`` stays byte-stable. If a production file starts
loading ``.success_metrics``, this test fails until the reserved description
and this file are updated together.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PDE_SRC = REPO_ROOT / "src" / "pde"
CONFIG_PY = PDE_SRC / "config.py"


def _attribute_loads(attr: str, *, exclude: set[Path]) -> list[tuple[Path, int]]:
    hits: list[tuple[Path, int]] = []
    py_files = list(PDE_SRC.rglob("*.py"))
    assert py_files, "src/pde has no Python files — the scan would be vacuous"
    for path in py_files:
        if path in exclude:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == attr:
                hits.append((path.relative_to(REPO_ROOT), node.lineno))
    return hits


class TestSuccessMetricsUnread:
    def test_no_production_reader_outside_config(self) -> None:
        hits = _attribute_loads("success_metrics", exclude={CONFIG_PY})
        assert hits == [], (
            "PDEGameConfig.success_metrics gained a production reader. "
            "Delete this reserved-field guard and the Field description's "
            "unread warning, then wire the consumer for real: " + repr(hits)
        )
