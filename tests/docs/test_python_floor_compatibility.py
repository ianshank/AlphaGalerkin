"""No unguarded use of stdlib newer than the declared Python floor.

``pyproject.toml`` declares ``requires-python = ">=3.10"`` and CI runs a 3.10 job,
but nothing checked that the tree actually *works* on 3.10 until that job ran --
and when it fails, it fails badly: an unimportable test module is a **collection**
error, which aborts the entire fast lane rather than failing one test.

That is not hypothetical. ``import tomllib`` (3.11+) and ``from datetime import
UTC`` (3.11+) both landed in this repository and took the 3.10 fast lane down at
collection. Neither is caught by ruff: ``target-version = "py310"`` governs which
*rewrites* ruff suggests, not which stdlib you reach for.

Guarded use is fine and is not flagged -- an import inside ``try``/``except
ImportError`` or behind a ``sys.version_info`` check is the correct way to use a
newer API with a fallback.

The floor is read from ``pyproject.toml`` rather than hardcoded, so raising
``requires-python`` to 3.11 relaxes this guard automatically instead of leaving a
stale rule to be discovered and deleted by hand.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Roots scanned. ``hf_space/`` is excluded: it is a deploy bundle with its own
#: runtime, and an accepted charter deviation.
SCAN_ROOTS: Final[tuple[str, ...]] = ("src", "tests", "scripts", "dashboard", "config")

#: Module -> the Python version that first shipped it in the stdlib.
_MODULES_ADDED_IN: Final[dict[str, tuple[int, int]]] = {
    "tomllib": (3, 11),
}

#: ``(module, name)`` -> the Python version that first shipped that name.
_NAMES_ADDED_IN: Final[dict[tuple[str, str], tuple[int, int]]] = {
    ("datetime", "UTC"): (3, 11),
    ("enum", "StrEnum"): (3, 11),
    ("enum", "ReprEnum"): (3, 11),
    ("typing", "Self"): (3, 11),
    ("typing", "LiteralString"): (3, 11),
    ("typing", "Never"): (3, 11),
    ("typing", "assert_never"): (3, 11),
    ("typing", "assert_type"): (3, 11),
    ("typing", "TypeVarTuple"): (3, 11),
    ("typing", "Unpack"): (3, 11),
    ("typing", "override"): (3, 12),
    ("typing", "TypeAliasType"): (3, 12),
    ("asyncio", "TaskGroup"): (3, 11),
    ("asyncio", "timeout"): (3, 11),
    ("contextlib", "chdir"): (3, 11),
    ("hashlib", "file_digest"): (3, 11),
    ("itertools", "batched"): (3, 12),
}

#: Builtins added after 3.10.
_BUILTINS_ADDED_IN: Final[dict[str, tuple[int, int]]] = {
    "ExceptionGroup": (3, 11),
    "BaseExceptionGroup": (3, 11),
}

_REQUIRES_PYTHON: Final[re.Pattern[str]] = re.compile(
    r"^requires-python\s*=\s*[\"']>=\s*(\d+)\.(\d+)", re.MULTILINE
)


def _python_floor() -> tuple[int, int]:
    """The declared minimum Python, from ``pyproject.toml``."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = _REQUIRES_PYTHON.search(text)
    assert match is not None, "pyproject.toml declares no requires-python floor"
    return int(match.group(1)), int(match.group(2))


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        directory = REPO_ROOT / root
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    conftest = REPO_ROOT / "conftest.py"
    if conftest.exists():
        files.append(conftest)
    return files


def _guarded_line_ranges(tree: ast.Module) -> set[int]:
    """Line numbers inside ``try`` or ``if`` bodies -- i.e. deliberately guarded."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try | ast.If):
            for child in ast.walk(node):
                if hasattr(child, "lineno"):
                    guarded.add(child.lineno)
    return guarded


def _violations(path: Path, floor: tuple[int, int]) -> list[str]:
    """Unguarded uses of stdlib newer than ``floor`` in ``path``."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a broken file fails elsewhere
        return []
    guarded = _guarded_line_ranges(tree)
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:  # a file outside the repo (unit-test fixtures)
        relative = path.as_posix()
    found: list[str] = []

    for node in ast.walk(tree):
        # ast.Module and a few others carry no lineno; they are never violations.
        lineno = getattr(node, "lineno", None)
        if lineno is None or lineno in guarded:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                added = _MODULES_ADDED_IN.get(alias.name.split(".")[0])
                if added and added > floor:
                    found.append(
                        f"{relative}:{node.lineno}: `import {alias.name}` needs "
                        f"Python {added[0]}.{added[1]}"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            base = node.module.split(".")[0]
            module_added = _MODULES_ADDED_IN.get(base)
            if module_added and module_added > floor:
                found.append(
                    f"{relative}:{node.lineno}: `from {node.module} import ...` needs "
                    f"Python {module_added[0]}.{module_added[1]}"
                )
                continue
            for alias in node.names:
                added = _NAMES_ADDED_IN.get((base, alias.name))
                if added and added > floor:
                    found.append(
                        f"{relative}:{node.lineno}: `from {base} import {alias.name}` "
                        f"needs Python {added[0]}.{added[1]}"
                    )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            added = _NAMES_ADDED_IN.get((node.value.id, node.attr))
            if added and added > floor:
                found.append(
                    f"{relative}:{node.lineno}: `{node.value.id}.{node.attr}` needs "
                    f"Python {added[0]}.{added[1]}"
                )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            added = _BUILTINS_ADDED_IN.get(node.id)
            if added and added > floor:
                found.append(
                    f"{relative}:{node.lineno}: `{node.id}` needs Python {added[0]}.{added[1]}"
                )
    return found


class TestPythonFloorCompatibility:
    def test_floor_is_declared(self) -> None:
        major, minor = _python_floor()
        assert (major, minor) >= (3, 8), f"implausible floor {major}.{minor}"

    def test_no_stdlib_newer_than_the_floor(self) -> None:
        floor = _python_floor()
        offenders = [
            violation for path in _scanned_files() for violation in _violations(path, floor)
        ]
        assert not offenders, (
            f"stdlib newer than the declared floor (Python {floor[0]}.{floor[1]}):\n  "
            + "\n  ".join(sorted(offenders))
            + "\n\nAn unimportable module is a *collection* error, so this takes the "
            "whole test run down on the oldest supported Python, not just one test. "
            "Use a backport, guard it behind sys.version_info, or raise "
            "requires-python."
        )

    def test_the_scan_actually_reaches_the_tree(self) -> None:
        """A scanner that finds no files passes everything."""
        files = _scanned_files()
        assert len(files) > 500, f"expected to scan the whole tree, found {len(files)}"
        assert any("src/" in f.as_posix() for f in files)
        assert any("tests/" in f.as_posix() for f in files)
        assert any("scripts/" in f.as_posix() for f in files)

    def test_guarded_imports_are_not_flagged(self, tmp_path: Path) -> None:
        """try/except and version checks are the correct way to use a newer API."""
        guarded = tmp_path / "guarded.py"
        guarded.write_text(
            "try:\n    import tomllib\nexcept ModuleNotFoundError:\n    import tomli as tomllib\n",
            encoding="utf-8",
        )
        assert not _violations(guarded, (3, 10))

    def test_an_unguarded_newer_import_is_flagged(self, tmp_path: Path) -> None:
        """The positive control: without it, passing proves nothing."""
        offending = tmp_path / "offending.py"
        offending.write_text("import tomllib\nfrom datetime import UTC\n", encoding="utf-8")
        violations = _violations(offending, (3, 10))
        assert len(violations) == 2, violations
        assert any("tomllib" in v for v in violations)
        assert any("UTC" in v for v in violations)

    def test_nothing_is_flagged_when_the_floor_is_new_enough(self, tmp_path: Path) -> None:
        """Raising requires-python must relax the guard, not strand a stale rule."""
        offending = tmp_path / "offending.py"
        offending.write_text("import tomllib\n", encoding="utf-8")
        assert not _violations(offending, (3, 12))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
