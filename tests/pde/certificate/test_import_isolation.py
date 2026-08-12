"""Gate 5 (spec AC4) — certificates are batch artifacts, never on the hot path.

Two AST-level invariants:

1. ``src/pde/certificate/**`` must **not** import anything from ``src/mcts/**``.
   If it did, cyclic imports could pull certificate code into an MCTS module.
2. ``src/mcts/**`` must **not** import anything from ``src/pde/certificate/**``.
   Certification is minutes-to-hours per solution (arXiv:2603.19165 Tables
   2/5); running it during rollouts would defeat MCTS.

The guard is static (AST-only): dynamic ``importlib`` calls can bypass it.
This matches the ``stochastic_galerkin_nke`` AC7 precedent — a documented
limitation that catches the accidental import case, not the adversarial one.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFICATE_PKG = REPO_ROOT / "src" / "pde" / "certificate"
MCTS_PKG = REPO_ROOT / "src" / "mcts"


def _iter_imports(py_file: Path) -> list[str]:
    """Return the dotted names of every ``import`` / ``from ... import`` in a file."""
    tree = ast.parse(py_file.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def _iter_py(pkg: Path) -> list[Path]:
    return sorted(p for p in pkg.rglob("*.py") if not p.name.startswith("."))


def test_certificate_pkg_does_not_import_mcts() -> None:
    """The certificate subpackage must not depend on MCTS internals."""
    offenders: list[tuple[str, str]] = []
    for py in _iter_py(CERTIFICATE_PKG):
        for mod in _iter_imports(py):
            if mod == "src.mcts" or mod.startswith("src.mcts."):
                offenders.append((str(py.relative_to(REPO_ROOT)), mod))
    assert not offenders, (
        f"certificate subpackage imports MCTS (spec AC4 / Gate 5 violated): {offenders!r}"
    )


def test_mcts_pkg_does_not_import_certificate() -> None:
    """MCTS rollout paths must not depend on certificate machinery.

    A dependency here would pull certificate cost (minutes-hours per solution
    per spec §5 / arXiv:2603.19165 Tables 2 & 5) onto the rollout hot path.
    """
    offenders: list[tuple[str, str]] = []
    for py in _iter_py(MCTS_PKG):
        for mod in _iter_imports(py):
            if mod == "src.pde.certificate" or mod.startswith("src.pde.certificate."):
                offenders.append((str(py.relative_to(REPO_ROOT)), mod))
    assert not offenders, (
        f"MCTS pkg imports certificate module (spec AC4 / Gate 5 violated): {offenders!r}"
    )


def test_certificate_pkg_files_present() -> None:
    """Sanity: the four foundation modules exist.

    Guards against a mis-merge that wipes the subpackage but leaves the tests
    behind.
    """
    for expected in ("__init__.py", "certificate.py", "config.py", "stability.py", "logging.py"):
        assert (CERTIFICATE_PKG / expected).is_file(), f"missing {expected}"
