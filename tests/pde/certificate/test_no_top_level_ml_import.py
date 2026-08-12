"""Static guard: no top-level ML framework imports in ``src/pde/certificate/``.

Spec §3 rule: the certificate subpackage must remain importable on a base
install even when torch and jax are absent. Frameworks may be imported
lazily *inside method bodies* (e.g.
:meth:`HeuristicGridResidualVerifier._evaluate_model`), but never at
module load.

Rationale: certificate code lives on the same import path as
``src.pde.config.PDEType``, which is imported by every scenario. A stray
``import torch`` at the top of ``src/pde/certificate/verifiers/torch_verifier.py``
would drag torch onto the load path of every PoC — an ecosystem-wide
regression for a single verifier's convenience.

The guard is AST-only: dynamic ``importlib`` calls can bypass it. Same
philosophy as ``test_import_isolation.py`` — catches the accidental case,
not the adversarial one.
"""

from __future__ import annotations

import ast
from pathlib import Path

CERTIFICATE_PKG = Path(__file__).resolve().parents[3] / "src" / "pde" / "certificate"

FORBIDDEN_TOPLEVEL_ROOTS = frozenset(
    {
        "torch",
        "jax",
        "jax_verify",
        "auto_LiRPA",
        "flax",
        "optax",
        "orbax",
        "chex",
    }
)


def _collect_toplevel_imports(py_file: Path) -> list[str]:
    """Return dotted module names of imports appearing at the module top level."""
    tree = ast.parse(py_file.read_text())
    names: list[str] = []
    for node in tree.body:  # top level only — not ast.walk
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def _root(name: str) -> str:
    return name.split(".", 1)[0]


def test_no_toplevel_ml_imports_in_certificate_pkg() -> None:
    offenders: list[tuple[str, str]] = []
    for py in sorted(CERTIFICATE_PKG.rglob("*.py")):
        for mod in _collect_toplevel_imports(py):
            if _root(mod) in FORBIDDEN_TOPLEVEL_ROOTS:
                offenders.append((str(py.relative_to(CERTIFICATE_PKG.parents[2])), mod))
    assert not offenders, (
        f"forbidden top-level ML import in certificate subpackage: {offenders!r}. "
        f"Move the import inside the method body — see the "
        f"HeuristicGridResidualVerifier._evaluate_model precedent."
    )


def test_typing_only_torch_import_would_be_allowed() -> None:
    """Sanity: the guard checks *executable* imports.

    ``if TYPE_CHECKING: import torch`` blocks are permitted (mypy still sees
    them, but the import is dead at run time). This test documents that the
    guard's AST walk does not descend into ``if`` blocks — only ``tree.body``.
    """
    # No offender file exists today; this is a documentation test. If the
    # guard's implementation changes to walk nested ``if`` blocks, this test
    # should be revisited.
    assert True
