"""Static import-graph helpers shared by the repository's import guards.

Two guards need the same three primitives -- resolve a file to its dotted
module name, enumerate what it imports (with relative imports resolved), and
match a module against a *module-boundary* prefix. They were written once for
``tests/pde/stochastic/test_import_isolation.py``; this is that implementation
lifted out so the architectural contracts in
``tests/regression/test_import_contracts.py`` extend it rather than fork it.

**Static direction only, stated plainly.** These helpers walk `import`
statements with `ast`; they never consult `sys.modules`. That is deliberate,
not a shortcut: importing anything under `src.pde` executes
``src/pde/__init__.py``, whose suppressed ``register_games`` import pulls game
and MCTS modules at runtime regardless, so a runtime check would report a
violation caused by a parent package's side effect rather than by the module
under test.
"""

from __future__ import annotations

import ast
from pathlib import Path

__all__ = ["imported_modules", "matches_module_prefix", "module_name_for", "python_files_under"]


def module_name_for(path: Path, repo_root: Path) -> str:
    """Return the dotted module name a repo-relative source file resolves to."""
    return ".".join(path.relative_to(repo_root).with_suffix("").parts)


def imported_modules(path: Path, repo_root: Path) -> set[str]:
    """Every absolute module name ``path`` imports, with relative imports resolved.

    ``from . import x`` inside ``src/a/b.py`` resolves to ``src.a``;
    ``from ..c import y`` resolves to ``src.c``. A bare ``from . import name``
    (no module) resolves to the package itself, which is the conservative
    reading -- it cannot be told apart statically from importing a submodule.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    own_package_parts = module_name_for(path, repo_root).split(".")[:-1]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    imported.add(node.module)
            else:
                base_parts = own_package_parts[: len(own_package_parts) - node.level + 1]
                base = ".".join(base_parts)
                imported.add(f"{base}.{node.module}" if node.module else base)
    return imported


def matches_module_prefix(module: str, prefix: str) -> bool:
    """Match on the module *boundary*, never on a raw string prefix.

    Without the boundary, ``src.pde.game`` would match ``src.pde.games`` and a
    guard would forbid a package nobody named. The two are distinct modules and
    must be listed separately when both are meant.
    """
    return module == prefix or module.startswith(prefix + ".")


def python_files_under(root: Path) -> list[Path]:
    """All ``*.py`` under ``root``, excluding bytecode caches, sorted."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
