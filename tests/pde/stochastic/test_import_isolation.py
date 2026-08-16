"""AC7: the stochastic layer does not import MCTS/self-play code paths.

Scope (stated honestly): this guard proves **static import direction only** —
an AST walk over the new modules' import statements. It deliberately does NOT
use ``sys.modules``: importing anything under ``src.pde`` executes
``src/pde/__init__.py``, whose (suppressed) ``register_games`` import pulls
game/MCTS modules at runtime regardless — a runtime check would false-positive
on that parent-package side effect. Companion: the MCTS and F0/F1 regression
surfaces stay green (run in CI, not asserted here).

Spec: specs/stochastic_galerkin_nke.spec.md (AC7, change-doc task 1.9).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# Modules owned by this change. Paths that do not exist yet are skipped, so
# the guard automatically covers the harness/scenario/CLI when they land.
GUARDED_GLOBS = [
    "src/pde/stochastic/*.py",
    "src/research/stochastic_galerkin_compare.py",
    "src/poc/scenarios/stochastic_galerkin_compare*.py",
    "scripts/run_stochastic_galerkin_compare.py",
]

# Module-boundary prefixes: `f` forbids module `m` iff m == f or m starts
# with f + "." (so `src.pde.game` does NOT match `src.pde.games` by accident —
# both are listed explicitly).
FORBIDDEN_PREFIXES = [
    "src.mcts",
    "src.games",
    "src.refinement",
    "src.training.self_play",
    "src.training.trainer",
    "src.pde.mcts_adapter",
    "src.pde.game",
    "src.pde.games",
    "src.pde.game_interface",
    "src.pde.register_games",
    "src.pde.trainer",
]


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for pattern in GUARDED_GLOBS:
        files.extend(REPO_ROOT.glob(pattern))
    return sorted(f for f in files if f.name != "__pycache__")


def _module_of(path: Path) -> str:
    return ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)


def _imported_modules(path: Path) -> set[str]:
    """All absolute module names imported by ``path`` (relative imports resolved)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    own_module = _module_of(path)
    own_package_parts = own_module.split(".")[:-1]
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


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


class TestImportIsolation:
    def test_guarded_files_exist(self):
        files = _guarded_files()
        assert files, "no guarded files found — the glob list is broken"
        assert any("stochastic" in str(f) for f in files)

    def test_no_forbidden_imports(self):
        violations: list[str] = []
        for path in _guarded_files():
            for module in _imported_modules(path):
                for prefix in FORBIDDEN_PREFIXES:
                    if _matches(module, prefix):
                        violations.append(
                            f"{path.relative_to(REPO_ROOT)} imports {module} "
                            f"(forbidden prefix {prefix})"
                        )
        assert not violations, "MCTS/self-play import leakage:\n" + "\n".join(violations)

    @pytest.mark.parametrize(
        ("module", "prefix", "expected"),
        [
            ("src.mcts", "src.mcts", True),
            ("src.mcts.search", "src.mcts", True),
            ("src.mctsx", "src.mcts", False),  # boundary, not raw prefix
            ("src.pde.games", "src.pde.game", False),  # listed separately
            ("src.pde.games.basis_selection", "src.pde.games", True),
            ("src.pde.gamestate", "src.pde.game", False),
        ],
    )
    def test_matcher_respects_module_boundaries(self, module, prefix, expected):
        assert _matches(module, prefix) is expected

    def test_allowed_dependencies_are_importable_names(self):
        """The layer's declared dependency surface stays small and explicit."""
        allowed_src_prefixes = (
            "src.pde.stochastic",
            "src.pde.time_stepping",
            "src.templates",
            "src.math_kernel",
            "src.constants",
            "src.poc",
            "src.research",
            "src.experiments",
        )
        for path in _guarded_files():
            for module in _imported_modules(path):
                if module.startswith("src."):
                    assert module.startswith(allowed_src_prefixes), (
                        f"{path.relative_to(REPO_ROOT)} imports {module}, which is "
                        "outside the declared dependency surface — extend the "
                        "allowlist deliberately if this is intended"
                    )
