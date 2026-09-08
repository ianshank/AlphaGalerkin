"""Import-graph: refinement games must not register from package ``__init__``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

INIT_FILES = (
    "src/pde/__init__.py",
    "src/pde/games/__init__.py",
    "src/refinement/__init__.py",
    "src/research/substrates/__init__.py",
)


def _imports_register_refinement_games_module(path: Path) -> bool:
    """True iff ``path`` imports the side-effect registration *module*.

    Re-exporting the decorator ``register_refinement_game`` from
    ``src.refinement.registry`` is fine and expected.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "src.pde.register_refinement_games" or mod.endswith(
                ".register_refinement_games"
            ):
                return True
            if mod == "register_refinement_games":
                return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src.pde.register_refinement_games" or alias.name.endswith(
                    ".register_refinement_games"
                ):
                    return True
    return False


@pytest.mark.parametrize("rel", INIT_FILES)
def test_init_does_not_import_register_refinement_games(rel: str) -> None:
    path = REPO_ROOT / rel
    assert path.is_file(), f"missing {rel}"
    assert not _imports_register_refinement_games_module(path), (
        f"{rel} must not import src.pde.register_refinement_games — use an explicit "
        f"`import src.pde.register_refinement_games` at the call site "
        f"(SIGSEGV / coverage-tracer class documented in src/pde/games/__init__.py)"
    )


def test_register_module_exists_and_registers() -> None:
    import src.pde.register_refinement_games  # noqa: F401
    from src.pde.games.substrate_refinement import GAME_REGISTRY_NAME
    from src.refinement.registry import RefinementGameRegistry

    assert RefinementGameRegistry().get(GAME_REGISTRY_NAME) is not None
