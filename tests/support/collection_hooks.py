"""Reading and driving the conftest code that decides what pytest collects.

Selected by a CI step is not the same as collected. A conftest can drop a module
(``collect_ignore``, ``pytest_ignore_collect``), drop or deselect items
(``pytest_collection_modifyitems``, ``pytest_collection_finish``), replace collection or the run
loop outright, or load hooks from a plugin module that no scan of conftests reads
(``pytest_plugins``) -- while every check that reads paths and steps stays green. Two guards use
this module:

* ``tests/docs/test_fem_required_visibility.py`` reads every conftest on a fem module's path,
  the root ``conftest.py`` included (clause (f)), and each fem module's own source (clause (g));
* ``tests/docs/test_root_conftest_hooks.py`` drives the root ``conftest.py``'s one collection
  hook. That hook is legitimate -- it gates the optional extras -- so it is run in every
  branch rather than exempted, which is what (f) used to do with the whole file (Copilot
  review 2, PR #160).
"""

from __future__ import annotations

import ast
import importlib.util
import io
import tokenize
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Final

#: The file name pytest reads hooks and ``collect_ignore`` from, in every directory.
CONFTEST: Final[str] = "conftest.py"

#: The hook that may rearrange collected items; the root's is driven, not read.
COLLECTION_HOOK: Final[str] = "pytest_collection_modifyitems"

#: Every hook pytest calls on a conftest is named with this prefix, and so is ``pytest_plugins``,
#: which loads hooks from a module no scan of conftests reads.
HOOK_PREFIX: Final[str] = "pytest_"

#: The two non-hook names pytest reads from a conftest to leave whole paths uncollected.
IGNORE_LISTS: Final[frozenset[str]] = frozenset({"collect_ignore", "collect_ignore_glob"})

#: Hooks a conftest may define without being reported, ``name -> the review that admitted it``.
#: Every other hook is reported: a list of the hooks that can drop, skip or re-rate a test grew
#: with each review -- ``pytest_collection_finish`` was its third miss (Copilot review 4, PR
#: #160), and pytest calls 52 hooks. Bodies are not read, since every hook receives objects it
#: could mutate, so an entry is a review decision, and
#: ``tests/docs/test_fem_required_visibility.py`` fails one that no conftest it reads defines.
REVIEWED_HOOKS: Final[dict[str, str]] = {
    "pytest_terminal_summary": (
        "called after every test has run, to add lines to the summary: the root's skip "
        "counts per optional extra and the E2E tier's resolved device"
    ),
}

#: What the root ``conftest.py`` may bind: its hook, because a guard drives it in every branch.
ROOT_DRIVEN_HOOKS: Final[frozenset[str]] = frozenset({COLLECTION_HOOK})

#: Reported for ``from module import *``: which names it binds is not readable here.
STAR_IMPORT: Final[str] = "*"
#: Reported for a call that can bind any name at run time (``globals()["collect_ignore"]
#: = ...``, ``exec``, ``setattr`` on the module); named, unlike a binding, by the builtin called.
DYNAMIC_BINDERS: Final[frozenset[str]] = frozenset({"globals", "vars", "exec", "setattr"})
#: ``from __future__ import annotations`` (PEP 563): annotations become strings that never run.
_FUTURE_MODULE: Final[str] = "__future__"
_POSTPONED_ANNOTATIONS: Final[str] = "annotations"
#: Nodes that bind the name in their ``name`` field (``None`` for a bare ``except:``).
_NAMED_BINDINGS: Final[tuple[type[ast.AST], ...]] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.ExceptHandler,
    ast.MatchAs,
    ast.MatchStar,
)


def _imported_names(node: ast.Import | ast.ImportFrom) -> set[str]:
    """Names an import binds: ``import a.b`` binds ``a``; ``from m import x as y`` binds ``y``."""
    if isinstance(node, ast.Import):
        return {alias.asname or alias.name.partition(".")[0] for alias in node.names}
    return {alias.asname or alias.name for alias in node.names}


def bound_names(tree: ast.AST) -> set[str]:
    """Every name ``tree`` binds, at any depth, however it is bound.

    ``=`` in any form (tuple unpacking, ``for``/``with`` targets, ``:=``), ``def`` and
    ``class``, ``except ... as``, a ``match`` capture, and imports with their aliases.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            found.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            found |= _imported_names(node)
        elif isinstance(node, _NAMED_BINDINGS) and node.name:
            found.add(node.name)
    return found


def collection_filters(source: str) -> set[str]:
    """Names a conftest binds, at any depth, that can change what pytest collects or runs.

    Every hook it defines (a :data:`HOOK_PREFIX` name, ``pytest_plugins`` included) unless
    :data:`REVIEWED_HOOKS` admits it, and the two :data:`IGNORE_LISTS`. Imports count: ``from
    helpers import collect_ignore`` binds a filter as surely as ``=`` (Copilot review 3, PR #160).
    A star import (:data:`STAR_IMPORT`) and a call that binds at run time
    (:data:`DYNAMIC_BINDERS`, reported by name) are reported, since what they bind is not
    readable here.
    """
    tree = ast.parse(source)
    bound = bound_names(tree)
    found = hooks_defined(source) - set(REVIEWED_HOOKS)
    found |= bound & (IGNORE_LISTS | {STAR_IMPORT})
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            found |= {node.func.id} & DYNAMIC_BINDERS
    return found


def hooks_defined(source: str) -> set[str]:
    """Every hook a conftest binds, reviewed or not: what :data:`REVIEWED_HOOKS` must match."""
    return {name for name in bound_names(ast.parse(source)) if name.startswith(HOOK_PREFIX)}


def conftests_on_path(path: Path, root: Path) -> list[Path]:
    """Every ``conftest.py`` pytest reads for ``path``, from its directory up to ``root``'s."""
    found: list[Path] = []
    directory = path.parent
    while directory == root or root in directory.parents:
        if (directory / CONFTEST).exists():
            found.append(directory / CONFTEST)
        if directory == root:
            break
        directory = directory.parent
    return found


def conftest_filters_on_path(path: Path, root: Path) -> dict[str, set[str]]:
    """The collection filters each conftest from ``path``'s directory up to ``root`` binds.

    Args:
        path: A test module (or any file) under ``root``.
        root: The repository root. Its own ``conftest.py`` is read too; only the hook in
            :data:`ROOT_DRIVEN_HOOKS` is left out there, because a guard drives it.

    Returns:
        Root-relative conftest path -> the filters it binds, for conftests binding any.

    """
    offenders: dict[str, set[str]] = {}
    for conftest in conftests_on_path(path, root):
        names = collection_filters(conftest.read_text(encoding="utf-8"))
        names -= ROOT_DRIVEN_HOOKS if conftest.parent == root else set()
        if names:
            offenders[conftest.relative_to(root).as_posix()] = names
    return offenders


def name_lines(source: str, name: str) -> list[int]:
    """Lines where ``name`` is a NAME token -- code only, never a string or comment.

    Alias-proof where an AST call-name match is not: ``from pytest import
    importorskip as need`` still spells the name once.
    """
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [t.start[0] for t in tokens if t.type == tokenize.NAME and t.string == name]


def postpones_annotations(tree: ast.Module) -> bool:
    """Whether ``from __future__ import annotations`` makes the module's annotations strings."""
    return any(
        isinstance(statement, ast.ImportFrom)
        and statement.module == _FUTURE_MODULE
        and any(alias.name == _POSTPONED_ANNOTATIONS for alias in statement.names)
        for statement in tree.body
    )


def annotations_of(node: ast.AST) -> list[ast.expr]:
    """The annotations ``node`` carries: a def's parameters and return, or an ``x: T`` target."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = node.args
        parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        parameters += [p for p in (arguments.vararg, arguments.kwarg) if p is not None]
        found = [p.annotation for p in parameters if p.annotation is not None]
        return [*found, node.returns] if node.returns is not None else found
    if isinstance(node, ast.AnnAssign):
        return [node.annotation]
    return []


def import_time_calls_with(source: str, keyword: str) -> list[int]:
    """Lines of calls passing ``keyword=`` that can run while the module is imported.

    Module and class bodies, decorators and argument defaults run at import, and so do
    annotations -- eagerly, on the Pythons CI runs -- unless ``from __future__ import
    annotations`` postpones them: ``def test_x() -> pytest.skip(..., allow_module_level=True)``
    skipped its module before collection (Copilot review 4, PR #160). A function body runs when
    called, and the scan does not follow calls, so a call in any body counts too: a helper the
    module calls at import is the same gap one level down. Only postponed annotations are left
    out, since nothing in them runs.
    """
    tree = ast.parse(source)
    postponed: set[int] = set()
    if postpones_annotations(tree):
        postponed = {
            id(part)
            for node in ast.walk(tree)
            for annotation in annotations_of(node)
            for part in ast.walk(annotation)
        }
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and id(node) not in postponed
        and any(k.arg == keyword for k in node.keywords)
    )


class FakeItem:
    """The two methods of ``pytest.Item`` the root hook calls, recording the markers it adds."""

    def __init__(self, markers: set[str]) -> None:
        self.markers = markers
        self.added: list[object] = []

    def get_closest_marker(self, name: str) -> object | None:
        return name if name in self.markers else None

    def add_marker(self, marker: object) -> None:
        self.added.append(marker)


class _DeselectionRecorder:
    """``config.hook``, as far as reporting deselected items goes."""

    def __init__(self) -> None:
        self.deselected: list[object] = []

    def pytest_deselected(self, *, items: list[object]) -> None:
        self.deselected.extend(items)


class FakeConfig:
    """A ``config`` whose ``hook.pytest_deselected`` records what a hook deselects."""

    def __init__(self) -> None:
        self.hook = _DeselectionRecorder()


def load_module(path: Path, name: str) -> ModuleType:
    """A fresh, unregistered copy of the module at ``path``, executed now.

    It reads the environment at this call, and ``sys.modules`` never sees it, so setting its
    globals cannot leak into the session.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rearrangements(
    hook: Callable[[FakeConfig, list[FakeItem]], object], items: list[FakeItem]
) -> list[str]:
    """Run a collection hook over ``items``, and report how it changed which of them run.

    Removing, adding or reordering items in place, or reporting any as deselected, hides a
    test from every step that selects it. Adding a marker changes how an item runs, not
    whether it is collected, so markers are left to the caller. Exceptions propagate.
    """
    before = list(items)
    config = FakeConfig()
    hook(config, items)
    problems: list[str] = []
    if len(items) != len(before) or any(now is not was for now, was in zip(items, before)):
        problems.append(f"it removed, added or reordered items ({len(before)} -> {len(items)})")
    if config.hook.deselected:
        problems.append(f"it reported {len(config.hook.deselected)} item(s) deselected")
    return problems
