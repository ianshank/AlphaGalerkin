"""Reading and driving the conftest code that decides what pytest collects.

Selected by a CI step is not the same as collected. A conftest can drop a module
(``collect_ignore``, ``pytest_ignore_collect``), drop or deselect items
(``pytest_collection_modifyitems``), replace collection outright, or load hooks from a plugin
module that no scan of conftests reads (``pytest_plugins``) -- while every check that reads
paths and steps stays green. Two guards use this module:

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

from tests.support.marker_expr import assigned_names

#: The file name pytest reads hooks and ``collect_ignore`` from, in every directory.
CONFTEST: Final[str] = "conftest.py"

#: The hook that may rearrange collected items; the root's is driven, not read.
COLLECTION_HOOK: Final[str] = "pytest_collection_modifyitems"

#: Conftest names that can remove a module or an item from collection, or replace collection:
#: the two ignore lists, every collection hook able to drop or replace what is collected
#: (``firstresult`` hooks that return an empty collector included), and ``pytest_plugins``,
#: which loads hooks from a module no scan of conftests reads.
COLLECTION_FILTERS: Final[frozenset[str]] = frozenset(
    {
        "collect_ignore",
        "collect_ignore_glob",
        "pytest_ignore_collect",
        COLLECTION_HOOK,
        "pytest_collection",
        "pytest_collect_directory",
        "pytest_make_collect_report",
        "pytest_pycollect_makemodule",
        "pytest_pycollect_makeitem",
        "pytest_plugins",
    }
)

#: What the root ``conftest.py`` may bind: its hook, because a guard drives it in every branch.
ROOT_DRIVEN_HOOKS: Final[frozenset[str]] = frozenset({COLLECTION_HOOK})


def collection_filters(source: str) -> set[str]:
    """Collection-filtering names a conftest binds or defines, at any depth."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found |= {node.name} & COLLECTION_FILTERS
        elif isinstance(node, ast.stmt):
            found |= assigned_names(node) & COLLECTION_FILTERS
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
    directory = path.parent
    while directory == root or root in directory.parents:
        conftest = directory / CONFTEST
        if conftest.exists():
            names = collection_filters(conftest.read_text(encoding="utf-8"))
            names -= ROOT_DRIVEN_HOOKS if directory == root else set()
            if names:
                offenders[conftest.relative_to(root).as_posix()] = names
        if directory == root:
            break
        directory = directory.parent
    return offenders


def name_lines(source: str, name: str) -> list[int]:
    """Lines where ``name`` is a NAME token -- code only, never a string or comment.

    Alias-proof where an AST call-name match is not: ``from pytest import
    importorskip as need`` still spells the name once.
    """
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [t.start[0] for t in tokens if t.type == tokenize.NAME and t.string == name]


def import_time_calls_with(source: str, keyword: str) -> list[int]:
    """Lines of calls passing ``keyword=`` that run at import: outside every function body.

    Decorators and argument defaults run at import too, so they are walked; a body
    is not, since it runs only when called.
    """
    found: list[int] = []

    def walk(node: ast.AST) -> None:
        if isinstance(node, ast.Call) and any(k.arg == keyword for k in node.keywords):
            found.append(node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
            for part in [*getattr(node, "decorator_list", []), *defaults]:
                walk(part)
            return
        for child in ast.iter_child_nodes(node):
            walk(child)

    walk(ast.parse(source))
    return found


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
