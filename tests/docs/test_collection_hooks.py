"""Unit tests for ``tests/support/collection_hooks.py``'s two source scans.

``tests/docs/test_fem_required_visibility.py`` asks these scans two questions: which names a
conftest on a fem module's path binds that can change what pytest collects or runs (clause (f),
:func:`~tests.support.collection_hooks.collection_filters`), and which calls in a fem module can
skip it while it is imported (clause (g),
:func:`~tests.support.collection_hooks.import_time_calls_with`). The guard checks the live tree
and re-plants defects on copies; this file pins each scan on synthetic sources, including the
forms the live tree never contains. Hermetic: parses Python, runs nothing.

Copilot review 4 of PR #160 found one gap in each, both reproduced first with real pytest: a
conftest ``pytest_collection_finish`` that emptied ``session.items`` left a failing fem test
unrun ("1 passed") while the scan reported nothing, and ``def test_x() ->
pytest.skip(..., allow_module_level=True)`` skipped its module ("1 skipped") while the scan,
which never read annotations, reported nothing. The same module with ``from __future__ import
annotations`` ran its test, which failed as it should: the control.

The fix's checks weakened in place, 9/9 caught by named tests (2026-10-15; "the guard" is
``test_fem_required_visibility.py``):

1. Hooks back to the old fixed list -> ``test_names_that_can_change_what_runs`` for
   ``[collection-finish-hook]``, ``[runtestloop-hook]`` and ``[sessionfinish-hook]``, and the
   guard's two hook plants.
2. ``REVIEWED_HOOKS`` ignored -> ``[reviewed-display-hook]`` and every root-conftest plant in
   the guard (the live root defines ``pytest_terminal_summary``).
3. Annotations never read -> the four eager-annotation rows and the guard's
   ``[eager-annotation]`` plant.
4. Annotations read even when postponed -> the three postponed rows and
   ``[postponed-annotation]``.
5. Function bodies not read -> ``[helper-called-at-import]``, here and in the guard.
6. Any ``__future__`` import read as postponing ->
   ``test_only_a_future_import_postpones_annotations``.
7. ``*args`` / ``**kwargs`` annotations dropped ->
   ``test_annotations_of_reads_every_annotation_position``.
8. ``conftests_on_path`` stopping below the root -> every root-conftest plant in the guard.
9. A review admitted for a hook no conftest defines ->
   ``test_every_reviewed_hook_is_still_defined_on_a_fem_path``.
"""

from __future__ import annotations

import ast

import pytest

from tests.support.collection_hooks import (
    HOOK_PREFIX,
    REVIEWED_HOOKS,
    STAR_IMPORT,
    annotations_of,
    collection_filters,
    hooks_defined,
    import_time_calls_with,
    postpones_annotations,
)

#: The keyword clause (g) asks about.
ALLOW_MODULE_LEVEL = "allow_module_level"

#: ``name -> (conftest source, the names collection_filters must report)``.
FILTER_CASES: dict[str, tuple[str, set[str]]] = {
    "ignore-list": ('collect_ignore = ["tests/pde"]\n', {"collect_ignore"}),
    "ignore-glob": ('collect_ignore_glob = ["*fem*"]\n', {"collect_ignore_glob"}),
    "collection-finish-hook": (
        "def pytest_collection_finish(session):\n    session.items[:] = []\n",
        {"pytest_collection_finish"},
    ),
    "runtestloop-hook": (
        "def pytest_runtestloop(session):\n    return True\n",
        {"pytest_runtestloop"},
    ),
    "sessionfinish-hook": (
        "def pytest_sessionfinish(session):\n    session.exitstatus = 0\n",
        {"pytest_sessionfinish"},
    ),
    "plugins": ('pytest_plugins = ["helpers.hide"]\n', {"pytest_plugins"}),
    "imported-hook": (
        "from helpers import hide as pytest_ignore_collect\n",
        {"pytest_ignore_collect"},
    ),
    "imported-list": ("from helpers import collect_ignore\n", {"collect_ignore"}),
    "tuple-unpacking": ('collect_ignore, _ = ["x"], None\n', {"collect_ignore"}),
    "star-import": ("from helpers import *\n", {STAR_IMPORT}),
    "dynamic-binding": ('globals()["collect_ignore"] = ["x"]\n', {"globals"}),
    "reviewed-display-hook": ("def pytest_terminal_summary(terminalreporter):\n    pass\n", set()),
    "fixtures-only": (
        "import pytest\n\n\n@pytest.fixture\ndef board():\n    return 9\n",
        set(),
    ),
}

#: ``name -> (module source, lines import_time_calls_with must report)``.
IMPORT_TIME_CASES: dict[str, tuple[str, list[int]]] = {
    "module-level-call": ('import pytest\npytest.skip("x", allow_module_level=True)\n', [2]),
    "decorator": ("@deco(allow_module_level=True)\ndef f():\n    pass\n", [1]),
    "argument-default": ("def f(x=g(allow_module_level=True)):\n    pass\n", [1]),
    "class-body": ("class T:\n    x = g(allow_module_level=True)\n", [2]),
    "eager-return-annotation": (
        'import pytest\ndef test_x() -> pytest.skip("x", allow_module_level=True):\n    pass\n',
        [2],
    ),
    "eager-parameter-annotation": (
        "def f(x: g(allow_module_level=True)):\n    pass\n",
        [1],
    ),
    "eager-variadic-annotations": (
        "def f(*a: g(allow_module_level=1), k: g(allow_module_level=1), "
        "**kw: g(allow_module_level=1)):\n    pass\n",
        [1, 1, 1],
    ),
    "eager-variable-annotation": ("x: g(allow_module_level=True) = 1\n", [1]),
    "postponed-return-annotation": (
        "from __future__ import annotations\nimport pytest\n"
        'def test_x() -> pytest.skip("x", allow_module_level=True):\n    pass\n',
        [],
    ),
    "postponed-parameter-annotation": (
        "from __future__ import annotations\ndef f(x: g(allow_module_level=True)):\n    pass\n",
        [],
    ),
    "postponed-variable-annotation": (
        "from __future__ import annotations\nx: g(allow_module_level=True) = 1\n",
        [],
    ),
    "postponed-default-still-runs": (
        "from __future__ import annotations\ndef f(x=g(allow_module_level=True)):\n    pass\n",
        [2],
    ),
    "helper-called-at-import": (
        'import pytest\ndef _need():\n    pytest.skip("x", allow_module_level=True)\n_need()\n',
        [3],
    ),
    "skip-without-the-keyword": ('import pytest\ndef test_x():\n    pytest.skip("v")\n', []),
}


@pytest.mark.parametrize("case", sorted(FILTER_CASES))
def test_names_that_can_change_what_runs(case: str) -> None:
    source, expected = FILTER_CASES[case]
    assert collection_filters(source) == expected


@pytest.mark.parametrize("case", sorted(IMPORT_TIME_CASES))
def test_calls_that_can_run_at_import(case: str) -> None:
    source, expected = IMPORT_TIME_CASES[case]
    assert import_time_calls_with(source, ALLOW_MODULE_LEVEL) == expected


def test_only_a_future_import_postpones_annotations() -> None:
    assert postpones_annotations(ast.parse("from __future__ import annotations\n"))
    assert not postpones_annotations(ast.parse("from __future__ import division\n"))
    assert not postpones_annotations(ast.parse("import annotations\n"))


def test_annotations_of_reads_every_annotation_position() -> None:
    (function,) = ast.parse("def f(a: A, /, b: B, *c: C, d: D, **e: E) -> R:\n    pass\n").body
    assert [ast.unparse(a) for a in annotations_of(function)] == ["A", "B", "D", "C", "E", "R"]
    (variable,) = ast.parse("x: X = 1\n").body
    assert [ast.unparse(a) for a in annotations_of(variable)] == ["X"]
    assert annotations_of(ast.parse("x = 1\n").body[0]) == []


def test_hooks_defined_counts_reviewed_hooks_too() -> None:
    """What ``REVIEWED_HOOKS`` is matched against: every hook, admitted or not."""
    source = "def pytest_terminal_summary(t):\n    pass\n\n\ndef pytest_runtestloop(s):\n    pass\n"
    assert hooks_defined(source) == {"pytest_terminal_summary", "pytest_runtestloop"}
    assert hooks_defined("def helper():\n    pass\n") == set()


def test_every_reviewed_hook_is_a_hook_with_a_reason() -> None:
    """An entry names a hook pytest would call, and says why it cannot hide a test."""
    for name, reason in REVIEWED_HOOKS.items():
        assert name.startswith(HOOK_PREFIX), name
        assert len(reason.split()) >= 8, f"{name}'s review is too thin to audit: {reason!r}"


def test_both_outcomes_are_exercised() -> None:
    """Vacuity: a scan that reports everything, or nothing, fails one half of each table."""
    assert {bool(expected) for _, expected in FILTER_CASES.values()} == {True, False}
    assert {bool(expected) for _, expected in IMPORT_TIME_CASES.values()} == {True, False}
