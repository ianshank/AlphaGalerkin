"""Guards that ``tests/integrations/eval_harness/`` cannot become uncountable again.

**Defect class:** an optional-extra test suite whose absence-handling yields zero
items, so no hook can count it, no ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` can fail
it, and no coverage gate can measure it -- while the job that installs the
extra "specifically to un-skip" it reports green.

Until 2026-09-11 (R-13, hygiene B37) that was the exact state of this suite:
eight files opened with ``pytest.importorskip("eval_harness")``, the
``test-extras`` job installed ``.[eval-harness]`` and then selected none of
them, ``src/integrations/eval_harness/*`` sat in pyproject.toml's global
coverage ``omit`` with no overriding gate, and the integrity guard's exemption
for it had been silenced with a citation to a Regression Surface row that never
existed. The eighth recorded instance of this repo's invisibility defect, found
-- again -- by a person reading a file.

What this file asserts, each clause cheap and exact:

* **(a)** no file under the suite calls ``importorskip("eval_harness")`` or
  imports ``eval_harness`` at module scope -- either shape yields zero items;
* **(b)** every ``test_*.py`` there carries ``eval_harness_required`` via a
  module-level ``pytestmark``, so the root hook and ``-m`` can see it;
* **(c)** the root ``conftest.py`` probes the extra with
  ``importlib.util.find_spec("eval_harness")``, and its hook -- **driven**, not
  grepped -- hard-fails under ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` naming the
  install command, and otherwise skips with a count that reaches the terminal
  summary;
* **(d)** a ``test-extras`` step selects the directory, requires the marker
  positively, gates ``--cov=src/integrations/eval_harness`` with a positive
  ``--cov-fail-under``, overrides the omit with ``--cov-config=``, and sets
  ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` -- and ``test-extras`` is a hard merge gate;
  the ``Makefile`` mirror carries the same shape and is driven with a stub
  ``PYTEST`` so a rewrite that preserves the spelling cannot hide;
* **(e)** collection of the suite succeeds in a subprocess on *this* install,
  with or without the extra, and under ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` the
  outcome matches the install (usage error without it, clean with it).

Mutations planted and killed (harden-a-guard; each reverted after the kill):

1. ``pytest.importorskip("eval_harness")`` restored at module scope in
   ``test_scorers.py`` -> ``test_no_module_level_importorskip_of_the_extra[test_scorers.py]``.
2. ``pytestmark`` line deleted from ``test_plugins.py`` ->
   ``test_every_test_file_carries_the_marker[test_plugins.py]``.
3. The whole ``ci.yml`` gate step deleted ->
   ``test_a_test_extras_step_gates_the_package`` (and, independently,
   ``test_coverage_gate_integrity.py::test_every_omit_entry_is_gated_somewhere``,
   which is why the exemption was deleted rather than reworded).
4. ``raise pytest.UsageError(...)`` in the root hook replaced by ``pass`` ->
   ``TestTheHookIsDriven::test_hard_fails_under_require_extras``.
5. The step's ``-m "eval_harness_required"`` flipped to
   ``-m "not eval_harness_required"`` ->
   ``test_the_gate_step_selects_the_marker_positively``.
6. ``--cov-fail-under=1`` lowered to ``--cov-fail-under=0`` (a gate that cannot
   fail) -> ``test_the_gate_threshold_is_positive``.

Hermetic except clause (e) and the Makefile drive, which spawn one pytest
collection / one ``make`` with a stub each. No network.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

from tests.docs.test_coverage_gate_integrity import _OMIT_WITHOUT_A_CI_GATE
from tests.docs.test_marker_vocabulary import registered_markers
from tests.support.marker_expr import (
    expression_excludes,
    expression_requires,
    marker_expressions,
)
from tests.support.workflows import (
    CI_SUCCESS_JOB,
    CI_WORKFLOW,
    MAKEFILE,
    REPO_ROOT,
    hard_gate_jobs,
    iter_commands,
    job_needs,
    load_workflow,
    makefile_target_recipe,
)

# --------------------------------------------------------------------------- #
# Named constants -- nothing below spells these inline.                        #
# --------------------------------------------------------------------------- #

#: The suite this file guards.
SUITE_DIR: Final[Path] = REPO_ROOT / "tests" / "integrations" / "eval_harness"

#: Glob for the suite's test modules, as pytest collects them.
TEST_GLOB: Final[str] = "test_*.py"

#: The importable name of the optional extra.
EXTRA_MODULE: Final[str] = "eval_harness"

#: The marker that makes the suite countable.
MARKER: Final[str] = "eval_harness_required"

#: The package the CI step must gate.
COV_TARGET: Final[str] = "src/integrations/eval_harness"

#: The job that installs the extra and therefore owns the gate.
GATE_JOB: Final[str] = "test-extras"

#: The Makefile target mirroring that step.
MAKE_TARGET: Final[str] = "test-eval-harness"

#: The environment variable that turns a missing extra into a collection error.
REQUIRE_EXTRAS_ENV: Final[str] = "ALPHAGALERKIN_REQUIRE_EXTRAS"

#: The exact ``pip`` invocation the hook must name.
INSTALL_HINT: Final[str] = "pip install -e '.[eval-harness]'"

#: The root conftest under test.
ROOT_CONFTEST: Final[Path] = REPO_ROOT / "conftest.py"

#: The pyproject omit pattern that must never regain an exemption.
OMIT_PATTERN: Final[str] = "src/integrations/eval_harness/*"

#: The two ways a shell argument can name the suite directory. A token merely
#: *prefixed* by one of these (a single file) is deliberately not a match.
SUITE_DIRECTORY_TOKENS: Final[frozenset[str]] = frozenset(
    {"tests/integrations/eval_harness", "tests/integrations/eval_harness/"}
)

#: Vacuity floors. The suite had 11 files on 2026-09-11 (8 of which used to
#: importorskip the extra); a scan that finds fewer is looking at the wrong
#: directory.
MIN_TEST_FILES: Final[int] = 8
#: Items the suite collects on a base install (39 measured 2026-09-11). A
#: collection that yields fewer than this has lost files, not gained skips.
MIN_COLLECTED_ITEMS: Final[int] = 30
#: Marked items the hook must report skipped when the extra is absent (39
#: measured 2026-09-11, every file being marked).
MIN_MARKED_ITEMS: Final[int] = 20

#: pytest's exit code for a ``UsageError`` raised during collection.
USAGE_ERROR_EXIT: Final[int] = int(pytest.ExitCode.USAGE_ERROR)

#: Whether the extra is importable *here*. Clause (e) branches on it, so both
#: install states are asserted rather than one being vacuous.
EXTRA_INSTALLED: Final[bool] = importlib.util.find_spec(EXTRA_MODULE) is not None

_COV_FAIL_UNDER = re.compile(r"--cov-fail-under=(?P<n>\d+)")
_COLLECTED = re.compile(r"(?P<n>\d+) tests? collected")
_COLLECTION_ERRORS = re.compile(r"\berrors?\b")
_SKIP_SUMMARY = re.compile(rf"{MARKER}: skipped (?P<n>\d+) test\(s\)")


# --------------------------------------------------------------------------- #
# AST helpers, unit-tested below so a parser that matches nothing cannot pass. #
# --------------------------------------------------------------------------- #


def suite_files() -> list[Path]:
    """Every ``test_*.py`` in the suite, sorted for stable parametrisation."""
    return sorted(SUITE_DIR.glob(TEST_GLOB))


def _call_name(node: ast.Call) -> str:
    """``pytest.importorskip`` / ``importorskip`` / ``find_spec`` -> the last attribute."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings, for resolving a Name argument."""
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                found[target.id] = node.value.value
    return found


def _first_arg_string(node: ast.Call, constants: dict[str, str]) -> str | None:
    """The first positional argument as a string, resolving module constants."""
    if not node.args:
        return None
    arg = node.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        return constants.get(arg.id)
    return None


def importorskip_targets(source: str) -> list[str]:
    """Every module name passed to an ``importorskip`` call anywhere in ``source``.

    Anywhere, not just module scope: an ``importorskip`` inside a fixture still
    turns a marked test into an *unmarked* skip that the hook cannot count.
    """
    tree = ast.parse(source)
    constants = _string_constants(tree)
    return [
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "importorskip"
        and (name := _first_arg_string(node, constants)) is not None
    ]


def module_scope_imports_of(source: str, package: str) -> list[str]:
    """Top-level ``import x`` / ``from x import`` statements naming ``package``."""
    tree = ast.parse(source)
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(
                alias.name
                for alias in node.names
                if alias.name == package or alias.name.startswith(package + ".")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == package or node.module.startswith(package + "."):
                found.append(node.module)
    return found


def _mark_names(node: ast.expr) -> list[str]:
    """``pytest.mark.x`` -> ``["x"]``; a list/tuple of them -> all names."""
    if isinstance(node, ast.Attribute):
        owner = node.value
        if isinstance(owner, ast.Attribute) and owner.attr == "mark":
            return [node.attr]
        return []
    if isinstance(node, (ast.List, ast.Tuple)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_mark_names(element))
        return names
    return []


def pytestmark_names(source: str) -> set[str]:
    """Marker names applied by a module-level ``pytestmark`` assignment."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            names.update(_mark_names(node.value))
    return names


def find_spec_targets(source: str) -> list[str]:
    """Every module name passed to a ``find_spec`` call in ``source``."""
    tree = ast.parse(source)
    constants = _string_constants(tree)
    return [
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _call_name(node) == "find_spec"
        and (name := _first_arg_string(node, constants)) is not None
    ]


class TestTheParsers:
    """Synthetic-input tests so the live checks cannot be vacuous."""

    def test_importorskip_is_found_at_module_scope_and_inside_a_fixture(self) -> None:
        source = (
            "import pytest\n"
            'pytest.importorskip("eval_harness")\n'
            "def f():\n"
            '    pytest.importorskip("torch")\n'
        )
        assert importorskip_targets(source) == ["eval_harness", "torch"]

    def test_importorskip_resolves_a_module_constant(self) -> None:
        source = 'import pytest\nNAME = "eval_harness"\npytest.importorskip(NAME)\n'
        assert importorskip_targets(source) == ["eval_harness"]

    def test_a_file_without_importorskip_yields_nothing(self) -> None:
        assert importorskip_targets("import pytest\npytestmark = pytest.mark.x\n") == []

    def test_module_scope_imports_match_the_package_and_its_submodules_only(self) -> None:
        source = (
            "import eval_harness\n"
            "from eval_harness.core import types\n"
            "from eval_harness_extra import y\n"
            "import evaluation\n"
            "def f():\n"
            "    from eval_harness.plugins import z\n"
        )
        assert module_scope_imports_of(source, "eval_harness") == [
            "eval_harness",
            "eval_harness.core",
        ]

    def test_pytestmark_single_and_list_forms(self) -> None:
        assert pytestmark_names("import pytest\npytestmark = pytest.mark.a\n") == {"a"}
        assert pytestmark_names("import pytest\npytestmark = [pytest.mark.a, pytest.mark.b]\n") == {
            "a",
            "b",
        }
        assert pytestmark_names("import pytest\nother = pytest.mark.a\n") == set()

    def test_find_spec_literal_and_constant_forms(self) -> None:
        literal = 'import importlib.util\nx = importlib.util.find_spec("eval_harness")\n'
        constant = 'import importlib.util\nM = "eval_harness"\nx = importlib.util.find_spec(M)\n'
        assert find_spec_targets(literal) == ["eval_harness"]
        assert find_spec_targets(constant) == ["eval_harness"]
        assert find_spec_targets("x = 1\n") == []


# --------------------------------------------------------------------------- #
# Vacuity                                                                      #
# --------------------------------------------------------------------------- #


def test_the_suite_directory_is_not_empty() -> None:
    """Without this, every parametrised assertion below iterates nothing."""
    assert SUITE_DIR.is_dir(), f"{SUITE_DIR} is missing -- the guard has no subject"
    assert len(suite_files()) >= MIN_TEST_FILES, (
        f"expected at least {MIN_TEST_FILES} test files under {SUITE_DIR}, "
        f"found {len(suite_files())}"
    )


def test_the_marker_is_registered() -> None:
    """The marker must be in pyproject.toml's registered vocabulary.

    ``--strict-markers`` rejects an unregistered marker on a test, so this is
    what lets clause (b) be satisfiable at all -- and what stops a typo in the
    CI step's ``-m`` from selecting nothing.
    """
    assert MARKER in registered_markers(), f"{MARKER!r} is not in pyproject.toml's markers"


# --------------------------------------------------------------------------- #
# (a) no importorskip / module-scope import of the extra                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", suite_files(), ids=lambda p: p.name)
def test_no_module_level_importorskip_of_the_extra(path: Path) -> None:
    """A ``pytest.importorskip("eval_harness")`` yields zero items: uncountable.

    Mutation 1: restored in ``test_scorers.py`` -> this test, for that id.
    """
    source = path.read_text(encoding="utf-8")
    assert EXTRA_MODULE not in importorskip_targets(source), (
        f"{path.name} calls importorskip({EXTRA_MODULE!r}); use the "
        f"{MARKER} marker so the root conftest can count or hard-fail it"
    )
    assert module_scope_imports_of(source, EXTRA_MODULE) == [], (
        f"{path.name} imports {EXTRA_MODULE} at module scope, which is a collection "
        "error without the extra -- move it into a fixture"
    )


# --------------------------------------------------------------------------- #
# (b) every file carries the marker                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", suite_files(), ids=lambda p: p.name)
def test_every_test_file_carries_the_marker(path: Path) -> None:
    """Mutation 2: ``pytestmark`` deleted from ``test_plugins.py`` -> this test.

    Files that need only the base install (``test_config.py``, ``test_oracle.py``,
    ``test_target.py``) are required to carry it too, and the reason is a trap
    this repo has already fallen into: the gate step selects
    ``-m "eval_harness_required"``, so an unmarked file's tests would run in the
    fast lane and sit **outside** the measurement of the very package the gate
    exists for (``config.py`` / ``oracle.py`` / ``target.py`` are in the global
    omit, so nothing else measures them). The price is that those 11 tests now
    run in ``test-extras`` -- a hard merge gate on every pull request, but
    skipped on the nightly ``schedule`` -- and are counted skips on a base
    install rather than silent ones. Disclosed here and in CLAUDE.md.
    """
    source = path.read_text(encoding="utf-8")
    assert MARKER in pytestmark_names(source), (
        f"{path.name} has no module-level `pytestmark = pytest.mark.{MARKER}`"
    )


# --------------------------------------------------------------------------- #
# (c) the root hook -- spelled correctly AND driven                            #
# --------------------------------------------------------------------------- #


def test_root_conftest_probes_the_extra_with_find_spec() -> None:
    source = ROOT_CONFTEST.read_text(encoding="utf-8")
    assert EXTRA_MODULE in find_spec_targets(source), (
        f"root conftest.py must probe the extra with find_spec({EXTRA_MODULE!r})"
    )
    assert MARKER in source, f"root conftest.py never mentions the {MARKER} marker"
    assert INSTALL_HINT in source, (
        f"root conftest.py must name the install command {INSTALL_HINT!r}"
    )


@dataclass
class _FakeItem:
    """The two methods of ``pytest.Item`` the hook touches."""

    markers: set[str]
    added: list[Any] = field(default_factory=list)

    def get_closest_marker(self, name: str) -> object | None:
        return object() if name in self.markers else None

    def add_marker(self, marker: Any) -> None:
        self.added.append(marker)


def _load_root_conftest() -> ModuleType:
    """Import the root conftest as a *private* module instance.

    Monkeypatching the live ``conftest`` would alter the running session's
    hook; a separate instance lets each branch be forced without side effects.
    """
    spec = importlib.util.spec_from_file_location("_root_conftest_under_test", ROOT_CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheHookIsDriven:
    """Both branches of the hook, forced, on this install."""

    @pytest.fixture
    def hook(self) -> ModuleType:
        return _load_root_conftest()

    def test_hard_fails_under_require_extras(
        self, hook: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation 4: ``raise pytest.UsageError`` -> ``pass`` -> this test."""
        monkeypatch.setattr(hook, "_HAS_EVAL_HARNESS", False)
        monkeypatch.setattr(hook, "_REQUIRE_EXTRAS", True)
        items = [_FakeItem({MARKER}), _FakeItem({MARKER}), _FakeItem(set())]
        with pytest.raises(pytest.UsageError) as excinfo:
            hook.pytest_collection_modifyitems(None, items)
        message = str(excinfo.value)
        assert f"{REQUIRE_EXTRAS_ENV}=1" in message
        assert "2 " + MARKER in message, "the error must count the marked tests"
        assert INSTALL_HINT in message, "the error must name the install command"

    def test_skips_with_a_count_when_not_required(
        self, hook: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hook, "_HAS_EVAL_HARNESS", False)
        monkeypatch.setattr(hook, "_REQUIRE_EXTRAS", False)
        monkeypatch.setattr(hook, "_eval_harness_skip_count", 0)
        # No gpu_required here: the CUDA leg of the same hook would add its own
        # skip on a CPU box and the count below would measure two hooks at once.
        marked = [_FakeItem({MARKER}), _FakeItem({MARKER, "integration"})]
        unmarked = _FakeItem({"fem_required"})
        monkeypatch.setattr(hook, "_HAS_SKFEM", True)  # isolate the fem leg
        hook.pytest_collection_modifyitems(None, [*marked, unmarked])
        for item in marked:
            reasons = [m.kwargs.get("reason", "") for m in item.added if m.name == "skip"]
            assert any("langfuse-eval-harness" in reason for reason in reasons), item
        assert unmarked.added == [], "an unmarked item must be left alone"
        assert hook._eval_harness_skip_count == len(marked)

    def test_the_count_reaches_the_terminal_summary(
        self, hook: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A count nobody prints is not visible; this is the visibility half."""
        monkeypatch.setattr(hook, "_eval_harness_skip_count", 7)
        monkeypatch.setattr(hook, "_fem_skip_count", 0)
        monkeypatch.setattr(hook, "_gpu_skip_count", 0)
        lines: list[str] = []

        class _Reporter:
            def write_line(self, line: str, **_: Any) -> None:
                lines.append(line)

        hook.pytest_terminal_summary(_Reporter(), 0, None)
        (line,) = lines
        assert line.startswith(f"{MARKER}: skipped 7 test(s)")
        assert INSTALL_HINT in line

    def test_does_nothing_when_the_extra_is_present(
        self, hook: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(hook, "_HAS_EVAL_HARNESS", True)
        monkeypatch.setattr(hook, "_REQUIRE_EXTRAS", True)
        monkeypatch.setattr(hook, "_HAS_SKFEM", True)
        item = _FakeItem({MARKER})
        hook.pytest_collection_modifyitems(None, [item])
        assert item.added == []


# --------------------------------------------------------------------------- #
# (d) the CI step and its Makefile mirror                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateStep:
    """One ``test-extras`` step that runs pytest over the suite with a gate."""

    name: str
    command: str
    env: dict[str, str]


def _names_the_suite(command: str) -> bool:
    return any(token in SUITE_DIRECTORY_TOKENS for token in command.split())


def _gates_the_package(command: str) -> bool:
    return re.search(rf"--cov={re.escape(COV_TARGET)}(?=\s|$)", command) is not None


def gate_steps() -> list[GateStep]:
    """Every ``test-extras`` step whose script selects the suite AND gates the package."""
    document = load_workflow(CI_WORKFLOW)
    job = (document.get("jobs") or {}).get(GATE_JOB) or {}
    found: list[GateStep] = []
    for index, step in enumerate(job.get("steps") or []):
        script = step.get("run")
        if not isinstance(script, str):
            continue
        env = {str(k): str(v) for k, v in (step.get("env") or {}).items()}
        for command in iter_commands(script):
            if _names_the_suite(command) and _gates_the_package(command):
                found.append(GateStep(step.get("name") or f"step[{index}]", command, env))
    return found


def _threshold(command: str) -> int | None:
    match = _COV_FAIL_UNDER.search(command)
    return int(match.group("n")) if match else None


def _requires_extras(step_env: dict[str, str], command: str) -> bool:
    """``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` via the step ``env:`` or an inline prefix."""
    if step_env.get(REQUIRE_EXTRAS_ENV) == "1":
        return True
    return f"{REQUIRE_EXTRAS_ENV}=1" in command.split()


def test_a_test_extras_step_gates_the_package() -> None:
    """Mutation 3: the whole step deleted -> this test."""
    steps = gate_steps()
    assert steps, (
        f"no `{GATE_JOB}` step in ci.yml both selects {SUITE_DIR.relative_to(REPO_ROOT)} "
        f"and passes --cov={COV_TARGET}; the suite is measured by nothing"
    )
    for step in steps:
        assert _threshold(step.command) is not None, f"{step.name}: no --cov-fail-under"
        assert "--cov-config=" in step.command, (
            f"{step.name}: {COV_TARGET} is in pyproject's omit; without --cov-config= "
            "the step measures 0.00% and cannot fail"
        )


def test_the_gate_threshold_is_positive() -> None:
    """Mutation 6: ``--cov-fail-under=0`` -> this test. Zero can never fail."""
    for step in gate_steps():
        threshold = _threshold(step.command)
        assert threshold is not None and threshold >= 1, (
            f"{step.name}: --cov-fail-under={threshold} cannot fail on an omit collision"
        )


def test_the_gate_step_selects_the_marker_positively() -> None:
    """Mutation 5: ``-m "eval_harness_required"`` -> ``"not ..."`` -> this test.

    Selecting the marker is what makes the step's *test set* the suite. With
    every file marked, an exclusion selects nothing and reports green on an
    empty run -- the same green, measuring nothing.
    """
    for step in gate_steps():
        expressions = marker_expressions(step.command)
        assert expressions, f"{step.name}: no -m expression"
        assert all(expression_requires(e, MARKER) for e in expressions), (
            f"{step.name}: -m {expressions} does not positively require {MARKER}"
        )
        assert not any(expression_excludes(e, MARKER) for e in expressions)


def test_the_gate_step_requires_the_extra() -> None:
    """The step must run under ``ALPHAGALERKIN_REQUIRE_EXTRAS=1``.

    Without it a half-installed extra skips every marked test and the step
    fails on "coverage too low" -- naming the symptom, not the cause.
    """
    for step in gate_steps():
        assert _requires_extras(step.env, step.command), (
            f"{step.name}: {REQUIRE_EXTRAS_ENV}=1 is set neither in env: nor inline"
        )


def test_the_gate_job_blocks_the_merge() -> None:
    """A gate in a job that cannot fail ``ci-success`` is a report."""
    document = load_workflow(CI_WORKFLOW)
    assert GATE_JOB in job_needs(document, CI_SUCCESS_JOB)
    ci_success = (document.get("jobs") or {})[CI_SUCCESS_JOB]
    scripts = [s.get("run") for s in ci_success.get("steps") or [] if isinstance(s, dict)]
    gated = set()
    for script in scripts:
        if isinstance(script, str):
            gated |= hard_gate_jobs(script)
    assert GATE_JOB in gated, f"{GATE_JOB} is in needs but has no `exit 1` block"


def test_the_omit_exemption_did_not_come_back() -> None:
    """The integrity-guard exemption may not return as a substitute for the gate.

    Self-expiring in the other direction: the exemption that hid B37 for months
    was deleted, not reworded, so a deleted gate fails the integrity guard.
    """
    assert OMIT_PATTERN not in _OMIT_WITHOUT_A_CI_GATE


def _raw_recipe_lines(target: str, makefile: Path = MAKEFILE) -> list[str]:
    """Recipe lines of ``target`` with Make's prefixes intact (for the ``-`` check)."""
    lines: list[str] = []
    collecting = False
    for line in makefile.read_text(encoding="utf-8").splitlines():
        if re.match(rf"^{re.escape(target)}\s*:(?!=)", line):
            collecting = True
            continue
        if not collecting:
            continue
        if line.startswith("\t"):
            lines.append(line[1:])
        elif line.strip() and not line.lstrip().startswith("#"):
            break
    return lines


def test_the_makefile_mirror_matches_the_ci_step() -> None:
    recipe = makefile_target_recipe(MAKE_TARGET)
    assert recipe, f"Makefile has no `{MAKE_TARGET}` target"
    (command,) = [c for c in recipe if _names_the_suite(c)]
    assert _gates_the_package(command)
    assert "--cov-config=" in command
    assert "--cov-fail-under=" in command
    assert all(expression_requires(e, MARKER) for e in marker_expressions(command))
    assert _requires_extras({}, command)
    for line in _raw_recipe_lines(MAKE_TARGET):
        assert not line.startswith("-"), (
            f"`{MAKE_TARGET}` carries Make's `-` prefix, which discards the gate's exit status"
        )


def test_the_makefile_mirror_is_driven_by_the_stub(tmp_path: Path) -> None:
    """Drive ``make`` with a stub ``PYTEST`` instead of asserting its spelling.

    A rewrite that preserves the text but drops the exit status (``-`` prefix,
    ``A; B``), the marker or the target cannot pass this; ~50 ms, no pytest.
    """
    record = tmp_path / "argv.txt"
    stub = tmp_path / "stub-pytest"
    stub.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$@" > "{record}"\n'
        f'printf "ENV=%s\\n" "${REQUIRE_EXTRAS_ENV}" >> "{record}"\n'
        "exit 3\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    result = subprocess.run(
        ["make", "-s", MAKE_TARGET, f"PYTEST={stub}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, "the stub exited 3 but make reported success"
    argv = record.read_text(encoding="utf-8").splitlines()
    assert f"--cov={COV_TARGET}" in argv
    assert MARKER in argv, f"the stub did not receive -m {MARKER}"
    assert any(token in SUITE_DIRECTORY_TOKENS for token in argv)
    assert "ENV=1" in argv, f"{REQUIRE_EXTRAS_ENV}=1 did not reach the child"


# --------------------------------------------------------------------------- #
# (e) collection really succeeds, on this install                              #
# --------------------------------------------------------------------------- #


def _collect(require_extras: bool) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != REQUIRE_EXTRAS_ENV}
    if require_extras:
        env[REQUIRE_EXTRAS_ENV] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(SUITE_DIR.relative_to(REPO_ROOT)),
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_suite_collects_without_the_extra_being_required() -> None:
    """The property importorskip broke: every file yields items on a base install."""
    result = _collect(require_extras=False)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    match = _COLLECTED.search(output)
    assert match, output
    assert int(match.group("n")) >= MIN_COLLECTED_ITEMS, output
    assert not _COLLECTION_ERRORS.search(output.splitlines()[-1]), output
    summary = _SKIP_SUMMARY.search(output)
    if EXTRA_INSTALLED:
        assert summary is None, "the extra is installed, yet the hook reported skips"
    else:
        assert summary is not None, "the extra is absent, yet the hook reported no skips"
        assert int(summary.group("n")) >= MIN_MARKED_ITEMS, output


def test_require_extras_outcome_matches_the_install() -> None:
    """Usage error without the extra; clean collection with it. Never a skip."""
    result = _collect(require_extras=True)
    output = result.stdout + result.stderr
    if EXTRA_INSTALLED:
        assert result.returncode == 0, output
        assert _SKIP_SUMMARY.search(output) is None
    else:
        assert result.returncode == USAGE_ERROR_EXIT, output
        assert INSTALL_HINT in output, output
        assert MARKER in output, output
