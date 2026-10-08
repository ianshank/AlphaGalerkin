"""Guards that no ``fem_required`` test can be invisible to CI.

**Defect class:** a test carrying ``fem_required`` that no step of a
scikit-fem-installing CI job selects has never run. The fast lane lacks the
extra, so the root ``conftest.py`` turns the marker into a *counted skip* --
visible as a number, green as a result -- and ``test-extras``, the only job that
installs ``[fem]``, names its suites file by file.

The instance this file was written for: ``TestSkfemTriMctsSmoke`` in
``tests/pde/games/test_substrate_refinement_game.py`` landed with PR #148
(``a85d265``, 2026-09-08), and no revision of ``ci.yml`` ever named the file, so
its one-step MCTS search on ``skfem_tri`` never executed in CI. The ninth
recorded instance of this repo's invisibility defect (CLAUDE.md's R-13 row
records the eighth) -- and, like the other eight, found by a person.

Clauses, each cheap and each falsifiable:

* **(a)** Discovery is by AST and finds the marker wherever it is *applied* --
  decorators, ``pytestmark`` at module or class level, ``pytest.param(marks=)``
  and ``add_marker``/``applymarker`` -- and never in prose. Vacuity floors on the
  scan, on the marker-applying modules (7 on 2026-10-08) and on a prose corpus
  that must stay disjoint from them.
* **(b)** Every ``.fem_required`` attribute access in code is one of those
  applications, in a module pytest collects. ``tokenize`` is the independent
  oracle: an alias (``fem = pytest.mark.fem_required``) or a helper module is a
  spelling the scan cannot follow, so it fails here instead of hiding.
* **(c)** Every fem test -- a static model of pytest's discovery, module and
  class markers included -- is *selected* by a pytest command in a job that
  installs scikit-fem: named by path, directory ancestor or node id; not
  ``--ignore``d, ``--ignore-glob``bed or ``--deselect``ed; no ``-k``; and the
  effective ``-m`` (``addopts`` first) evaluated by pytest's own compiler
  against the test's full marker set, so ``not fem_required`` *and* a filter
  narrowed to some other marker both fail.
* **(d)** ...by a step that fails loud: ``ALPHAGALERKIN_REQUIRE_EXTRAS=1`` in
  effect, no ``continue-on-error``, no ``if:`` that can skip it, in a hard merge
  gate. The root hook is *driven* to show the variable turns a missing
  scikit-fem into a usage error rather than a skip.
* **(e)** A ``[fem]`` job is one whose ``pip install`` names an extra that
  ``pyproject.toml`` declares with scikit-fem -- derived, never spelled --
  before the step runs; ``test-extras`` must be one.
* **(f)** No conftest on a fem module's path filters collection
  (``collect_ignore``, ``collect_ignore_glob``, ``pytest_ignore_collect``,
  ``pytest_collection_modifyitems``): the edit that would hide a module from
  every step while (c) stays green.

The required mutations -- the selecting step deleted, its ``-m`` flipped to
``not fem_required``, ``ALPHAGALERKIN_REQUIRE_EXTRAS`` dropped, a marker in a
module outside every selected path, and a docstring-only mention that must *not*
be flagged -- are re-planted on copies by :class:`TestPlantedDefects`, so they
keep running in CI.

Hermetic except the root-conftest drive, which imports ``conftest.py``.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
import posixpath
import re
import shlex
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from tests.support.marker_expr import (
    FORM_DYNAMIC,
    FORM_PYTESTMARK,
    MODULE_OWNER,
    NODEID_SEPARATOR,
    MarkerApplication,
    ModuleMarkers,
    PytestInvocation,
    assigned_names,
    attribute_access_lines,
    pytest_invocation,
    scan_module_markers,
)
from tests.support.workflows import (
    CI_WORKFLOW,
    CI_WORKFLOW_FILENAME,
    NOT_SELECTED_BY_PATH,
    REPO_ROOT,
    WorkflowStep,
    blocking_jobs,
    iter_commands,
    load_workflow,
    pip_install,
    selection_obstacles,
    workflow_steps,
)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, the declared floor: pytest itself requires tomli
    import tomli as tomllib

# --------------------------------------------------------------------------- #
# Named constants -- nothing below spells these inline.                        #
# --------------------------------------------------------------------------- #

#: The marker this file guards.
FEM_MARKER: Final[str] = "fem_required"

#: The distribution the ``[fem]`` extra exists to install, canonicalised.
FEM_DISTRIBUTION: Final[str] = "scikit-fem"

#: The job that installs it today; (e) requires it to stay a ``[fem]`` job.
EXPECTED_FEM_JOB: Final[str] = "test-extras"

#: The command the root hook names when scikit-fem is missing. Its extra must be
#: one ``pyproject.toml`` derives as a ``[fem]`` extra.
FEM_INSTALL_HINT: Final[str] = "pip install -e '.[fem]'"

#: Turns a missing extra into a collection error in the root ``conftest.py``.
REQUIRE_EXTRAS_ENV: Final[str] = "ALPHAGALERKIN_REQUIRE_EXTRAS"
REQUIRE_EXTRAS_ON: Final[str] = "1"

#: pytest inserts this variable's options after the ini ``addopts``.
PYTEST_ADDOPTS_ENV: Final[str] = "PYTEST_ADDOPTS"

#: Step ``if:`` conditions that run whenever the job runs.
ALWAYS_RUN_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"always()", "${{ always() }}", "success()", "${{ success() }}"}
)

#: Conftest names that remove modules or items from collection.
COLLECTION_FILTERS: Final[frozenset[str]] = frozenset(
    {
        "collect_ignore",
        "collect_ignore_glob",
        "pytest_ignore_collect",
        "pytest_collection_modifyitems",
    }
)

TESTS_ROOT: Final[Path] = REPO_ROOT / "tests"
PYPROJECT: Final[Path] = REPO_ROOT / "pyproject.toml"
ROOT_CONFTEST: Final[Path] = REPO_ROOT / "conftest.py"
CONFTEST: Final[str] = "conftest.py"

#: The module this file was written for; the planted defects mutate its selection.
INSTANCE_MODULE: Final[str] = "tests/pde/games/test_substrate_refinement_game.py"

#: How a scanned file participates in collection.
KIND_TEST: Final[str] = "test module"
KIND_CONFTEST: Final[str] = "conftest"
KIND_OTHER: Final[str] = "module pytest does not collect"

#: Vacuity floors. 595 files, 7 marker-applying modules and 19 prose-only files
#: on 2026-10-08; the module floor is today's count, so losing one fails.
MIN_SCANNED_FILES: Final[int] = 500
MIN_FEM_MODULES: Final[int] = 7
MIN_PROSE_CORPUS: Final[int] = 10

#: A quoted ``-m`` filter inside a ``run:`` script (planted defects only).
_QUOTED_MARKER_FILTER: Final[re.Pattern[str]] = re.compile(r"""-m\s+(["'])[^"']*\1""")


# --------------------------------------------------------------------------- #
# pyproject.toml                                                               #
# --------------------------------------------------------------------------- #


def load_pyproject(path: Path = PYPROJECT) -> dict[str, Any]:
    """The parsed ``pyproject.toml``."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def ini_values(ini: dict[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """An ini option that pytest reads as a list (whitespace-split when a string)."""
    value = ini.get(key)
    if value is None:
        return default
    return tuple(value.split()) if isinstance(value, str) else tuple(value)


def _requirement(text: str) -> Requirement | None:
    try:
        return Requirement(text)
    except InvalidRequirement:
        return None


def _requirement_name(text: str) -> str | None:
    requirement = _requirement(text)
    return None if requirement is None else canonicalize_name(requirement.name)


def fem_extras(pyproject: dict[str, Any]) -> frozenset[str]:
    """Extras that install scikit-fem, directly or through ``<project>[extra]``."""
    project = pyproject["project"]
    own = canonicalize_name(project["name"])
    optional: dict[str, list[str]] = project.get("optional-dependencies", {})
    found: set[str] = set()
    changed = True
    while changed:
        changed = False
        for extra, texts in optional.items():
            if canonicalize_name(extra) in found:
                continue
            for requirement in filter(None, map(_requirement, texts)):
                name = canonicalize_name(requirement.name)
                via = {canonicalize_name(e) for e in requirement.extras} & found
                if name == FEM_DISTRIBUTION or (name == own and via):
                    found.add(canonicalize_name(extra))
                    changed = True
                    break
    return frozenset(found)


PYPROJECT_DATA: Final[dict[str, Any]] = load_pyproject()
INI: Final[dict[str, Any]] = PYPROJECT_DATA["tool"]["pytest"]["ini_options"]
FEM_EXTRAS: Final[frozenset[str]] = fem_extras(PYPROJECT_DATA)


# --------------------------------------------------------------------------- #
# (a) and (b): scanning the test tree                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScannedFile:
    """One file under ``tests/`` that mentions the marker at all."""

    path: str
    kind: str
    markers: ModuleMarkers
    accesses: Counter[int]

    def applications(self) -> list[MarkerApplication]:
        return [a for a in self.markers.applications if a.marker == FEM_MARKER]


@dataclass(frozen=True)
class FemTarget:
    """A unit some ``[fem]`` step must run: one test, or all of a module/directory."""

    path: str
    nodeid: str
    markers: frozenset[str]

    def __str__(self) -> str:
        return f"{self.path}{NODEID_SEPARATOR}{self.nodeid}" if self.nodeid else self.path


def scan_tree(root: Path, relative_to: Path, ini: dict[str, Any]) -> list[ScannedFile]:
    """Every ``*.py`` under ``root`` whose text mentions the marker, scanned.

    The substring prefilter loses nothing: an application has to spell the
    marker's name somewhere, and the scan below decides whether it applies it.
    """
    python_files = ini_values(ini, "python_files", ("test_*.py", "*_test.py"))
    functions = ini_values(ini, "python_functions", ("test",))
    classes = ini_values(ini, "python_classes", ("Test",))
    found: list[ScannedFile] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if FEM_MARKER not in text:
            continue
        if path.name == CONFTEST:
            kind = KIND_CONFTEST
        elif any(fnmatch.fnmatch(path.name, pattern) for pattern in python_files):
            kind = KIND_TEST
        else:
            kind = KIND_OTHER
        markers = scan_module_markers(text, function_patterns=functions, class_patterns=classes)
        relative = path.relative_to(relative_to).as_posix()
        found.append(ScannedFile(relative, kind, markers, attribute_access_lines(text, FEM_MARKER)))
    return found


def misplaced(scanned: ScannedFile) -> list[str]:
    """Why a file's references to the marker cannot be trusted -- empty if they can.

    A test module may apply the marker in any recognised form; a conftest only
    through ``add_marker``/``applymarker`` (``pytestmark`` and decorators there
    reach no test); any other module not at all.
    """
    if scanned.kind == KIND_TEST:
        effective = scanned.applications()
    elif scanned.kind == KIND_CONFTEST:
        effective = [a for a in scanned.applications() if a.form == FORM_DYNAMIC]
    else:
        effective = []
    problems = [
        f"line {a.line}: {a.form} application in a {scanned.kind}, which reaches no test"
        for a in scanned.applications()
        if a not in effective
    ]
    stray = scanned.accesses - Counter(a.line for a in effective)
    problems.extend(
        f"line {line}: `.{FEM_MARKER}` is not a recognised application (an alias, a helper, "
        f"or a spelling the scan cannot follow); spell `@pytest.mark.{FEM_MARKER}` at the "
        "application site"
        for line in sorted(stray.elements())
    )
    return problems


def _explained(application: MarkerApplication, tests: list[FemTarget]) -> bool:
    """Whether the static test list accounts for an application's effect."""
    if application.form == FORM_DYNAMIC:
        return False
    if application.owner == MODULE_OWNER:
        return application.form == FORM_PYTESTMARK and bool(tests)
    prefix = application.owner + NODEID_SEPARATOR
    return any(t.nodeid == application.owner or t.nodeid.startswith(prefix) for t in tests)


def fem_targets(scanned: ScannedFile) -> list[FemTarget]:
    """What some ``[fem]`` step must run because of this file.

    One target per fem-carrying test. An application the static model cannot
    attribute -- dynamic, a module-level ``pytest.param``, a marker on a helper
    or a mixin -- widens to the whole module (a conftest's: its directory),
    carrying the module's own ``pytestmark``.
    """
    applications = scanned.applications()
    if not applications or scanned.kind == KIND_OTHER:
        return []
    if scanned.kind == KIND_CONFTEST:
        if any(a.form == FORM_DYNAMIC for a in applications):
            return [FemTarget(posixpath.dirname(scanned.path), "", frozenset({FEM_MARKER}))]
        return []
    tests = [
        FemTarget(scanned.path, t.nodeid, t.markers)
        for t in scanned.markers.tests
        if FEM_MARKER in t.markers
    ]
    if not all(_explained(a, tests) for a in applications):
        module_marks = {
            a.marker
            for a in scanned.markers.applications
            if a.form == FORM_PYTESTMARK and a.owner == MODULE_OWNER
        }
        tests.append(FemTarget(scanned.path, "", frozenset({FEM_MARKER, *module_marks})))
    return list(dict.fromkeys(tests))


# --------------------------------------------------------------------------- #
# (c), (d), (e): what the [fem] steps run                                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Candidate:
    """One pytest command in a ``[fem]`` job, with the step that runs it."""

    step: WorkflowStep
    invocation: PytestInvocation


def fem_install_steps(steps: list[WorkflowStep], extras: frozenset[str]) -> dict[str, int]:
    """Job -> index of its first step that installs scikit-fem."""
    found: dict[str, int] = {}
    for step in steps:
        if step.script is None or step.job in found:
            continue
        for command in iter_commands(step.script):
            install = pip_install(command)
            if install is None:
                continue
            requested = {canonicalize_name(e) for e in install.project_extras}
            named = {_requirement_name(r) for r in install.requirements}
            if requested & extras or FEM_DISTRIBUTION in named:
                found[step.job] = step.index
                break
    return found


def fem_candidates(workflow: Path, extras: frozenset[str], ini: dict[str, Any]) -> list[Candidate]:
    """Every pytest command that runs after scikit-fem is installed in its job."""
    steps = workflow_steps(workflow)
    installed_at = fem_install_steps(steps, extras)
    addopts = shlex.split(str(ini.get("addopts", "")))
    found: list[Candidate] = []
    for step in steps:
        installed = installed_at.get(step.job)
        if step.script is None or installed is None or step.index <= installed:
            continue
        for command in iter_commands(step.script):
            first = pytest_invocation(command)
            if first is None:
                continue
            extra = shlex.split({**step.env, **first.env}.get(PYTEST_ADDOPTS_ENV, ""))
            invocation = pytest_invocation(command, addopts=(*addopts, *extra))
            assert invocation is not None  # the same command parsed a line above
            found.append(Candidate(step, invocation))
    return found


def obstacles_for(target: FemTarget, candidate: Candidate, ini: dict[str, Any]) -> list[str]:
    """Why ``candidate`` would not run ``target`` -- empty if it would."""
    return selection_obstacles(
        candidate.invocation,
        path=target.path,
        nodeid=target.nodeid,
        markers=target.markers,
        working_directory=candidate.step.working_directory,
        testpaths=ini_values(ini, "testpaths", ()),
    )


def loudness_obstacles(candidate: Candidate, blocking: set[str]) -> list[str]:
    """Why a selecting step could stay green with scikit-fem missing or its tests red."""
    step = candidate.step
    value = {**step.env, **candidate.invocation.env}.get(REQUIRE_EXTRAS_ENV)
    obstacles: list[str] = []
    if value != REQUIRE_EXTRAS_ON:
        obstacles.append(
            f"{REQUIRE_EXTRAS_ENV}={value!r}, not {REQUIRE_EXTRAS_ON!r}: a missing scikit-fem "
            "skips every test instead of failing"
        )
    if step.continue_on_error:
        obstacles.append("continue-on-error: the step cannot fail its job")
    if step.condition is not None and step.condition.strip() not in ALWAYS_RUN_CONDITIONS:
        obstacles.append(f"if: {step.condition!r} can skip the step")
    if step.job not in blocking:
        obstacles.append(f"job {step.job!r} is not a hard gate in ci-success")
    return obstacles


def uncovered(
    targets: list[FemTarget],
    candidates: list[Candidate],
    ini: dict[str, Any],
    blocking: set[str] | None = None,
) -> dict[str, list[str]]:
    """Targets no candidate runs (loudly, when ``blocking`` is given), with diagnostics.

    Diagnostics name only the candidates that reach the target by path, so a
    failure reads as "this step names it, and here is why it does not count".
    """
    failures: dict[str, list[str]] = {}
    for target in targets:
        notes: list[str] = []
        for candidate in candidates:
            obstacles = obstacles_for(target, candidate, ini)
            if not obstacles and blocking is not None:
                obstacles = loudness_obstacles(candidate, blocking)
            if not obstacles:
                break
            if NOT_SELECTED_BY_PATH not in obstacles:
                notes.append(f"{candidate.step}: " + "; ".join(obstacles))
        else:
            failures[str(target)] = notes
    return failures


def _report(failures: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for target, notes in failures.items():
        lines.append(f"  {target}")
        lines.extend(f"      {note}" for note in notes or ["no [fem] step names it at all"])
    return "\n".join(lines)


SCANNED: Final[list[ScannedFile]] = scan_tree(TESTS_ROOT, REPO_ROOT, INI)
SCANNED_BY_PATH: Final[dict[str, ScannedFile]] = {s.path: s for s in SCANNED}
TARGETS: Final[dict[str, list[FemTarget]]] = {s.path: fem_targets(s) for s in SCANNED}
FEM_MODULES: Final[list[str]] = sorted(path for path, targets in TARGETS.items() if targets)
REFERENCING_FILES: Final[list[str]] = sorted(
    s.path for s in SCANNED if s.accesses or s.applications()
)
CANDIDATES: Final[list[Candidate]] = fem_candidates(CI_WORKFLOW, FEM_EXTRAS, INI)
BLOCKING: Final[set[str]] = blocking_jobs(load_workflow(CI_WORKFLOW))

_SELECTION_ADVICE: Final[str] = (
    "Select it in a `test-extras` step under ALPHAGALERKIN_REQUIRE_EXTRAS=1 -- by path, "
    '`-m "fem_required and not gpu_required"` to run only its fem half (see the '
    "`Run fem_required tests from fast-lane modules` step)."
)


# --------------------------------------------------------------------------- #
# (a) discovery, by AST, with vacuity floors                                   #
# --------------------------------------------------------------------------- #


def test_the_scan_reads_the_whole_test_tree() -> None:
    """A scan rooted at the wrong directory passes every assertion in this file."""
    assert len(list(TESTS_ROOT.rglob("*.py"))) >= MIN_SCANNED_FILES


def test_enough_modules_apply_the_fem_marker() -> None:
    """Non-vacuity, at today's count: losing a module is a change to review, not drift.

    Also the backstop for a re-spelling the scan cannot follow -- the module
    drops out of discovery, and the count drops below the floor.
    """
    assert len(FEM_MODULES) >= MIN_FEM_MODULES, (
        f"only {len(FEM_MODULES)} modules apply `{FEM_MARKER}` ({FEM_MODULES}); "
        f"{MIN_FEM_MODULES} did on 2026-10-08. A removed marker is a reviewed change "
        "(lower the floor); a re-spelled one is a hole -- spell it `@pytest.mark.<name>`."
    )


def test_prose_mentions_are_never_applications() -> None:
    """The false-positive corpus, judged by an oracle the scan does not share.

    ``tokenize`` says these files never access ``.fem_required`` in code, so a
    decorator, ``pytestmark`` or ``pytest.param`` application in any of them
    can only be a scan that reads prose -- a grep in an AST's clothing.
    """
    corpus = [s for s in SCANNED if not s.accesses]
    assert len(corpus) >= MIN_PROSE_CORPUS, f"only {len(corpus)} prose-only files"
    leaked = {s.path: [a.form for a in s.applications() if a.form != FORM_DYNAMIC] for s in corpus}
    assert not {path: forms for path, forms in leaked.items() if forms}


# --------------------------------------------------------------------------- #
# (b) every reference is an application the scan understood                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", REFERENCING_FILES)
def test_every_fem_reference_is_a_recognised_application(path: str) -> None:
    """An access the scan cannot place is a spelling it cannot follow."""
    problems = misplaced(SCANNED_BY_PATH[path])
    assert not problems, f"{path}:\n  " + "\n  ".join(problems)


# --------------------------------------------------------------------------- #
# (c) and (d): selected, by a step that fails loud                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module", FEM_MODULES)
def test_every_fem_test_is_selected_by_a_fem_job_step(module: str) -> None:
    """Path, ``--ignore``, ``--deselect``, ``-k`` and ``-m`` defects all fail here."""
    failures = uncovered(TARGETS[module], CANDIDATES, INI)
    assert not failures, (
        f"fem_required test(s) in {module} that no step of a scikit-fem job runs:\n"
        f"{_report(failures)}\n{_SELECTION_ADVICE}"
    )


@pytest.mark.parametrize("module", FEM_MODULES)
def test_a_step_selecting_it_fails_loud(module: str) -> None:
    """Selected is half of it: the step must also be able to fail the build."""
    failures = uncovered(TARGETS[module], CANDIDATES, INI, BLOCKING)
    assert not failures, (
        f"fem_required test(s) in {module} run only by steps that cannot fail loud:\n"
        f"{_report(failures)}"
    )


class _FakeItem:
    """The two methods of ``pytest.Item`` the root hook calls."""

    def __init__(self, markers: set[str]) -> None:
        self.markers = markers
        self.added: list[object] = []

    def get_closest_marker(self, name: str) -> object | None:
        return name if name in self.markers else None

    def add_marker(self, marker: object) -> None:
        self.added.append(marker)


def _load_root_conftest() -> ModuleType:
    """A fresh copy of the root ``conftest.py``, reading the environment now."""
    spec = importlib.util.spec_from_file_location("_fem_guard_root_conftest", ROOT_CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _driven_hook(monkeypatch: pytest.MonkeyPatch, require: bool) -> ModuleType:
    if require:
        monkeypatch.setenv(REQUIRE_EXTRAS_ENV, REQUIRE_EXTRAS_ON)
    else:
        monkeypatch.delenv(REQUIRE_EXTRAS_ENV, raising=False)
    hook = _load_root_conftest()
    monkeypatch.setattr(hook, "_HAS_SKFEM", False)
    monkeypatch.setattr(hook, "_HAS_EVAL_HARNESS", True)  # isolate the fem branch
    return hook


def test_require_extras_turns_a_missing_scikit_fem_into_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What makes (d)'s variable worth requiring: driven, not grepped."""
    hook = _driven_hook(monkeypatch, require=True)
    with pytest.raises(pytest.UsageError, match=re.escape(FEM_INSTALL_HINT)):
        hook.pytest_collection_modifyitems(None, [_FakeItem({FEM_MARKER})])


def test_without_require_extras_a_missing_scikit_fem_is_a_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fast lane's behaviour -- the skip (d) exists to keep out of test-extras."""
    hook = _driven_hook(monkeypatch, require=False)
    item = _FakeItem({FEM_MARKER})
    hook.pytest_collection_modifyitems(None, [item])
    assert len(item.added) == 1


# --------------------------------------------------------------------------- #
# (e) the [fem] job is derived from pyproject.toml                             #
# --------------------------------------------------------------------------- #


def test_the_fem_extras_are_derived_from_pyproject() -> None:
    """The extra the root hook tells people to install must be one this file derives."""
    match = re.search(r"\[(?P<extra>[^\]]+)\]", FEM_INSTALL_HINT)
    assert match is not None
    assert canonicalize_name(match.group("extra")) in FEM_EXTRAS, sorted(FEM_EXTRAS)


def test_a_ci_job_installs_scikit_fem() -> None:
    """Without a ``[fem]`` job there are no candidates, and every selection test fails."""
    jobs = fem_install_steps(workflow_steps(CI_WORKFLOW), FEM_EXTRAS)
    assert EXPECTED_FEM_JOB in jobs, (
        f"{EXPECTED_FEM_JOB} no longer installs an extra carrying {FEM_DISTRIBUTION} "
        f"({sorted(FEM_EXTRAS)}); found [fem] jobs: {sorted(jobs)}"
    )


def test_at_least_one_fem_step_is_recognised() -> None:
    """Non-vacuity for (c) and (d): a loud ``[fem]`` step the parsers can read."""
    loud = [c for c in CANDIDATES if not loudness_obstacles(c, BLOCKING)]
    assert {c.step.job for c in loud} >= {EXPECTED_FEM_JOB}, [str(c.step) for c in CANDIDATES]


# --------------------------------------------------------------------------- #
# (f) no conftest filters a fem module out of collection                       #
# --------------------------------------------------------------------------- #


def collection_filters(source: str) -> set[str]:
    """Collection-filtering names a conftest binds or defines, at any depth."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found |= {node.name} & COLLECTION_FILTERS
        elif isinstance(node, ast.stmt):
            found |= assigned_names(node) & COLLECTION_FILTERS
    return found


@pytest.mark.parametrize("module", FEM_MODULES)
def test_no_conftest_on_its_path_filters_collection(module: str) -> None:
    """The root ``conftest.py`` is exempt: its hook is the one driven in (d)."""
    directory = (REPO_ROOT / module).parent
    offenders: dict[str, set[str]] = {}
    while directory != REPO_ROOT and REPO_ROOT in directory.parents:
        conftest = directory / CONFTEST
        if conftest.exists():
            names = collection_filters(conftest.read_text(encoding="utf-8"))
            if names:
                offenders[conftest.relative_to(REPO_ROOT).as_posix()] = names
        directory = directory.parent
    assert not offenders, f"conftest(s) on {module}'s path filter collection: {offenders}"


# --------------------------------------------------------------------------- #
# The mutations, re-planted on copies so they keep running in CI               #
# --------------------------------------------------------------------------- #


def _instance_targets() -> list[FemTarget]:
    assert INSTANCE_MODULE in FEM_MODULES, f"{INSTANCE_MODULE} is no longer discovered"
    return TARGETS[INSTANCE_MODULE]


def _selecting_steps() -> list[tuple[str, int]]:
    """``(job, index)`` of every live step that runs the instance module's fem tests."""
    found = {
        (c.step.job, c.step.index)
        for c in CANDIDATES
        for t in _instance_targets()
        if not obstacles_for(t, c, INI)
    }
    assert found, f"no live step runs {INSTANCE_MODULE} -- nothing to mutate"
    return sorted(found)


def _mutated(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> list[Candidate]:
    document = load_workflow(CI_WORKFLOW)
    mutate(document)
    path = tmp_path / CI_WORKFLOW_FILENAME
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return fem_candidates(path, FEM_EXTRAS, INI)


def _plant(tmp_path: Path, relative: str, source: str) -> ScannedFile:
    planted = tmp_path / relative
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(source, encoding="utf-8")
    (scanned,) = scan_tree(tmp_path / "tests", tmp_path, INI)
    return scanned


class TestPlantedDefects:
    """The five required mutations, planted on copies so they keep running in CI.

    A refactor that stopped reporting one would otherwise pass every live
    assertion above while the protection quietly went away.
    """

    def test_the_unmutated_copy_is_clean(self, tmp_path: Path) -> None:
        """The control: a round-tripped copy must cover the instance loudly."""
        candidates = _mutated(tmp_path, lambda document: None)
        assert not uncovered(_instance_targets(), candidates, INI, BLOCKING)

    def test_deleting_the_selecting_step_is_reported(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            for job, index in reversed(_selecting_steps()):
                del document["jobs"][job]["steps"][index]

        assert uncovered(_instance_targets(), _mutated(tmp_path, mutate), INI)

    def test_a_not_fem_required_filter_is_reported(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            for job, index in _selecting_steps():
                step = document["jobs"][job]["steps"][index]
                step["run"], count = _QUOTED_MARKER_FILTER.subn(
                    f'-m "not {FEM_MARKER}"', step["run"]
                )
                assert count, f"anchor not found in {job}[{index}]: the mutation would no-op"

        assert uncovered(_instance_targets(), _mutated(tmp_path, mutate), INI)

    def test_dropping_require_extras_is_reported(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            for job, index in _selecting_steps():
                step = document["jobs"][job]["steps"][index]
                inline = f"{REQUIRE_EXTRAS_ENV}={REQUIRE_EXTRAS_ON} "
                removed = (step.get("env") or {}).pop(REQUIRE_EXTRAS_ENV, None)
                assert removed is not None or inline in step["run"], "anchor not found"
                step["run"] = step["run"].replace(inline, "")

        candidates = _mutated(tmp_path, mutate)
        assert not uncovered(_instance_targets(), candidates, INI), "still selected"
        assert uncovered(_instance_targets(), candidates, INI, BLOCKING), "but not loud"

    def test_a_marker_outside_every_selected_path_is_reported(self, tmp_path: Path) -> None:
        source = f"import pytest\n\n\n@pytest.mark.{FEM_MARKER}\ndef test_x() -> None:\n    pass\n"
        scanned = _plant(tmp_path, "tests/pde/test_planted_fem.py", source)
        targets = fem_targets(scanned)
        assert targets and not misplaced(scanned)
        assert uncovered(targets, CANDIDATES, INI) == {f"{scanned.path}::test_x": []}

    def test_a_docstring_only_mention_is_not_an_application(self, tmp_path: Path) -> None:
        source = (
            f'"""Mentions ``@pytest.mark.{FEM_MARKER}`` in prose only."""\n'
            f"# pytestmark = pytest.mark.{FEM_MARKER}\n"
            f'NOTE = "pytest.mark.{FEM_MARKER}"\n\n\ndef test_x() -> None:\n    pass\n'
        )
        scanned = _plant(tmp_path, "tests/test_prose_only.py", source)
        assert (scanned.applications(), scanned.accesses, fem_targets(scanned)) == ([], {}, [])
