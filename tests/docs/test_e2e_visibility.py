"""Guards that the ``tests/e2e/`` tier cannot silently become CI-invisible again.

Until this branch, ``tests/e2e/`` was excluded from every step in ``ci.yml``
except one chess smoke: the fast lane passed **both** ``--ignore=tests/e2e/``
and ``-m "not e2e"``, and the ``coverage`` job repeated both. Eleven files and
81 passing tests gated nothing, and ``make test-e2e`` -- which ``make pre-pr``
chains -- selected 3 of them via a ``test_user_journey_*.py`` glob.

That is the *third* recorded instance of the same defect class in this repo
(``tests/demos/`` + ``tests/notebooks/``, 226 tests never executed until
``18f533d``; ``src/backend``, 213 passing tests measured nowhere). Every one was
found by a person reading the workflow. This file is the check that would have
found them, applied to the tier that was fixed last:

* **(a)** every ``tests/e2e/test_*.py`` carries the ``e2e`` marker, so ``-m e2e``
  can actually select it;
* **(b)** some ``ci.yml`` step selects the *directory* without ``--ignore``-ing
  it, under a ``-m`` expression that does not deselect ordinary ``e2e`` tests;
* **(c)** the job owning that step is in ``ci-success``'s ``needs`` **and** has a
  hard ``exit 1`` gate -- a job that runs, reports and cannot fail the build is
  a report, not a gate;
* **(d)** the ``Makefile`` target selects the whole directory, not a glob subset;
* **(e)** no ``gpu_required`` marker appears in the tier outside one allowlisted
  sentinel, because a device-agnostic tier must not be skippable on device;
* **(f)** a step *positively* selects the ``fem_required`` half, which
  ``test-e2e`` deliberately excludes -- excluded in both places is precisely the
  invisibility this file exists to prevent.

Hermetic: parses Python, YAML and Make text. Runs nothing -- no pytest, no
subprocess, no network. Guards ``docs/E2E_TEST_PLAN.md`` §3 (Phase 0).
"""

from __future__ import annotations

import ast
import shlex
from pathlib import Path
from typing import Final

import pytest

from tests.support.marker_expr import (
    expression_excludes,
    expression_requires,
    expression_selects_plainly,
    marker_expressions,
)
from tests.support.workflows import (
    CI_SUCCESS_JOB,
    CI_WORKFLOW,
    CI_WORKFLOW_FILENAME,
    MAKEFILE,
    REPO_ROOT,
    RunScript,
    hard_gate_jobs,
    iter_commands,
    iter_run_scripts,
    job_needs,
    load_workflow,
    makefile_target_recipe,
)

#: The tier this file guards.
E2E_DIR = REPO_ROOT / "tests" / "e2e"

#: Glob for the tier's test modules, as pytest collects them.
E2E_TEST_GLOB = "test_*.py"

#: The two ways a shell argument can name the directory itself. A token that is
#: merely *prefixed* by one of these (``tests/e2e/test_chess_training_e2e.py``,
#: ``--ignore=tests/e2e/``) is deliberately not a match.
E2E_DIRECTORY_TOKENS = frozenset({"tests/e2e", "tests/e2e/"})

#: Markers this file reasons about, named rather than spelled inline.
E2E_MARKER = "e2e"
GPU_MARKER = "gpu_required"
FEM_MARKER = "fem_required"

#: The Makefile target that must mirror the CI job.
MAKEFILE_E2E_TARGET = "test-e2e"

#: Substring that identifies a glob-narrowed selection of the tier -- the exact
#: shape ``make test-e2e`` used to carry (``tests/e2e/test_user_journey_*.py``).
GLOB_MARKER = "*"

#: The **only** test name in ``tests/e2e/`` permitted to carry ``gpu_required``.
#:
#: Rationale (``docs/E2E_TEST_PLAN.md`` §1, "Sentinel" and "Marker rule"): the
#: marker *skips*, so a device-agnostic tier that carries it is a tier that can
#: silently not run on the device it claims to be agnostic about. The one
#: exception is a sentinel asserting that the resolved device and CUDA
#: availability agree -- which has nothing to assert on a CPU host.
#:
#: This is a **forward reference**: Phase 0 adds the sentinel, so today the
#: allowlist matches nothing and the guard passes vacuously. That is intended --
#: the assertion is "every ``gpu_required`` site in the tier is allowlisted",
#: which still fails the moment a *new* one appears. ``TestTheAllowlist`` below
#: stops the entry becoming a free pass once the sentinel does exist.
GPU_SENTINEL_TEST_NAMES = frozenset({"test_resolved_device_matches_cuda_availability"})

#: Jobs that run the E2E tier but are knowingly **not** blocking, each with a
#: reason. Empty, and meant to stay that way: a tier wired into a non-blocking
#: job is the exact failure this file exists to catch, so an entry here is a
#: disclosure, not a fix.
#:
#: Self-expiring in both directions, the ``_OMIT_WITHOUT_A_CI_GATE`` idiom from
#: ``tests/docs/test_coverage_gate_integrity.py``: an exemption for a job that
#: has *become* blocking fails (so it cannot rot into a licence nobody reviews),
#: and an exemption naming a job that does not exist fails as vacuous. A comment
#: in the workflow is explicitly **not** an accepted substitute -- a stale
#: comment passes forever, which is how the previous exclusion survived.
JOBS_EXEMPT_FROM_BLOCKING: dict[str, str] = {}

#: Minimum number of files the tier must contain for the parametrised marker
#: check to mean anything. A guard that iterates an empty glob passes everything.
MIN_E2E_TEST_FILES = 5


# --------------------------------------------------------------------------
# Marker extraction (AST). Unit-tested on synthetic source in TestMarkerSites,
# so a walker that stops recognising decorators cannot report as "covered"
# merely because the live files happen to be well-formed.
# --------------------------------------------------------------------------

#: Owner name recorded for a module-level ``pytestmark`` assignment.
MODULE_OWNER = "<module>"


def _marker_name(node: ast.expr) -> str | None:
    """Extract the marker name from a ``pytest.mark.X`` expression node.

    Args:
        node: Any expression node, e.g. a decorator or an assignment value.

    Returns:
        The marker name for ``pytest.mark.X`` and ``pytest.mark.X(...)``,
        otherwise ``None``.

    """
    target = node.func if isinstance(node, ast.Call) else node
    if not isinstance(target, ast.Attribute):
        return None
    parent = target.value
    if not isinstance(parent, ast.Attribute) or parent.attr != "mark":
        return None
    if not isinstance(parent.value, ast.Name) or parent.value.id != "pytest":
        return None
    return target.attr


def _pytestmark_values(node: ast.Assign) -> list[ast.expr]:
    """The marker expressions of a ``pytestmark = ...`` assignment.

    Args:
        node: An assignment statement.

    Returns:
        Each assigned marker expression (one, or the elements of a list/tuple).
        Empty for assignments to any other name.

    """
    if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
        return []
    value = node.value
    if isinstance(value, (ast.List, ast.Tuple)):
        return list(value.elts)
    return [value]


def marker_sites(source: str) -> dict[str, set[str]]:
    """Map every pytest marker used in ``source`` to the names carrying it.

    Recognised: module- and class-level ``pytestmark`` assignments (including
    list form) and ``@pytest.mark.X`` decorators on functions and classes.

    Args:
        source: Python source text of a test module.

    Returns:
        Marker name -> owner names, where an owner is a function or class name,
        or :data:`MODULE_OWNER` for a module-level ``pytestmark``.

    """
    sites: dict[str, set[str]] = {}

    def record(marker: str | None, owner: str) -> None:
        if marker is not None:
            sites.setdefault(marker, set()).add(owner)

    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for value in _pytestmark_values(statement):
                record(_marker_name(value), MODULE_OWNER)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for decorator in node.decorator_list:
                record(_marker_name(decorator), node.name)
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    for value in _pytestmark_values(statement):
                        record(_marker_name(value), node.name)
    return sites


# --------------------------------------------------------------------------
# Workflow / Makefile selection predicates.
# --------------------------------------------------------------------------


def selects_e2e_directory(command: str) -> bool:
    """Whether a shell command selects ``tests/e2e/`` as a whole directory.

    Token equality, not substring containment: ``--ignore=tests/e2e/`` and
    ``tests/e2e/test_chess_training_e2e.py`` both *contain* the path and both
    select nothing of the tier at large. Only a standalone path token counts.

    Args:
        command: One logical shell command.

    Returns:
        True if some argument is exactly ``tests/e2e`` or ``tests/e2e/``.

    """
    return any(token in E2E_DIRECTORY_TOKENS for token in command.split())


def effective_marker_expression(command: str) -> str:
    """The ``-m`` expression pytest would actually apply to a command.

    Args:
        command: One logical shell command.

    Returns:
        The last marker expression in the command (pytest keeps the last
        ``-m``), or the empty string when there is none -- which selects
        everything.

    """
    expressions = marker_expressions(command)
    return expressions[-1] if expressions else ""


def e2e_selecting_commands() -> list[tuple[RunScript, str]]:
    """Every ``ci.yml`` command that runs the E2E tier broadly.

    "Broadly" excludes a command narrowed to another positively-required
    marker: ``-m "fem_required and not gpu_required"`` runs only the
    optional-extra subset and cannot stand in as proof the tier is executed.

    Returns:
        ``(run script, marker expression)`` pairs, one per qualifying command.

    """
    found: list[tuple[RunScript, str]] = []
    for run in iter_run_scripts():
        if run.workflow != CI_WORKFLOW_FILENAME:
            continue
        for command in iter_commands(run.script):
            if not selects_e2e_directory(command):
                continue
            expression = effective_marker_expression(command)
            if expression_selects_plainly(expression, E2E_MARKER):
                found.append((run, expression))
    return found


def e2e_selecting_ci_commands() -> list[tuple[RunScript, str]]:
    """The same qualifying steps as :func:`e2e_selecting_commands`, as raw commands.

    The sibling returns the *marker* expression, which is what the visibility
    clauses ask about. The partition clause needs the command text itself, to
    read its ``-k``.

    Returns:
        ``(run script, command)`` pairs, one per qualifying command.

    """
    found: list[tuple[RunScript, str]] = []
    for run in iter_run_scripts():
        if run.workflow != CI_WORKFLOW_FILENAME:
            continue
        for command in iter_commands(run.script):
            if not selects_e2e_directory(command):
                continue
            if expression_selects_plainly(effective_marker_expression(command), E2E_MARKER):
                found.append((run, command))
    return found


def fem_selecting_commands() -> list[tuple[RunScript, str]]:
    """Every ``ci.yml`` command that positively selects the ``fem_required`` half.

    Returns:
        ``(run script, marker expression)`` pairs whose command names the
        directory and whose expression requires ``fem_required`` un-negated.

    """
    found: list[tuple[RunScript, str]] = []
    for run in iter_run_scripts():
        if run.workflow != CI_WORKFLOW_FILENAME:
            continue
        for command in iter_commands(run.script):
            if not selects_e2e_directory(command):
                continue
            expression = effective_marker_expression(command)
            if expression_requires(expression, FEM_MARKER):
                found.append((run, expression))
    return found


def ci_success_blocking_jobs() -> set[str]:
    """Jobs ``ci-success`` both depends on and hard-fails the build for.

    Returns:
        The intersection of ``ci-success``'s ``needs:`` list and the jobs named
        in an ``if`` condition whose body reaches ``exit 1``.

    """
    document = load_workflow(CI_WORKFLOW)
    needs = set(job_needs(document, CI_SUCCESS_JOB))
    gated: set[str] = set()
    for run in iter_run_scripts():
        if run.workflow == CI_WORKFLOW_FILENAME and run.job == CI_SUCCESS_JOB:
            gated |= hard_gate_jobs(run.script)
    return needs & gated


E2E_TEST_FILES = sorted(E2E_DIR.glob(E2E_TEST_GLOB))
E2E_PYTHON_FILES = sorted(E2E_DIR.glob("*.py"))


# --------------------------------------------------------------------------
# (a) Every file in the tier is selectable by `-m e2e`.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", E2E_TEST_FILES, ids=lambda p: p.name)
def test_every_e2e_test_file_carries_the_e2e_marker(path: Path) -> None:
    """An unmarked file cannot be selected by ``-m e2e``, only ignored by path.

    Two files (``test_chess_training_e2e.py``, ``test_cli_journey.py``) were in
    exactly that state: ``-m e2e`` could not reach them, so the only thing
    keeping them out of the fast lane was the ``--ignore``, and the only thing
    that could put them *in* a job was naming the file. Marker coverage is what
    makes the tier addressable as a tier.
    """
    sites = marker_sites(path.read_text(encoding="utf-8"))
    assert E2E_MARKER in sites, (
        f"{path.relative_to(REPO_ROOT)} carries no `{E2E_MARKER}` marker, so "
        f"`-m {E2E_MARKER}` cannot select it. Add `pytestmark = pytest.mark.{E2E_MARKER}` "
        "at module level, or decorate each test."
    )


def test_the_e2e_directory_is_not_empty() -> None:
    """Non-vacuity: a parametrised guard over an empty glob passes everything."""
    assert len(E2E_TEST_FILES) >= MIN_E2E_TEST_FILES, (
        f"only {len(E2E_TEST_FILES)} files matched {E2E_TEST_GLOB} in {E2E_DIR} -- "
        "the marker guard above is now near-vacuous; check the discovery glob."
    )


# --------------------------------------------------------------------------
# (b) A CI step really runs the tier.
# --------------------------------------------------------------------------


def test_a_ci_step_runs_the_e2e_directory() -> None:
    """Some ``ci.yml`` command must select the tier without deselecting it.

    The historical state this rejects: the directory named nowhere except
    behind ``--ignore=``, plus a single file named in ``test-chess``. Both of
    those *mention* ``tests/e2e`` in the workflow while running none of it.
    """
    commands = e2e_selecting_commands()
    assert commands, (
        f"no step in {CI_WORKFLOW_FILENAME} runs `tests/e2e/` as a directory under a "
        f"`-m` expression that keeps ordinary `{E2E_MARKER}` tests. The tier is "
        "CI-invisible again: add a step (see the `test-e2e` job) or restore its `-m`."
    )


def test_the_e2e_step_does_not_exclude_the_e2e_marker() -> None:
    """The specific mutation: ``-m "... and not e2e"`` on the job that runs the tier.

    Separated from the test above so the failure message names the mechanism.
    A step keeping ``--ignore``-free path selection while adding ``not e2e``
    would still *look* like the tier is wired.
    """
    excluded = [
        (run, expression)
        for run in iter_run_scripts()
        if run.workflow == CI_WORKFLOW_FILENAME
        for command in iter_commands(run.script)
        if selects_e2e_directory(command)
        for expression in [effective_marker_expression(command)]
        if expression_excludes(expression, E2E_MARKER)
    ]
    assert not excluded, (
        "step(s) select `tests/e2e/` and then deselect the whole tier with "
        f"`not {E2E_MARKER}`:\n"
        + "\n".join(f"  {run} -m {expression!r}" for run, expression in excluded)
    )


# --------------------------------------------------------------------------
# (c) The job that runs it can fail the build.
# --------------------------------------------------------------------------


def test_the_job_running_e2e_is_blocking() -> None:
    """Being in ``needs:`` is half a gate; the ``exit 1`` block is the other half.

    ``ci-success`` echoes every job's result and then hard-fails on a subset.
    A job present in ``needs`` but absent from the ``if ... exit 1`` sequence
    reports red in the log and green to the branch protection rule -- exactly
    the B38 finding (``test-integration``, ``test-jax``, ``test-chess`` ran on
    every PR and could not block a merge).
    """
    commands = e2e_selecting_commands()
    assert commands, "no step runs the tier at all -- see test_a_ci_step_runs_the_e2e_directory"
    blocking = ci_success_blocking_jobs()
    offenders = sorted(
        {
            run.job
            for run, _ in commands
            if run.job not in blocking and run.job not in JOBS_EXEMPT_FROM_BLOCKING
        }
    )
    assert not offenders, (
        f"job(s) running the E2E tier that cannot fail the build: {offenders}. "
        f"Each must appear in `{CI_SUCCESS_JOB}`'s `needs:` AND in an "
        '`if [[ "${{ needs.<job>.result }}" != "success" ]]; then ... exit 1; fi` block. '
        "A workflow comment is not accepted: a stale comment passes forever."
    )


def test_every_blocking_exemption_names_a_real_job() -> None:
    """An exemption for a job that does not exist guards nothing."""
    document = load_workflow(CI_WORKFLOW)
    jobs = document.get("jobs")
    known = set(jobs) if isinstance(jobs, dict) else set()
    unknown = sorted(set(JOBS_EXEMPT_FROM_BLOCKING) - known)
    assert not unknown, (
        f"JOBS_EXEMPT_FROM_BLOCKING names job(s) absent from {CI_WORKFLOW_FILENAME}: "
        f"{unknown} -- delete the entries or fix the names."
    )


def test_every_blocking_exemption_states_a_reason() -> None:
    """A reason is what separates an exemption from a hole someone tolerated."""
    for job, reason in JOBS_EXEMPT_FROM_BLOCKING.items():
        assert reason.strip(), f"{job} is exempted from blocking with no reason"
        assert len(reason.split()) >= 8, f"{job}'s reason is too thin to review: {reason!r}"


def test_every_blocking_exemption_is_still_needed() -> None:
    """Self-expiry: an exemption for a job that *is* now blocking must be deleted.

    Without this the dict becomes a licence nobody re-reads -- the same rot the
    ``_STAGED_FOR_UPCOMING_TASK`` and ``FORWARD_REFERENCES`` staleness checks
    exist to prevent elsewhere in this repo.
    """
    blocking = ci_success_blocking_jobs()
    stale = sorted(job for job in JOBS_EXEMPT_FROM_BLOCKING if job in blocking)
    assert not stale, (
        f"these jobs are exempted as non-blocking but now block the build: {stale}. "
        "Delete their JOBS_EXEMPT_FROM_BLOCKING entries."
    )


# --------------------------------------------------------------------------
# (d) The Makefile mirrors CI.
# --------------------------------------------------------------------------


def test_makefile_e2e_target_selects_the_whole_directory() -> None:
    """``make test-e2e`` must run the tier, not a glob subset of it.

    It previously ran ``tests/e2e/test_user_journey_*.py`` -- 3 of 81 tests --
    and ``make pre-pr`` chains it, so a PR was certified against three E2E
    tests. A glob is indistinguishable from the directory at a glance, which is
    why this is asserted rather than reviewed.
    """
    recipe = makefile_target_recipe(MAKEFILE_E2E_TARGET)
    assert recipe, (
        f"no `{MAKEFILE_E2E_TARGET}` target found in {MAKEFILE.name} -- "
        "the developer-facing mirror of the CI job is gone."
    )
    assert any(selects_e2e_directory(command) for command in recipe), (
        f"`make {MAKEFILE_E2E_TARGET}` does not select the `tests/e2e/` directory: "
        f"{recipe}. A glob subset (e.g. `tests/e2e/test_user_journey_*.py`) runs "
        "a fraction of the tier while reading like the whole of it."
    )
    globbed = [
        token
        for command in recipe
        for token in command.split()
        if token.startswith("tests/e2e/") and GLOB_MARKER in token
    ]
    assert not globbed, (
        f"`make {MAKEFILE_E2E_TARGET}` narrows the tier with glob(s) {globbed}; "
        "select the directory and filter with `-m` as CI does."
    )


# --------------------------------------------------------------------------
# (e) Nothing device-agnostic is skippable on device.
# --------------------------------------------------------------------------


def gpu_required_sites_in_tier() -> dict[Path, set[str]]:
    """Every ``gpu_required`` marker site in the tier, by file.

    Returns:
        Path -> owner names (test/class names, or :data:`MODULE_OWNER`).

    """
    found: dict[Path, set[str]] = {}
    for path in E2E_PYTHON_FILES:
        owners = marker_sites(path.read_text(encoding="utf-8")).get(GPU_MARKER)
        if owners:
            found[path] = owners
    return found


def test_no_unallowlisted_gpu_required_marker_in_the_e2e_tier() -> None:
    """``gpu_required`` *skips*, so it must not reach a device-agnostic tier.

    Passes vacuously today -- the sentinel Phase 0 adds does not exist yet, and
    no other site does either. It is still a live guard: the assertion is that
    every site found is allowlisted, so the first unlisted one fails here rather
    than quietly turning a journey into a test that has never run on CI.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): sorted(owners - GPU_SENTINEL_TEST_NAMES)
        for path, owners in gpu_required_sites_in_tier().items()
        if owners - GPU_SENTINEL_TEST_NAMES
    }
    assert not offenders, (
        f"`{GPU_MARKER}` in the E2E tier outside the sentinel allowlist: {offenders}. "
        "The marker skips, and every workflow runs on `ubuntu-latest`, so a marked "
        "journey has never executed in CI. Take the device from the `e2e_device` "
        "fixture instead (docs/E2E_TEST_PLAN.md §1)."
    )


class TestTheAllowlist:
    """The sentinel allowlist must not become a free pass."""

    def test_the_allowlist_is_not_empty(self) -> None:
        """An empty allowlist would make the guard above unfalsifiable in reverse."""
        assert GPU_SENTINEL_TEST_NAMES

    def test_each_allowlisted_name_is_a_sentinel_or_still_a_forward_reference(self) -> None:
        """An allowlisted name may only exist *carrying* the marker it excuses.

        The entry is a forward reference until Phase 0 lands the sentinel. Once
        a test of that name exists, it must actually be the ``gpu_required``
        sentinel -- otherwise the allowlist would silently excuse an ordinary
        journey that happened to be given the same name.
        """
        marked = {owner for owners in gpu_required_sites_in_tier().values() for owner in owners}
        defined: set[str] = set()
        for path in E2E_PYTHON_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined.add(node.name)
        for name in GPU_SENTINEL_TEST_NAMES:
            if name in defined:
                assert name in marked, (
                    f"{name} exists in the E2E tier but does not carry `{GPU_MARKER}` -- "
                    "it is allowlisted as the GPU sentinel, so either mark it or remove "
                    "the allowlist entry."
                )


# --------------------------------------------------------------------------
# (f) The optional-extra half is selected somewhere.
# --------------------------------------------------------------------------


def test_a_ci_step_positively_selects_the_fem_required_e2e_half() -> None:
    """``test-e2e`` excludes ``fem_required``; some job must positively select it.

    Excluded in the only job that runs the tier and selected nowhere else is
    the invisibility bug with an extra step: the tests exist, collect, and never
    run. ``test-extras`` is the one job that installs the ``[fem]`` extra, so it
    owns the positive selection.
    """
    commands = fem_selecting_commands()
    assert commands, (
        f"no step in {CI_WORKFLOW_FILENAME} runs `tests/e2e/` with a `-m` expression "
        f"that positively requires `{FEM_MARKER}`. The `test-e2e` job excludes that "
        "half (it does not install the extra), so without this step it runs nowhere. "
        "Restore the `Run fem_required E2E journeys` step in `test-extras`."
    )


# --------------------------------------------------------------------------
# Meta-guards: the parsers, driven on synthetic input.
# --------------------------------------------------------------------------


class TestMarkerSites:
    """Unit-tests the AST walker, so it cannot pass by matching nothing.

    Per the ``TestGatePredicate`` precedent in
    ``tests/research/test_amr_arena_interpretability.py``: a predicate exercised
    only through live data whose current shape never enters half its branches is
    a predicate nothing checks.
    """

    def test_module_level_pytestmark(self) -> None:
        assert marker_sites("import pytest\npytestmark = pytest.mark.e2e\n") == {
            "e2e": {MODULE_OWNER}
        }

    def test_module_level_pytestmark_list(self) -> None:
        source = "import pytest\npytestmark = [pytest.mark.e2e, pytest.mark.slow]\n"
        assert marker_sites(source) == {"e2e": {MODULE_OWNER}, "slow": {MODULE_OWNER}}

    def test_function_decorator(self) -> None:
        source = "import pytest\n@pytest.mark.e2e\ndef test_x() -> None:\n    pass\n"
        assert marker_sites(source) == {"e2e": {"test_x"}}

    def test_decorator_with_arguments(self) -> None:
        source = (
            "import pytest\n"
            '@pytest.mark.skipif(True, reason="x")\n'
            "def test_x() -> None:\n    pass\n"
        )
        assert marker_sites(source) == {"skipif": {"test_x"}}

    def test_class_level_pytestmark(self) -> None:
        source = "import pytest\nclass TestX:\n    pytestmark = pytest.mark.gpu_required\n"
        assert marker_sites(source) == {"gpu_required": {"TestX"}}

    def test_unmarked_module_yields_nothing(self) -> None:
        assert marker_sites("def test_x() -> None:\n    pass\n") == {}

    def test_a_lookalike_decorator_is_not_a_marker(self) -> None:
        """``mark.e2e`` without the ``pytest`` root, and unrelated decorators."""
        source = (
            "import functools\n@functools.cache\ndef helper() -> None:\n    pass\n"
            "other = mark.e2e\n"
        )
        assert marker_sites(source) == {}

    def test_a_mention_in_a_comment_or_string_is_not_a_marker(self) -> None:
        """Why this is an AST walk and not a grep.

        ``tests/e2e/conftest.py`` discusses ``gpu_required`` in prose; a text
        scan would report the whole tier as GPU-marked.
        """
        source = '# pytest.mark.gpu_required\nDOC = "pytest.mark.e2e"\n'
        assert marker_sites(source) == {}


class TestSelectionPredicates:
    """Unit-tests the workflow predicates on synthetic commands."""

    def test_directory_token_is_selected(self) -> None:
        assert selects_e2e_directory("pytest tests/e2e/ -m 'not slow' -q")
        assert selects_e2e_directory("pytest tests/e2e -q")

    def test_an_ignore_of_the_directory_is_not_a_selection(self) -> None:
        """The historical state: the path appears, the tier does not run."""
        assert not selects_e2e_directory("pytest tests/ --ignore=tests/e2e/ -q")

    def test_a_single_file_is_not_a_directory_selection(self) -> None:
        """``test-chess`` names one file; that is not the tier."""
        assert not selects_e2e_directory("pytest tests/e2e/test_chess_training_e2e.py -q")

    def test_effective_expression_is_the_last_one(self) -> None:
        assert effective_marker_expression('pytest tests/e2e/ -m "a" -m "b"') == "b"

    def test_absent_expression_selects_everything(self) -> None:
        assert effective_marker_expression("pytest tests/e2e/ -q") == ""
        assert expression_selects_plainly("", E2E_MARKER)

    def test_a_module_invocation_is_not_an_expression(self) -> None:
        """``python -m pytest`` must not read as ``-m pytest`` the marker filter."""
        assert effective_marker_expression("python -m pytest tests/e2e/ -q") == ""


class TestTheGuardItself:
    """Meta-guards. A parser matching nothing passes every assertion above."""

    def test_the_workflow_parser_finds_the_e2e_job(self) -> None:
        """Pins the discovery, so a YAML shape change cannot silently empty it."""
        jobs = {run.job for run, _ in e2e_selecting_commands()}
        assert jobs, "no job found running the tier -- the command parser is broken"

    def test_ci_success_blocking_jobs_are_recognised(self) -> None:
        """The blocking-set reader must find the gates that demonstrably exist.

        ``lint`` and ``test-fast`` have carried hard ``exit 1`` gates since long
        before this branch. If they stop being recognised, every ``(c)``
        assertion above is passing on an empty set.
        """
        blocking = ci_success_blocking_jobs()
        assert {"lint", "test-fast"} <= blocking, (
            f"the ci-success gate parser found only {sorted(blocking)}"
        )

    def test_a_reported_but_ungated_job_is_not_counted_as_blocking(self) -> None:
        """``echo`` of a result is not a gate; only ``if ... exit 1`` is.

        ``transfer-baseline-regression`` is deliberately soft: it is echoed and
        commented as non-blocking. It must not be read as blocking, or the
        distinction the whole of ``(c)`` rests on does not exist.
        """
        assert "transfer-baseline-regression" not in ci_success_blocking_jobs()

    def test_the_makefile_parser_reads_a_known_target(self) -> None:
        assert makefile_target_recipe(MAKEFILE_E2E_TARGET), "the Makefile parser found no recipe"
        assert not makefile_target_recipe("no-such-target-exists")

    def test_hard_gate_jobs_requires_an_exit(self) -> None:
        gated = 'if [[ "${{ needs.a.result }}" != "success" ]]; then\n  exit 1\nfi\n'
        echoed = 'if [[ "${{ needs.b.result }}" != "success" ]]; then\n  echo "note"\nfi\n'
        assert hard_gate_jobs(gated) == {"a"}
        assert hard_gate_jobs(echoed) == set()


# --------------------------------------------------------------------------- #
# Clause (g): the memory-workaround split must stay exhaustive                 #
# --------------------------------------------------------------------------- #

#: The ``-k`` partition the `test-e2e` job runs the tier under.
#:
#: The tier cannot currently run as a single pytest process: the parent's RSS
#: climbs monotonically and never releases, so the job dies (on a GitHub runner
#: with "The runner has received a shutdown signal"; locally OOM-killed at a
#: 13.6 GB peak). Splitting at the first in-process file caps each process --
#: measured 1,451 MB and 3,614 MB respectively. See docs/E2E_TEST_PLAN.md 12.4;
#: the leak is pre-existing chess/MCTS work, not the journeys'.
#:
#: The pair is exhaustive by construction (``X`` and ``not X``), so no test can
#: fall between the two steps. What this constant guards is the *other* failure
#: mode: deleting one step, which would silently retire half the tier -- exactly
#: the invisibility this whole file exists to prevent, reintroduced through the
#: workaround for a different problem.
E2E_REQUIRED_K_PARTITION: Final[frozenset[str]] = frozenset({"chess", "not chess"})

#: Set when the tier is run by a single unfiltered step. Kept as an accepted
#: shape so that fixing the leak and collapsing the two steps back into one does
#: NOT require editing this guard -- only deleting the partition constant.
E2E_UNFILTERED: Final[frozenset[str]] = frozenset({""})


def effective_k_expression(command: str) -> str:
    """The ``-k`` expression pytest would apply to *command*.

    Mirrors :func:`effective_marker_expression`: pytest keeps the last ``-k``
    when several are given.

    Args:
        command: One logical shell command.

    Returns:
        The expression, or ``""`` when the command has no ``-k``.

    """
    tokens = shlex.split(command)
    expression = ""
    for index, token in enumerate(tokens[:-1]):
        if token == "-k":
            expression = tokens[index + 1]
    return expression


def test_the_e2e_k_partition_is_complete() -> None:
    """Every step running the tier together covers all of it.

    Two accepted shapes: one unfiltered step, or the complementary ``-k`` pair
    recorded in :data:`E2E_REQUIRED_K_PARTITION`. Anything else -- one half
    deleted, or a third filter added -- means some journeys run nowhere.
    """
    observed = {effective_k_expression(command) for _run, command in e2e_selecting_ci_commands()}
    assert observed in (E2E_UNFILTERED, E2E_REQUIRED_K_PARTITION), (
        f"the `-k` filters on the steps running tests/e2e/ are {sorted(observed)!r}, which is "
        f"neither a single unfiltered step nor the complete partition "
        f"{sorted(E2E_REQUIRED_K_PARTITION)!r}. Half the tier is running nowhere -- the "
        "invisibility this file guards, reintroduced through the memory workaround."
    )


class TestKExpressionParsing:
    """Unit-test the ``-k`` extractor on synthetic input.

    Written because the live data currently has exactly two shapes; a parser
    that returned ``""`` for everything would satisfy neither branch of the
    assertion above by accident, but would silently accept a third step.
    """

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ('pytest tests/e2e/ -k "not chess"', "not chess"),
            ("pytest tests/e2e/ -k chess", "chess"),
            ("pytest tests/e2e/", ""),
            ('pytest tests/e2e/ -k "a" -k "b"', "b"),
            ("pytest tests/e2e/ -m e2e", ""),
            ("pytest -k", ""),
        ],
    )
    def test_reads_the_last_k_expression(self, command: str, expected: str) -> None:
        """The last ``-k`` wins; a missing or dangling one yields the empty string."""
        assert effective_k_expression(command) == expected
