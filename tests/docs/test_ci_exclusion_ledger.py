"""Guard the CI ``--ignore`` / ``--deselect`` ledger against a silent fourth copy.

Defect class: a flag that appears in ``ci.yml`` but not in the ledger (or the
reverse) is how exclusions rot — the next invisibility incident, named in
prose, enforced by nothing.

The three invocation sites (``test-fast``, ``coverage``, Makefile
``CI_TEST_EXCLUDES``) must carry the same ignore/deselect set as
``docs/ci-exclusion-ledger.md``'s YAML block. Extracting those sites onto one
shared args file is hygiene B7 and is deliberately not done here.

Hermetic: parses YAML, the workflow, and the Makefile. Runs no pytest
children.

Mutations (named tests, not ``gpu_required`` / ``fem_required``):

1. Drop a ledger ignore that CI still has — ``test_ledger_matches_ci`` fails.
2. Add a ``--deselect`` to a synthetic CI script the parser would accept —
   ``test_flag_extraction_sees_deselect`` still finds it; the live equality
   test fails if it lands in ``ci.yml`` without a ledger row.
3. Make the coverage list differ from test-fast —
   ``test_fast_and_coverage_exclusions_match`` fails.
4. Empty the YAML block — ``test_the_ledger_is_not_empty`` fails.
5. Leave ``reopen`` blank — ``test_every_ledger_row_has_owner_reason_reopen``
   fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Final

import yaml

from tests.support.workflows import (
    CI_WORKFLOW,
    CI_WORKFLOW_FILENAME,
    MAKEFILE,
    REPO_ROOT,
    iter_commands,
    iter_run_scripts,
    join_line_continuations,
    strip_shell_comments,
)

LEDGER_PATH: Final[Path] = REPO_ROOT / "docs" / "ci-exclusion-ledger.md"

FAST_JOB: Final[str] = "test-fast"
FAST_STEP: Final[str] = "Run fast unit tests"
COVERAGE_JOB: Final[str] = "coverage"
COVERAGE_STEP: Final[str] = "Run tests with coverage"
MAKE_EXCLUDES_NAME: Final[str] = "CI_TEST_EXCLUDES"

_LEDGER_FENCE = re.compile(
    r"<!-- ci-exclusion-ledger:begin -->\s*```yaml\n(?P<body>.*?)\n```\s*"
    r"<!-- ci-exclusion-ledger:end -->",
    re.S,
)
_FLAG = re.compile(r"--(?P<kind>ignore|deselect)=(?P<value>\S+)")
_MAKE_EXCLUDES = re.compile(
    rf"^{MAKE_EXCLUDES_NAME}\s*:=\s*(?P<body>(?:.*\\\n)*.*)$",
    re.M,
)


def extract_exclusion_flags(script: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(--ignore=...,)`` and ``(--deselect=...,)`` in source order.

    Args:
        script: A workflow ``run:`` body or a joined Makefile assignment.

    Returns:
        Two tuples of the path/nodeid values (the part after ``=``).

    """
    normalised = join_line_continuations(strip_shell_comments(script))
    ignores: list[str] = []
    deselects: list[str] = []
    for match in _FLAG.finditer(normalised):
        value = match.group("value")
        if match.group("kind") == "ignore":
            ignores.append(value)
        else:
            deselects.append(value)
    return tuple(ignores), tuple(deselects)


def _step_script(job: str, step: str) -> str:
    found = [
        run.script
        for run in iter_run_scripts()
        if run.workflow == CI_WORKFLOW_FILENAME and run.job == job and run.step == step
    ]
    assert found, f"no ci.yml step {job!r}/{step!r}"
    assert len(found) == 1, f"ci.yml step {job!r}/{step!r} is not unique: {len(found)}"
    return found[0]


def _makefile_ci_test_excludes(text: str) -> str:
    """The ``CI_TEST_EXCLUDES :=`` assignment, continuations preserved.

    Must not go through ``makefile_commands``: continuation lines of a
    variable start with a tab, and that helper strips Make recipe prefixes
    (``-``), which would turn ``--ignore`` into ``-ignore``.
    """
    match = _MAKE_EXCLUDES.search(text)
    assert match is not None, f"{MAKE_EXCLUDES_NAME} assignment not found in Makefile"
    return match.group("body")


def load_ledger(text: str) -> dict[str, Any]:
    """Parse the YAML fence in the ledger document.

    Args:
        text: Full markdown source of ``docs/ci-exclusion-ledger.md``.

    Returns:
        The parsed mapping (``ignores`` / ``deselects`` lists).

    """
    match = _LEDGER_FENCE.search(text)
    assert match is not None, (
        "ledger YAML fence not found; expected "
        "<!-- ci-exclusion-ledger:begin --> … ```yaml … ``` "
        "<!-- ci-exclusion-ledger:end -->"
    )
    document = yaml.safe_load(match.group("body"))
    assert isinstance(document, dict), "ledger YAML must be a mapping"
    return document


def _required_string(row: dict[str, Any], key: str, label: str) -> str:
    value = row.get(key)
    assert isinstance(value, str) and value.strip(), f"{label} missing non-empty {key!r}"
    return value.strip()


def ledger_ignore_paths(document: dict[str, Any]) -> tuple[str, ...]:
    rows = document.get("ignores")
    assert isinstance(rows, list), "ledger ignores must be a list"
    return tuple(_required_string(row, "path", "ignore") for row in rows)


def ledger_deselect_nodeids(document: dict[str, Any]) -> tuple[str, ...]:
    rows = document.get("deselects")
    assert isinstance(rows, list), "ledger deselects must be a list"
    return tuple(_required_string(row, "nodeid", "deselect") for row in rows)


# --------------------------------------------------------------------------
# Vacuity
# --------------------------------------------------------------------------


def test_the_ledger_file_exists() -> None:
    """Without this, every parse below iterates nothing."""
    assert LEDGER_PATH.is_file(), f"missing {LEDGER_PATH}"


def test_the_ledger_is_not_empty() -> None:
    """A YAML fence that parses to empty lists matches an empty CI list forever."""
    document = load_ledger(LEDGER_PATH.read_text(encoding="utf-8"))
    ignores = ledger_ignore_paths(document)
    deselects = ledger_deselect_nodeids(document)
    assert ignores, "ledger ignores is empty"
    assert deselects, "ledger deselects is empty"


def test_fast_and_coverage_steps_exist() -> None:
    """Renaming the CI step without updating this file would skip the equality."""
    fast = _step_script(FAST_JOB, FAST_STEP)
    coverage = _step_script(COVERAGE_JOB, COVERAGE_STEP)
    assert "pytest" in fast
    assert "pytest" in coverage


# --------------------------------------------------------------------------
# Parser unit tests (synthetic; so a rewrite of extract_exclusion_flags fails
# a named test rather than only the live equality).
# --------------------------------------------------------------------------


class TestFlagExtraction:
    def test_flag_extraction_sees_ignore_and_deselect(self) -> None:
        nodeid = "tests/data/test_dataset.py::TestReplayDataset::test_iteration_with_dataloader"
        script = f"pytest tests/ \\\n  --ignore=tests/e2e/ \\\n  --deselect={nodeid}\n"
        ignores, deselects = extract_exclusion_flags(script)
        assert ignores == ("tests/e2e/",)
        assert deselects == (
            "tests/data/test_dataset.py::TestReplayDataset::test_iteration_with_dataloader",
        )

    def test_flag_extraction_ignores_shell_comments(self) -> None:
        script = "pytest tests/ --ignore=tests/e2e/  # --ignore=tests/secret/\n"
        ignores, deselects = extract_exclusion_flags(script)
        assert ignores == ("tests/e2e/",)
        assert deselects == ()

    def test_makefile_variable_keeps_double_dash(self) -> None:
        """Regression: tab-prefixed continuation must not strip ``-`` off ``--ignore``."""
        body = _makefile_ci_test_excludes(
            "CI_TEST_EXCLUDES := \\\n\t--ignore=tests/e2e/ \\\n\t--deselect=a::b\n"
        )
        ignores, deselects = extract_exclusion_flags(body)
        assert ignores == ("tests/e2e/",)
        assert deselects == ("a::b",)


# --------------------------------------------------------------------------
# Live lockstep
# --------------------------------------------------------------------------


def test_fast_and_coverage_exclusions_match() -> None:
    """The two ci.yml copies are the duplication this ledger exists to name.

    A change that updates only test-fast (or only coverage) is the defect
    Makefile comments already warn about. This named test fails on that
    split rather than waiting for a human to diff the YAML.
    """
    fast = extract_exclusion_flags(_step_script(FAST_JOB, FAST_STEP))
    coverage = extract_exclusion_flags(_step_script(COVERAGE_JOB, COVERAGE_STEP))
    assert fast[0], "test-fast ignore list is empty"
    assert fast[1], "test-fast deselect list is empty"
    assert fast == coverage, (
        "test-fast and coverage ignore/deselect lists differ; "
        "update both ci.yml steps and the ledger together"
    )


def test_makefile_excludes_match_ci() -> None:
    """``make test-fast`` must not be narrower than CI (the pre-pr lie)."""
    ci = extract_exclusion_flags(_step_script(FAST_JOB, FAST_STEP))
    make = extract_exclusion_flags(_makefile_ci_test_excludes(MAKEFILE.read_text(encoding="utf-8")))
    assert ci == make, (
        "Makefile CI_TEST_EXCLUDES does not match ci.yml test-fast; the third copy drifted"
    )


def test_ledger_matches_ci() -> None:
    """The ledger is not documentation of a list it does not equal.

    Order is part of the contract: a reshuffle that keeps the set but
    changes pytest's application order is still a three-way edit.
    """
    document = load_ledger(LEDGER_PATH.read_text(encoding="utf-8"))
    ledger_ignores = ledger_ignore_paths(document)
    ledger_deselects = ledger_deselect_nodeids(document)
    ci_ignores, ci_deselects = extract_exclusion_flags(_step_script(FAST_JOB, FAST_STEP))
    assert ledger_ignores == ci_ignores
    assert ledger_deselects == ci_deselects


def test_every_ledger_row_has_owner_reason_reopen() -> None:
    """A row without reopen criteria is a freeze nobody can lift."""
    document = load_ledger(LEDGER_PATH.read_text(encoding="utf-8"))
    for row in document["ignores"]:
        label = row.get("path", "<missing path>")
        _required_string(row, "owner", f"ignore {label}")
        _required_string(row, "reason", f"ignore {label}")
        _required_string(row, "reopen", f"ignore {label}")
        assert "covered_by" in row, f"ignore {label} missing covered_by (use ~ if none)"
    for row in document["deselects"]:
        label = row.get("nodeid", "<missing nodeid>")
        _required_string(row, "owner", f"deselect {label}")
        _required_string(row, "reason", f"deselect {label}")
        _required_string(row, "reopen", f"deselect {label}")
        assert "covered_by" in row, f"deselect {label} missing covered_by (use ~ if none)"


def test_every_ignore_path_exists() -> None:
    """A stale ignore of a deleted tree is an exemption that no longer pays rent."""
    document = load_ledger(LEDGER_PATH.read_text(encoding="utf-8"))
    for path in ledger_ignore_paths(document):
        target = REPO_ROOT / path
        assert target.exists(), f"ignored path does not exist on disk: {path}"


def test_every_deselect_nodeid_still_names_a_test() -> None:
    """Deselecting a renamed test is a silent no-op (pytest warns, CI stays green)."""
    document = load_ledger(LEDGER_PATH.read_text(encoding="utf-8"))
    for nodeid in ledger_deselect_nodeids(document):
        file_part, _, rest = nodeid.partition("::")
        assert file_part.endswith(".py"), f"deselect is not a nodeid: {nodeid}"
        source = (REPO_ROOT / file_part).read_text(encoding="utf-8")
        func = rest.rsplit("::", 1)[-1]
        assert f"def {func}(" in source, (
            f"deselect {nodeid} names {func} which is not defined in {file_part}"
        )


def test_ci_yml_points_at_the_ledger() -> None:
    """A comment the next editor will see, not a doc nobody opens while editing YAML."""
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "docs/ci-exclusion-ledger.md" in text, (
        "ci.yml has no pointer to the ledger; the next ignore will be added in YAML only"
    )


def test_makefile_points_at_the_ledger() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "docs/ci-exclusion-ledger.md" in text


def test_b7_extract_is_not_in_this_change() -> None:
    """The optional shared-args extraction must not sneak in as a CI rewrite.

    B7 is the right long-term move and the wrong one to bundle with a CI-green
    restore. This asserts we did not introduce a pytest addopts / args-file
    that the two jobs source — the invocations still spell the flags inline.
    """
    fast = _step_script(FAST_JOB, FAST_STEP)
    coverage = _step_script(COVERAGE_JOB, COVERAGE_STEP)
    for script, label in ((fast, FAST_STEP), (coverage, COVERAGE_STEP)):
        commands = iter_commands(script)
        pytest_cmds = [
            cmd for cmd in commands if cmd.split()[0].endswith("pytest") or cmd.startswith("pytest")
        ]
        assert pytest_cmds, f"{label} has no pytest command"
        joined = " ".join(pytest_cmds)
        assert "--ignore=" in joined
        assert "xargs" not in joined
        assert "pytest-fast-excludes" not in joined
