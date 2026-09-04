"""E2E journeys for the abstraction-audit governance CLI.

Guards the CLAUDE.md Regression Surface row *"Abstraction audit (F0/F1
screen)"* and ``docs/E2E_TEST_PLAN.md`` §6.2.

The row and the workflow are two statements of the same gate, and nothing
previously held them together: the row names four roots
(``src/mcts src/refinement src/pde src/research``) and ``.github/workflows/
ci.yml`` names four roots, but a fifth added to either would not have failed
anything. :func:`test_audit_abstractions_with_the_ci_gated_argv_is_clean`
parses the workflow, asserts the root *set* equals the four the row names, and
then runs exactly that argv — so the documentation, the gate's scope, and the
gate's verdict are asserted together.

**Parsing scope, deliberately narrow.** The ``lint`` job runs
``scripts.audit_abstractions`` three times. Only the *first* has explicit roots
plus ``--fail-on-missing``; the second expands ``$(ls -d src/*/ | grep -v ...)``
in the shell and so is not hermetically parseable (a test that tried would have
to reimplement ``ls`` + ``grep``, which is a second, divergent copy of the
scope), and the third is ``continue-on-error`` and report-only. Only the first
is parsed, and :data:`SHELL_SUBSTITUTION` is what makes "explicit" checkable
rather than assumed.

**Device (plan §1, flow (c)):** ``scripts/audit_abstractions.py`` is a pure AST
walk over source files. It imports no torch, resolves no device, and takes no
device flag; no device is forwarded and no device assertion is fabricated.
Nothing is written to disk, so ``tmp_path`` is not needed here.
"""

from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from tests.e2e.conftest import E2E_TRIVIAL_TIMEOUT_S, PROJECT_ROOT

if TYPE_CHECKING:
    from tests.e2e.conftest import CLIRunnerType

pytestmark = pytest.mark.e2e

#: The entry point under test.
AUDIT_MODULE = "scripts.audit_abstractions"

#: Exit codes asserted here. The audit exits non-zero *only* under
#: ``--fail-on-missing``, which is what makes the report-only journey's ``0``
#: an assertion about the flag's semantics rather than about the findings.
EXIT_OK = 0

#: The workflow carrying the blocking gate, and the job the audit runs in.
CI_WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
LINT_JOB = "lint"

#: The flag that turns the audit from a report into a gate.
FAIL_ON_MISSING_FLAG = "--fail-on-missing"

#: Marker for a shell command substitution. Its presence is what disqualifies
#: the second audit step from being parsed as "explicit roots".
SHELL_SUBSTITUTION = "$("

#: The roots CLAUDE.md's *"Abstraction audit (F0/F1 screen)"* row declares
#: gated. Compared as a set against the workflow, so a fifth root in either
#: place, or a dropped one, fails.
CLAUDE_MD_GATED_ROOTS = frozenset({"src/mcts", "src/refinement", "src/pde", "src/research"})

#: The report-only root: the ``src/backend`` domain-PoC backlog, which the
#: workflow runs with ``continue-on-error`` precisely because it has findings.
REPORT_ONLY_ROOT = "src/backend"

#: Lines the audit prints. ``CLEAN_REPORT_PREFIX`` is emitted only when nothing
#: is missing; ``FINDINGS_HEADERS`` are the two "here is what has no caller"
#: banners, either of which proves the report-only run reached real subjects.
CLEAN_REPORT_PREFIX = "OK: every abstract method"
FINDINGS_HEADERS = (
    "Protocol members with NO reader:",
    "Abstract methods with NO call site (dead abstractions):",
)

#: Shape the parsed command must have before its roots are trusted.
EXPECTED_COMMAND_PREFIX = ("python", "-m", AUDIT_MODULE)

_CONTINUATION = re.compile(r"\\\n\s*")


def _lint_job_steps() -> list[dict[str, Any]]:
    """Return the ``lint`` job's steps, parsed from ``ci.yml``.

    Returns:
        One mapping per step, in workflow order.

    """
    document: Any = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{CI_WORKFLOW} did not parse to a mapping"
    jobs = document.get("jobs")
    assert isinstance(jobs, dict), f"{CI_WORKFLOW} declares no jobs"
    job = jobs.get(LINT_JOB)
    assert isinstance(job, dict), f"{CI_WORKFLOW} has no '{LINT_JOB}' job"
    steps = job.get("steps")
    assert isinstance(steps, list), f"{CI_WORKFLOW}::{LINT_JOB} has no steps"
    return [step for step in steps if isinstance(step, dict)]


def _first_gated_audit_command() -> list[str]:
    """The first ``lint`` command that gates on the audit with explicit roots.

    Backslash continuations are folded first, because the workflow wraps the
    invocation across two lines and the roots and the flag are only on one
    logical line after that.

    Returns:
        The command's tokens, as :func:`shlex.split` produces them.

    Raises:
        AssertionError: No such command exists — which would make every
            assertion built on it vacuous, so it fails loudly rather than
            returning an empty list.

    """
    for step in _lint_job_steps():
        script = step.get("run")
        if not isinstance(script, str):
            continue
        for line in _CONTINUATION.sub(" ", script).splitlines():
            command = line.strip()
            if AUDIT_MODULE not in command:
                continue
            if FAIL_ON_MISSING_FLAG not in command or SHELL_SUBSTITUTION in command:
                continue
            return shlex.split(command)
    raise AssertionError(
        f"{CI_WORKFLOW}::{LINT_JOB} has no '{AUDIT_MODULE} ... {FAIL_ON_MISSING_FLAG}' "
        "step with explicit (non-substituted) roots"
    )


def _roots_of(command: list[str]) -> list[str]:
    """Positional root arguments of a parsed audit command.

    Args:
        command: Tokens of the invocation.

    Returns:
        Every token after the module name that is not an option flag, in order
        (a list, not a set, so a duplicated root is still visible to callers).

    """
    assert tuple(command[:3]) == EXPECTED_COMMAND_PREFIX, (
        f"unexpected invocation shape: {command!r}"
    )
    return [token for token in command[3:] if not token.startswith("-")]


def test_audit_abstractions_with_the_ci_gated_argv_is_clean(cli_runner: CLIRunnerType) -> None:
    """The workflow's gated audit argv covers the documented roots and passes.

    Guards the CLAUDE.md row *"Abstraction audit (F0/F1 screen)"*, which states
    that ``src/mcts src/refinement src/pde src/research`` are gated with
    ``--fail-on-missing``. Two independent failure modes are covered:

    1. **Scope drift** — the parsed root set must equal
       :data:`CLAUDE_MD_GATED_ROOTS` exactly. A fifth root added to the
       workflow, or one silently dropped, fails here, so the row and the
       workflow cannot diverge unnoticed.
    2. **Verdict** — that exact argv is then run as a process and must exit 0,
       so a newly dead abstraction under any gated root fails this test as well
       as CI.

    The workflow is the source of the argv (never a hardcoded copy of it), and
    CLAUDE.md is the source of the expected scope; asserting one against the
    other is the point.

    Device-irrelevant surface (plan §1, flow (c)): a pure AST walk.
    """
    command = _first_gated_audit_command()
    roots = _roots_of(command)

    assert roots, "parsed no roots -- the run below would audit the default scope, not the gate"
    assert len(roots) == len(set(roots)), f"duplicate root in the workflow argv: {roots!r}"
    assert set(roots) == set(CLAUDE_MD_GATED_ROOTS), (
        "ci.yml's gated audit roots and CLAUDE.md's 'Abstraction audit (F0/F1 screen)' "
        f"row disagree: workflow={sorted(roots)} vs documented={sorted(CLAUDE_MD_GATED_ROOTS)}"
    )
    for root in roots:
        assert (PROJECT_ROOT / root).is_dir(), f"gated root does not exist on disk: {root}"

    result = cli_runner(AUDIT_MODULE, [*roots, FAIL_ON_MISSING_FLAG], E2E_TRIVIAL_TIMEOUT_S, None)

    assert result.returncode == EXIT_OK, result.output
    assert CLEAN_REPORT_PREFIX in result.output


def test_audit_report_only_root_exits_zero_with_findings(cli_runner: CLIRunnerType) -> None:
    """Without ``--fail-on-missing`` the audit reports and still exits 0.

    Guards the CLAUDE.md row *"Abstraction audit (F0/F1 screen)"*'s second
    half: *"The rest of ``src/`` (notably the ``backend`` domain-PoC backlog)
    runs report-only in the same job."* The property under test is that the
    gate is opt-in — findings alone must not fail the build — because the
    workflow relies on it (the step is ``continue-on-error``, which would mask
    a change here, so this test is the only place the ``0`` is actually
    asserted).

    The findings assertion is also this test's non-vacuity proof: an audit that
    scanned nothing would print no banner, and a bare ``exit 0`` would then be
    an assertion about an empty scan rather than about the flag.

    Device-irrelevant surface (plan §1, flow (c)): a pure AST walk.
    """
    assert (PROJECT_ROOT / REPORT_ONLY_ROOT).is_dir()

    result = cli_runner(AUDIT_MODULE, [REPORT_ONLY_ROOT], E2E_TRIVIAL_TIMEOUT_S, None)

    assert result.returncode == EXIT_OK, result.output
    assert any(header in result.output for header in FINDINGS_HEADERS), (
        f"expected findings under {REPORT_ONLY_ROOT}; got: {result.output}"
    )
