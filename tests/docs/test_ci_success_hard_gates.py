"""Guards the merge gate's *membership* (plan R-09).

``ci-success`` is the only job branch protection can require, so a job is a
merge gate exactly when it is in ``ci-success``'s ``needs:`` list **and** named
in an ``if`` block there whose body reaches ``exit 1``. Both halves are needed:
a job in ``needs`` that is only ``echo``ed is a report, and an ``exit 1`` block
naming a job that is not in ``needs`` reads an empty result and never fires.

``focus`` (scope containment) and ``secrets`` (gitleaks) each landed outside
the gate on purpose -- a brand-new check promoted into someone's merge path by
the pull request that introduces it gives a first red no way to be triaged --
with a comment promising promotion once green. Promises in comments rot; this
module turns the promotion into a check. ``focus`` is ``pull_request``-only, so
the gate must accept ``skipped`` from it on a push and reject it on a pull
request that does not carry the visible ``focus-override`` label, and the
*shape* of that condition is asserted, not only the job's presence.

The final test closes the class rather than the instance: every job in
``ci.yml`` is either a hard gate or an entry in :data:`DISCLOSED_NOT_HARD`
with a reason, and each entry is asserted to still be needed, so a job that
is later promoted expires its own exemption instead of keeping a stale one.

Mutation kills (each planted, run, reverted):

* drop ``focus`` from ``ci-success.needs`` ->
  ``test_promoted_job_is_in_ci_success_needs[focus]`` and
  ``test_every_job_is_a_hard_gate_or_a_disclosed_exception``
* ``exit 1`` -> ``echo`` in the ``secrets`` block ->
  ``test_promoted_job_is_hard_gated[secrets]``
* bare ``!= "success"`` for ``focus`` (drop the ``skipped`` clause and the
  labelled-skip block) -> ``test_focus_gate_accepts_skipped_only_when_explained``
* a new ``if: github.event_name != 'schedule'`` job outside ``needs`` ->
  ``test_every_job_is_a_hard_gate_or_a_disclosed_exception``
* ``!= "success"`` -> ``== "failure"`` on the ``secrets`` block (Copilot review,
  PR #151: cancelled/skipped would merge) ->
  ``test_hard_gate_rejects_every_non_success_result[secrets]``
* ``exit 1`` -> ``echo "would exit 1"`` (Copilot review: a substring match
  credited text as a gate) -> ``test_promoted_job_is_hard_gated[secrets]``,
  because the parser now requires a standalone ``exit <non-zero>`` command
"""

from __future__ import annotations

from typing import Final

import pytest

from tests.support.workflows import (
    CI_SUCCESS_JOB,
    CI_WORKFLOW,
    CI_WORKFLOW_FILENAME,
    body_exits_nonzero,
    hard_gate_conditions,
    hard_gate_jobs,
    iter_run_scripts,
    job_needs,
    load_workflow,
)

#: Jobs this cycle placed into the merge gate: ``focus`` and ``secrets`` were
#: promoted by R-09; ``typecheck`` was split out of ``lint`` by R-12a and
#: carries the abstraction audit, which was already a hard gate there. Each
#: must be in ``needs`` and hard.
PROMOTED_JOBS: Final[tuple[str, ...]] = ("focus", "secrets", "typecheck")

#: The pull-request-only job whose ``skipped`` result needs explaining.
FOCUS_JOB: Final[str] = "focus"

#: The visible escape hatch. It must be spelled identically in the ``focus``
#: job's own ``if:`` and in ``ci-success``'s skip clause, or the two disagree.
FOCUS_OVERRIDE_LABEL: Final[str] = "focus-override"

#: Tokens the event-aware skip clause must carry. ``skipped`` is the result
#: being excused, ``github.event_name`` is what makes the excuse conditional,
#: and the label is the only legitimate reason for a pull-request skip.
FOCUS_SKIP_CLAUSE_TOKENS: Final[tuple[str, ...]] = (
    "skipped",
    "github.event_name",
    FOCUS_OVERRIDE_LABEL,
)

#: Jobs that fire in ``ci.yml`` and are deliberately NOT hard merge gates.
#: Every entry states why, and every entry is checked to still be needed --
#: an exemption for a job that has since become a hard gate fails the suite,
#: so this list cannot rot in either direction.
DISCLOSED_NOT_HARD: Final[dict[str, str]] = {
    "test-slow": (
        "PR-invisible by design: fires on schedule, the default branch, "
        "[full-test] or workflow_dispatch, never on a plain pull request"
    ),
    "transfer-baseline-regression": (
        "disclosed soft job (in needs, echo-only) until its cross-runner "
        "stability is characterised; plan R-07 decides hard-or-deviation"
    ),
}

#: A floor on the number of hard gates, so a parser that silently matches
#: nothing cannot make every membership assertion below vacuous.
MIN_HARD_GATES: Final[int] = 10


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _ci_document() -> dict[str, object]:
    return load_workflow(CI_WORKFLOW)


def _ci_success_script() -> str:
    """Every ``run:`` body of ``ci-success``, concatenated in step order."""
    scripts = [
        run.script
        for run in iter_run_scripts()
        if run.workflow == CI_WORKFLOW_FILENAME and run.job == CI_SUCCESS_JOB
    ]
    assert scripts, f"{CI_WORKFLOW_FILENAME} job {CI_SUCCESS_JOB!r} has no run: step"
    return "\n".join(scripts)


def _ci_success_needs() -> set[str]:
    return set(job_needs(_ci_document(), CI_SUCCESS_JOB))


def _hard_gates() -> set[str]:
    return hard_gate_jobs(_ci_success_script())


def _all_jobs() -> set[str]:
    jobs = _ci_document().get("jobs")
    assert isinstance(jobs, dict) and jobs, f"{CI_WORKFLOW_FILENAME} parsed to no jobs"
    return {str(name) for name in jobs}


def _job_if(job: str) -> str:
    jobs = _ci_document().get("jobs")
    assert isinstance(jobs, dict)
    entry = jobs.get(job)
    assert isinstance(entry, dict), f"{CI_WORKFLOW_FILENAME} has no job {job!r}"
    expr = entry.get("if")
    return str(expr) if expr is not None else ""


def _conditions_naming(job: str) -> list[str]:
    needle = f"needs.{job}.result"
    return [cond for cond in hard_gate_conditions(_ci_success_script()) if needle in cond]


# --------------------------------------------------------------------------
# Vacuity
# --------------------------------------------------------------------------


def test_the_gate_parser_finds_a_plausible_number_of_hard_gates() -> None:
    """A parser that matched nothing would pass every ``in`` assertion trivially."""
    gates = _hard_gates()
    assert len(gates) >= MIN_HARD_GATES, (
        f"only {len(gates)} hard gates parsed from {CI_SUCCESS_JOB}: {sorted(gates)}"
    )


class TestHardGateConditionsParser:
    """Unit tests for the shared helper on synthetic input."""

    def test_returns_only_exit_1_blocks(self) -> None:
        script = (
            'if [[ "${{ needs.a.result }}" != "success" ]]; then\n'
            '  echo "NOTE: soft"\n'
            "fi\n"
            'if [[ "${{ needs.b.result }}" != "success" ]]; then\n'
            '  echo "FAIL"\n'
            "  exit 1\n"
            "fi\n"
        )
        conditions = hard_gate_conditions(script)
        assert len(conditions) == 1
        assert "needs.b.result" in conditions[0]
        assert "needs.a.result" not in conditions[0]

    def test_exit_1_in_text_only_is_not_a_gate(self) -> None:
        """``echo "would exit 1"`` and a comment are not an ``exit`` command."""
        script = (
            'if [[ "${{ needs.a.result }}" != "success" ]]; then\n'
            '  echo "NOTE: soft -- a hard gate would exit 1 here"\n'
            "  # exit 1\n"
            "fi\n"
            'if [[ "${{ needs.b.result }}" != "success" ]]; then\n'
            "  exit 2\n"
            "fi\n"
            'if [[ "${{ needs.c.result }}" != "success" ]]; then\n'
            "  exit 0\n"
            "fi\n"
        )
        conditions = hard_gate_conditions(script)
        assert [("needs.b.result" in c) for c in conditions] == [True]
        assert body_exits_nonzero('echo "x"; exit 1')
        assert not body_exits_nonzero('echo "exit 1"')

    def test_a_multi_line_condition_is_returned_whole(self) -> None:
        script = (
            'if [[ "${{ needs.a.result }}" == "skipped" && "${{ github.event_name }}" == "x" \\\n'
            '      && "${{ contains(y, \'z\') }}" != "true" ]]; then\n'
            "  exit 1\n"
            "fi\n"
        )
        (condition,) = hard_gate_conditions(script)
        assert "github.event_name" in condition
        assert "contains(y, 'z')" in condition

    def test_agrees_with_hard_gate_jobs(self) -> None:
        script = _ci_success_script()
        named: set[str] = set()
        for condition in hard_gate_conditions(script):
            named.update(
                token.removeprefix("needs.").removesuffix(".result")
                for token in condition.replace('"', " ").replace("}", " ").split()
                if token.startswith("needs.") and token.endswith(".result")
            )
        assert named == hard_gate_jobs(script)


# --------------------------------------------------------------------------
# (1)+(2) The promoted jobs are in `needs` and hard-gated
# --------------------------------------------------------------------------


@pytest.mark.parametrize("job", PROMOTED_JOBS)
def test_promoted_job_is_in_ci_success_needs(job: str) -> None:
    assert job in _ci_success_needs(), (
        f"{job!r} is not in {CI_SUCCESS_JOB}.needs; its result is unreadable there "
        "and any `exit 1` block naming it never fires"
    )


@pytest.mark.parametrize("job", PROMOTED_JOBS)
def test_promoted_job_is_hard_gated(job: str) -> None:
    assert job in _hard_gates(), (
        f"{job!r} is echoed but not gated in {CI_SUCCESS_JOB}: no `if ... exit 1` block names "
        "needs.{job}.result, so it reports and cannot fail the build"
    )


@pytest.mark.parametrize("job", sorted(hard_gate_jobs(_ci_success_script())), ids=str)
def test_hard_gate_rejects_every_non_success_result(job: str) -> None:
    """A gate written ``== "failure"`` lets ``cancelled`` and ``skipped`` merge.

    ``hard_gate_jobs`` reports any exit-1 block naming the job; this asserts
    the *semantics* (Copilot review, PR #151): at least one such block must
    reject everything that is not ``success``. The ``focus`` skip clause is a
    second block and is checked for its own shape separately.
    """
    conditions = _conditions_naming(job)
    assert any('!= "success"' in c for c in conditions), (
        f'no exit-1 condition naming needs.{job}.result is of the form `!= "success"`; '
        f'a positive `== "failure"` test passes on cancelled/skipped: {conditions}'
    )


@pytest.mark.parametrize("job", PROMOTED_JOBS)
def test_promoted_job_exists(job: str) -> None:
    """A renamed job would leave a dangling `needs` entry, which GitHub rejects at parse."""
    assert job in _all_jobs()


# --------------------------------------------------------------------------
# (3) The `focus` gate is event-aware
# --------------------------------------------------------------------------


def test_focus_gate_accepts_skipped_only_when_explained() -> None:
    """``focus`` is pull_request-only, so a bare ``!= success`` is wrong twice.

    It would fail every push run (the job is ``skipped`` there), and the
    tempting fix -- ``== failure`` -- would let a pull request whose ``focus``
    job was skipped by accident merge. The gate therefore needs two clauses:
    one that rejects any result other than ``success``/``skipped``, and one
    that rejects ``skipped`` on a pull request without the override label.
    """
    conditions = _conditions_naming(FOCUS_JOB)
    assert conditions, f"no exit-1 block in {CI_SUCCESS_JOB} names needs.{FOCUS_JOB}.result"

    failure_clauses = [c for c in conditions if '!= "success"' in c and '!= "skipped"' in c]
    assert failure_clauses, (
        f"the {FOCUS_JOB} gate has no clause accepting `skipped`; a push run (where the "
        "pull_request-only job is skipped) would fail ci-success"
    )

    skip_clauses = [c for c in conditions if all(token in c for token in FOCUS_SKIP_CLAUSE_TOKENS)]
    assert skip_clauses, (
        f"the {FOCUS_JOB} gate accepts `skipped` unconditionally; a skip on a pull request "
        f"without the {FOCUS_OVERRIDE_LABEL!r} label must fail. Expected a clause carrying all "
        f"of {FOCUS_SKIP_CLAUSE_TOKENS}, got: {conditions}"
    )


def test_focus_override_label_is_spelled_identically_in_both_places() -> None:
    """The job skips itself under the label; ci-success excuses the skip under the label.

    If the two spellings drift, one side stops recognising the other and the
    escape hatch either fails the build or opens it silently.
    """
    job_if = _job_if(FOCUS_JOB)
    assert FOCUS_OVERRIDE_LABEL in job_if, f"{FOCUS_JOB}'s own if: no longer names the label"
    assert "pull_request" in job_if, f"{FOCUS_JOB} is no longer pull_request-only: {job_if!r}"
    assert any(FOCUS_OVERRIDE_LABEL in c for c in _conditions_naming(FOCUS_JOB))


# --------------------------------------------------------------------------
# (4) Every job is a hard gate or a disclosed exception
# --------------------------------------------------------------------------


def test_every_job_is_a_hard_gate_or_a_disclosed_exception() -> None:
    """Closes the class: a new job cannot land as a silent report.

    This is how ``focus``, ``secrets``, ``test-integration``, ``test-jax`` and
    ``test-chess`` all came to run on every pull request while blocking
    nothing (B38): each was added without a ``needs`` entry, and nothing
    noticed. A job that should not gate gets an entry in
    :data:`DISCLOSED_NOT_HARD` with a reason, which is visible in review.
    """
    blocking = _ci_success_needs() & _hard_gates()
    undisclosed = _all_jobs() - {CI_SUCCESS_JOB} - blocking - set(DISCLOSED_NOT_HARD)
    assert not undisclosed, (
        f"jobs that run but cannot fail the build and are not disclosed: {sorted(undisclosed)}; "
        f"add each to {CI_SUCCESS_JOB}.needs and its exit-1 block, or to DISCLOSED_NOT_HARD "
        "with a reason"
    )


@pytest.mark.parametrize("job", sorted(DISCLOSED_NOT_HARD), ids=str)
def test_disclosed_exception_is_still_needed(job: str) -> None:
    """Self-expiring: an exemption for a job that is now a hard gate is stale."""
    assert job in _all_jobs(), f"DISCLOSED_NOT_HARD names a job that no longer exists: {job!r}"
    assert job not in (_ci_success_needs() & _hard_gates()), (
        f"{job!r} is now a hard gate; delete its DISCLOSED_NOT_HARD entry"
    )


@pytest.mark.parametrize("job", sorted(DISCLOSED_NOT_HARD), ids=str)
def test_disclosed_exception_states_a_reason(job: str) -> None:
    assert len(DISCLOSED_NOT_HARD[job].strip()) >= 20, f"{job!r}: reason is a placeholder"
