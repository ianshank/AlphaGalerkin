"""Guard B23: the nightly cron must not re-run the PR matrix.

Defect class: a ``schedule:`` trigger that exists so ``test-slow`` can run
will, by default, fire every other job too. On an unchanged SHA that is
~55 minutes of duplicate work (hygiene audit B23). The contract is:

* ``test-slow`` runs on ``schedule`` even when ``test-fast`` is skipped.
* every other job in ``ci.yml`` skips ``schedule``.

Hermetic: parses the workflow as YAML. Runs nothing.

Mutations (named tests):

1. Drop ``if: github.event_name != 'schedule'`` from ``lint`` —
   ``test_every_non_slow_job_skips_the_nightly_schedule`` fails.
2. Remove ``github.event_name == 'schedule'`` from ``test-slow`` —
   ``test_test_slow_still_runs_on_schedule`` fails.
3. Empty the jobs map — ``test_the_workflow_has_jobs`` fails first.
"""

from __future__ import annotations

from typing import Any, Final

from tests.support.workflows import CI_WORKFLOW, load_workflow

_NIGHTLY_SKIP: Final[str] = "github.event_name != 'schedule'"
_PULL_REQUEST_ONLY: Final[str] = "github.event_name == 'pull_request'"
_SCHEDULE_RUN: Final[str] = "github.event_name == 'schedule'"
_SLOW_JOB: Final[str] = "test-slow"


def _jobs() -> dict[str, Any]:
    document = load_workflow(CI_WORKFLOW)
    jobs = document.get("jobs")
    assert isinstance(jobs, dict), "ci.yml has no jobs mapping"
    return jobs


def _job_if(job: Any) -> str:
    if not isinstance(job, dict):
        return ""
    raw = job.get("if", "")
    if isinstance(raw, bool):
        return str(raw).lower()
    return str(raw)


def test_the_workflow_has_jobs() -> None:
    """Vacuity: a missing jobs map would make every assertion below iterate nothing."""
    jobs = _jobs()
    assert _SLOW_JOB in jobs
    assert len(jobs) > 2


def test_every_non_slow_job_skips_the_nightly_schedule() -> None:
    """B23: cron exists so test-slow runs; the rest of the matrix must not.

    ``focus`` is already pull_request-only (schedule never fires it). Every
    other non-slow job must name the schedule skip explicitly.
    """
    missing: list[str] = []
    for name, job in _jobs().items():
        if name == _SLOW_JOB:
            continue
        expr = _job_if(job)
        skips = _NIGHTLY_SKIP in expr or _PULL_REQUEST_ONLY in expr
        if not skips:
            missing.append(name)
    assert not missing, (
        "these jobs would re-run on the nightly cron; add "
        f"{_NIGHTLY_SKIP!r} (or keep them pull_request-only): {missing}"
    )


def test_test_slow_still_runs_on_schedule() -> None:
    """A skip copied onto test-slow would retire the reason the cron exists.

    ``!cancelled()`` is load-bearing: on schedule, ``test-fast`` is skipped,
    and GitHub skips dependents unless the ``if:`` continues past a skipped
    needed job.
    """
    expr = _job_if(_jobs()[_SLOW_JOB])
    assert _SCHEDULE_RUN in expr, (
        f"{_SLOW_JOB} if: does not mention schedule, so the nightly runs nothing: {expr!r}"
    )
    assert "!cancelled()" in expr, (
        f"{_SLOW_JOB} must use !cancelled() so a skipped test-fast does not "
        f"skip the nightly: {expr!r}"
    )
