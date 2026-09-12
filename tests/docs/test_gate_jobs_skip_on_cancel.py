"""Every aggregate gate job skips on cancellation instead of failing.

A gate job -- one that fans in over ``needs:`` and ``exit 1``s on a bad result
-- must not run under a bare ``if: always()``. When the workflow's concurrency
group supersedes a run, the needed jobs are ``cancelled``; under ``always()``
the gate still executes, reads ``cancelled``, and staples a red X to a SHA
nobody was waiting on. ``ci.yml``'s ``ci-success`` was fixed to ``!cancelled()``
on 2026-09-08 after this was mis-triaged as a CI failure twice in one day
(B38); ``regression-surface.yml`` and ``sbir-demo-smoke.yml`` carried the
same ``always()`` and produced the same red X on 2026-09-11 (run
34659426510). This guard closes the class across every workflow.

Mutation kills (each planted, run, reverted):

* ``if: ${{ !cancelled() }}`` -> ``if: always()`` on ``surface-success`` ->
  ``test_gate_job_does_not_run_on_cancellation[regression-surface.yml::surface-success]``
* delete the ``if:`` from ``smoke-success`` entirely (a gate with no status
  function is skipped when a need fails, so it can never report) ->
  ``test_gate_job_declares_a_status_function[sbir-demo-smoke.yml::smoke-success]``
"""

from __future__ import annotations

from typing import Any, Final

import pytest

from tests.support.workflows import (
    WORKFLOW_DIR,
    body_exits_nonzero,
    job_needs,
    load_workflow,
)

#: Minimum fan-in for a job to count as an aggregate gate. A single-need job
#: is a stage, not a gate, and may legitimately depend on a plain success.
MIN_GATE_FAN_IN: Final[int] = 2

#: The status-function form that runs the gate on failure and skips it on
#: cancellation. GitHub treats any status function in ``if:`` as opting out
#: of the implicit ``success()``.
CANCEL_SAFE_FORM: Final[str] = "!cancelled()"

#: A gate written with this form executes on a superseded run.
CANCEL_UNSAFE_FORM: Final[str] = "always()"

#: Vacuity floor: three workflows carry a gate today.
MIN_GATE_JOBS: Final[int] = 3


def _gate_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    """Every ``(workflow, job, job_doc)`` that fans in and exits non-zero."""
    found: list[tuple[str, str, dict[str, Any]]] = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        document = load_workflow(path)
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            continue
        for name, job in jobs.items():
            if not isinstance(job, dict) or len(job_needs(document, str(name))) < MIN_GATE_FAN_IN:
                continue
            scripts = [
                s.get("run") for s in job.get("steps", []) if isinstance(s, dict) and s.get("run")
            ]
            if any(body_exits_nonzero(str(script)) for script in scripts):
                found.append((path.name, str(name), job))
    return found


_GATES: Final[list[tuple[str, str, dict[str, Any]]]] = _gate_jobs()
_IDS: Final[list[str]] = [f"{wf}::{job}" for wf, job, _ in _GATES]


def test_the_scan_finds_the_known_gates() -> None:
    """Vacuity: a parser that found no gates would pass every test below."""
    assert len(_GATES) >= MIN_GATE_JOBS, f"only {len(_GATES)} gate jobs found: {_IDS}"
    assert "ci.yml::ci-success" in _IDS


@pytest.mark.parametrize(("workflow", "job", "doc"), _GATES, ids=_IDS)
def test_gate_job_declares_a_status_function(workflow: str, job: str, doc: dict[str, Any]) -> None:
    """Without a status function the gate is skipped whenever a need fails."""
    expr = str(doc.get("if", ""))
    assert CANCEL_SAFE_FORM in expr or CANCEL_UNSAFE_FORM in expr, (
        f"{workflow}::{job}: if: {expr!r} has no status function; GitHub applies the implicit "
        "success(), so the gate is skipped -- and reports nothing -- exactly when a need fails"
    )


@pytest.mark.parametrize(("workflow", "job", "doc"), _GATES, ids=_IDS)
def test_gate_job_does_not_run_on_cancellation(
    workflow: str, job: str, doc: dict[str, Any]
) -> None:
    expr = str(doc.get("if", ""))
    assert CANCEL_UNSAFE_FORM not in expr, (
        f"{workflow}::{job}: if: {expr!r} uses {CANCEL_UNSAFE_FORM}; a run superseded by the "
        f"concurrency group reports a red X for cancelled needs. Use {CANCEL_SAFE_FORM}."
    )
    assert CANCEL_SAFE_FORM in expr, (
        f"{workflow}::{job}: if: {expr!r} must carry {CANCEL_SAFE_FORM}"
    )
