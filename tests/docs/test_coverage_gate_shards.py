"""Guards the sharding of the ``coverage-gates`` job (plan R-12b).

The 44 per-module coverage gate steps used to run serially in one job and were
the long pole of every pull request. They now run under a ``strategy.matrix``
with one ``shard`` axis, and each gate step carries ``matrix.shard == N`` in
its ``if:``. The steps deliberately stay in ``ci.yml`` rather than moving to a
script: ``tests/docs/test_charter_alignment.py`` and
``tests/docs/test_coverage_gate_integrity.py`` read the ``run:`` bodies to
prove each documented gate is enforced, and a script would take those bodies
out of their reach.

Sharding by step condition has three ways to be silently wrong, and this
module closes each:

* a gate step with **no** shard condition runs on every shard -- four times
  the work, and the "balance" the matrix exists for is fiction;
* a gate step naming a shard **outside** the matrix never runs on any shard
  -- the gate is still in the file, still documented, and enforces nothing,
  which is this repo's recurring invisibility defect in a new coat;
* a shard with **no** steps is a runner that checks out, installs torch, and
  exits green having measured nothing.

Mutation kills (each planted, run, reverted):

* drop ``matrix.shard == N &&`` from one gate step ->
  ``test_every_gate_step_names_exactly_one_shard``
* ``matrix.shard == 5`` on one gate step ->
  ``test_every_gate_step_names_an_existing_shard``
* add ``5`` to the matrix with no step assigned ->
  ``test_every_shard_has_at_least_one_gate_step``
* ``always() || matrix.shard == 1`` on one gate step (Copilot review, PR #151:
  the clause is present, one shard exists, and the step runs everywhere) ->
  ``test_shard_clause_restricts_rather_than_decorates``
* rename one gate step away from the ``Per-module coverage gate`` prefix
  (Copilot review: a name-derived gate set silently shrinks) ->
  ``test_named_gate_steps_and_coverage_commands_agree`` -- and because the
  gate set is now derived from the coverage *command*, the renamed step is
  still held to exactly one shard by every per-step test above
"""

from __future__ import annotations

import re
from typing import Any, Final

import pytest

from tests.support.workflows import CI_WORKFLOW, iter_commands, load_workflow

#: The sharded job.
COVERAGE_GATES_JOB: Final[str] = "coverage-gates"

#: The matrix axis name.
SHARD_AXIS: Final[str] = "shard"

#: Prefix every per-module gate step's ``name:`` carries (a naming convention,
#: cross-checked against the command-derived gate set; not the source of truth).
GATE_STEP_PREFIX: Final[str] = "Per-module coverage gate"

#: Tokens whose presence in a command makes a step a coverage gate: the
#: pytest-cov threshold flag and the native runner's report threshold.
COVERAGE_GATE_TOKENS: Final[tuple[str, ...]] = ("--cov-fail-under", "--fail-under")

#: The condition fragment that pins a step to a shard.
_SHARD_CONDITION: Final[re.Pattern[str]] = re.compile(rf"matrix\.{SHARD_AXIS}\s*==\s*(?P<n>\d+)")

#: Vacuity floor: the job carried 44 gate steps when sharded. A parser that
#: found only a handful would make every per-step assertion below cheap.
MIN_GATE_STEPS: Final[int] = 30

#: Balance tolerance: no shard may carry more than this multiple of the mean
#: step count. Step count is a proxy for wall-clock (the assignment was
#: balanced by test-function count), so the bound is loose on purpose -- it
#: catches "everything landed on shard 1", not a 10% drift.
MAX_SHARD_LOAD_RATIO: Final[float] = 2.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _job() -> dict[str, Any]:
    jobs = load_workflow(CI_WORKFLOW).get("jobs")
    assert isinstance(jobs, dict)
    job = jobs.get(COVERAGE_GATES_JOB)
    assert isinstance(job, dict), f"{CI_WORKFLOW.name} has no job {COVERAGE_GATES_JOB!r}"
    return job


def _matrix_shards() -> list[int]:
    strategy = _job().get("strategy")
    assert isinstance(strategy, dict), f"{COVERAGE_GATES_JOB} has no strategy: block"
    matrix = strategy.get("matrix")
    assert isinstance(matrix, dict), f"{COVERAGE_GATES_JOB}.strategy has no matrix:"
    shards = matrix.get(SHARD_AXIS)
    assert isinstance(shards, list), f"{COVERAGE_GATES_JOB}.strategy.matrix has no {SHARD_AXIS}:"
    return [int(s) for s in shards]


def _steps() -> list[dict[str, Any]]:
    steps = _job().get("steps")
    assert isinstance(steps, list) and steps
    return [s for s in steps if isinstance(s, dict)]


def _runs_a_coverage_gate(step: dict[str, Any]) -> bool:
    """A step is a gate iff one of its *commands* carries a coverage threshold.

    Derived from the ``run:`` body, not the display name: a renamed step, or a
    coverage invocation added under some other name, must still be held to
    exactly one shard. The name prefix is checked separately, as a
    consistency cross-check, never as the source of truth.
    """
    script = step.get("run")
    if not isinstance(script, str):
        return False
    return any(
        any(token in command for token in COVERAGE_GATE_TOKENS) for command in iter_commands(script)
    )


def _is_named_gate_step(step: dict[str, Any]) -> bool:
    return str(step.get("name", "")).startswith(GATE_STEP_PREFIX)


def _gate_steps() -> list[dict[str, Any]]:
    return [s for s in _steps() if _runs_a_coverage_gate(s)]


def _setup_steps() -> list[dict[str, Any]]:
    return [s for s in _steps() if not _runs_a_coverage_gate(s)]


def shard_clause_is_required(condition: str | None) -> bool:
    """True iff ``matrix.shard == N`` is a top-level ``&&`` conjunct of ``condition``.

    Presence is not restriction: ``always() || matrix.shard == 1`` names one
    existing shard and still runs on every shard. The clause must be joined to
    the rest of the expression by ``&&`` only, with no ``||`` anywhere.
    """
    if not condition or "||" in condition:
        return False
    conjuncts = [part.strip() for part in condition.split("&&")]
    return any(_SHARD_CONDITION.fullmatch(part) for part in conjuncts)


def shards_named(condition: str | None) -> list[int]:
    """Every ``matrix.shard == N`` in a step's ``if:``, in order.

    Args:
        condition: The raw ``if:`` expression, or ``None`` when absent.

    Returns:
        The shard numbers the condition pins the step to (empty if none).

    """
    if not condition:
        return []
    return [int(m.group("n")) for m in _SHARD_CONDITION.finditer(condition)]


def _step_id(step: dict[str, Any]) -> str:
    return str(step.get("name", "<unnamed>"))


# --------------------------------------------------------------------------
# Vacuity and parser
# --------------------------------------------------------------------------


def test_the_matrix_is_non_empty() -> None:
    shards = _matrix_shards()
    assert shards, f"{COVERAGE_GATES_JOB}.strategy.matrix.{SHARD_AXIS} is empty"
    assert len(shards) == len(set(shards)), f"duplicate shard ids: {shards}"


def test_the_job_carries_a_plausible_number_of_gate_steps() -> None:
    n = len(_gate_steps())
    assert n >= MIN_GATE_STEPS, f"only {n} steps named `{GATE_STEP_PREFIX}…` parsed"


class TestShardsNamedParser:
    def test_absent_condition(self) -> None:
        assert shards_named(None) == []
        assert shards_named("") == []

    def test_single_shard(self) -> None:
        assert shards_named("always() && matrix.shard == 3 && hashFiles('x') != ''") == [3]

    def test_two_shards_are_both_reported(self) -> None:
        assert shards_named("matrix.shard == 1 || matrix.shard == 2") == [1, 2]

    def test_other_matrix_axes_are_ignored(self) -> None:
        assert shards_named("matrix.python-version == '3.10'") == []


# --------------------------------------------------------------------------
# Every gate step pins exactly one existing shard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("step", _gate_steps(), ids=_step_id)
def test_every_gate_step_names_exactly_one_shard(step: dict[str, Any]) -> None:
    named = shards_named(step.get("if"))
    assert len(named) == 1, (
        f"{_step_id(step)!r}: if: names {len(named)} shard(s) ({named}); a step with none runs "
        "on every shard, one with several runs on each of them"
    )


@pytest.mark.parametrize("step", _gate_steps(), ids=_step_id)
def test_every_gate_step_names_an_existing_shard(step: dict[str, Any]) -> None:
    named = shards_named(step.get("if"))
    matrix = _matrix_shards()
    assert all(n in matrix for n in named), (
        f"{_step_id(step)!r}: if: names shard {named} but the matrix is {matrix}; the step "
        "runs on no shard at all and the gate it carries enforces nothing"
    )


@pytest.mark.parametrize("step", _gate_steps(), ids=_step_id)
def test_shard_clause_restricts_rather_than_decorates(step: dict[str, Any]) -> None:
    """``always() || matrix.shard == 1`` names one existing shard and runs on all four.

    Counting the clause is not enough; it must be a top-level ``&&`` conjunct
    with no ``||`` in the expression, or the shard assignment is decoration.
    """
    condition = step.get("if")
    assert shard_clause_is_required(condition), (
        f"{_step_id(step)!r}: if: {condition!r} does not *require* its shard clause "
        "(an `||` or a missing `&&` conjunct lets the step run on every shard)"
    )


def test_named_gate_steps_and_coverage_commands_agree() -> None:
    """The name prefix is a convention; the coverage command is the fact.

    A step renamed away from the prefix, or a coverage threshold added under
    another name, would otherwise drift out of the human-readable set while
    the per-step assertions above (driven by the command) still cover it --
    or, before this cross-check existed, the reverse.
    """
    by_name = {_step_id(s) for s in _steps() if _is_named_gate_step(s)}
    by_command = {_step_id(s) for s in _gate_steps()}
    assert by_name == by_command, (
        f"named-but-not-gating: {sorted(by_name - by_command)}; "
        f"gating-but-not-named: {sorted(by_command - by_name)}"
    )


class TestShardClauseIsRequired:
    def test_conjunct_form_passes(self) -> None:
        assert shard_clause_is_required("always() && matrix.shard == 2 && hashFiles('x') != ''")

    def test_disjunction_fails(self) -> None:
        assert not shard_clause_is_required("always() || matrix.shard == 1")

    def test_disjunction_anywhere_fails(self) -> None:
        assert not shard_clause_is_required("matrix.shard == 1 && (a || b)")

    def test_absent_fails(self) -> None:
        assert not shard_clause_is_required("always() && hashFiles('x') != ''")
        assert not shard_clause_is_required(None)


@pytest.mark.parametrize("step", _gate_steps(), ids=_step_id)
def test_every_gate_step_still_runs_after_an_earlier_failure(step: dict[str, Any]) -> None:
    """``always()`` keeps the remaining gates reporting once one has failed.

    Sharding must not lose this: a shard whose first gate fails should still
    report every other gate on that shard, so a red run lists all failures.
    """
    assert "always()" in str(step.get("if", "")), f"{_step_id(step)!r}: if: lost always()"


# --------------------------------------------------------------------------
# Every shard does work; setup runs on all of them
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shard", _matrix_shards(), ids=lambda s: f"shard-{s}")
def test_every_shard_has_at_least_one_gate_step(shard: int) -> None:
    assigned = [s for s in _gate_steps() if shards_named(s.get("if")) == [shard]]
    assert assigned, (
        f"matrix shard {shard} has no gate step: a runner that installs torch and exits "
        "green having measured nothing"
    )


def test_shard_load_is_not_grossly_unbalanced() -> None:
    shards = _matrix_shards()
    counts = {
        shard: sum(1 for s in _gate_steps() if shards_named(s.get("if")) == [shard])
        for shard in shards
    }
    mean = sum(counts.values()) / len(counts)
    assert max(counts.values()) <= MAX_SHARD_LOAD_RATIO * mean, (
        f"shard load {counts} exceeds {MAX_SHARD_LOAD_RATIO}x the mean ({mean:.1f}); "
        "rebalance the `matrix.shard == N` assignments"
    )


@pytest.mark.parametrize("step", _setup_steps(), ids=_step_id)
def test_setup_steps_carry_no_shard_condition(step: dict[str, Any]) -> None:
    """Checkout / setup-python / install must run on every shard."""
    assert shards_named(step.get("if")) == [], (
        f"{_step_id(step)!r} is pinned to a shard; every other shard would run its gates "
        "without a checkout or an install"
    )


def test_fail_fast_is_off() -> None:
    """One red shard must not cancel the others: a red run should list every failure."""
    strategy = _job().get("strategy")
    assert isinstance(strategy, dict)
    assert strategy.get("fail-fast") is False, (
        f"{COVERAGE_GATES_JOB}.strategy.fail-fast must be false"
    )
