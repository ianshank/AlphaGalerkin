"""E2E: the record-baseline / diff regression gate, as separate processes.

Guards CLAUDE.md's *"PoC baseline harness (WS2)"* Regression Surface row and
``specs/headline_runs.spec.md``. This is the local, hard-asserting twin of CI's
``transfer-baseline-regression`` job, which is soft-gated and therefore cannot
fail a merge.

What this adds over ``tests/scripts/test_run_*.py``: those call ``main(argv)``
in-process. Here each harness runs as a real process, so the exit code a shell
sees, the argument parser, and the on-disk baseline document are all exercised
together -- which is how CI and a human actually use these scripts.

Device: every case forwards ``--device <e2e_device>``, so the same file runs
unchanged on a CUDA host. See ``tests/e2e/conftest.py`` for the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

from tests.e2e.conftest import (
    E2E_BENCHMARK_TIMEOUT_S,
    E2E_TRAINING_TIMEOUT_S,
    NO_CUDA_ENV,
    CLIRunnerType,
)

pytestmark = pytest.mark.e2e

# --------------------------------------------------------------------------- #
# Tolerances                                                                   #
# --------------------------------------------------------------------------- #

#: Tolerance for a re-run against its own freshly recorded baseline, on CPU.
#:
#: MEASURED, not guessed: with the seed pinned by the shipped config, all three
#: harnesses reproduce their stable metrics **exactly** on this CPU container --
#: a re-run diffed at ``--tolerance-pct 0`` reports ``delta_pct=0.0`` for every
#: entry and exits 0. 1.0% therefore carries real headroom while still being
#: tight enough that a genuine change in the numbers fails.
#:
#: This is deliberately far tighter than the ``--tolerance-pct 100000`` used by
#: the in-process round-trip in ``tests/scripts/test_run_transfer_baseline_compare.py``,
#: whose comment attributes it to "CPU-matmul run-to-run drift". That drift does
#: not reproduce here at these budgets; if it reappears on another runner this
#: constant is the single place to widen, and widening it is a finding worth
#: recording rather than a silent edit.
RERUN_TOLERANCE_PCT_CPU: Final[float] = 1.0

#: Tolerance for the same comparison on a non-CPU device. GPU kernels may
#: reassociate reductions, and ``src.seeding.set_global_seeds`` deliberately does
#: not set the cuDNN determinism flags, so bitwise equality is not promised
#: there. Applying the CPU value on CUDA would be asserting a guarantee the code
#: does not make.
RERUN_TOLERANCE_PCT_NON_CPU: Final[float] = 25.0

#: Factor applied to a baseline entry when constructing a guaranteed regression.
#: Halving a ``lower_better`` value means the observed run is twice the baseline,
#: which at ``tolerance_pct = 0`` must be reported.
REGRESSION_TIGHTEN_FACTOR: Final[float] = 0.5

#: Tolerance written into the tightened baseline. Zero, so the regression cannot
#: be absorbed.
REGRESSION_TOLERANCE_PCT: Final[float] = 0.0

#: Metric-name substrings that are never chosen as the tightened metric. A
#: metric whose value is 0.0 does not regress when halved (0.5 * 0 == 0), and
#: the registry floors a zero denominator -- so picking one would produce a test
#: that passes for the wrong reason. Win fractions and seed spreads are exactly
#: those at a small seed count.
UNTIGHTENABLE_SUBSTRINGS: Final[tuple[str, ...]] = ("_win_fraction", "_seed_std")


@dataclass(frozen=True)
class HarnessCase:
    """One harness script and the argv that drives it cheaply.

    Attributes:
        module: Dotted module path run via ``python -m``.
        config: Shipped scenario YAML, used as shipped.
        scenario_name: Key the harness records metrics under.
        overrides: Budget-shrinking flags. Every one is an existing flag on that
            script; values respect the schema's own floors (notably
            ``n_train_samples >= 64``).
        timeout: Tier constant sized to the measured runtime.
        label: pytest id.

    """

    module: str
    config: str
    scenario_name: str
    overrides: tuple[str, ...]
    timeout: int
    label: str
    extra_env: dict[str, str] = field(default_factory=dict)


#: The three harnesses that expose the gate. Measured runtimes on a 4-CPU
#: container: lshape ~6 s, stochastic ~13 s, transfer ~27 s.
#:
#: ``run_lshape_amr`` had NO baseline flags until this change -- it was the one
#: committed headline that could not be regression-gated from its own CLI. It is
#: included here rather than left as a documented gap, which is what turns
#: "should have the flags" into a test that fails without them.
HARNESS_CASES: Final[tuple[HarnessCase, ...]] = (
    HarnessCase(
        module="scripts.run_lshape_amr",
        config="config/scenarios/lshape_amr_compare_cpu.yaml",
        scenario_name="lshape_amr_compare",
        overrides=("--max-dof", "120", "--n-simulations", "2", "--seed", "1"),
        timeout=E2E_BENCHMARK_TIMEOUT_S,
        label="lshape_amr",
    ),
    HarnessCase(
        module="scripts.run_stochastic_galerkin_compare",
        config="config/scenarios/stochastic_galerkin_compare_ci.yaml",
        scenario_name="stochastic_galerkin_compare",
        overrides=("--n-epochs", "1", "--n-seeds", "1", "--grid-n", "8"),
        timeout=E2E_TRAINING_TIMEOUT_S,
        label="stochastic_galerkin",
    ),
    HarnessCase(
        module="scripts.run_transfer_baseline_compare",
        config="config/scenarios/transfer_baseline_compare_ci.yaml",
        scenario_name="transfer_baseline_compare",
        # n_train_samples has a schema floor of 64 (ge=64); 8 -- the value this
        # plan originally proposed -- is a ValidationError before any run.
        overrides=(
            "--n-epochs",
            "1",
            "--n-seeds",
            "1",
            "--n-train-samples",
            "64",
            "--target-resolution",
            "13",
        ),
        timeout=E2E_TRAINING_TIMEOUT_S,
        label="transfer_baseline",
    ),
)

#: The cheapest case, used for the assertions that cost a whole extra run for
#: little extra signal. Chosen by measured runtime, not by hand, so it follows
#: the budgets rather than going stale next to them.
CHEAPEST_CASE: Final[HarnessCase] = min(HARNESS_CASES, key=lambda case: case.timeout)


def _argv(case: HarnessCase, *, output_dir: Path, device: str, extra: tuple[str, ...]) -> list[str]:
    """Build the full argv for *case*.

    Args:
        case: The harness to run.
        output_dir: Artifact destination (always under ``tmp_path``).
        device: Concrete device string from the ``e2e_device`` fixture.
        extra: Flags appended after the shared ones.

    Returns:
        The argument list, without the ``python -m <module>`` prefix.

    """
    return [
        "--config",
        case.config,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        *case.overrides,
        *extra,
    ]


def _tightened(document: dict[str, object]) -> tuple[dict[str, object], str]:
    """Return *document* with exactly one entry made impossible to satisfy.

    Halves one strictly-positive ``lower_better`` entry and zeroes its tolerance,
    so the observed run -- twice the recorded value -- must be reported as a
    regression.

    Args:
        document: A loaded baseline document.

    Returns:
        ``(mutated_document, metric_name)``.

    Raises:
        AssertionError: No entry is eligible, which would make the test vacuous.

    """
    entries = document["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        name = str(entry["metric_name"])
        value = float(entry["value"])
        if value > 0.0 and not any(sub in name for sub in UNTIGHTENABLE_SUBSTRINGS):
            entry["value"] = value * REGRESSION_TIGHTEN_FACTOR
            entry["tolerance_pct"] = REGRESSION_TOLERANCE_PCT
            return document, name
    raise AssertionError(
        "no strictly-positive, tightenable entry in the baseline -- the "
        "regression test would be vacuous"
    )


@pytest.fixture(scope="module", params=HARNESS_CASES, ids=lambda case: case.label)
def harness_case(request: pytest.FixtureRequest) -> HarnessCase:
    """Parametrise the module over the three harness scripts.

    Module-scoped so the recorded-baseline fixture below runs each harness once
    rather than once per test.

    Returns:
        The case under test.

    """
    case: HarnessCase = request.param
    return case


@pytest.fixture(scope="module")
def recorded(
    harness_case: HarnessCase,
    cli_runner: CLIRunnerType,
    e2e_device: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[HarnessCase, Path, Path, int]:
    """Run one harness once with ``--record-baseline``.

    Returns:
        ``(case, baseline_path, output_dir, returncode)``.

    """
    output_dir = tmp_path_factory.mktemp(f"record_{harness_case.label}")
    baseline_path = output_dir / "baseline.json"
    result = cli_runner(
        harness_case.module,
        _argv(
            harness_case,
            output_dir=output_dir,
            device=e2e_device,
            extra=("--record-baseline", str(baseline_path)),
        ),
        harness_case.timeout,
        None,
    )
    return harness_case, baseline_path, output_dir, result.returncode


def test_record_baseline_exits_zero(recorded: tuple[HarnessCase, Path, Path, int]) -> None:
    """Recording exits 0 regardless of the run's own acceptance verdict.

    The harnesses ignore the threshold verdict under ``--record-baseline`` (you
    are capturing what the numbers *are*, not asserting they are good), so 0 is
    the contract even for the L-shape case whose headline ratio fails its gate
    at full budget.
    """
    case, baseline_path, _output_dir, returncode = recorded
    assert returncode == 0, f"{case.module} --record-baseline did not exit 0"
    assert baseline_path.is_file(), f"{case.module} recorded no baseline file"


def test_recorded_document_is_a_valid_baseline(
    recorded: tuple[HarnessCase, Path, Path, int],
) -> None:
    """The recorded document loads through the versioned registry API.

    Not merely "the JSON parses": it must survive ``ScenarioBaselineRegistry.load``,
    which applies ``migrate_baseline_document``. A harness writing a shape the
    loader rejects would otherwise only be discovered by whoever next tried to
    gate against it.
    """
    from src.poc.baselines import ScenarioBaselineRegistry

    case, baseline_path, _output_dir, _rc = recorded
    registry = ScenarioBaselineRegistry.load(baseline_path)

    document = json.loads(baseline_path.read_text(encoding="utf-8"))
    entries = document["entries"]
    assert entries, f"{case.module} recorded an empty baseline -- nothing to gate on"
    # Every recorded entry belongs to this harness's scenario. The filter that
    # picks stable metrics is per-harness policy, so the assertion is
    # "everything recorded is this scenario's", not a fixed metric count.
    assert {entry["scenario_name"] for entry in entries} == {case.scenario_name}
    assert registry is not None


def test_rerun_against_own_baseline_is_within_measured_drift(
    recorded: tuple[HarnessCase, Path, Path, int],
    harness_case: HarnessCase,
    cli_runner: CLIRunnerType,
    e2e_device: str,
    e2e_device_type: str,
    tmp_path: Path,
) -> None:
    """A second run reproduces the first within the measured tolerance.

    ``--baseline`` re-executes the harness, so this is a genuine reproducibility
    check rather than a comparison of a result with itself. On CPU the measured
    drift is exactly zero (see ``RERUN_TOLERANCE_PCT_CPU``); on CUDA the
    tolerance widens because the repo sets no cuDNN determinism flags.
    """
    _case, baseline_path, _output_dir, _rc = recorded
    tolerance = RERUN_TOLERANCE_PCT_CPU if e2e_device_type == "cpu" else RERUN_TOLERANCE_PCT_NON_CPU
    retuned = tmp_path / "retuned.json"
    document = json.loads(baseline_path.read_text(encoding="utf-8"))
    for entry in document["entries"]:
        entry["tolerance_pct"] = tolerance
    retuned.write_text(json.dumps(document), encoding="utf-8")

    result = cli_runner(
        harness_case.module,
        _argv(
            harness_case,
            output_dir=tmp_path / "out",
            device=e2e_device,
            extra=("--baseline", str(retuned)),
        ),
        harness_case.timeout,
        None,
    )
    assert result.returncode == 0, (
        f"{harness_case.module} did not reproduce within {tolerance}%: {result.output[-2000:]}"
    )


def test_tightened_baseline_exits_one_and_names_the_metric(
    recorded: tuple[HarnessCase, Path, Path, int],
    harness_case: HarnessCase,
    cli_runner: CLIRunnerType,
    e2e_device: str,
    tmp_path: Path,
) -> None:
    """A baseline the run cannot meet exits 1 and reports the offending metric.

    The mirror of the test above: it proves the gate can go red at all. Without
    it, a harness that ignored ``--baseline`` entirely would pass every other
    assertion here.
    """
    _case, baseline_path, _output_dir, _rc = recorded
    document = json.loads(baseline_path.read_text(encoding="utf-8"))
    document, tightened_metric = _tightened(document)
    tightened_path = tmp_path / "tightened.json"
    tightened_path.write_text(json.dumps(document), encoding="utf-8")

    result = cli_runner(
        harness_case.module,
        _argv(
            harness_case,
            output_dir=tmp_path / "out",
            device=e2e_device,
            extra=("--baseline", str(tightened_path)),
        ),
        harness_case.timeout,
        None,
    )
    assert result.returncode == 1, (
        f"{harness_case.module} did not report a regression it must detect"
    )
    assert tightened_metric in result.output


def test_missing_baseline_file_exits_nonzero(
    cli_runner: CLIRunnerType,
    e2e_device: str,
    tmp_path: Path,
) -> None:
    """A ``--baseline`` path that does not exist fails loudly, naming the path.

    Run on the cheapest harness only. The baseline is loaded *after* the
    scenario executes, so this assertion costs an entire benchmark run; paying
    that three times would add runtime without adding signal, since the load
    path is shared by all three via ``handle_baseline_flags``.
    """
    absent = tmp_path / "does_not_exist.json"
    result = cli_runner(
        CHEAPEST_CASE.module,
        _argv(
            CHEAPEST_CASE,
            output_dir=tmp_path / "out",
            device=e2e_device,
            extra=("--baseline", str(absent)),
        ),
        CHEAPEST_CASE.timeout,
        None,
    )
    assert result.returncode != 0
    assert absent.name in result.output


def test_cuda_request_fails_loud_when_no_cuda_is_visible(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """``--device cuda`` with no visible CUDA device fails; it never falls back.

    Runs identically on a CPU-only runner and on a GPU host: ``CUDA_VISIBLE_DEVICES=""``
    hides the device from the child either way. On a GPU host it additionally
    proves the child honoured the flag rather than quietly using the accelerator.

    Guards the fail-loud half of ``src/poc/device.py``'s stated policy, which is
    what the whole E2E device contract rests on.
    """
    result = cli_runner(
        CHEAPEST_CASE.module,
        _argv(CHEAPEST_CASE, output_dir=tmp_path / "out", device="cuda", extra=()),
        CHEAPEST_CASE.timeout,
        NO_CUDA_ENV,
    )
    assert result.returncode == 1
    assert "CUDA is not available" in result.output
