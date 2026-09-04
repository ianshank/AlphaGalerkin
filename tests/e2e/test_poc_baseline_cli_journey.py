"""E2E: ``poc.cli run`` -> ``record-baseline`` -> ``diff`` as three processes.

Guards CLAUDE.md's *"PoC baseline harness (WS2)"* Regression Surface row.

The distinction from ``test_baseline_gate_journey.py``: that file drives the
per-harness scripts, where ``--baseline`` *re-executes* the run, so a comparison
there is a reproducibility check. Here the chain records and diffs the **same
persisted run id**, so the diff is exact by construction on any device -- no
tolerance is involved and none is asserted. Both shapes are worth having; only
this one can prove the record/diff arithmetic itself rather than the harness's
determinism.

The scenario is pinned onto ``e2e_device`` through ``pin_scenario_yaml`` because
``poc.cli run`` exposes no ``--device`` flag; see ``tests/e2e/conftest.py`` for
why adding one was rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

import pytest

from tests.e2e.conftest import (
    E2E_BENCHMARK_TIMEOUT_S,
    E2E_TRIVIAL_TIMEOUT_S,
    CLIRunnerType,
    ScenarioYamlPinnerType,
)

pytestmark = pytest.mark.e2e

#: Shipped CPU config driving the chain. Chosen because it is a real scenario
#: with real metrics that finishes in ~8 s, and because it declares a ``device``
#: key -- ``pin_scenario_yaml`` refuses to pin a key the config does not have,
#: so a config without one could not be steered onto ``e2e_device``.
SCENARIO_CONFIG: Final[str] = "config/scenarios/noyron_basis_cpu.yaml"

#: The scenario's registered name, as it appears in the persisted result.
SCENARIO_NAME: Final[str] = "noyron_basis"

#: Tolerance recorded by ``record-baseline``. Any positive value works for an
#: exact self-diff; a small one is chosen so the clean-diff assertion is not
#: passing merely because the tolerance is enormous.
RECORD_TOLERANCE_PCT: Final[str] = "10"

#: Factor and tolerance used to force a regression, mirroring
#: ``test_baseline_gate_journey.py``. Halving a lower-better value means the
#: observed run is twice the baseline, which at zero tolerance must be reported.
REGRESSION_TIGHTEN_FACTOR: Final[float] = 0.5
REGRESSION_TOLERANCE_PCT: Final[float] = 0.0

#: A run id that cannot exist.
ABSENT_RUN_ID: Final[str] = "no_such_run_id"


def _run_id_of(output_dir: Path) -> str:
    """Return the single run id directory name under ``<output_dir>/results``.

    Args:
        output_dir: The ``--output-dir`` the run was given.

    Returns:
        The run id.

    Raises:
        AssertionError: If zero or several run directories exist -- either would
            make the rest of the chain ambiguous.

    """
    results = output_dir / "results"
    run_dirs = sorted(path for path in results.iterdir() if path.is_dir())
    assert len(run_dirs) == 1, f"expected exactly one run under {results}, found {run_dirs}"
    return run_dirs[0].name


def _result_payload(output_dir: Path, run_id: str) -> dict[str, Any]:
    """Load the single persisted ScenarioResult JSON for *run_id*.

    Args:
        output_dir: The run's output directory.
        run_id: Run id returned by :func:`_run_id_of`.

    Returns:
        The parsed result document.

    """
    payloads = sorted((output_dir / "results" / run_id).glob("*.json"))
    assert len(payloads) == 1, f"expected one result JSON, found {payloads}"
    parsed: dict[str, Any] = json.loads(payloads[0].read_text(encoding="utf-8"))
    return parsed


@pytest.fixture(scope="module")
def completed_run(
    cli_runner: CLIRunnerType,
    e2e_device: str,
    pin_scenario_yaml: ScenarioYamlPinnerType,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str, int]:
    """Run the scenario once through ``poc.cli run``.

    Module-scoped: every test below reads the same run, so re-running it per
    test would triple the cost for no added signal.

    Returns:
        ``(output_dir, run_id, returncode)``.

    """
    output_dir = tmp_path_factory.mktemp("poc_chain")
    config = pin_scenario_yaml(SCENARIO_CONFIG, device=e2e_device)
    result = cli_runner(
        "src.poc.cli",
        ["run", "--config", str(config), "--output-dir", str(output_dir)],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    if result.returncode != 0:
        pytest.fail(f"poc.cli run did not exit 0:\n{result.output[-3000:]}")
    return output_dir, _run_id_of(output_dir), result.returncode


def test_run_persists_a_result_that_round_trips(completed_run: tuple[Path, str, int]) -> None:
    """The persisted JSON re-validates as a ``ScenarioResult`` in a fresh process.

    ``ResultCollector`` writes ``model_dump(mode="json")`` with ``default=str``,
    which is exactly the kind of serialisation that can quietly stop round
    tripping. Asserting "the file parses as JSON" would not catch that; asserting
    the model accepts it back does.
    """
    from src.poc.config import ScenarioResult

    output_dir, run_id, returncode = completed_run
    assert returncode == 0
    payload = _result_payload(output_dir, run_id)
    assert payload["scenario_name"] == SCENARIO_NAME
    assert payload["status"] == "passed"
    assert payload["passed"] is True
    assert ScenarioResult.model_validate(payload) is not None


def test_result_records_the_device_the_run_actually_used(
    completed_run: tuple[Path, str, int],
    e2e_device: str,
) -> None:
    """``ScenarioResult.device`` reports the execution device, not host availability.

    Before ``BaseScenario.execution_device_label`` this field was filled with
    ``"cuda" if torch.cuda.is_available() else "cpu"``, so a ``device: cpu``
    config on a CUDA host persisted ``"cuda"`` -- the inverse of the field's
    documented meaning, in the one artifact a reader would consult to find out
    where a run executed.

    On a CPU-only runner this passes both before and after that fix; it is the
    CUDA host, or a run pinned with ``E2E_DEVICE=cpu`` on one, that discriminates.
    That asymmetry is why the assertion compares against the *pinned* device
    rather than against CUDA availability.
    """
    output_dir, run_id, _rc = completed_run
    payload = _result_payload(output_dir, run_id)
    assert payload["device"] == e2e_device


def test_record_then_diff_is_clean_on_the_same_run(
    completed_run: tuple[Path, str, int],
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """Recording a run and diffing that same run reports no regression.

    The exact self-diff: no re-execution, so this is device-independent and
    tolerance-independent. A non-zero exit here means the record/diff arithmetic
    itself is wrong, which no reproducibility-flavoured test could isolate.
    """
    output_dir, run_id, _rc = completed_run
    baseline = tmp_path / "baseline.json"

    record = cli_runner(
        "src.poc.cli",
        [
            "record-baseline",
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--out",
            str(baseline),
            "--tolerance-pct",
            RECORD_TOLERANCE_PCT,
        ],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert record.returncode == 0, record.output
    document = json.loads(baseline.read_text(encoding="utf-8"))
    assert document["entries"], "record-baseline wrote no entries -- the diff would be vacuous"

    diff = cli_runner(
        "src.poc.cli",
        [
            "diff",
            "--baseline",
            str(baseline),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
        ],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert diff.returncode == 0, diff.output


def test_diff_against_a_tightened_baseline_exits_one(
    completed_run: tuple[Path, str, int],
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """A baseline the run cannot meet exits 1.

    The mirror of the clean self-diff above. Without it, a ``diff`` that ignored
    its baseline entirely would pass every other assertion in this file.
    """
    output_dir, run_id, _rc = completed_run
    baseline = tmp_path / "baseline.json"
    record = cli_runner(
        "src.poc.cli",
        [
            "record-baseline",
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--out",
            str(baseline),
            "--tolerance-pct",
            RECORD_TOLERANCE_PCT,
        ],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert record.returncode == 0, record.output

    document = json.loads(baseline.read_text(encoding="utf-8"))
    tightened_name: str | None = None
    for entry in document["entries"]:
        value = float(entry["value"])
        if entry["direction"] == "lower_better" and value > 0.0:
            entry["value"] = value * REGRESSION_TIGHTEN_FACTOR
            entry["tolerance_pct"] = REGRESSION_TOLERANCE_PCT
            tightened_name = str(entry["metric_name"])
            break
    assert tightened_name is not None, (
        "no strictly-positive lower_better entry to tighten -- the test would be vacuous"
    )
    tightened = tmp_path / "tightened.json"
    tightened.write_text(json.dumps(document), encoding="utf-8")

    diff = cli_runner(
        "src.poc.cli",
        [
            "diff",
            "--baseline",
            str(tightened),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
        ],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert diff.returncode == 1, diff.output
    assert tightened_name in diff.output


def test_record_with_unknown_run_id_exits_one_and_names_it(
    completed_run: tuple[Path, str, int],
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """An unknown ``--run-id`` fails, and the message names the id.

    Cheap here -- unlike the harness scripts, ``record-baseline`` reads a
    persisted run rather than executing one, so the error path costs nothing.
    """
    output_dir, _run_id, _rc = completed_run
    result = cli_runner(
        "src.poc.cli",
        [
            "record-baseline",
            "--output-dir",
            str(output_dir),
            "--run-id",
            ABSENT_RUN_ID,
            "--out",
            str(tmp_path / "unused.json"),
        ],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert result.returncode == 1
    assert ABSENT_RUN_ID in result.output
