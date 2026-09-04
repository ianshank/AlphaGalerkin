"""E2E journey: the shipped L-shape AMR CPU YAML through both entry points.

Guards (per ``docs/E2E_TEST_PLAN.md`` §4.2):

- ``specs/lshape_amr_compare.spec.md`` -- AC3 (the three honest ratios are all
  reported), and the acceptance gate's *exit-code* contract.
- CLAUDE.md Regression Surface row *"L-shape AMR MCTS-vs-Dörfler baseline"*.
- Plan §1's device contract: ``ScenarioResult.device`` must record the
  **execution** device, and a ``cuda`` request on a CUDA-less host must fail
  loud rather than fall back silently.

**Surface: numpy-only, with a recorded device.** ``src/research/lshape_amr_compare.py``
solves with numpy; the scenario resolves a device and logs it but places nothing
on it. The device assertions below are therefore *record-only* -- this journey
is never described as exercising a GPU (plan §1, honesty rule).

Two parsers live here because ``scripts/run_lshape_amr`` **prints** its metrics
rather than persisting a result JSON (only ``poc.cli run`` persists one). Both
are unit-tested on synthetic text in :class:`TestPrintedOutputParsers`: a parser
that silently returns ``{}`` would make every assertion in this file vacuous,
and the trap is real -- ``ScenarioResult.summary()`` prints its *own*
indented ``Metrics:`` block, with different formatting, immediately above the
one this file must read.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.e2e.conftest import (
    E2E_BENCHMARK_TIMEOUT_S,
    NO_CUDA_ENV,
    CLIResult,
    CLIRunnerType,
    PyRunnerType,
    ScenarioYamlPinnerType,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.e2e

#: Entry points under test.
SCRIPT_MODULE: str = "scripts.run_lshape_amr"
POC_CLI_MODULE: str = "src.poc.cli"

#: The shipped CPU config, used as shipped by the script journey and only via
#: ``pin_scenario_yaml`` (a tmp copy) by the CLI journey.
SHIPPED_CPU_CONFIG: str = "config/scenarios/lshape_amr_compare_cpu.yaml"

#: Reduced budget. Both entry points get the same one so the two journeys
#: describe the same run.
MAX_DOF: int = 120
N_SIMULATIONS: int = 2
SEED: int = 1

#: Exit codes, exact.
EXIT_PASSED: int = 0
EXIT_NOT_PASSED: int = 1

#: ``ScenarioStatus`` values as they appear in the printed summary and the
#: persisted JSON.
STATUS_PASSED: str = "passed"
STATUS_FAILED: str = "failed"
STATUS_ERROR: str = "error"

#: The statuses that represent a *verdict* (a run that completed and was
#: judged), as opposed to a run that crashed.
VERDICT_STATUSES: frozenset[str] = frozenset({STATUS_PASSED, STATUS_FAILED})

#: The three honest comparisons AC3 requires the harness to report.
RATIO_METRIC_KEYS: tuple[str, ...] = (
    "l2_error_ratio_at_matched_dof",
    "l2_error_ratio_at_matched_solves",
    "error_per_dof_ratio_mcts_over_dorfler",
)

#: Per-seed spread keys. Asserted *present*, never asserted non-zero: at this
#: budget every seed produces the identical trajectory (measured
#: ``l2_ratio_seed_std == 0``), which is a property of the reduced budget rather
#: than a defect.
SPREAD_METRIC_KEYS: tuple[str, ...] = (
    "l2_ratio_seed_min",
    "l2_ratio_seed_max",
    "l2_ratio_seed_std",
)

#: CSV schema.
CSV_ARM_COLUMN: str = "method"
EXPECTED_ARMS: frozenset[str] = frozenset({"dorfler", "mcts"})

#: The one place a device literal is allowed in this tier (plan §10): the
#: fail-loud negative test, paired with ``NO_CUDA_ENV`` so it runs identically
#: on a CPU-only runner and on a GPU box.
CUDA_REQUEST: str = "cuda"
CUDA_FAILURE_SUBSTRING: str = "but CUDA is not available"

# --------------------------------------------------------------------------- #
# Parsers for the script's printed output                                      #
# --------------------------------------------------------------------------- #

#: ``main()`` prints this header at column 0. ``ScenarioResult.summary()``
#: prints its own ``"   Metrics:"`` (three-space indent, six-decimal values,
#: optional " PASS"/" FAIL" suffix) earlier in the same stream, so the anchor
#: must be the *exact* line, not a stripped or substring match.
METRICS_HEADER: str = "Metrics:"

#: Terminates the metrics block. Same exact-line rule.
ARTIFACTS_HEADER: str = "Artifacts:"

#: ``  <name>: <value>`` -- exactly two leading spaces, as ``main()`` writes it.
_METRIC_LINE = re.compile(r"^ {2}(?P<name>[A-Za-z_][A-Za-z0-9_]*): (?P<value>\S+)$")

#: The summary's status line, at any indentation.
_STATUS_LINE = re.compile(r"^\s*Status:\s*(?P<status>\S+)\s*$")


class PrintedOutputError(ValueError):
    """The script's output lacked a block this journey must read.

    Raised rather than returning an empty mapping: a parser that degrades to
    ``{}`` turns every downstream assertion into a tautology, which is the exact
    failure mode this file exists to avoid.
    """


def parse_metrics_block(output: str) -> dict[str, float]:
    """Parse ``main()``'s printed ``Metrics:`` block.

    Anchored on the header line, never on line position: ``structlog`` writes
    ``debug`` lines into the same stream even at ``--log-level WARNING``, so the
    block's offset from the top of the output is not a contract.

    Args:
        output: Combined stdout+stderr of the script.

    Returns:
        Metric name to value. Empty when the block itself is empty (an errored
        run prints the header with nothing under it).

    Raises:
        PrintedOutputError: If no ``Metrics:`` header was printed at all.

    """
    lines = output.splitlines()
    try:
        start = lines.index(METRICS_HEADER)
    except ValueError as exc:
        raise PrintedOutputError(
            f"no {METRICS_HEADER!r} header in the script output; "
            "refusing to report an empty metric set as a successful parse"
        ) from exc

    metrics: dict[str, float] = {}
    for line in lines[start + 1 :]:
        if line == ARTIFACTS_HEADER:
            break
        match = _METRIC_LINE.match(line)
        if match is not None:
            metrics[match.group("name")] = float(match.group("value"))
    return metrics


def parse_status(output: str) -> str:
    """Parse the verdict from ``ScenarioResult.summary()``'s status line.

    Args:
        output: Combined stdout+stderr of the script.

    Returns:
        The ``ScenarioStatus`` value, e.g. ``"passed"`` / ``"failed"`` /
        ``"error"``.

    Raises:
        PrintedOutputError: If no status line was printed.

    """
    for line in output.splitlines():
        match = _STATUS_LINE.match(line)
        if match is not None:
            return match.group("status")
    raise PrintedOutputError("no 'Status:' line in the script output")


def expected_exit_code(status: str) -> int:
    """The exit code ``main()`` must produce for *status*.

    ``main()`` returns ``0 if result.passed else 1`` and ``passed`` is what sets
    a non-error status, so the mapping is total.

    Args:
        status: A ``ScenarioStatus`` value.

    Returns:
        ``0`` when the run passed, ``1`` otherwise.

    """
    return EXIT_PASSED if status == STATUS_PASSED else EXIT_NOT_PASSED


@dataclass(frozen=True)
class LShapeScriptRun:
    """One completed ``scripts.run_lshape_amr`` process and its output dir."""

    result: CLIResult
    output_dir: Path

    @property
    def status(self) -> str:
        """The scenario's verdict, parsed from the printed summary."""
        return parse_status(self.result.output)

    @property
    def metrics(self) -> dict[str, float]:
        """The printed metrics block."""
        return parse_metrics_block(self.result.output)


@pytest.fixture
def lshape_script_run(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
    e2e_device: str,
) -> LShapeScriptRun:
    """Run the artifact-writing script once, entirely inside ``tmp_path``.

    ``--output-dir`` is overridden because the shipped config writes to the
    committed ``results/`` directory; without it this journey would dirty the
    working tree (plan §2 rule 3).

    Returns:
        The completed run.

    """
    output_dir = tmp_path / "artifacts"
    result = cli_runner(
        SCRIPT_MODULE,
        [
            "--config",
            SHIPPED_CPU_CONFIG,
            "--output-dir",
            str(output_dir),
            "--max-dof",
            str(MAX_DOF),
            "--n-simulations",
            str(N_SIMULATIONS),
            "--seed",
            str(SEED),
            "--device",
            e2e_device,
        ],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    return LShapeScriptRun(result=result, output_dir=output_dir)


class TestPrintedOutputParsers:
    """Unit-test the parsers above on synthetic text.

    Without these, a parser that returned ``{}`` (or picked up
    ``summary()``'s block instead of ``main()``'s) would make every
    solve-driven assertion in this file pass while measuring nothing.
    """

    #: A faithful miniature of a real run: structlog noise, then
    #: ``summary()``'s indented block with *different* values, then
    #: ``main()``'s block, then the artifacts block.
    SAMPLE = "\n".join(
        [
            "2026-09-04 [debug    ] scenario_registered  name=lshape_amr_compare",
            "PASS lshape_amr_compare",
            "   Status: passed",
            "   Duration: 1.90s",
            "   Metrics:",
            "     l2_error_ratio_at_matched_dof: 0.111111 PASS",
            "     n_seeds: 3.000000",
            "",
            "Metrics:",
            "  l2_error_ratio_at_matched_dof: 0.962654",
            "  l2_ratio_seed_std: 0",
            "  n_seeds: 3",
            "",
            "Artifacts:",
            "  csv: /tmp/x.csv",
            "  png: /tmp/x.png",
        ]
    )

    def test_reads_the_scripts_block_not_the_summarys(self) -> None:
        """The discriminating case: two ``Metrics:`` blocks, different values."""
        metrics = parse_metrics_block(self.SAMPLE)
        assert metrics["l2_error_ratio_at_matched_dof"] == pytest.approx(0.962654)
        assert metrics["n_seeds"] == pytest.approx(3.0)
        assert metrics["l2_ratio_seed_std"] == pytest.approx(0.0)

    def test_stops_at_the_artifacts_header(self) -> None:
        """Artifact paths are not metrics; a greedy parser would choke on them."""
        assert "csv" not in parse_metrics_block(self.SAMPLE)
        assert "png" not in parse_metrics_block(self.SAMPLE)

    def test_missing_header_raises_rather_than_returning_empty(self) -> None:
        with pytest.raises(PrintedOutputError, match="Metrics:"):
            parse_metrics_block("nothing to see here\n")

    def test_an_empty_block_parses_to_no_metrics(self) -> None:
        """An errored run prints the header with nothing beneath it."""
        assert parse_metrics_block("Metrics:\n\nArtifacts:\n") == {}

    def test_status_is_parsed_from_the_summary(self) -> None:
        assert parse_status(self.SAMPLE) == STATUS_PASSED

    def test_missing_status_raises(self) -> None:
        with pytest.raises(PrintedOutputError, match="Status:"):
            parse_status("Metrics:\n  a: 1\n")

    @pytest.mark.parametrize(
        ("status", "code"),
        [(STATUS_PASSED, EXIT_PASSED), (STATUS_FAILED, EXIT_NOT_PASSED)],
    )
    def test_exit_code_mapping(self, status: str, code: int) -> None:
        assert expected_exit_code(status) == code


def test_script_exit_code_tracks_the_verdict(lshape_script_run: LShapeScriptRun) -> None:
    """The shell-visible exit code equals the run's own verdict.

    Guards ``specs/lshape_amr_compare.spec.md``'s acceptance contract: exit 0
    **iff** ``l2_error_ratio_at_matched_dof < 1.0`` passed.

    Deliberately **not** ``== 0``. That gate is a research question: the honest
    headline (median 1.0996) fails it, while this reduced budget happens to pass
    (measured 0.9627). Asserting a literal 0 would encode a research outcome as
    a software contract -- the defect class CLAUDE.md records under the
    2026-08-16 retraction. The expected code is therefore derived from the
    verdict the run itself printed.
    """
    run = lshape_script_run
    status = run.status
    assert status in VERDICT_STATUSES, (
        f"the run must reach a verdict, not crash; status={status!r}\n{run.result.output}"
    )
    assert run.result.returncode == expected_exit_code(status), run.result.output


def test_all_three_ratios_are_printed_and_finite(lshape_script_run: LShapeScriptRun) -> None:
    """AC3: all three honest comparisons are reported, not just the gated one.

    The matched-DOF ratio is the gate; matched-*solves* is the honest
    matched-compute axis; matched-wall-clock is recorded, not gated. A harness
    that printed only the flattering one would still pass its own gate, which is
    why presence is asserted separately from the verdict.

    Spread keys are asserted **present only**. At this budget all seeds are
    identical (``l2_ratio_seed_std == 0`` measured), so a non-zero assertion
    would be a statement about the budget, not about the code.
    """
    metrics = lshape_script_run.metrics
    for key in RATIO_METRIC_KEYS:
        assert key in metrics, f"missing {key!r}; got {sorted(metrics)}"
        assert math.isfinite(metrics[key]), f"{key} = {metrics[key]!r}"
    for key in SPREAD_METRIC_KEYS:
        assert key in metrics, f"missing per-seed spread key {key!r}"


def test_csv_and_png_land_in_output_dir(lshape_script_run: LShapeScriptRun) -> None:
    """Both committed artifacts are written where argv said, with both arms.

    Guards the Regression Surface row's artifact half. The basename is not
    hardcoded: exactly one CSV and one PNG must appear under the requested
    directory, which also proves nothing escaped into the working tree.
    """
    run = lshape_script_run
    assert run.result.returncode == expected_exit_code(run.status), run.result.output

    csvs = sorted(run.output_dir.glob("*.csv"))
    pngs = sorted(run.output_dir.glob("*.png"))
    assert len(csvs) == 1, f"expected exactly one CSV under {run.output_dir}, got {csvs}"
    assert len(pngs) == 1, f"expected exactly one PNG under {run.output_dir}, got {pngs}"

    with csvs[0].open(newline="", encoding="utf-8") as handle:
        arms = {row[CSV_ARM_COLUMN] for row in csv.DictReader(handle)}
    assert arms == EXPECTED_ARMS, f"the artifact must contain both arms; got {sorted(arms)}"


def test_poc_cli_persists_json_with_verdict_and_device(
    cli_runner: CLIRunnerType,
    py_runner: PyRunnerType,
    pin_scenario_yaml: ScenarioYamlPinnerType,
    tmp_path: Path,
    e2e_device: str,
) -> None:
    """The generic CLI persists a result JSON whose verdict drives the exit code.

    Guards plan §1's ``ScenarioResult.device`` fix: the field is documented as
    "Computation device used" but used to record host CUDA *availability*, so a
    ``device: cpu`` config on a CUDA host persisted ``"cuda"``. It must now equal
    the device the run was asked for. **Record-only**: the L-shape solve is
    numpy, so this asserts the device was *recorded*, not that a GPU did work.

    Also re-validates the persisted document in a **fresh process**:
    ``ResultCollector`` writes ``model_dump(mode="json")`` with ``default=str``,
    exactly the kind of write that can silently stop round-tripping.
    """
    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"
    pinned = pin_scenario_yaml(
        SHIPPED_CPU_CONFIG,
        device=e2e_device,
        max_dof=MAX_DOF,
        n_simulations=N_SIMULATIONS,
        output_dir=str(artifacts_dir),
    )

    result = cli_runner(
        POC_CLI_MODULE,
        ["run", "--config", str(pinned), "--output-dir", str(runs_dir)],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )

    payloads = sorted((runs_dir / "results").glob("*/*.json"))
    assert len(payloads) == 1, f"expected one persisted result under {runs_dir}, got {payloads}"
    payload: Mapping[str, object] = json.loads(payloads[0].read_text(encoding="utf-8"))

    assert payload["status"] in VERDICT_STATUSES, payload["status"]
    assert payload["passed"] is (payload["status"] == STATUS_PASSED)
    assert result.returncode == expected_exit_code(str(payload["status"])), result.output
    assert payload["device"] == e2e_device

    round_trip = py_runner(
        "import json, sys\n"
        "from src.poc.config import ScenarioResult\n"
        f"payload = json.loads(open({str(payloads[0])!r}, encoding='utf-8').read())\n"
        "ScenarioResult.model_validate(payload)\n"
        "print('round-tripped')\n",
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert round_trip.returncode == EXIT_PASSED, round_trip.output
    assert "round-tripped" in round_trip.stdout


def test_cuda_request_fails_loud_without_cuda(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """A ``cuda`` request with no CUDA fails loudly instead of falling back.

    Guards plan §1's third property. ``CUDA_VISIBLE_DEVICES=""`` makes this run
    identically on a CPU-only runner and on a GPU box -- and on the latter it
    additionally proves the child honoured the flag rather than quietly using
    the GPU it can see.

    Exit ``1`` and status ``error`` together: ``1`` alone would also be produced
    by a run that completed and failed its threshold, which is a different
    outcome.
    """
    output_dir = tmp_path / "artifacts"
    result = cli_runner(
        SCRIPT_MODULE,
        [
            "--config",
            SHIPPED_CPU_CONFIG,
            "--output-dir",
            str(output_dir),
            "--max-dof",
            str(MAX_DOF),
            "--n-simulations",
            str(N_SIMULATIONS),
            "--seed",
            str(SEED),
            "--device",
            CUDA_REQUEST,
        ],
        E2E_BENCHMARK_TIMEOUT_S,
        dict(NO_CUDA_ENV),
    )

    assert result.returncode == EXIT_NOT_PASSED, result.output
    assert parse_status(result.output) == STATUS_ERROR, result.output
    assert CUDA_FAILURE_SUBSTRING in result.output, result.output
