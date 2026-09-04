"""E2E journey: ``scripts.run_adaptive_vs_uniform`` -> CSV -> provenance sidecar.

Guards (per ``docs/E2E_TEST_PLAN.md`` §4.1):

- The charter evidence row *"L-shape adaptive Dörfler vs uniform at matched
  DOF"* (``openspec/specs/project-charter/spec.md``) and the artifact it cites,
  ``results/lshape_adaptive_vs_uniform.csv`` plus its ``.run.json`` sidecar.
- ``specs/refinement_substrate.spec.md`` AC5 (both arms measured through one
  solver) and AC7 (the rate separation is *reported*, per arm).
- CLAUDE.md Regression Surface row *"L-shape AMR MCTS-vs-Dörfler baseline"*,
  whose shared substrate this script exercises.

Why an E2E tier at all for this script: ``tests/scripts/test_run_adaptive_vs_uniform.py``
never imports ``main``. It covers ``build_parser`` / ``run_uniform_arm`` /
``compare`` / ``export_csv`` only, so **the entry point and the provenance
sidecar write are exercised by nothing** -- the gap peer review found in v1 of
the plan. Everything here therefore goes through the real process.

**Surface: numpy-only.** ``scripts/run_adaptive_vs_uniform`` and
``src/research/substrates/*`` import no torch, the script has no ``--device``
flag, and nothing here places a tensor anywhere. No device is passed and none is
asserted -- fabricating a device assertion on this surface would claim more than
the code does (plan §1, flow (c)).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.research.run_manifest import (
    UNKNOWN,
    load_run_manifest,
    manifest_path_for,
    migrate_run_manifest,
)
from tests.e2e.conftest import (
    E2E_BENCHMARK_TIMEOUT_S,
    E2E_TRIVIAL_TIMEOUT_S,
    CLIResult,
    CLIRunnerType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.e2e

#: The entry point under test, as a shell would spell it.
SCRIPT_MODULE: str = "scripts.run_adaptive_vs_uniform"

#: Argv budget. Small enough that both arms finish in a few seconds, large
#: enough that each arm produces >= 3 levels (a rate fit through fewer points
#: always "succeeds" and always means nothing).
INITIAL_SIDE: int = 4
MAX_DOF: int = 120
MARKING_FRACTION: float = 0.5

#: ``--scale`` is not passed; this is the parser's default, and the sidecar must
#: echo the *resolved* config rather than only what argv mentioned.
DEFAULT_SCALE: float = 1.0

#: CSV schema the charter guard reads.
CSV_ARM_COLUMN: str = "method"
CSV_DOF_COLUMN: str = "n_dof"
EXPECTED_ARMS: frozenset[str] = frozenset({"uniform", "dorfler"})

#: ``RunManifest.harness`` must name the module that produced the artifact.
EXPECTED_HARNESS: str = SCRIPT_MODULE

#: Every numeric ``--flag`` this journey passes, keyed by its ``argparse`` dest
#: (which is the key the sidecar's ``config`` echoes). Compared as parsed
#: numbers, never as strings: ``"0.5"`` and ``0.5`` are the same configuration.
EXPECTED_NUMERIC_CONFIG: dict[str, float] = {
    "initial_side": float(INITIAL_SIDE),
    "max_dof": float(MAX_DOF),
    "marking_fraction": MARKING_FRACTION,
    "scale": DEFAULT_SCALE,
}

#: The metric keys the charter row is written against.
CHARTER_METRIC_KEYS: tuple[str, ...] = (
    "uniform_convergence_exponent",
    "dorfler_convergence_exponent",
    "dorfler_over_uniform_min",
    "dorfler_over_uniform_max",
)

#: Exit codes, exact. ``2`` is argparse's usage error and nothing else.
EXIT_SUCCESS: int = 0
EXIT_ARGPARSE_USAGE: int = 2


@dataclass(frozen=True)
class AdaptiveVsUniformRun:
    """One completed run of the script, with the paths its argv named."""

    result: CLIResult
    csv_path: Path
    manifest_path: Path
    argv_output: str


def _rows_by_arm(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Group the artifact's rows by the arm column, preserving file order.

    Args:
        csv_path: The CSV the script wrote.

    Returns:
        Mapping from arm name to its rows, in the order they appear.

    """
    grouped: dict[str, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row[CSV_ARM_COLUMN], []).append(row)
    return grouped


def _dofs(rows: Sequence[dict[str, str]]) -> list[int]:
    """DOF counts of *rows*, in file order.

    Args:
        rows: Rows for a single arm.

    Returns:
        The ``n_dof`` column parsed as integers.

    """
    return [int(row[CSV_DOF_COLUMN]) for row in rows]


@pytest.fixture
def adaptive_vs_uniform_run(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> AdaptiveVsUniformRun:
    """Run the script once into ``tmp_path`` and hand back its paths.

    Every output -- CSV and sidecar alike -- lands under ``tmp_path``, so the
    working tree stays clean (plan §2 rule 3). No ``--device``: the script has
    no such flag and the surface is numpy-only.

    Returns:
        The completed run.

    """
    csv_path = tmp_path / "adaptive_vs_uniform.csv"
    argv_output = str(csv_path)
    result = cli_runner(
        SCRIPT_MODULE,
        [
            "--output",
            argv_output,
            "--initial-side",
            str(INITIAL_SIDE),
            "--max-dof",
            str(MAX_DOF),
            "--marking-fraction",
            str(MARKING_FRACTION),
        ],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    return AdaptiveVsUniformRun(
        result=result,
        csv_path=csv_path,
        manifest_path=manifest_path_for(csv_path),
        argv_output=argv_output,
    )


def test_script_writes_csv_and_sidecar(adaptive_vs_uniform_run: AdaptiveVsUniformRun) -> None:
    """The entry point produces both halves of the charter's evidence.

    Guards the charter evidence row *"L-shape adaptive Dörfler vs uniform at
    matched DOF"*: the cited CSV must carry **both** arms in the column the
    charter guard reads, and the provenance sidecar -- which nothing else in the
    repo exercises -- must exist beside it.
    """
    run = adaptive_vs_uniform_run
    assert run.result.returncode == EXIT_SUCCESS, run.result.output

    assert run.csv_path.is_file(), f"no CSV at {run.csv_path}"
    grouped = _rows_by_arm(run.csv_path)
    assert set(grouped) == EXPECTED_ARMS, (
        f"the artifact must contain the comparison, not one arm; got {sorted(grouped)}"
    )
    assert all(grouped[arm] for arm in EXPECTED_ARMS)

    assert run.manifest_path.is_file(), (
        f"no provenance sidecar at {run.manifest_path}; the charter requires every "
        "numeric claim to cite an artifact that says how it was produced"
    )


def test_budget_is_a_stopping_rule(adaptive_vs_uniform_run: AdaptiveVsUniformRun) -> None:
    """``--max-dof`` stops the sweep; it does **not** cap DOF.

    ``run_uniform_arm`` appends the row *then* checks ``n_dof >= max_dof``
    (``scripts/run_adaptive_vs_uniform.py``), so the final level always
    overshoots -- at ``--max-dof 120`` the uniform arm reaches 208 DOF. v1 of the
    plan asserted "no row exceeds ``--max-dof``", which would have failed on
    unmutated code. This asserts the *documented* semantics instead, per plan
    §4.1, so the test tracks the contract rather than a misreading of it.

    Also pins the matched-DOF window's upper end to the smaller of the two arms'
    final DOF: neither arm may be credited for a level the other never reached.
    """
    run = adaptive_vs_uniform_run
    assert run.result.returncode == EXIT_SUCCESS, run.result.output
    grouped = _rows_by_arm(run.csv_path)

    finals: dict[str, int] = {}
    for arm in sorted(EXPECTED_ARMS):
        dofs = _dofs(grouped[arm])
        assert len(dofs) >= 2, f"arm {arm!r} produced too few levels to test a stopping rule"
        assert all(dof < MAX_DOF for dof in dofs[:-1]), (
            f"arm {arm!r}: every level but the last must be below the budget; got {dofs}"
        )
        assert dofs[-1] >= MAX_DOF, (
            f"arm {arm!r}: the budget is a stopping rule, so the last level must "
            f"reach or overshoot it; got {dofs}"
        )
        finals[arm] = dofs[-1]

    manifest = load_run_manifest(run.manifest_path)
    assert manifest.metrics["matched_dof_max"] == pytest.approx(float(min(finals.values())))


def test_sidecar_round_trips_and_echoes_argv(
    adaptive_vs_uniform_run: AdaptiveVsUniformRun,
) -> None:
    """The sidecar reproduces the run: harness, resolved config, artifact path.

    Guards the provenance half of ``specs/refinement_substrate.spec.md`` AC5 --
    a committed artifact must say how it was produced. ``config`` is compared as
    parsed numbers rather than strings, and ``artifacts["csv"]`` verbatim against
    argv (the script writes ``str(output)``, i.e. exactly what was passed, which
    is absolute here because argv was).
    """
    run = adaptive_vs_uniform_run
    assert run.result.returncode == EXIT_SUCCESS, run.result.output

    manifest = load_run_manifest(run.manifest_path)
    assert manifest.harness == EXPECTED_HARNESS

    assert set(manifest.config) == set(EXPECTED_NUMERIC_CONFIG) | {"output"}, (
        f"the sidecar must echo the whole resolved config; got {sorted(manifest.config)}"
    )
    for key, expected in EXPECTED_NUMERIC_CONFIG.items():
        assert float(manifest.config[key]) == pytest.approx(expected), (
            f"config[{key!r}] = {manifest.config[key]!r}"
        )
    assert manifest.config["output"] == run.argv_output

    assert manifest.artifacts["csv"] == run.argv_output

    raw = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    once = migrate_run_manifest(raw)
    assert migrate_run_manifest(once) == once, "migration must be idempotent"
    assert raw == json.loads(run.manifest_path.read_text(encoding="utf-8")), (
        "migrate_run_manifest must not mutate its caller's document"
    )


def test_sidecar_metrics_have_the_charter_shape(
    adaptive_vs_uniform_run: AdaptiveVsUniformRun,
) -> None:
    """Both arms' rates and the matched-DOF ratios are recorded and finite.

    Guards ``specs/refinement_substrate.spec.md`` AC7's *reporting* half: the
    rate separation must be published per arm, not merely computed.

    **No sign assertion, deliberately.** At this reduced budget the direction of
    ``dorfler_over_uniform_*`` is a research outcome, not a contract; the pinned
    adequacy verdict lives in
    ``tests/research/test_amr_arena_interpretability.py`` and in this tier's
    ``test_refinement_substrate_journey.py``, over the pinned DOF window.
    """
    run = adaptive_vs_uniform_run
    assert run.result.returncode == EXIT_SUCCESS, run.result.output

    metrics = load_run_manifest(run.manifest_path).metrics
    for key in CHARTER_METRIC_KEYS:
        assert key in metrics, f"missing charter metric {key!r}; got {sorted(metrics)}"
        assert math.isfinite(metrics[key]), f"{key} = {metrics[key]!r}"

    assert metrics["matched_dof_min"] <= metrics["matched_dof_max"]


def test_hardware_tag_records_the_host_the_run_measured_on(
    adaptive_vs_uniform_run: AdaptiveVsUniformRun,
) -> None:
    """``hardware_tag`` names a real host, not the "we did not record it" default.

    The one provenance field a device-agnostic tier needs: a wall-clock or
    accelerator-sensitive number is uninterpretable without it.

    The assertion is ``!= UNKNOWN`` rather than the ``tag.strip()`` this test
    originally carried, because ``UNKNOWN`` is itself a non-empty string --
    so the original passed identically whether the harness populated the field
    or not. Verified: deleting ``hardware_tag=collect_hardware_tag()`` from
    ``scripts/run_adaptive_vs_uniform.py`` left the whole file green. The
    *content* of the tag is pinned by ``tests/research/test_hardware_tag.py``;
    what this asserts is that the harness reaches the collector at all.
    """
    run = adaptive_vs_uniform_run
    assert run.result.returncode == EXIT_SUCCESS, run.result.output

    tag = load_run_manifest(run.manifest_path).hardware_tag
    assert tag != UNKNOWN, (
        "the harness wrote the RunManifest default: collect_hardware_tag() was "
        "never called, so this run's numbers cannot be attributed to a host"
    )
    assert tag.strip()


def test_unknown_flag_exits_two(cli_runner: CLIRunnerType, tmp_path: Path) -> None:
    """An unrecognised flag is an argparse usage error: exactly ``2``.

    Exact code, not a set (plan §2 rule 2): ``1`` would mean the script ran and
    failed, which is a different contract.
    """
    result = cli_runner(
        SCRIPT_MODULE,
        ["--output", str(tmp_path / "unused.csv"), "--no-such-flag"],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert result.returncode == EXIT_ARGPARSE_USAGE, result.output
    assert not (tmp_path / "unused.csv").exists()
