"""E2E: arena CLI --help without FEM; tiny skfem compare on the extras job.

The ``fem_required`` journeys run in CI's ``test-extras`` job (``pytest
tests/e2e/ -m "fem_required and not gpu_required"``), the only job with
scikit-fem. A ``fem_required`` test anywhere else in the arena suites would be
selected by no CI step, which is why the skfem greedy-control check lives here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.e2e.conftest import E2E_BENCHMARK_TIMEOUT_S, E2E_TRIVIAL_TIMEOUT_S, CLIRunnerType

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YAML = REPO_ROOT / "config" / "scenarios" / "mcts_classical_amr_arena_ci.yaml"
HEADLINE_YAML = REPO_ROOT / "config" / "scenarios" / "mcts_classical_amr_arena.yaml"
HEADLINE_BASENAME = "mcts_classical_amr_arena"


def test_script_help_exits_zero(cli_runner: CLIRunnerType) -> None:
    """``--help`` must not import scikit-fem (runs on the CPU e2e job)."""
    result = cli_runner(
        "scripts.run_mcts_classical_amr_arena",
        ["--help"],
        E2E_TRIVIAL_TIMEOUT_S,
        None,
    )
    assert result.returncode == 0, result.output
    assert "--proposal-grade" in result.output
    assert "mcts" in result.output.lower() or "arena" in result.output.lower()


@pytest.mark.fem_required
def test_ci_yaml_completes_with_a_verdict(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """Mechanism smoke on skfem_tri. Exit code tracks the threshold, not a crash."""
    result = cli_runner(
        "scripts.run_mcts_classical_amr_arena",
        [
            "--config",
            str(CI_YAML),
            "--output-dir",
            str(tmp_path),
            "--skip-adequacy",
        ],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert result.returncode in (0, 1), result.output
    csv_files = list(tmp_path.glob("*.csv"))
    assert csv_files, result.output
    sidecar = csv_files[0].with_suffix(".run.json")
    assert sidecar.exists(), result.output


@pytest.mark.fem_required
def test_one_simulation_on_the_shipped_yaml_is_greedy(
    cli_runner: CLIRunnerType,
    tmp_path: Path,
) -> None:
    """Mechanism check on skfem_tri: one simulation per step reproduces greedy.

    With ``top_k_actions > 0`` the legal set is indicator-ranked. PUCT at an
    unvisited root scores every child 0 (its exploration term scales with the
    square root of the parent's visits) and takes the first, ``legal[0]`` --
    the greedy choice. At a reused root with one visit it takes the largest
    prior, a monotone softmax of the same indicators. So one simulation cannot
    choose anything but greedy: ``decisions_diverging_from_greedy`` must read 0
    and every seed's MCTS curve must equal the greedy control's. This asserts
    the mechanism only; nothing here reads the committed ``n_simulations=8``
    result, which is a research outcome.
    """
    result = cli_runner(
        "scripts.run_mcts_classical_amr_arena",
        [
            "--config",
            str(HEADLINE_YAML),
            "--n-simulations",
            "1",
            "--output-dir",
            str(tmp_path),
            "--log-level",
            "WARNING",
        ],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    sidecar = tmp_path / f"{HEADLINE_BASENAME}.run.json"
    assert sidecar.is_file(), result.output
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    metrics = manifest["metrics"]
    # Exact exit code, derived from the run's own verdict rather than asserted.
    gate = manifest["config"]["max_l2_ratio_at_matched_dof"]
    expected_exit = 0 if metrics["l2_error_ratio_at_matched_dof"] < gate else 1
    assert result.returncode == expected_exit, result.output
    assert manifest["config"]["n_simulations"] == 1
    assert manifest["config"]["include_greedy_control"] is True
    assert metrics["decisions_diverging_from_greedy"] == 0.0
    assert metrics["decisions_diverging_from_greedy_max"] == 0.0

    with (tmp_path / f"{HEADLINE_BASENAME}.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    greedy = [(row["n_dof"], row["l2_error"]) for row in rows if row["method"] == "greedy"]
    mcts: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if row["method"] == "mcts":
            mcts.setdefault(row["seed"], []).append((row["n_dof"], row["l2_error"]))
    assert len(greedy) > 1, result.output
    assert sorted(mcts) == sorted(str(seed) for seed in manifest["seeds"])
    for seed, curve in mcts.items():
        assert curve == greedy, f"seed {seed}: MCTS at one simulation diverged from greedy"
    assert metrics["l2_error_ratio_mcts_over_greedy_at_matched_dof"] == 1.0
