"""E2E: the Gate 1 CLI as a process -- ``--help`` on CPU, one tiny skfem smoke.

The ``fem_required`` journey runs in CI's ``test-extras`` job (``pytest
tests/e2e/ -m "fem_required and not gpu_required"``), the only job with
scikit-fem; a ``fem_required`` test outside ``tests/e2e/`` would be selected by
no CI step. The smoke drives the shipped Z-testbed YAML with every budget
shrunk and the adequacy precondition off (the adequacy gate has its own suite);
it asserts the mechanism -- exit code 0 whatever the verdict, every arm in the
CSV, the verdict and each criterion in the sidecar -- never a research outcome.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from tests.e2e.conftest import E2E_BENCHMARK_TIMEOUT_S, E2E_TRIVIAL_TIMEOUT_S, CLIRunnerType

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
ZSHAPE_YAML = REPO_ROOT / "config" / "scenarios" / "lookahead_vs_greedy_zshape.yaml"
MODULE = "scripts.run_lookahead_vs_greedy"
#: Budgets small enough for the E2E benchmark timeout; nothing here is scored.
SMOKE_OVERRIDES = {
    "max_steps": 2,
    "n_simulations": 3,
    "top_k_actions": 2,
    "robust_n_seeds": 2,
    "min_robust_wins": 1,
    "classical_max_dof": 150,
    "max_refinements_classical": 60,
    "require_adequacy_precondition": False,
    "artifact_basename": "lvg_e2e",
}
GATED = (
    "primary_l2_ratio_vs_best_classical",
    "primary_decisions_diverging_from_greedy",
    "robust_median_l2_ratio_vs_best_classical",
    "robust_seeds_below_best_classical",
    "primary_dof_saving_vs_greedy",
)


def test_help_exits_zero_and_states_the_exit_codes(cli_runner: CLIRunnerType) -> None:
    """``--help`` must not need scikit-fem (runs on the CPU e2e job)."""
    result = cli_runner(MODULE, ["--help"], E2E_TRIVIAL_TIMEOUT_S, None)
    assert result.returncode == 0, result.output
    assert "--proposal-grade" in result.output
    assert "3 adequacy abort" in result.output


@pytest.mark.fem_required
def test_a_tiny_z_testbed_run_completes_with_a_verdict(
    cli_runner: CLIRunnerType, tmp_path: Path
) -> None:
    raw = yaml.safe_load(ZSHAPE_YAML.read_text(encoding="utf-8"))
    entry = raw["scenarios"][0]
    entry.update(SMOKE_OVERRIDES)
    entry["substrate"]["initial_refinements"] = 1
    entry["substrate"]["solve_cache_max_entries"] = 512
    config = tmp_path / "lvg_e2e.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    result = cli_runner(
        MODULE,
        ["--config", str(config), "--output-dir", str(tmp_path), "--log-level", "WARNING"],
        E2E_BENCHMARK_TIMEOUT_S,
        None,
    )
    assert result.returncode == 0, result.output

    with (tmp_path / "lvg_e2e.csv").open(encoding="utf-8", newline="") as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert methods == {
        "uniform",
        "dorfler_theta0.1",
        "dorfler_theta0.3",
        "dorfler_theta0.5",
        "greedy",
        "mcts_primary",
        "mcts_robust",
    }
    manifest = json.loads((tmp_path / "lvg_e2e.run.json").read_text(encoding="utf-8"))
    assert manifest["config"]["operator_name"] == "zshape_poisson"
    assert set(GATED) <= set(manifest["metrics"])
    assert manifest["metrics"]["verdict_go"] in (0.0, 1.0)
    primary = next(arm for arm in manifest["arms"] if arm["name"].startswith("mcts_primary"))
    assert len(primary["parameters"]["tree_depth_per_step"]) == SMOKE_OVERRIDES["max_steps"]
    assert ("GO" if manifest["metrics"]["verdict_go"] else "NO-GO") in manifest["notes"]
    assert (tmp_path / "lvg_e2e.png").is_file()
