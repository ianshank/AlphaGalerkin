"""E2E: arena CLI --help without FEM; tiny skfem compare on the extras job."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.conftest import E2E_BENCHMARK_TIMEOUT_S, E2E_TRIVIAL_TIMEOUT_S, CLIRunnerType

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YAML = REPO_ROOT / "config" / "scenarios" / "mcts_classical_amr_arena_ci.yaml"


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
