"""E2E tests for the PoC scenario framework.

Tests the scenario CLI user journey:
1. List available scenarios
2. Show scenario info
3. Run scenarios
4. Compare runs
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.e2e.conftest import (
    E2E_TRAINING_TIMEOUT_S,
    E2E_TRIVIAL_TIMEOUT_S,
)

if TYPE_CHECKING:
    from tests.e2e.conftest import CLIRunnerType


@pytest.mark.e2e
def test_poc_cli_list(cli_runner: CLIRunnerType) -> None:
    """Verify 'list' command shows all scenarios."""
    result = cli_runner("src.poc.cli", ["list"])
    assert result.success, f"Failed: {result.stderr}"
    # Should list registered scenarios
    assert "transfer" in result.stdout.lower() or "scenario" in result.stdout.lower()


@pytest.mark.e2e
def test_poc_cli_help(cli_runner: CLIRunnerType) -> None:
    """Verify help command works."""
    result = cli_runner("src.poc.cli", ["--help"])
    assert result.success, f"Failed: {result.stderr}"
    assert "usage" in result.stdout.lower() or "poc" in result.stdout.lower()


@pytest.mark.e2e
def test_poc_cli_info_transfer(cli_runner: CLIRunnerType) -> None:
    """Verify 'info' command shows scenario details."""
    result = cli_runner("src.poc.cli", ["info", "transfer"])
    # `in [0, 1, 2]` is every code argparse can produce, so it asserted nothing:
    # it passed whether the scenario was found, missing, or the CLI crashed.
    # `transfer` is a built-in scenario registered at import, so 0 is the contract.
    assert result.returncode == 0, f"Unexpected error: {result.stderr}"
    assert "transfer" in result.output.lower()


@pytest.mark.e2e
def test_poc_cli_run_help(cli_runner: CLIRunnerType) -> None:
    """Verify 'run' subcommand shows help."""
    result = cli_runner("src.poc.cli", ["run", "--help"])
    assert result.success, f"Failed: {result.stderr}"


@pytest.mark.e2e
def test_poc_cli_invalid_scenario(cli_runner: CLIRunnerType) -> None:
    """Verify invalid scenario name is handled."""
    result = cli_runner(
        "src.poc.cli",
        ["info", "nonexistent_scenario_xyz"],
        timeout=E2E_TRIVIAL_TIMEOUT_S,
    )
    # The previous `not success or "not found" in ...` disjunction passed on any
    # non-zero exit, including a crash. An unknown scenario is a handled error: 1.
    assert result.returncode == 1, f"Expected a handled error: {result.output}"
    assert "not found" in result.output.lower()


@pytest.mark.e2e
def test_poc_cli_compare_help(cli_runner: CLIRunnerType) -> None:
    """Verify 'compare' subcommand shows help."""
    result = cli_runner("src.poc.cli", ["compare", "--help"])
    # `compare` is a registered subparser; --help on it exits 0.
    assert result.returncode == 0, f"Unexpected error: {result.stderr}"


@pytest.mark.e2e
@pytest.mark.slow
def test_poc_cli_run_tier_filter(
    cli_runner: CLIRunnerType,
    temp_output_dir: Path,
) -> None:
    """Verify tier filtering works."""
    result = cli_runner(
        "src.poc.cli",
        ["run", "--tier", "unit", "--output-dir", str(temp_output_dir)],
        timeout=E2E_TRAINING_TIMEOUT_S,
    )
    # `cmd_run` returns `all(r.passed)`, so 0 means every selected scenario
    # passed its thresholds. No unit-tier scenario is registered today, and an
    # empty selection is vacuously all-passed -- still 0, not "either".
    assert result.returncode == 0, f"Unexpected error: {result.stderr}"


@pytest.mark.e2e
def test_poc_cli_config_path(cli_runner: CLIRunnerType, config_dir: Path) -> None:
    """Verify config file path is handled."""
    # Check if poc_quick.yaml exists
    config_file = config_dir / "scenarios" / "poc_quick.yaml"
    if not config_file.exists():
        pytest.skip("poc_quick.yaml not found")

    result = cli_runner(
        "src.poc.cli",
        ["run", "--config", str(config_file), "--help"],
        timeout=E2E_TRIVIAL_TIMEOUT_S,
    )
    # --help short-circuits before the config is read; it exits 0.
    assert result.returncode == 0, f"Unexpected error: {result.stderr}"
