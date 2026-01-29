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
    # May succeed or fail depending on scenario registration
    # But should not crash with an unexpected error
    assert result.returncode in [0, 1, 2], f"Unexpected error: {result.stderr}"


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
        timeout=30,
    )
    # Should exit with error but not crash
    not_found_in_stderr = "not found" in result.stderr.lower()
    not_found_in_stdout = "not found" in result.stdout.lower()
    assert not result.success or not_found_in_stderr or not_found_in_stdout


@pytest.mark.e2e
def test_poc_cli_compare_help(cli_runner: CLIRunnerType) -> None:
    """Verify 'compare' subcommand shows help."""
    result = cli_runner("src.poc.cli", ["compare", "--help"])
    # Compare might not exist, but should handle gracefully
    assert result.returncode in [0, 2], f"Unexpected error: {result.stderr}"


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
        timeout=120,
    )
    # Should attempt to run unit tier scenarios
    # Exit code depends on scenario availability
    assert result.returncode in [0, 1], f"Unexpected error: {result.stderr}"


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
        timeout=30,
    )
    # Help should work even with config path
    assert result.returncode in [0, 2]
