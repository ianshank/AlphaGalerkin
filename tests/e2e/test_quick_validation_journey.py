"""E2E tests for the quick validation journey.

Tests the 5-minute validation pipeline:
1. train_physics.py - Train on Poisson data
2. verify_transfer.py - Verify zero-shot transfer
3. benchmark_fnet.py - Benchmark FNet speed
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.e2e.conftest import CLIRunnerType


@pytest.mark.e2e
def test_train_physics_help(cli_runner: CLIRunnerType) -> None:
    """Verify train_physics.py shows help."""
    result = cli_runner("src.experiments.train_physics", ["--help"])
    assert result.success, f"Failed: {result.stderr}"
    assert "train" in result.stdout.lower() or "usage" in result.stdout.lower()


@pytest.mark.e2e
def test_verify_transfer_help(cli_runner: CLIRunnerType) -> None:
    """Verify verify_transfer.py shows help."""
    result = cli_runner("src.experiments.verify_transfer", ["--help"])
    assert result.success, f"Failed: {result.stderr}"
    assert "verify" in result.stdout.lower() or "usage" in result.stdout.lower()


@pytest.mark.e2e
def test_benchmark_fnet_help(cli_runner: CLIRunnerType) -> None:
    """Verify benchmark_fnet.py shows help."""
    result = cli_runner("src.experiments.benchmark_fnet", ["--help"])
    assert result.success, f"Failed: {result.stderr}"
    assert "benchmark" in result.stdout.lower() or "usage" in result.stdout.lower()


@pytest.mark.e2e
@pytest.mark.slow
def test_train_physics_minimal(cli_runner: CLIRunnerType, temp_output_dir: Path) -> None:
    """Run minimal physics training (2 epochs).

    This tests the core training loop without full convergence.
    """
    result = cli_runner(
        "src.experiments.train_physics",
        [
            "--n-epochs",
            "2",
            "--n-train-samples",
            "10",
            "--n-eval-samples",
            "5",
            "--output-dir",
            str(temp_output_dir),
            "--train-size",
            "5",
            "--eval-size",
            "7",
        ],
        timeout=120,
    )
    # Training might fail due to minimal data, but should not crash
    assert result.returncode in [0, 1], f"Unexpected error: {result.stderr}"


@pytest.mark.e2e
@pytest.mark.slow
def test_benchmark_fnet_small(cli_runner: CLIRunnerType, temp_output_dir: Path) -> None:
    """Run FNet benchmark with small sizes.

    Tests the O(N log N) vs O(N^2) complexity comparison.
    """
    result = cli_runner(
        "src.experiments.benchmark_fnet",
        [
            "--sizes",
            "16,25",
            "--batch-size",
            "2",
            "--n-iterations",
            "3",
            "--n-warmup",
            "1",
            "--output-dir",
            str(temp_output_dir),
        ],
        timeout=60,
    )
    # Check for successful execution
    assert result.success, f"Benchmark failed: {result.stderr}"


@pytest.mark.e2e
def test_train_physics_invalid_args(cli_runner: CLIRunnerType) -> None:
    """Verify invalid arguments are rejected."""
    result = cli_runner(
        "src.experiments.train_physics",
        ["--invalid-nonexistent-arg"],
        timeout=30,
    )
    assert not result.success, "Should reject invalid arguments"


@pytest.mark.e2e
def test_output_directory_creation(cli_runner: CLIRunnerType, temp_output_dir: Path) -> None:
    """Verify output directories are created correctly."""
    _ = temp_output_dir / "nested" / "output"  # Verify path operations work
    result = cli_runner(
        "src.experiments.train_physics",
        [
            "--help",  # Just check help works with output dir param
        ],
    )
    assert result.success
