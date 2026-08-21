"""E2E test fixtures and configuration.

Provides shared fixtures for end-to-end testing of CLI commands
and user journeys.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

# Subprocess budgets for the CLI journeys, scaled by one env var.
#
# These were six bare literals across two modules (30 / 60 / 120), and the
# 120 s one is not merely untidy: it is why
# `test_quick_validation_journey.py::test_train_physics_minimal` fails on a
# loaded machine. It shells out to a real training run, so the budget is a bet
# about how fast the host is -- and the test asserts `returncode in [0, 1]`,
# which a timeout (-1) fails. A slow box therefore reports a training defect
# that does not exist.
#
# `E2E_TIMEOUT_SCALE` multiplies all three so a slow or contended runner can be
# accommodated without editing source or loosening an assertion. Left at 1.0 the
# values are byte-for-byte what they were.
#
# The three tiers are ordered by what the subprocess actually does, and that
# ordering is the part worth preserving if these are ever retuned:
#   TRIVIAL  -- argument parsing / --help; process startup dominates.
#   BENCH    -- a bounded measurement loop; work is real but capped by argv.
#   TRAINING -- a real training run; the only one whose cost tracks host speed.
E2E_TIMEOUT_SCALE: float = float(os.environ.get("E2E_TIMEOUT_SCALE", "1.0"))

E2E_TRIVIAL_TIMEOUT_S: int = int(30 * E2E_TIMEOUT_SCALE)
E2E_BENCHMARK_TIMEOUT_S: int = int(60 * E2E_TIMEOUT_SCALE)
E2E_TRAINING_TIMEOUT_S: int = int(120 * E2E_TIMEOUT_SCALE)

# Type alias for the CLI runner fixture
CLIRunnerType = Callable[
    [str, list[str] | None, int, dict[str, str] | None],
    "CLIResult",
]


@dataclass
class CLIResult:
    """Result from running a CLI command."""

    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.returncode == 0


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs.

    Yields:
        Path to temporary directory (cleaned up after test).

    """
    with tempfile.TemporaryDirectory(prefix="alphagalerkin_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cli_runner() -> CLIRunnerType:
    """Create a CLI command runner.

    Returns:
        Function to run CLI commands and capture output.

    """

    def run_command(
        module: str,
        args: list[str] | None = None,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> CLIResult:
        """Run a Python module command.

        Args:
            module: Module to run (e.g., "src.poc.cli").
            args: Command-line arguments.
            timeout: Timeout in seconds.
            env: Additional environment variables.

        Returns:
            CLIResult with command output.

        """
        cmd = [sys.executable, "-m", module]
        if args:
            cmd.extend(args)

        import os

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
                cwd=Path(__file__).parents[2],  # Project root
            )
            return CLIResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=cmd,
            )
        except subprocess.TimeoutExpired:
            return CLIResult(
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                command=cmd,
            )

    return run_command


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root.

    """
    return Path(__file__).parents[2]


@pytest.fixture
def config_dir(project_root: Path) -> Path:
    """Get the config directory.

    Returns:
        Path to config directory.

    """
    return project_root / "config"
