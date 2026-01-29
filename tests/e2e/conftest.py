"""E2E test fixtures and configuration.

Provides shared fixtures for end-to-end testing of CLI commands
and user journeys.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import pytest

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
