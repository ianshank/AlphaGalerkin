"""Smoke tests for command-line interface entrypoints.

This ensures that all CLI entrypoints can be executed with --help without
crashing or hanging, verifying basic command structure.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Define the root directory to check for existence of scripts
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_cli_entrypoints() -> list[str]:
    """Get a list of CLI modules to test.

    Checks for file existence to ensure we don't fail on missing optional CLIs.
    """
    entrypoints = []

    # Always expect this one based on requirements, but safe to check
    if (ROOT_DIR / "src" / "poc" / "cli.py").exists():
        entrypoints.append("src.poc.cli")

    if (ROOT_DIR / "src" / "tools" / "cli.py").exists():
        entrypoints.append("src.tools.cli")

    if (ROOT_DIR / "src" / "agents" / "cli.py").exists():
        entrypoints.append("src.agents.cli")

    if (ROOT_DIR / "scripts" / "benchmark_codec.py").exists():
        # Note: scripts are usually not in a package, but python -m scripts.benchmark_codec works
        # if scripts has an __init__.py or if it's run from the root.
        entrypoints.append("scripts.benchmark_codec")

    return entrypoints


CLI_ENTRYPOINTS = get_cli_entrypoints()


@pytest.mark.parametrize("cli_module", CLI_ENTRYPOINTS, ids=CLI_ENTRYPOINTS)
def test_cli_help(cli_module: str) -> None:
    """Test that the CLI module can be run with --help and returns exit code 0."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", cli_module, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(ROOT_DIR),
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"CLI '{cli_module} --help' timed out after 10 seconds.")

    assert result.returncode == 0, (
        f"CLI '{cli_module} --help' failed with exit code {result.returncode}.\n"
        f"Stdout: {result.stdout}\nStderr: {result.stderr}"
    )

    # Check that output looks like a help message (either argparse or click)
    output = result.stdout.lower() + result.stderr.lower()
    assert "usage:" in output or "options:" in output or "help" in output, (
        f"CLI '{cli_module} --help' did not output typical help text.\nOutput: {result.stdout}"
    )
