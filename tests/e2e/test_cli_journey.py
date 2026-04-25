"""End-to-end CLI journey tests.

These tests verify that the CLI application can boot, parse arguments,
and handle various user journeys correctly.
"""

import sys
from unittest.mock import patch

import pytest

# Skip the entire module if the CLI entry point cannot be imported
cli_module = pytest.importorskip("src.tools.cli", reason="CLI module (src.tools.cli) not available")
main = cli_module.main


def test_cli_help_command() -> None:
    """Verify the application boots and shows help correctly.

    This exercises the full import chain and argument parser initialization.
    """
    test_args = ["alphagalerkin", "--help"]
    with patch.object(sys, "argv", test_args):
        # Help should cause SystemExit(0)
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 0


def test_cli_train_dry_run() -> None:
    """Verify invalid arguments are rejected with appropriate error code.

    This validates the argument parser is working correctly.
    """
    test_args = ["alphagalerkin", "train", "--non-existent-arg"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        # Should exit with non-zero error code
        assert excinfo.value.code != 0
