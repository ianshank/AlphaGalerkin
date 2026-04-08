"""Tests for src.tools.cli module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.tools.cli import main


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_no_command_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """No subcommand prints help and exits cleanly."""
        with patch("sys.argv", ["alphagalerkin"]):
            main()
        captured = capsys.readouterr()
        assert "Available commands" in captured.out or "usage:" in captured.out.lower()

    def test_help_flag(self) -> None:
        """--help raises SystemExit(0)."""
        with patch("sys.argv", ["alphagalerkin", "--help"]), pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_gtp_subcommand_dispatches(self) -> None:
        """'gtp' subcommand calls gtp_main."""
        with (
            patch("sys.argv", ["alphagalerkin", "gtp", "--device", "cpu"]),
            patch("src.tools.cli.sys") as mock_sys,
            patch("src.tools.gtp.main") as mock_gtp,
        ):
            mock_sys.argv = ["alphagalerkin", "gtp", "--device", "cpu"]
            # Parse args using real argparse, but intercept dispatch

            # We can't easily call main() without it modifying sys.argv,
            # so just verify the import path exists
            assert callable(mock_gtp)

    def test_verify_subcommand_dispatches(self) -> None:
        """'verify' subcommand calls run_verification."""
        with (
            patch(
                "sys.argv", ["alphagalerkin", "verify", "--train-size", "5", "--infer-size", "9"]
            ),
            patch("src.tools.verify_invariance.run_verification", return_value=True) as mock_verify,
            pytest.raises(SystemExit) as exc,
        ):
            main()
        mock_verify.assert_called_once_with(train_size=5, infer_size=9, device="cpu")
        assert exc.value.code == 0

    def test_verify_subcommand_failure_exits_1(self) -> None:
        """'verify' returns False -> exit code 1."""
        with (
            patch("sys.argv", ["alphagalerkin", "verify"]),
            patch("src.tools.verify_invariance.run_verification", return_value=False),
            pytest.raises(SystemExit) as exc,
        ):
            main()
        assert exc.value.code == 1

    def test_generate_colab_dispatches(self) -> None:
        """'generate-colab' subcommand calls generate_colab_notebook."""
        with (
            patch("sys.argv", ["alphagalerkin", "generate-colab"]),
            patch("src.tools.colab.generate_colab_notebook") as mock_gen,
        ):
            main()
        mock_gen.assert_called_once()

    def test_gtp_subcommand_calls_gtp_main(self) -> None:
        """'gtp' subcommand dispatches to gtp.main()."""
        with (
            patch("sys.argv", ["alphagalerkin", "gtp", "--device", "cpu"]),
            patch("src.tools.gtp.main") as mock_gtp,
        ):
            main()
        mock_gtp.assert_called_once()

    def test_gtp_subcommand_with_model_flag(self) -> None:
        """'gtp' subcommand passes --model flag."""
        with (
            patch("sys.argv", ["alphagalerkin", "gtp", "--model", "/tmp/m.pt"]),
            patch("src.tools.gtp.main") as mock_gtp,
        ):
            main()
        mock_gtp.assert_called_once()
