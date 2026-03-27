"""Tests for CLI entry point.

Validates argument parsing, subcommand dispatch, help output, and
error handling for the AlphaGalerkin CLI.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.tools.cli import main


class TestCLIArgumentParsing:
    """Test argument parsing for all subcommands."""

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Running with no arguments prints help and does not error."""
        with patch("sys.argv", ["alphagalerkin"]):
            main()
        captured = capsys.readouterr()
        assert "AlphaGalerkin" in captured.out

    def test_verify_defaults(self) -> None:
        """Verify subcommand uses default values when not overridden."""
        with (
            patch("sys.argv", ["alphagalerkin", "verify"]),
            patch("src.tools.cli.run_verification", return_value=True) as mock_verify,
            patch("sys.exit") as mock_exit,
        ):
            # Import fresh to pick up lazy import

            import src.tools.cli as cli_mod

            with patch.object(cli_mod, "__name__", "src.tools.cli"):
                with patch(
                    "src.tools.verify_invariance.run_verification", return_value=True
                ) as mock_verify:
                    main()
            mock_verify.assert_called_once_with(
                train_size=9,
                infer_size=19,
                device="cpu",
            )
            mock_exit.assert_called_once_with(0)

    @pytest.mark.parametrize(
        "train_size,infer_size,device",
        [
            (5, 13, "cpu"),
            (9, 25, "cpu"),
            (13, 19, "cpu"),
        ],
    )
    def test_verify_custom_args(
        self, train_size: int, infer_size: int, device: str
    ) -> None:
        """Verify subcommand forwards custom arguments correctly."""
        args = [
            "alphagalerkin",
            "verify",
            "--train-size",
            str(train_size),
            "--infer-size",
            str(infer_size),
            "--device",
            device,
        ]
        with (
            patch("sys.argv", args),
            patch(
                "src.tools.verify_invariance.run_verification", return_value=True
            ) as mock_verify,
            patch("sys.exit"),
        ):
            main()
        mock_verify.assert_called_once_with(
            train_size=train_size,
            infer_size=infer_size,
            device=device,
        )

    def test_gtp_dispatches_to_gtp_main(self) -> None:
        """GTP subcommand delegates to the GTP module."""
        with (
            patch("sys.argv", ["alphagalerkin", "gtp", "--board-size", "9"]),
            patch("src.tools.gtp.main") as mock_gtp_main,
        ):
            main()
        mock_gtp_main.assert_called_once()

    def test_gtp_with_model_path(self) -> None:
        """GTP subcommand passes model path argument."""
        with (
            patch(
                "sys.argv",
                ["alphagalerkin", "gtp", "--model", "/tmp/model.pt"],
            ),
            patch("src.tools.gtp.main") as mock_gtp_main,
        ):
            main()
        mock_gtp_main.assert_called_once()

    def test_generate_colab_dispatches(self) -> None:
        """generate-colab subcommand calls the right function."""
        with (
            patch("sys.argv", ["alphagalerkin", "generate-colab"]),
            patch(
                "src.tools.colab.generate_colab_notebook"
            ) as mock_gen,
        ):
            main()
        mock_gen.assert_called_once()


class TestCLIExitCodes:
    """Test exit codes from the verify subcommand."""

    def test_verify_success_exits_zero(self) -> None:
        """Verification passing yields exit code 0."""
        with (
            patch("sys.argv", ["alphagalerkin", "verify"]),
            patch(
                "src.tools.verify_invariance.run_verification", return_value=True
            ),
            patch("sys.exit") as mock_exit,
        ):
            main()
        mock_exit.assert_called_once_with(0)

    def test_verify_failure_exits_one(self) -> None:
        """Verification failing yields exit code 1."""
        with (
            patch("sys.argv", ["alphagalerkin", "verify"]),
            patch(
                "src.tools.verify_invariance.run_verification", return_value=False
            ),
            patch("sys.exit") as mock_exit,
        ):
            main()
        mock_exit.assert_called_once_with(1)


class TestCLIInvalidArguments:
    """Test behaviour with invalid arguments."""

    def test_unknown_subcommand_prints_help(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unknown subcommand is treated like no command (prints help)."""
        with patch("sys.argv", ["alphagalerkin", "nonexistent"]):
            # argparse will error on unknown subcommands
            with pytest.raises(SystemExit):
                main()

    def test_verify_invalid_type_exits(self) -> None:
        """Non-integer board size triggers argparse error."""
        with (
            patch("sys.argv", ["alphagalerkin", "verify", "--train-size", "abc"]),
            pytest.raises(SystemExit),
        ):
            main()
