"""Tests for the CLI entry point (src/alphagalerkin/cli.py)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.alphagalerkin.cli import _parse_value, main

# -------------------------------------------------------------------
# _parse_value
# -------------------------------------------------------------------


class TestParseValue:
    """Tests for the _parse_value helper function."""

    def test_true_lowercase(self) -> None:
        assert _parse_value("true") is True

    def test_true_mixedcase(self) -> None:
        assert _parse_value("True") is True

    def test_true_uppercase(self) -> None:
        assert _parse_value("TRUE") is True

    def test_false_lowercase(self) -> None:
        assert _parse_value("false") is False

    def test_false_mixedcase(self) -> None:
        assert _parse_value("False") is False

    def test_false_uppercase(self) -> None:
        assert _parse_value("FALSE") is False

    def test_integer_positive(self) -> None:
        result = _parse_value("42")
        assert result == 42
        assert isinstance(result, int)

    def test_integer_negative(self) -> None:
        result = _parse_value("-7")
        assert result == -7
        assert isinstance(result, int)

    def test_integer_zero(self) -> None:
        result = _parse_value("0")
        assert result == 0
        assert isinstance(result, int)

    def test_float_with_decimal(self) -> None:
        result = _parse_value("3.14")
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_float_negative(self) -> None:
        result = _parse_value("-0.5")
        assert result == pytest.approx(-0.5)
        assert isinstance(result, float)

    def test_float_scientific_notation(self) -> None:
        result = _parse_value("1e-3")
        assert result == pytest.approx(0.001)
        assert isinstance(result, float)

    def test_string_passthrough(self) -> None:
        result = _parse_value("hello")
        assert result == "hello"
        assert isinstance(result, str)

    def test_string_with_spaces(self) -> None:
        result = _parse_value("hello world")
        assert result == "hello world"
        assert isinstance(result, str)

    def test_empty_string(self) -> None:
        result = _parse_value("")
        assert result == ""
        assert isinstance(result, str)


# -------------------------------------------------------------------
# CLI group help
# -------------------------------------------------------------------


class TestCLIGroup:
    """Tests for the main CLI group."""

    def test_main_help(self) -> None:
        """--help should exit 0 and mention AlphaGalerkin."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AlphaGalerkin" in result.output

    def test_train_help(self) -> None:
        """Train --help should exit 0 and mention training."""
        runner = CliRunner()
        result = runner.invoke(main, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train" in result.output or "train" in result.output.lower()

    def test_validate_config_help(self) -> None:
        """validate-config --help should exit 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate-config", "--help"])
        assert result.exit_code == 0
        assert "Validate" in result.output or "config" in result.output.lower()

    def test_evaluate_help(self) -> None:
        """Evaluate --help should exit 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["evaluate", "--help"])
        assert result.exit_code == 0


# -------------------------------------------------------------------
# train --dry-run
# -------------------------------------------------------------------


class TestTrainDryRun:
    """Tests for the train command with --dry-run."""

    @patch("src.alphagalerkin.cli.AlphaGalerkinConfig")
    @patch("src.alphagalerkin.cli.configure_logging")
    def test_dry_run_prints_config_summary(
        self,
        mock_logging: MagicMock,
        mock_config_cls: MagicMock,
    ) -> None:
        """--dry-run should print config summary without training."""
        # Build a mock config returned by from_dict
        mock_config = MagicMock()
        mock_config.physics.pde_type = "elliptic"
        mock_config.device = "cpu"
        mock_config.mcts.num_simulations = 100
        mock_config.training.total_steps = 10000
        mock_config_cls.from_dict.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(main, ["train", "--dry-run"])

        assert result.exit_code == 0
        assert "Config validated successfully" in result.output
        assert "elliptic" in result.output

    @patch("src.alphagalerkin.cli.AlphaGalerkinConfig")
    @patch("src.alphagalerkin.cli.configure_logging")
    def test_dry_run_with_overrides(
        self,
        mock_logging: MagicMock,
        mock_config_cls: MagicMock,
    ) -> None:
        """--dry-run should parse overrides correctly."""
        mock_config = MagicMock()
        mock_config.physics.pde_type = "parabolic"
        mock_config.device = "cpu"
        mock_config.mcts.num_simulations = 50
        mock_config.training.total_steps = 5000
        mock_config_cls.from_dict.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "train",
                "--dry-run",
                "--pde-type", "parabolic",
                "--override", "training.batch_size=32",
            ],
        )

        assert result.exit_code == 0
        # Verify from_dict was called with the override values
        call_args = mock_config_cls.from_dict.call_args
        overrides = call_args[0][0]
        assert overrides["training"]["batch_size"] == 32
        assert overrides["physics"]["pde_type"] == "parabolic"

    @patch("src.alphagalerkin.cli.AlphaGalerkinConfig")
    @patch("src.alphagalerkin.cli.configure_logging")
    def test_dry_run_with_boolean_override(
        self,
        mock_logging: MagicMock,
        mock_config_cls: MagicMock,
    ) -> None:
        """Boolean override values are parsed correctly."""
        mock_config = MagicMock()
        mock_config.physics.pde_type = "elliptic"
        mock_config.device = "cpu"
        mock_config.mcts.num_simulations = 100
        mock_config.training.total_steps = 10000
        mock_config_cls.from_dict.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "train",
                "--dry-run",
                "--override", "training.mixed_precision=true",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_config_cls.from_dict.call_args
        overrides = call_args[0][0]
        assert overrides["training"]["mixed_precision"] is True

    @patch("src.alphagalerkin.cli.AlphaGalerkinConfig")
    @patch("src.alphagalerkin.cli.configure_logging")
    def test_dry_run_shows_mcts_simulations(
        self,
        mock_logging: MagicMock,
        mock_config_cls: MagicMock,
    ) -> None:
        """--dry-run output includes MCTS simulations count."""
        mock_config = MagicMock()
        mock_config.physics.pde_type = "elliptic"
        mock_config.device = "cpu"
        mock_config.mcts.num_simulations = 200
        mock_config.training.total_steps = 5000
        mock_config_cls.from_dict.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(main, ["train", "--dry-run"])

        assert result.exit_code == 0
        assert "MCTS simulations: 200" in result.output

    @patch("src.alphagalerkin.cli.AlphaGalerkinConfig")
    @patch("src.alphagalerkin.cli.configure_logging")
    def test_dry_run_shows_training_steps(
        self,
        mock_logging: MagicMock,
        mock_config_cls: MagicMock,
    ) -> None:
        """--dry-run output includes total training steps."""
        mock_config = MagicMock()
        mock_config.physics.pde_type = "elliptic"
        mock_config.device = "cpu"
        mock_config.mcts.num_simulations = 100
        mock_config.training.total_steps = 77777
        mock_config_cls.from_dict.return_value = mock_config

        runner = CliRunner()
        result = runner.invoke(main, ["train", "--dry-run"])

        assert result.exit_code == 0
        assert "Training steps: 77777" in result.output
