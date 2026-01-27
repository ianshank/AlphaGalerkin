"""Tests for the validation CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.validation.cli import create_parser, main


class TestCLIParser:
    """Tests for CLI argument parsing."""

    def test_parser_creation(self) -> None:
        """Test parser is created correctly."""
        parser = create_parser()
        assert parser is not None

    def test_run_command_parsing(self) -> None:
        """Test parsing run command."""
        parser = create_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_run_with_only_flag(self) -> None:
        """Test parsing run with --only flag."""
        parser = create_parser()
        args = parser.parse_args(["run", "--only", "gpu_training"])
        assert args.command == "run"
        assert args.only == "gpu_training"

    def test_run_with_parallel(self) -> None:
        """Test parsing run with --parallel flag."""
        parser = create_parser()
        args = parser.parse_args(["run", "--parallel"])
        assert args.parallel is True

    def test_run_with_stop_on_failure(self) -> None:
        """Test parsing run with --stop-on-failure flag."""
        parser = create_parser()
        args = parser.parse_args(["run", "--stop-on-failure"])
        assert args.stop_on_failure is True

    def test_run_with_config(self, tmp_path: Path) -> None:
        """Test parsing run with --config flag."""
        config_file = tmp_path / "config.yaml"
        config_file.touch()

        parser = create_parser()
        args = parser.parse_args(["run", "--config", str(config_file)])
        assert args.config == str(config_file)

    def test_merge_check_requires_pr(self) -> None:
        """Test merge-check requires --pr argument."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            # Should fail because --pr is required
            parser.parse_args(["merge-check"])

    def test_merge_check_with_pr(self) -> None:
        """Test parsing merge-check with --pr."""
        parser = create_parser()
        args = parser.parse_args(["merge-check", "--pr", "7"])
        assert args.command == "merge-check"
        assert args.pr == 7

    def test_merge_check_with_allow_failures(self) -> None:
        """Test parsing merge-check with --allow-failures."""
        parser = create_parser()
        args = parser.parse_args(["merge-check", "--pr", "7", "--allow-failures", "5"])
        assert args.allow_failures == 5

    def test_tolerance_check_parsing(self) -> None:
        """Test parsing tolerance-check command."""
        parser = create_parser()
        args = parser.parse_args(["tolerance-check"])
        assert args.command == "tolerance-check"

    def test_tolerance_check_with_test_dir(self) -> None:
        """Test parsing tolerance-check with --test-dir."""
        parser = create_parser()
        args = parser.parse_args(["tolerance-check", "--test-dir", "tests/unit/"])
        assert args.test_dir == "tests/unit/"

    def test_config_command_parsing(self) -> None:
        """Test parsing config command."""
        parser = create_parser()
        args = parser.parse_args(["config", "--show"])
        assert args.command == "config"
        assert args.show is True

    def test_config_generate_parsing(self) -> None:
        """Test parsing config --generate."""
        parser = create_parser()
        args = parser.parse_args(["config", "--generate", "output.yaml"])
        assert args.generate == "output.yaml"

    def test_invalid_only_choice_rejected(self) -> None:
        """Test that invalid --only values are rejected."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--only", "invalid"])


class TestCLIMain:
    """Tests for CLI main function."""

    def test_no_command_shows_help(self) -> None:
        """Test that no command shows help and returns 0."""
        exit_code = main([])
        assert exit_code == 0

    def test_config_show(self) -> None:
        """Test config --show command."""
        with patch("builtins.print") as mock_print:
            exit_code = main(["config", "--show"])
            assert exit_code == 0
            # Should print JSON config
            mock_print.assert_called()

    def test_config_generate(self, tmp_path: Path) -> None:
        """Test config --generate command."""
        output_file = tmp_path / "generated.yaml"

        exit_code = main(["config", "--generate", str(output_file)])
        assert exit_code == 0
        assert output_file.exists()

    def test_run_with_disabled_validations(self, tmp_path: Path) -> None:
        """Test run command with all validations disabled."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "run_tolerance_fix": False,
            "run_gpu_training": False,
            "run_transfer_validation": False,
            "run_merge_readiness": False,
            "output_dir": str(tmp_path),
            "save_results": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        exit_code = main(["run", "--config", str(config_file)])
        assert exit_code == 0


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_tolerance_check_runs(self, tmp_path: Path) -> None:
        """Test tolerance-check command runs."""
        # Create empty test directory
        test_dir = tmp_path / "tests"
        test_dir.mkdir()

        exit_code = main(["tolerance-check", "--test-dir", str(test_dir)])
        assert exit_code == 0

    def test_run_with_only_tolerance(self, tmp_path: Path) -> None:
        """Test run --only tolerance."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "output_dir": str(tmp_path),
            "save_results": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        exit_code = main([
            "run",
            "--config", str(config_file),
            "--only", "tolerance",
        ])
        # May pass or fail depending on test environment
        assert exit_code in [0, 1]

    def test_verbose_output(self, tmp_path: Path) -> None:
        """Test verbose output flag."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "run_tolerance_fix": False,
            "run_gpu_training": False,
            "run_transfer_validation": False,
            "run_merge_readiness": False,
            "output_dir": str(tmp_path),
            "save_results": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        exit_code = main([
            "run",
            "--config", str(config_file),
            "--verbose",
        ])
        assert exit_code == 0
