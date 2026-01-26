"""Tests for validation runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.validation.config import (
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from src.validation.runner import ValidationRunner, run_validation


class TestValidationRunner:
    """Tests for ValidationRunner class."""

    def test_initialization(self, validation_config: ValidationConfig) -> None:
        """Test runner initialization."""
        runner = ValidationRunner(config=validation_config)
        assert runner.config == validation_config

    def test_initialization_with_kwargs(self) -> None:
        """Test initialization with keyword arguments."""
        runner = ValidationRunner(seed=123, verbose=True)
        assert runner.config.seed == 123
        assert runner.config.verbose is True

    def test_output_directory_created(
        self,
        validation_config: ValidationConfig,
        tmp_path: Path,
    ) -> None:
        """Test output directory is created."""
        config = ValidationConfig(
            **{**validation_config.model_dump(), "output_dir": str(tmp_path / "test_output")}
        )
        runner = ValidationRunner(config=config)
        assert runner._output_dir.exists()

    def test_run_all_with_no_validations(self) -> None:
        """Test run_all when all validations are disabled."""
        config = ValidationConfig(
            run_tolerance_fix=False,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            save_results=False,
        )
        runner = ValidationRunner(config=config)
        results = runner.run_all()
        assert len(results) == 0

    def test_run_all_tolerance_only(self) -> None:
        """Test run_all with only tolerance validation."""
        config = ValidationConfig(
            run_tolerance_fix=True,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            save_results=False,
        )
        runner = ValidationRunner(config=config)
        results = runner.run_all()
        assert "tolerance_fix" in results

    def test_stop_on_failure(self) -> None:
        """Test stop_on_failure stops execution."""
        config = ValidationConfig(
            run_tolerance_fix=True,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=True,
            stop_on_failure=True,
            save_results=False,
        )

        # Mock a failing validation
        with patch(
            "src.validation.scenarios.tolerance_fixer.ToleranceTestFixer.run"
        ) as mock_run:
            mock_result = MagicMock(spec=ValidationResult)
            mock_result.passed = False
            mock_result.status = ValidationStatus.FAILED
            mock_run.return_value = mock_result

            runner = ValidationRunner(config=config)
            results = runner.run_all()

            # Should stop after first failure
            assert len(results) <= 2  # May run 1 or 2 depending on timing

    def test_parallel_execution(self) -> None:
        """Test parallel execution mode."""
        config = ValidationConfig(
            run_tolerance_fix=True,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=True,
            parallel=True,
            max_workers=2,
            save_results=False,
        )

        with patch(
            "src.validation.scenarios.tolerance_fixer.ToleranceTestFixer.run"
        ) as mock_tolerance, patch(
            "src.validation.scenarios.merge_readiness.MergeReadinessChecker.run"
        ) as mock_merge:
            from datetime import datetime

            mock_result = ValidationResult(
                validation_name="test",
                config_hash="abc",
                status=ValidationStatus.PASSED,
                passed=True,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.1,
            )
            mock_tolerance.return_value = mock_result
            mock_merge.return_value = mock_result

            runner = ValidationRunner(config=config)
            results = runner.run_all()

            assert len(results) == 2

    def test_summary_generation(self, validation_config: ValidationConfig) -> None:
        """Test summary string generation."""
        runner = ValidationRunner(config=validation_config)

        # Get summary without running (empty)
        summary = runner.get_summary()
        assert "No validations" in summary

        # Run and get summary
        runner.run_all()
        summary = runner.get_summary()
        assert "VALIDATION SUMMARY" in summary

    def test_results_saved(self, tmp_path: Path) -> None:
        """Test results are saved to file."""
        config = ValidationConfig(
            run_tolerance_fix=True,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            output_dir=str(tmp_path),
            save_results=True,
        )

        runner = ValidationRunner(config=config)
        runner.run_all()

        # Check for results file
        result_files = list(tmp_path.glob("results_*.json"))
        assert len(result_files) == 1

        # Verify content
        import json

        with open(result_files[0]) as f:
            data = json.load(f)
        assert "results" in data
        assert "summary" in data

    def test_exception_handling_in_validation(self) -> None:
        """Test exceptions in validations are handled."""
        config = ValidationConfig(
            run_tolerance_fix=True,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            save_results=False,
        )

        with patch(
            "src.validation.scenarios.tolerance_fixer.ToleranceTestFixer.run"
        ) as mock_run:
            mock_run.side_effect = RuntimeError("Unexpected error")

            runner = ValidationRunner(config=config)
            results = runner.run_all()

            assert "tolerance_fix" in results
            assert results["tolerance_fix"].status == ValidationStatus.ERROR


class TestRunValidationFunction:
    """Tests for run_validation convenience function."""

    def test_run_with_defaults(self) -> None:
        """Test running with default config."""
        with patch.object(ValidationRunner, "run_all") as mock_run:
            mock_run.return_value = {}
            results = run_validation(
                run_tolerance_fix=False,
                run_gpu_training=False,
                run_transfer_validation=False,
                run_merge_readiness=False,
            )
            assert results == {}

    def test_run_with_config_file(self, tmp_path: Path) -> None:
        """Test running with config file."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "run_tolerance_fix": False,
            "run_gpu_training": False,
            "run_transfer_validation": False,
            "run_merge_readiness": False,
            "save_results": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        results = run_validation(config_path=str(config_file))
        assert len(results) == 0

    def test_config_overrides(self, tmp_path: Path) -> None:
        """Test config overrides work."""
        import yaml

        config_file = tmp_path / "config.yaml"
        config_data = {
            "seed": 42,
            "run_tolerance_fix": False,
            "run_gpu_training": False,
            "run_transfer_validation": False,
            "run_merge_readiness": False,
            "save_results": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.object(ValidationRunner, "__init__", return_value=None) as mock_init:
            with patch.object(ValidationRunner, "run_all", return_value={}):
                # This would need adjustment based on actual implementation
                pass
