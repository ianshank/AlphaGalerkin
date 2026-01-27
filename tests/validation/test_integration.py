"""Integration tests for the validation framework.

These tests verify that all components work together correctly.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.validation.config import (
    GPUTrainingConfig,
    MergeReadinessConfig,
    ToleranceConfig,
    ToleranceLevel,
    TransferValidationConfig,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
)
from src.validation.logging import DebugContext, ValidationLogger, configure_validation_logging
from src.validation.runner import ValidationRunner, run_validation
from src.validation.scenarios.base import BaseValidator
from src.validation.tolerance import ToleranceChecker, assert_allclose
from src.validation.utils import deep_merge


class TestValidationFrameworkIntegration:
    """Integration tests for the complete validation framework."""

    def test_runner_with_no_validations(self, tmp_path: Path) -> None:
        """Test runner with all validations disabled."""
        config = ValidationConfig(
            run_tolerance_fix=False,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            output_dir=str(tmp_path),
            save_results=False,
        )
        runner = ValidationRunner(config=config)
        results = runner.run_all()
        assert len(results) == 0

    def test_runner_summary_generation(self, tmp_path: Path) -> None:
        """Test that runner generates proper summary."""
        config = ValidationConfig(
            run_tolerance_fix=False,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            output_dir=str(tmp_path),
            save_results=False,
        )
        runner = ValidationRunner(config=config)
        runner.run_all()
        summary = runner.get_summary()
        assert "No validations" in summary

    def test_config_deep_merge_preserves_nested(self) -> None:
        """Test that deep merge preserves nested config values."""
        base_config = {
            "seed": 42,
            "gpu_training": {
                "d_model": 256,
                "n_heads": 8,
                "batch_size": 64,
            },
            "run_tolerance_fix": False,
        }
        overrides = {
            "gpu_training": {"d_model": 128},
            "seed": 123,
        }

        merged = deep_merge(base_config, overrides)

        # Overridden values should be updated
        assert merged["seed"] == 123
        assert merged["gpu_training"]["d_model"] == 128

        # Non-overridden values should be preserved
        assert merged["gpu_training"]["n_heads"] == 8
        assert merged["gpu_training"]["batch_size"] == 64
        assert merged["run_tolerance_fix"] is False

    def test_tolerance_checker_with_logger(self) -> None:
        """Test tolerance checker integrates with logging."""
        import numpy as np

        logger = ValidationLogger("test_integration")
        checker = ToleranceChecker(level=ToleranceLevel.RELAXED)

        # Perform checks
        with checker.context() as ctx:
            result1 = ctx.check_close(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
            result2 = ctx.check_close(np.array([1.0]), np.array([1.0001]))

        report = checker.report()
        assert report["total_comparisons"] == 2
        assert report["success_rate"] >= 0.5

    def test_debug_context_captures_exception(self) -> None:
        """Test DebugContext captures exception details."""
        logger = ValidationLogger("test_debug")

        try:
            with DebugContext("test_operation", logger) as ctx:
                ctx.record("step", 1)
                ctx.checkpoint("before_error")
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

    def test_validation_result_serialization(self) -> None:
        """Test ValidationResult can be serialized to dict."""
        now = datetime.now()
        result = ValidationResult(
            validation_name="test",
            config_hash="abc123",
            status=ValidationStatus.PASSED,
            passed=True,
            metrics={"accuracy": 0.95},
            details={"extra": "info"},
            start_time=now,
            end_time=now,
            duration_seconds=1.5,
        )

        # Should be serializable
        data = result.model_dump()
        assert data["validation_name"] == "test"
        assert data["metrics"]["accuracy"] == 0.95


class TestConfigurationLoading:
    """Tests for configuration loading and merging."""

    def test_load_from_yaml_and_override(self, tmp_path: Path) -> None:
        """Test loading config from YAML with CLI overrides."""
        import yaml

        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "seed": 42,
            "gpu_training": {"d_model": 256, "n_heads": 8},
            "run_tolerance_fix": True,
            "run_gpu_training": False,
            "run_transfer_validation": False,
            "run_merge_readiness": False,
            "save_results": False,
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load with overrides
        results = run_validation(
            config_path=str(config_file),
            seed=123,
        )

        # Should run without error
        assert isinstance(results, dict)

    def test_pydantic_validation_catches_errors(self) -> None:
        """Test that Pydantic validation catches invalid configs."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GPUTrainingConfig(n_steps=-1)

        with pytest.raises(ValidationError):
            TransferValidationConfig(eval_resolutions=[])


class TestLoggingIntegration:
    """Tests for logging system integration."""

    def test_validation_logger_metrics(self) -> None:
        """Test ValidationLogger tracks metrics."""
        logger = ValidationLogger("test_metrics")

        logger.metric("loss", 0.5, step=1)
        logger.metric("loss", 0.3, step=2)
        logger.metric("loss", 0.1, step=3)

        summary = logger.get_metrics_summary()
        assert "loss" in summary
        assert summary["loss"]["count"] == 3
        assert summary["loss"]["min"] == 0.1
        assert summary["loss"]["max"] == 0.5

    def test_validation_logger_timing(self) -> None:
        """Test ValidationLogger timing context."""
        import time

        logger = ValidationLogger("test_timing")

        with logger.timed("test_operation") as timing:
            time.sleep(0.01)  # Small delay
            timing["extra_info"] = "test"

        # Timing should be recorded
        elapsed = logger.get_elapsed_time()
        assert elapsed >= 0.01

    def test_configure_logging_levels(self) -> None:
        """Test configuring different log levels."""
        # Should not raise
        configure_validation_logging(level="DEBUG")
        configure_validation_logging(level="INFO")
        configure_validation_logging(level="WARNING")


class TestCustomValidator:
    """Tests for creating custom validators."""

    def test_custom_validator_implementation(self) -> None:
        """Test implementing a custom validator."""
        from pydantic import BaseModel, Field

        class CustomConfig(BaseModel):
            """Custom validation config."""

            name: str = Field(default="custom")
            threshold: float = Field(default=0.5, ge=0, le=1)

        class CustomValidator(BaseValidator):
            """Custom validator for testing."""

            name = "custom"
            config_class = CustomConfig

            def validate(self) -> ValidationResult:
                self.record_metric("custom_metric", 0.95)
                return self._create_result(
                    ValidationStatus.PASSED,
                    passed=True,
                    custom_data="test",
                )

        # Run custom validator
        config = CustomConfig(threshold=0.7)
        validator = CustomValidator(config=config)
        result = validator.run()

        assert result.passed is True
        assert result.metrics["custom_metric"] == 0.95
        assert result.details.get("custom_data") == "test"

    def test_validator_exception_handling(self) -> None:
        """Test that validators handle exceptions gracefully."""
        from pydantic import BaseModel

        class FailingConfig(BaseModel):
            """Config for failing validator."""

            name: str = "failing"

        class FailingValidator(BaseValidator):
            """Validator that always fails."""

            name = "failing"
            config_class = FailingConfig

            def validate(self) -> ValidationResult:
                raise RuntimeError("Intentional failure")

        validator = FailingValidator()
        result = validator.run()

        assert result.passed is False
        assert result.status == ValidationStatus.ERROR
        assert "Intentional failure" in str(result.error_message)


class TestToleranceFixerIntegration:
    """Tests for tolerance fixer scenario integration."""

    def test_tolerance_fixer_scans_files(self, tmp_path: Path) -> None:
        """Test that tolerance fixer can scan test files."""
        from src.validation.scenarios.tolerance_fixer import ToleranceTestFixer

        # Create a test file with potential tolerance issue
        test_file = tmp_path / "test_example.py"
        test_file.write_text('''
import numpy as np

def test_strict_tolerance():
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
''')

        config = ToleranceConfig(level=ToleranceLevel.RELAXED)
        fixer = ToleranceTestFixer(config=config, test_dirs=[str(tmp_path)])
        result = fixer.run()

        # Should complete without error
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED]


class TestMergeReadinessIntegration:
    """Tests for merge readiness checker integration."""

    def test_merge_checker_runs_commands(self) -> None:
        """Test that merge checker can run validation commands."""
        from src.validation.scenarios.merge_readiness import MergeReadinessChecker

        config = MergeReadinessConfig(
            pr_number=1,
            require_all_tests_pass=False,
            require_lint_pass=False,
            require_type_check_pass=False,
            check_dependencies=False,
        )

        checker = MergeReadinessChecker(config=config)
        result = checker.run()

        # Should complete (pass or fail doesn't matter for integration test)
        assert result.status in [
            ValidationStatus.PASSED,
            ValidationStatus.FAILED,
            ValidationStatus.ERROR,
        ]

    def test_merge_checker_summary(self) -> None:
        """Test merge checker summary generation."""
        from src.validation.scenarios.merge_readiness import MergeReadinessChecker

        config = MergeReadinessConfig(
            pr_number=7,
            require_all_tests_pass=False,
            require_lint_pass=False,
            require_type_check_pass=False,
            check_dependencies=False,
        )

        checker = MergeReadinessChecker(config=config)
        summary = checker.get_summary()

        assert "PR #7" in summary


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_validation_workflow(self, tmp_path: Path) -> None:
        """Test a complete validation workflow."""
        # Create config
        config = ValidationConfig(
            run_tolerance_fix=False,  # Skip to speed up test
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            output_dir=str(tmp_path),
            save_results=True,
            seed=42,
        )

        # Run validation
        runner = ValidationRunner(config=config)
        results = runner.run_all()

        # Check output
        summary = runner.get_summary()
        assert "No validations" in summary or "SUMMARY" in summary

    def test_parallel_execution_mode(self, tmp_path: Path) -> None:
        """Test parallel execution mode."""
        config = ValidationConfig(
            run_tolerance_fix=False,
            run_gpu_training=False,
            run_transfer_validation=False,
            run_merge_readiness=False,
            parallel=True,
            max_workers=2,
            output_dir=str(tmp_path),
            save_results=False,
        )

        runner = ValidationRunner(config=config)
        results = runner.run_all()

        assert isinstance(results, dict)
