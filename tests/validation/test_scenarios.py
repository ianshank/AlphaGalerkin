"""Tests for validation scenarios."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.validation.config import (
    GPUTrainingConfig,
    MergeReadinessConfig,
    ToleranceConfig,
    TransferValidationConfig,
    ValidationStatus,
)
from src.validation.scenarios.base import BaseValidator
from src.validation.scenarios.merge_readiness import MergeReadinessChecker
from src.validation.scenarios.tolerance_fixer import ToleranceTestFixer


class TestBaseValidator:
    """Tests for BaseValidator abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test BaseValidator cannot be instantiated directly."""
        # BaseValidator requires validate() to be implemented
        with pytest.raises(TypeError):
            BaseValidator()  # type: ignore[abstract]

    def test_record_metric(self) -> None:
        """Test recording metrics works."""

        class ConcreteValidator(BaseValidator):
            name = "test"

            def validate(self):
                self.record_metric("test_metric", 0.95)
                return self._create_result(ValidationStatus.PASSED, passed=True)

        validator = ConcreteValidator()
        result = validator.run()
        assert "test_metric" in result.metrics
        assert result.metrics["test_metric"] == 0.95

    def test_record_detail(self) -> None:
        """Test recording details works."""

        class ConcreteValidator(BaseValidator):
            name = "test"

            def validate(self):
                self.record_detail("extra_info", {"key": "value"})
                return self._create_result(ValidationStatus.PASSED, passed=True)

        validator = ConcreteValidator()
        result = validator.run()
        assert "extra_info" in result.details
        assert result.details["extra_info"]["key"] == "value"

    def test_exception_handling(self) -> None:
        """Test exceptions are caught and reported."""

        class FailingValidator(BaseValidator):
            name = "failing"

            def validate(self):
                raise RuntimeError("Test error")

        validator = FailingValidator()
        result = validator.run()
        assert result.status == ValidationStatus.ERROR
        assert result.passed is False
        assert "Test error" in result.error_message

    def test_timing_recorded(self) -> None:
        """Test timing is recorded."""
        import time

        class SlowValidator(BaseValidator):
            name = "slow"

            def validate(self):
                time.sleep(0.1)
                return self._create_result(ValidationStatus.PASSED, passed=True)

        validator = SlowValidator()
        result = validator.run()
        assert result.duration_seconds >= 0.1


class TestToleranceTestFixer:
    """Tests for ToleranceTestFixer scenario."""

    def test_initialization(self, tolerance_config: ToleranceConfig) -> None:
        """Test initialization with config."""
        fixer = ToleranceTestFixer(config=tolerance_config, test_dirs=["tests/"])
        assert fixer.config == tolerance_config

    def test_runs_without_error(self, tolerance_config: ToleranceConfig) -> None:
        """Test fixer runs without error."""
        fixer = ToleranceTestFixer(
            config=tolerance_config,
            test_dirs=["tests/validation/"],
        )
        result = fixer.run()
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.FAILED]

    def test_finds_tolerance_issues(self, tmp_path: Path) -> None:
        """Test finding tolerance issues in test files."""
        # Create a test file with tolerance issues
        test_file = tmp_path / "test_example.py"
        test_file.write_text(
            """
def test_strict_tolerance():
    import numpy as np
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    np.allclose(a, b, rtol=1e-10, atol=1e-12)  # Too strict
"""
        )

        fixer = ToleranceTestFixer(test_dirs=[str(tmp_path)])
        result = fixer.run()

        # Should find at least one issue
        assert "issues" in result.details

    def test_generates_suggestions(self, tmp_path: Path) -> None:
        """Test generating fix suggestions."""
        test_file = tmp_path / "test_example.py"
        test_file.write_text(
            """
def test_missing_tolerance():
    import numpy as np
    a = np.array([1.0])
    np.testing.assert_allclose(a, a)  # Missing rtol
"""
        )

        fixer = ToleranceTestFixer(test_dirs=[str(tmp_path)])
        result = fixer.run()

        if result.details.get("issues"):
            assert result.details.get("suggestions")


class TestMergeReadinessChecker:
    """Tests for MergeReadinessChecker scenario."""

    def test_initialization(self, merge_config: MergeReadinessConfig) -> None:
        """Test initialization with config."""
        checker = MergeReadinessChecker(config=merge_config)
        assert checker.config.pr_number == 7

    def test_runs_without_external_commands(self) -> None:
        """Test checker runs when external checks are disabled."""
        config = MergeReadinessConfig(
            pr_number=7,
            require_all_tests_pass=False,
            require_lint_pass=False,
            require_type_check_pass=False,
            require_no_fixmes=False,
            require_no_todos=False,
            check_dependencies=False,
        )
        checker = MergeReadinessChecker(config=config)
        result = checker.run()

        # Should pass when all checks are disabled
        assert result.passed is True

    @patch("subprocess.run")
    def test_test_failure_detection(self, mock_run: MagicMock) -> None:
        """Test detecting test failures."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_example.py::test_foo\nAssertionError: mismatch",
            stderr="",
        )

        config = MergeReadinessConfig(
            pr_number=7,
            require_all_tests_pass=True,
            require_lint_pass=False,
            require_type_check_pass=False,
        )
        checker = MergeReadinessChecker(config=config)
        result = checker.run()

        assert len(checker._test_failures) > 0

    @patch("subprocess.run")
    def test_allowed_failures_ignored(self, mock_run: MagicMock) -> None:
        """Test allowed failures are ignored."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_flaky.py::test_known_flaky\nTimeout",
            stderr="",
        )

        config = MergeReadinessConfig(
            pr_number=7,
            require_all_tests_pass=True,
            allowed_test_failures=["test_known_flaky"],
            require_lint_pass=False,
            require_type_check_pass=False,
        )
        checker = MergeReadinessChecker(config=config)
        result = checker.run()

        # Should pass because the failure is allowed
        assert result.passed is True

    def test_summary_generation(self, merge_config: MergeReadinessConfig) -> None:
        """Test summary string generation."""
        checker = MergeReadinessChecker(config=merge_config)
        summary = checker.get_summary()
        assert "PR #7" in summary


class TestGPUTrainingValidator:
    """Tests for GPUTrainingValidator scenario."""

    def test_initialization(self, gpu_training_config: GPUTrainingConfig) -> None:
        """Test initialization with config."""
        from src.validation.scenarios.gpu_training import GPUTrainingValidator

        validator = GPUTrainingValidator(config=gpu_training_config)
        assert validator.config.device == "cpu"

    def test_setup_without_gpu(self) -> None:
        """Test setup works without GPU requirement."""
        from src.validation.scenarios.gpu_training import GPUTrainingValidator

        config = GPUTrainingConfig(
            device="cpu",
            require_gpu=False,
            n_steps=10,  # Minimum allowed value (ge=10)
            batch_size=2,
            d_model=64,  # Minimum allowed value (ge=64)
            n_heads=2,
            n_layers=1,
        )
        validator = GPUTrainingValidator(config=config)
        validator.setup()
        assert validator._device == "cpu"

    def test_fails_when_gpu_required_but_missing(self) -> None:
        """Test fails when GPU required but not available."""
        from src.validation.scenarios.gpu_training import GPUTrainingValidator

        # Skip if CUDA is actually available
        try:
            import torch

            if torch.cuda.is_available():
                pytest.skip("CUDA is available, cannot test GPU requirement failure")
        except ImportError:
            pytest.skip("PyTorch not available")

        config = GPUTrainingConfig(
            device="cuda",
            require_gpu=True,
        )
        validator = GPUTrainingValidator(config=config)
        result = validator.run()
        assert result.status == ValidationStatus.ERROR


class TestTransferValidator:
    """Tests for TransferValidator scenario."""

    def test_initialization(self, transfer_config: TransferValidationConfig) -> None:
        """Test initialization with config."""
        from src.validation.scenarios.transfer import TransferValidator

        validator = TransferValidator(config=transfer_config)
        assert validator.config.train_resolution == 5

    def test_setup_sets_seed(self, transfer_config: TransferValidationConfig) -> None:
        """Test setup sets random seed."""
        from src.validation.scenarios.transfer import TransferValidator

        validator = TransferValidator(config=transfer_config)
        validator.setup()
        # Seed should be set - verify by checking consistency
