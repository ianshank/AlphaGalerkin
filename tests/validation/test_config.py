"""Tests for validation configuration schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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


class TestToleranceConfig:
    """Tests for ToleranceConfig."""

    def test_default_creation(self) -> None:
        """Test creating config with defaults."""
        config = ToleranceConfig()
        assert config.level == ToleranceLevel.STANDARD
        assert config.check_dtype is True

    def test_preset_tolerances(self) -> None:
        """Test preset tolerance levels."""
        strict = ToleranceConfig(level=ToleranceLevel.STRICT)
        rtol, atol = strict.get_tolerance()
        assert rtol == 1e-7
        assert atol == 1e-9

        relaxed = ToleranceConfig(level=ToleranceLevel.RELAXED)
        rtol, atol = relaxed.get_tolerance()
        assert rtol == 1e-4
        assert atol == 1e-6

    def test_custom_tolerances(self) -> None:
        """Test custom tolerance values."""
        config = ToleranceConfig(rtol=1e-3, atol=1e-5)
        rtol, atol = config.get_tolerance()
        assert rtol == 1e-3
        assert atol == 1e-5

    def test_partial_custom_override(self) -> None:
        """Test partial override of preset."""
        config = ToleranceConfig(level=ToleranceLevel.STANDARD, rtol=1e-3)
        rtol, atol = config.get_tolerance()
        assert rtol == 1e-3
        assert atol == 1e-8  # From preset

    def test_invalid_tolerances_rejected(self) -> None:
        """Test that negative tolerances are rejected."""
        with pytest.raises(ValidationError):
            ToleranceConfig(rtol=-1e-5)

    def test_dtype_specific_tolerances(self) -> None:
        """Test dtype-specific tolerance values."""
        config = ToleranceConfig(
            float32_rtol=1e-4,
            float32_atol=1e-5,
            float64_rtol=1e-8,
            float64_atol=1e-10,
        )
        assert config.float32_rtol == 1e-4
        assert config.float64_rtol == 1e-8


class TestGPUTrainingConfig:
    """Tests for GPUTrainingConfig."""

    def test_default_creation(self) -> None:
        """Test creating config with defaults."""
        config = GPUTrainingConfig()
        assert config.name == "gpu_training"
        assert config.device == "auto"
        assert config.require_gpu is True

    def test_valid_board_sizes(self) -> None:
        """Test valid board sizes."""
        config = GPUTrainingConfig(board_sizes=[9, 13, 19])
        assert config.board_sizes == [9, 13, 19]

    def test_invalid_board_sizes_rejected(self) -> None:
        """Test that invalid board sizes are rejected."""
        with pytest.raises(ValidationError):
            GPUTrainingConfig(board_sizes=[3])  # Too small

        with pytest.raises(ValidationError):
            GPUTrainingConfig(board_sizes=[30])  # Too large

    def test_n_steps_validation(self) -> None:
        """Test n_steps validation."""
        config = GPUTrainingConfig(n_steps=100)
        assert config.n_steps == 100

        with pytest.raises(ValidationError):
            GPUTrainingConfig(n_steps=5)  # Too small (min 10)

    def test_learning_rate_positive(self) -> None:
        """Test learning rate must be positive."""
        with pytest.raises(ValidationError):
            GPUTrainingConfig(learning_rate=0)

        with pytest.raises(ValidationError):
            GPUTrainingConfig(learning_rate=-1e-4)


class TestTransferValidationConfig:
    """Tests for TransferValidationConfig."""

    def test_default_creation(self) -> None:
        """Test creating config with defaults."""
        config = TransferValidationConfig()
        assert config.train_resolution == 9
        assert config.primary_eval_resolution == 19
        assert 19 in config.eval_resolutions

    def test_primary_added_to_eval(self) -> None:
        """Test primary resolution is added to eval list."""
        config = TransferValidationConfig(
            eval_resolutions=[9, 13],
            primary_eval_resolution=19,
        )
        assert 19 in config.eval_resolutions

    def test_eval_resolutions_sorted(self) -> None:
        """Test eval resolutions are sorted."""
        config = TransferValidationConfig(eval_resolutions=[19, 9, 13])
        assert config.eval_resolutions == [9, 13, 19]

    def test_duplicates_removed(self) -> None:
        """Test duplicate resolutions are removed."""
        config = TransferValidationConfig(eval_resolutions=[9, 9, 13, 13, 19])
        assert config.eval_resolutions == [9, 13, 19]

    def test_empty_eval_resolutions_rejected(self) -> None:
        """Test empty eval resolutions are rejected."""
        with pytest.raises(ValidationError):
            TransferValidationConfig(eval_resolutions=[])

    def test_mse_threshold_positive(self) -> None:
        """Test MSE threshold must be positive."""
        with pytest.raises(ValidationError):
            TransferValidationConfig(mse_threshold=0)


class TestMergeReadinessConfig:
    """Tests for MergeReadinessConfig."""

    def test_default_creation(self) -> None:
        """Test creating config with defaults."""
        config = MergeReadinessConfig()
        assert config.pr_number == 7
        assert config.require_all_tests_pass is True

    def test_pr_number_positive(self) -> None:
        """Test PR number must be positive."""
        with pytest.raises(ValidationError):
            MergeReadinessConfig(pr_number=0)

    def test_allowed_test_failures(self) -> None:
        """Test allowed test failures list."""
        config = MergeReadinessConfig(
            allowed_test_failures=["test_known_flaky", "test_platform_specific"]
        )
        assert len(config.allowed_test_failures) == 2


class TestValidationConfig:
    """Tests for ValidationConfig root config."""

    def test_default_creation(self) -> None:
        """Test creating config with defaults."""
        config = ValidationConfig()
        assert config.run_tolerance_fix is True
        assert config.run_gpu_training is True
        assert config.seed == 42

    def test_compute_hash_deterministic(self) -> None:
        """Test config hash is deterministic."""
        config1 = ValidationConfig()
        config2 = ValidationConfig()
        assert config1.compute_hash() == config2.compute_hash()

    def test_different_configs_different_hash(self) -> None:
        """Test different configs have different hashes."""
        config1 = ValidationConfig(seed=42)
        config2 = ValidationConfig(seed=123)
        assert config1.compute_hash() != config2.compute_hash()

    def test_nested_config_access(self) -> None:
        """Test accessing nested configurations."""
        config = ValidationConfig()
        assert config.tolerance.level == ToleranceLevel.STANDARD
        assert config.gpu_training.device == "auto"


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_result_creation(self) -> None:
        """Test creating a validation result."""
        from datetime import datetime

        result = ValidationResult(
            validation_name="test",
            config_hash="abc123",
            status=ValidationStatus.PASSED,
            passed=True,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0,
        )
        assert result.passed is True
        assert result.status == ValidationStatus.PASSED

    def test_summary_generation(self) -> None:
        """Test summary string generation."""
        from datetime import datetime

        result = ValidationResult(
            validation_name="test_validation",
            config_hash="abc123",
            status=ValidationStatus.PASSED,
            passed=True,
            metrics={"accuracy": 0.95, "loss": 0.1},
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=10.5,
        )
        summary = result.summary()
        assert "test_validation" in summary
        assert "[PASS]" in summary
        assert "10.50s" in summary

    def test_failed_result_summary(self) -> None:
        """Test summary for failed result."""
        from datetime import datetime

        result = ValidationResult(
            validation_name="failing_test",
            config_hash="def456",
            status=ValidationStatus.FAILED,
            passed=False,
            error_message="Test failed due to tolerance",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=5.0,
        )
        summary = result.summary()
        assert "[FAIL]" in summary
        assert "tolerance" in summary.lower()
