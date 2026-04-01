"""Tests for model validation utilities.

Covers ValidationResult dataclass and ModelValidator initialization.
"""

from __future__ import annotations

from src.deployment.validate import ModelValidator, ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation(self) -> None:
        """Can create with all required fields."""
        result = ValidationResult(
            passed=True,
            max_policy_diff=1e-6,
            max_value_diff=1e-7,
            mean_policy_diff=1e-7,
            mean_value_diff=1e-8,
            pytorch_time_ms=10.0,
            onnx_time_ms=5.0,
            speedup_ratio=2.0,
            n_samples_tested=100,
            failed_samples=0,
        )
        assert result.passed is True
        assert result.max_policy_diff == 1e-6
        assert result.n_samples_tested == 100
        assert result.failed_samples == 0

    def test_default_error_message(self) -> None:
        """Default error_message is None."""
        result = ValidationResult(
            passed=True,
            max_policy_diff=0.0,
            max_value_diff=0.0,
            mean_policy_diff=0.0,
            mean_value_diff=0.0,
            pytorch_time_ms=1.0,
            onnx_time_ms=1.0,
            speedup_ratio=1.0,
            n_samples_tested=1,
            failed_samples=0,
        )
        assert result.error_message is None

    def test_with_error_message(self) -> None:
        """Can include error message."""
        result = ValidationResult(
            passed=False,
            max_policy_diff=0.1,
            max_value_diff=0.2,
            mean_policy_diff=0.05,
            mean_value_diff=0.1,
            pytorch_time_ms=10.0,
            onnx_time_ms=5.0,
            speedup_ratio=2.0,
            n_samples_tested=50,
            failed_samples=10,
            error_message="Output difference exceeds tolerance",
        )
        assert result.passed is False
        assert result.failed_samples == 10
        assert "tolerance" in result.error_message

    def test_speedup_ratio(self) -> None:
        """Speedup ratio reflects performance comparison."""
        result = ValidationResult(
            passed=True,
            max_policy_diff=0.0,
            max_value_diff=0.0,
            mean_policy_diff=0.0,
            mean_value_diff=0.0,
            pytorch_time_ms=20.0,
            onnx_time_ms=5.0,
            speedup_ratio=4.0,
            n_samples_tested=10,
            failed_samples=0,
        )
        assert result.speedup_ratio == 4.0


class TestModelValidator:
    """Tests for ModelValidator class."""

    def test_init_defaults(self) -> None:
        """Initializes with default tolerances."""
        validator = ModelValidator()
        assert validator.tolerance == 1e-5
        assert validator.relative_tolerance == 1e-4

    def test_init_custom_tolerances(self) -> None:
        """Initializes with custom tolerances."""
        validator = ModelValidator(tolerance=1e-3, relative_tolerance=1e-2)
        assert validator.tolerance == 1e-3
        assert validator.relative_tolerance == 1e-2

    def test_multiple_validators(self) -> None:
        """Multiple validators can have different tolerances."""
        v1 = ModelValidator(tolerance=1e-3)
        v2 = ModelValidator(tolerance=1e-6)
        assert v1.tolerance == 1e-3
        assert v2.tolerance == 1e-6
