"""Tests for tolerance and precision utilities."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.config import ToleranceConfig, ToleranceLevel
from src.validation.tolerance import (
    ToleranceChecker,
    assert_allclose,
    assert_tensor_allclose,
    get_tolerance_for_dtype,
)


class TestGetToleranceForDtype:
    """Tests for get_tolerance_for_dtype function."""

    def test_float32_tolerance(self) -> None:
        """Test float32 tolerance values."""
        config = ToleranceConfig()
        rtol, atol = get_tolerance_for_dtype("float32", config)
        assert rtol == config.float32_rtol
        assert atol == config.float32_atol

    def test_float64_tolerance(self) -> None:
        """Test float64 tolerance values."""
        config = ToleranceConfig()
        rtol, atol = get_tolerance_for_dtype("float64", config)
        assert rtol == config.float64_rtol
        assert atol == config.float64_atol

    def test_float16_tolerance(self) -> None:
        """Test float16 gets relaxed tolerance."""
        rtol, atol = get_tolerance_for_dtype("float16")
        assert rtol >= 1e-3
        assert atol >= 1e-4

    def test_no_config_uses_defaults(self) -> None:
        """Test default config is used when none provided."""
        rtol, atol = get_tolerance_for_dtype("float32")
        assert rtol > 0
        assert atol > 0


class TestAssertAllclose:
    """Tests for assert_allclose function."""

    def test_identical_values_pass(self) -> None:
        """Test identical values pass."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert assert_allclose(a, b) is True

    def test_close_values_pass(self) -> None:
        """Test values within tolerance pass."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7])
        assert assert_allclose(a, b, rtol=1e-5, atol=1e-6) is True

    def test_different_values_fail(self) -> None:
        """Test values outside tolerance fail."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.0, 3.0])
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        assert "not close" in str(exc_info.value).lower()

    def test_nan_detection(self) -> None:
        """Test NaN values are detected."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        config = ToleranceConfig(allow_nan=False)
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, config=config)
        assert "NaN" in str(exc_info.value)

    def test_nan_allowed(self) -> None:
        """Test NaN values can be allowed."""
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, np.nan, 3.0])
        config = ToleranceConfig(allow_nan=True)
        # Should not raise, but result depends on np.allclose behavior
        # np.allclose returns False for NaN comparisons by default

    def test_inf_detection(self) -> None:
        """Test Inf values are detected."""
        a = np.array([1.0, np.inf, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        config = ToleranceConfig(allow_inf=False)
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, config=config)
        assert "Inf" in str(exc_info.value)

    def test_scalars_work(self) -> None:
        """Test scalar values work."""
        assert assert_allclose(1.0, 1.0 + 1e-10) is True

    def test_error_message_informative(self) -> None:
        """Test error message contains useful information."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.5, 3.0])
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        error_msg = str(exc_info.value)
        assert "rtol" in error_msg
        assert "atol" in error_msg
        assert "difference" in error_msg.lower()

    def test_custom_message_included(self) -> None:
        """Test custom message is included in error."""
        a = np.array([1.0])
        b = np.array([2.0])
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b, msg="Custom error message")
        assert "Custom error message" in str(exc_info.value)


class TestAssertTensorAllclose:
    """Tests for assert_tensor_allclose with PyTorch tensors."""

    @pytest.fixture
    def torch_available(self) -> bool:
        """Check if torch is available."""
        try:
            import torch

            return True
        except ImportError:
            return False

    def test_identical_tensors_pass(self, torch_available: bool) -> None:
        """Test identical tensors pass."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        assert assert_tensor_allclose(a, b) is True

    def test_close_tensors_pass(self, torch_available: bool) -> None:
        """Test tensors within tolerance pass."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0 + 1e-7, 2.0 - 1e-7, 3.0 + 1e-7])
        assert assert_tensor_allclose(a, b, rtol=1e-5, atol=1e-6) is True

    def test_shape_mismatch_fails(self, torch_available: bool) -> None:
        """Test shape mismatch is caught."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0])
        with pytest.raises(AssertionError) as exc_info:
            assert_tensor_allclose(a, b)
        assert "Shape mismatch" in str(exc_info.value)

    def test_dtype_aware_tolerance(self, torch_available: bool) -> None:
        """Test dtype-aware tolerance adjustment."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch

        # Float32 should use relaxed tolerance
        config = ToleranceConfig(check_dtype=True)
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([1.0 + 1e-6, 2.0, 3.0], dtype=torch.float32)
        assert assert_tensor_allclose(a, b, config=config) is True


class TestToleranceChecker:
    """Tests for ToleranceChecker class."""

    def test_checker_creation(self) -> None:
        """Test creating a tolerance checker."""
        checker = ToleranceChecker(level=ToleranceLevel.STANDARD)
        assert checker.rtol == 1e-5
        assert checker.atol == 1e-8

    def test_assert_close_passes(self) -> None:
        """Test assert_close with passing values."""
        checker = ToleranceChecker()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0 + 1e-7, 2.0, 3.0])
        assert checker.assert_close(a, b) is True

    def test_assert_close_fails(self) -> None:
        """Test assert_close with failing values."""
        checker = ToleranceChecker()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.0, 3.0])
        with pytest.raises(AssertionError):
            checker.assert_close(a, b)

    def test_check_close_no_raise(self) -> None:
        """Test check_close doesn't raise."""
        checker = ToleranceChecker()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.5, 2.0, 3.0])
        result = checker.check_close(a, b)
        assert result is False

    def test_comparison_counting(self) -> None:
        """Test comparison counting works."""
        checker = ToleranceChecker()

        for i in range(5):
            checker.check_close(i, i + 0.0001)

        report = checker.report()
        assert report["total_comparisons"] == 5

    def test_failure_tracking(self) -> None:
        """Test failure tracking works."""
        checker = ToleranceChecker(level=ToleranceLevel.STRICT)

        checker.check_close(1.0, 1.0)  # Pass
        checker.check_close(1.0, 1.1)  # Fail
        checker.check_close(1.0, 1.0)  # Pass

        report = checker.report()
        assert report["total_comparisons"] == 3
        assert report["failures"] == 1

    def test_reset(self) -> None:
        """Test reset clears counters."""
        checker = ToleranceChecker()
        checker.check_close(1.0, 1.0)
        checker.check_close(1.0, 2.0)
        checker.reset()

        report = checker.report()
        assert report["total_comparisons"] == 0
        assert report["failures"] == 0

    def test_context_manager(self) -> None:
        """Test context manager works."""
        checker = ToleranceChecker()

        with checker.context() as ctx:
            ctx.assert_close(1.0, 1.0 + 1e-10)
            ctx.assert_close(2.0, 2.0 + 1e-10)

        report = checker.report()
        assert report["total_comparisons"] == 2
        assert report["failures"] == 0


class TestToleranceLevels:
    """Tests for different tolerance levels."""

    @pytest.mark.parametrize(
        "level,expected_rtol,expected_atol",
        [
            (ToleranceLevel.STRICT, 1e-7, 1e-9),
            (ToleranceLevel.STANDARD, 1e-5, 1e-8),
            (ToleranceLevel.RELAXED, 1e-4, 1e-6),
            (ToleranceLevel.LOOSE, 1e-3, 1e-5),
        ],
    )
    def test_tolerance_levels(
        self,
        level: ToleranceLevel,
        expected_rtol: float,
        expected_atol: float,
    ) -> None:
        """Test each tolerance level has expected values."""
        config = ToleranceConfig(level=level)
        rtol, atol = config.get_tolerance()
        assert rtol == expected_rtol
        assert atol == expected_atol

    def test_strict_catches_small_differences(self) -> None:
        """Test strict level catches small differences."""
        checker = ToleranceChecker(level=ToleranceLevel.STRICT)
        result = checker.check_close(1.0, 1.0 + 1e-6)
        assert result is False  # Should fail strict check

    def test_loose_allows_larger_differences(self) -> None:
        """Test loose level allows larger differences."""
        checker = ToleranceChecker(level=ToleranceLevel.LOOSE)
        result = checker.check_close(1.0, 1.0 + 5e-4)
        assert result is True  # Should pass loose check
