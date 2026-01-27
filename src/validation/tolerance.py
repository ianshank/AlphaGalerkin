"""Tolerance and precision utilities for numerical tests.

This module provides reusable utilities for handling numerical precision
in tests, addressing common tolerance/precision issues.

Design Principles:
    - Dtype-aware: Automatic tolerance adjustment based on tensor dtype
    - Configurable: All thresholds configurable via ToleranceConfig
    - Informative: Detailed error messages with mismatch locations
    - Composable: Works with numpy, torch, and scalar values
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.validation.config import ToleranceConfig, ToleranceLevel

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


def get_tolerance_for_dtype(
    dtype: Any,
    config: ToleranceConfig | None = None,
) -> tuple[float, float]:
    """Get appropriate tolerance values for a given dtype.

    Args:
        dtype: NumPy or PyTorch dtype.
        config: Optional tolerance configuration.

    Returns:
        Tuple of (rtol, atol) appropriate for the dtype.

    Example:
        >>> import torch
        >>> rtol, atol = get_tolerance_for_dtype(torch.float32)
        >>> rtol, atol
        (1e-05, 1e-06)
    """
    if config is None:
        config = ToleranceConfig()

    # Get dtype string for comparison
    dtype_str = str(dtype).lower()

    # Check for float32/float16 (lower precision)
    if "float16" in dtype_str or "half" in dtype_str:
        return (1e-3, 1e-4)
    elif "float32" in dtype_str or "single" in dtype_str:
        if config.check_dtype:
            return (config.float32_rtol, config.float32_atol)
        return config.get_tolerance()
    elif "float64" in dtype_str or "double" in dtype_str:
        if config.check_dtype:
            return (config.float64_rtol, config.float64_atol)
        return config.get_tolerance()
    else:
        # Default tolerance
        return config.get_tolerance()


def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: float | None = None,
    atol: float | None = None,
    config: ToleranceConfig | None = None,
    msg: str = "",
) -> bool:
    """Assert that two values are close within tolerance.

    Works with scalars, numpy arrays, and handles edge cases like NaN/Inf.

    Args:
        actual: Actual value.
        expected: Expected value.
        rtol: Relative tolerance (overrides config).
        atol: Absolute tolerance (overrides config).
        config: Tolerance configuration.
        msg: Additional error message.

    Returns:
        True if values are close.

    Raises:
        AssertionError: If values are not close within tolerance.

    Example:
        >>> assert_allclose(1.0000001, 1.0, rtol=1e-5)
        True
    """
    import numpy as np

    if config is None:
        config = ToleranceConfig()

    # Get tolerances
    if rtol is None or atol is None:
        default_rtol, default_atol = config.get_tolerance()
        rtol = rtol if rtol is not None else default_rtol
        atol = atol if atol is not None else default_atol

    # Convert to numpy for uniform handling
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)

    # Check for NaN/Inf
    if not config.allow_nan:
        if np.any(np.isnan(actual_np)) or np.any(np.isnan(expected_np)):
            raise AssertionError(
                f"NaN values found. {msg}\n"
                f"actual contains NaN: {np.any(np.isnan(actual_np))}\n"
                f"expected contains NaN: {np.any(np.isnan(expected_np))}"
            )

    if not config.allow_inf:
        if np.any(np.isinf(actual_np)) or np.any(np.isinf(expected_np)):
            raise AssertionError(
                f"Inf values found. {msg}\n"
                f"actual contains Inf: {np.any(np.isinf(actual_np))}\n"
                f"expected contains Inf: {np.any(np.isinf(expected_np))}"
            )

    # Perform comparison
    try:
        is_close = np.allclose(
            actual_np,
            expected_np,
            rtol=rtol,
            atol=atol,
            equal_nan=config.allow_nan,
        )
    except TypeError:
        # Handle cases where allclose doesn't work (complex, etc.)
        diff = np.abs(actual_np - expected_np)
        tolerance = atol + rtol * np.abs(expected_np)
        is_close = np.all(diff <= tolerance)

    if not is_close:
        # Generate detailed error message
        diff = np.abs(actual_np - expected_np)
        rel_diff = diff / (np.abs(expected_np) + 1e-12)

        # Find worst mismatches
        flat_diff = diff.flatten()
        flat_rel_diff = rel_diff.flatten()
        n_failures = config.max_failures_to_report

        worst_indices = np.argsort(flat_diff)[-n_failures:][::-1]

        error_lines = [
            f"Values not close within tolerance. {msg}",
            f"rtol={rtol}, atol={atol}",
            f"Shape: {actual_np.shape}",
            f"Max absolute difference: {float(np.max(diff)):.2e}",
            f"Max relative difference: {float(np.max(rel_diff)):.2e}",
            f"Mean absolute difference: {float(np.mean(diff)):.2e}",
            f"Worst mismatches (up to {n_failures}):",
        ]

        flat_actual = actual_np.flatten()
        flat_expected = expected_np.flatten()
        for idx in worst_indices:
            if flat_diff[idx] > atol + rtol * abs(flat_expected[idx]):
                error_lines.append(
                    f"  [{idx}]: actual={flat_actual[idx]:.6e}, "
                    f"expected={flat_expected[idx]:.6e}, "
                    f"diff={flat_diff[idx]:.2e}, "
                    f"rel_diff={flat_rel_diff[idx]:.2e}"
                )

        raise AssertionError("\n".join(error_lines))

    if config.verbose:
        logger.debug(
            "assert_allclose_passed",
            rtol=rtol,
            atol=atol,
            max_diff=float(np.max(np.abs(actual_np - expected_np))),
        )

    return True


def assert_tensor_allclose(
    actual: Any,
    expected: Any,
    rtol: float | None = None,
    atol: float | None = None,
    config: ToleranceConfig | None = None,
    msg: str = "",
    check_device: bool = True,
    check_dtype: bool = True,
) -> bool:
    """Assert that two PyTorch tensors are close within tolerance.

    Handles device placement and dtype-specific tolerances automatically.

    Args:
        actual: Actual tensor.
        expected: Expected tensor.
        rtol: Relative tolerance (overrides config).
        atol: Absolute tolerance (overrides config).
        config: Tolerance configuration.
        msg: Additional error message.
        check_device: Whether to check device matches.
        check_dtype: Whether to check dtype matches.

    Returns:
        True if tensors are close.

    Raises:
        AssertionError: If tensors are not close within tolerance.
        ImportError: If PyTorch is not available.

    Example:
        >>> import torch
        >>> a = torch.tensor([1.0, 2.0, 3.0])
        >>> b = torch.tensor([1.0, 2.0, 3.00001])
        >>> assert_tensor_allclose(a, b)
        True
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is required for assert_tensor_allclose") from e

    if config is None:
        config = ToleranceConfig()

    # Ensure both are tensors
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected)

    # Check shapes
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: actual={actual.shape}, expected={expected.shape}. {msg}"
        )

    # Check device
    if check_device and actual.device != expected.device:
        raise AssertionError(
            f"Device mismatch: actual={actual.device}, expected={expected.device}. {msg}"
        )

    # Check dtype (warn but don't fail)
    if check_dtype and actual.dtype != expected.dtype:
        logger.warning(
            "dtype_mismatch",
            actual_dtype=str(actual.dtype),
            expected_dtype=str(expected.dtype),
        )

    # Get dtype-appropriate tolerances
    if rtol is None or atol is None:
        dtype_rtol, dtype_atol = get_tolerance_for_dtype(actual.dtype, config)
        rtol = rtol if rtol is not None else dtype_rtol
        atol = atol if atol is not None else dtype_atol

    # Move to CPU for comparison
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()

    # Check for NaN/Inf
    if not config.allow_nan:
        if torch.any(torch.isnan(actual_cpu)) or torch.any(torch.isnan(expected_cpu)):
            raise AssertionError(
                f"NaN values found. {msg}\n"
                f"actual contains NaN: {bool(torch.any(torch.isnan(actual_cpu)))}\n"
                f"expected contains NaN: {bool(torch.any(torch.isnan(expected_cpu)))}"
            )

    if not config.allow_inf:
        if torch.any(torch.isinf(actual_cpu)) or torch.any(torch.isinf(expected_cpu)):
            raise AssertionError(
                f"Inf values found. {msg}\n"
                f"actual contains Inf: {bool(torch.any(torch.isinf(actual_cpu)))}\n"
                f"expected contains Inf: {bool(torch.any(torch.isinf(expected_cpu)))}"
            )

    # Perform comparison
    is_close = torch.allclose(
        actual_cpu,
        expected_cpu,
        rtol=rtol,
        atol=atol,
        equal_nan=config.allow_nan,
    )

    if not is_close:
        # Generate detailed error message
        diff = torch.abs(actual_cpu - expected_cpu)
        rel_diff = diff / (torch.abs(expected_cpu) + 1e-12)

        # Find worst mismatches
        flat_diff = diff.flatten()
        n_failures = config.max_failures_to_report

        _, worst_indices = torch.topk(flat_diff, min(n_failures, flat_diff.numel()))

        error_lines = [
            f"Tensors not close within tolerance. {msg}",
            f"rtol={rtol}, atol={atol}",
            f"Shape: {actual.shape}",
            f"Dtype: {actual.dtype}",
            f"Device: {actual.device}",
            f"Max absolute difference: {float(torch.max(diff)):.2e}",
            f"Max relative difference: {float(torch.max(rel_diff)):.2e}",
            f"Mean absolute difference: {float(torch.mean(diff)):.2e}",
            f"Worst mismatches (up to {n_failures}):",
        ]

        flat_actual = actual_cpu.flatten()
        flat_expected = expected_cpu.flatten()
        for idx in worst_indices:
            idx = int(idx)
            if flat_diff[idx] > atol + rtol * abs(float(flat_expected[idx])):
                error_lines.append(
                    f"  [{idx}]: actual={float(flat_actual[idx]):.6e}, "
                    f"expected={float(flat_expected[idx]):.6e}, "
                    f"diff={float(flat_diff[idx]):.2e}, "
                    f"rel_diff={float(rel_diff.flatten()[idx]):.2e}"
                )

        raise AssertionError("\n".join(error_lines))

    if config.verbose:
        logger.debug(
            "assert_tensor_allclose_passed",
            rtol=rtol,
            atol=atol,
            max_diff=float(torch.max(torch.abs(actual_cpu - expected_cpu))),
            dtype=str(actual.dtype),
        )

    return True


class ToleranceChecker:
    """Reusable tolerance checker for test suites.

    Provides a context manager and methods for consistent
    tolerance handling across tests.

    Example:
        >>> from src.validation.tolerance import ToleranceChecker, ToleranceLevel
        >>> checker = ToleranceChecker(level=ToleranceLevel.RELAXED)
        >>> with checker.context():
        ...     checker.assert_close(a, b)
    """

    def __init__(
        self,
        config: ToleranceConfig | None = None,
        level: ToleranceLevel | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> None:
        """Initialize tolerance checker.

        Args:
            config: Full tolerance configuration.
            level: Tolerance level (convenience, ignored if config provided).
            rtol: Override relative tolerance.
            atol: Override absolute tolerance.
        """
        if config is not None:
            self.config = config
        else:
            self.config = ToleranceConfig(
                level=level or ToleranceLevel.STANDARD,
                rtol=rtol,
                atol=atol,
            )

        self._comparison_count = 0
        self._failure_count = 0
        self._failures: list[dict[str, Any]] = []

    @property
    def rtol(self) -> float:
        """Get current relative tolerance."""
        return self.config.get_tolerance()[0]

    @property
    def atol(self) -> float:
        """Get current absolute tolerance."""
        return self.config.get_tolerance()[1]

    def assert_close(
        self,
        actual: Any,
        expected: Any,
        msg: str = "",
        **kwargs: Any,
    ) -> bool:
        """Assert values are close, using configured tolerance.

        Args:
            actual: Actual value.
            expected: Expected value.
            msg: Additional message.
            **kwargs: Additional arguments to pass to assert functions.

        Returns:
            True if close.

        Raises:
            AssertionError: If not close.
        """
        self._comparison_count += 1

        try:
            # Detect if tensor or array
            try:
                import torch

                if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
                    return assert_tensor_allclose(
                        actual, expected, config=self.config, msg=msg, **kwargs
                    )
            except ImportError:
                pass  # PyTorch not installed, fall back to numpy comparison

            return assert_allclose(
                actual, expected, config=self.config, msg=msg, **kwargs
            )
        except AssertionError as e:
            self._failure_count += 1
            self._failures.append({
                "msg": msg,
                "error": str(e),
                "comparison": self._comparison_count,
            })
            raise

    def check_close(
        self,
        actual: Any,
        expected: Any,
        msg: str = "",
        **kwargs: Any,
    ) -> bool:
        """Check if values are close, without raising.

        Args:
            actual: Actual value.
            expected: Expected value.
            msg: Additional message.
            **kwargs: Additional arguments.

        Returns:
            True if close, False otherwise.
        """
        try:
            self.assert_close(actual, expected, msg=msg, **kwargs)
            return True
        except AssertionError:
            return False

    def context(self) -> ToleranceContext:
        """Get a context manager for this checker.

        Returns:
            Context manager that tracks comparisons.
        """
        return ToleranceContext(self)

    def report(self) -> dict[str, Any]:
        """Generate a report of all comparisons.

        Returns:
            Dictionary with comparison statistics.
        """
        return {
            "total_comparisons": self._comparison_count,
            "failures": self._failure_count,
            "success_rate": (
                (self._comparison_count - self._failure_count) / self._comparison_count
                if self._comparison_count > 0
                else 1.0
            ),
            "rtol": self.rtol,
            "atol": self.atol,
            "failure_details": self._failures,
        }

    def reset(self) -> None:
        """Reset comparison counters."""
        self._comparison_count = 0
        self._failure_count = 0
        self._failures = []


class ToleranceContext:
    """Context manager for tolerance checking.

    Tracks comparisons within a context and provides summary.
    """

    def __init__(self, checker: ToleranceChecker) -> None:
        """Initialize context.

        Args:
            checker: Tolerance checker to use.
        """
        self.checker = checker
        self._initial_count = 0
        self._initial_failures = 0

    def __enter__(self) -> ToleranceChecker:
        """Enter context, recording initial state."""
        self._initial_count = self.checker._comparison_count
        self._initial_failures = self.checker._failure_count
        return self.checker

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Exit context, logging summary."""
        comparisons = self.checker._comparison_count - self._initial_count
        failures = self.checker._failure_count - self._initial_failures

        logger.info(
            "tolerance_context_complete",
            comparisons=comparisons,
            failures=failures,
            success_rate=((comparisons - failures) / comparisons if comparisons > 0 else 1.0),
        )

        return False  # Don't suppress exceptions


def create_tolerance_fixtures(level: ToleranceLevel = ToleranceLevel.STANDARD) -> dict[str, Any]:
    """Create pytest fixtures for tolerance testing.

    Returns a dictionary of fixtures that can be registered
    with pytest using pytest_plugins or conftest.py.

    Args:
        level: Default tolerance level.

    Returns:
        Dictionary of fixture functions.

    Example:
        # In conftest.py:
        from src.validation.tolerance import create_tolerance_fixtures
        fixtures = create_tolerance_fixtures()
        tolerance_checker = fixtures['tolerance_checker']
    """
    import pytest

    @pytest.fixture
    def tolerance_config() -> ToleranceConfig:
        """Provide tolerance configuration."""
        return ToleranceConfig(level=level)

    @pytest.fixture
    def tolerance_checker(tolerance_config: ToleranceConfig) -> ToleranceChecker:
        """Provide tolerance checker."""
        return ToleranceChecker(config=tolerance_config)

    @pytest.fixture
    def relaxed_tolerance() -> ToleranceChecker:
        """Provide relaxed tolerance checker for known precision issues."""
        return ToleranceChecker(level=ToleranceLevel.RELAXED)

    @pytest.fixture
    def strict_tolerance() -> ToleranceChecker:
        """Provide strict tolerance checker."""
        return ToleranceChecker(level=ToleranceLevel.STRICT)

    return {
        "tolerance_config": tolerance_config,
        "tolerance_checker": tolerance_checker,
        "relaxed_tolerance": relaxed_tolerance,
        "strict_tolerance": strict_tolerance,
    }
