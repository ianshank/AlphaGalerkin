"""Tests for backend debug utilities."""

from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


from src.backend.debug import (
    assert_dtype,
    assert_finite,
    assert_no_nans,
    assert_shape,
    check_gradient_health,
    log_tensor_stats,
)


class TestAssertShape:
    """Test shape assertion utilities."""

    def test_valid_shape(self):
        x = torch.zeros(2, 3, 4)
        assert_shape(x, (2, 3, 4))  # Should not raise

    def test_invalid_shape(self):
        x = torch.zeros(2, 3, 4)
        with pytest.raises(ValueError, match="size"):
            assert_shape(x, (2, 3, 5))

    def test_wildcard_shape(self):
        x = torch.zeros(2, 3, 4)
        assert_shape(x, (2, -1, 4))  # -1 matches any size

    def test_wrong_ndim(self):
        x = torch.zeros(2, 3)
        with pytest.raises(ValueError, match="dimensions"):
            assert_shape(x, (2, 3, 4))

    def test_named_assertion(self):
        x = torch.zeros(2, 3)
        with pytest.raises(ValueError, match="'my_tensor'"):
            assert_shape(x, (2, 4), name="my_tensor")


class TestAssertDtype:
    """Test dtype assertion."""

    def test_valid_dtype(self):
        x = torch.zeros(2, 3, dtype=torch.float32)
        assert_dtype(x, torch.float32)  # Should not raise

    def test_invalid_dtype(self):
        x = torch.zeros(2, 3, dtype=torch.float32)
        with pytest.raises(ValueError, match="dtype"):
            assert_dtype(x, torch.float64)


class TestAssertNoNans:
    """Test NaN assertion."""

    def test_clean_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert_no_nans(x)  # Should not raise

    def test_nan_tensor(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN"):
            assert_no_nans(x)


class TestAssertFinite:
    """Test finite assertion."""

    def test_clean_tensor(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert_finite(x)  # Should not raise

    def test_nan_tensor(self):
        x = torch.tensor([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN"):
            assert_finite(x)

    def test_inf_tensor(self):
        x = torch.tensor([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="Inf"):
            assert_finite(x)


class TestLogTensorStats:
    """Test tensor stats logging and return value."""

    def test_basic_stats(self):
        import structlog

        log = structlog.get_logger("test")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = log_tensor_stats(log, "test_tensor", x)
        assert stats["shape"] == (4,)
        assert "float" in stats["dtype"]
        assert abs(stats["min"] - 1.0) < 1e-5
        assert abs(stats["max"] - 4.0) < 1e-5
        assert abs(stats["mean"] - 2.5) < 1e-5
        assert stats["has_nan"] is False
        assert stats["has_inf"] is False

    def test_nan_detected(self):
        import structlog

        log = structlog.get_logger("test")
        x = torch.tensor([1.0, float("nan"), 3.0])
        stats = log_tensor_stats(log, "nan_tensor", x)
        assert stats["has_nan"] is True

    def test_log_tensor_stats_runs(self):
        """Just verify it doesn't crash on normal tensors."""
        import structlog

        log = structlog.get_logger("test")
        x = torch.randn(3, 4)
        stats = log_tensor_stats(log, "test_tensor", x)
        assert isinstance(stats, dict)
        assert "shape" in stats


class TestGradientHealth:
    """Test gradient health checking."""

    def test_healthy_grads(self):
        grads = {
            "layer1": torch.randn(3, 4),
            "layer2": torch.randn(
                5,
            ),
        }
        report = check_gradient_health(grads)
        assert report["healthy"] is True
        assert report["num_params"] == 2
        assert report["has_nan"] is False
        assert report["has_inf"] is False
        assert len(report["issues"]) == 0

    def test_nan_grads(self):
        grads = {
            "layer1": torch.tensor([float("nan"), 1.0]),
            "layer2": torch.randn(
                5,
            ),
        }
        report = check_gradient_health(grads)
        assert report["healthy"] is False
        assert report["has_nan"] is True
        assert any("layer1" in issue for issue in report["issues"])

    def test_exploding_grads(self):
        grads = {
            "layer1": torch.randn(3, 4) * 1000,
        }
        report = check_gradient_health(grads, max_norm=10.0)
        assert report["healthy"] is False
        assert len(report["issues"]) > 0
        assert any("exceeds" in issue for issue in report["issues"])

    def test_inf_grads(self):
        grads = {
            "layer1": torch.tensor([float("inf"), 1.0]),
        }
        report = check_gradient_health(grads)
        assert report["healthy"] is False
        assert report["has_inf"] is True

    def test_total_norm_computed(self):
        grads = {
            "layer1": torch.ones(3),
        }
        report = check_gradient_health(grads)
        expected_norm = (3.0) ** 0.5
        assert abs(report["total_norm"] - expected_norm) < 1e-4

    def test_grads_as_list(self):
        grads = [torch.randn(3), torch.randn(4)]
        report = check_gradient_health(grads)
        assert report["num_params"] == 2
        assert report["healthy"] is True

    def test_grads_as_single_tensor(self):
        report = check_gradient_health(torch.randn(5))
        assert report["num_params"] == 1
        assert report["healthy"] is True

    def test_grads_with_none_values(self):
        grads = {"layer1": torch.randn(3), "layer2": None}
        report = check_gradient_health(grads)
        assert report["healthy"] is True

    def test_gradient_health_with_logger(self):
        import structlog

        log = structlog.get_logger("test")
        grads = {"layer1": torch.randn(3, 4)}
        report = check_gradient_health(grads, logger=log)
        assert report["healthy"] is True

    def test_gradient_health_unhealthy_with_logger(self):
        import structlog

        log = structlog.get_logger("test")
        grads = {"layer1": torch.tensor([float("nan")])}
        report = check_gradient_health(grads, logger=log)
        assert report["healthy"] is False

    def test_gradient_explosion_with_logger(self):
        import structlog

        log = structlog.get_logger("test")
        grads = {"layer1": torch.ones(100) * 100}
        report = check_gradient_health(grads, logger=log, max_norm=1.0)
        assert report["healthy"] is False

    def test_empty_grads(self):
        report = check_gradient_health({})
        assert report["healthy"] is True
        assert report["num_params"] == 0


class TestAssertDtypeExtended:
    """Extended dtype assertion tests."""

    def test_matching_dtype_extended(self):
        x = torch.randn(3)
        assert_dtype(x, torch.float32)

    def test_mismatching_dtype_extended(self):
        x = torch.randn(3)
        with pytest.raises(ValueError, match="dtype"):
            assert_dtype(x, torch.float64)

    def test_matching_string_dtype(self):
        x = torch.randn(3)
        assert_dtype(x, "float32")

    def test_named_tensor_dtype(self):
        x = torch.randn(3).to(torch.float64)
        with pytest.raises(ValueError, match="my_tensor"):
            assert_dtype(x, torch.float32, name="my_tensor")


class TestNormalizeDtypeStr:
    """Test _normalize_dtype_str helper."""

    def test_torch_dtype(self):
        from src.backend.debug import _normalize_dtype_str

        result = _normalize_dtype_str(torch.float32)
        assert result == "float32"

    def test_string_dtype(self):
        from src.backend.debug import _normalize_dtype_str

        result = _normalize_dtype_str("float32")
        assert result == "float32"

    def test_torch_prefixed_string(self):
        from src.backend.debug import _normalize_dtype_str

        result = _normalize_dtype_str("torch.float32")
        assert result == "float32"

    def test_numpy_dtype(self):
        import numpy as np

        from src.backend.debug import _normalize_dtype_str

        result = _normalize_dtype_str(np.dtype("float32"))
        assert result == "float32"

    def test_numpy_class(self):
        import numpy as np

        from src.backend.debug import _normalize_dtype_str

        result = _normalize_dtype_str(np.float64)
        assert "float64" in result


class TestWithNumpy:
    """Test debug utilities with numpy arrays."""

    def test_assert_shape_numpy(self):
        import numpy as np

        x = np.ones((3, 4))
        assert_shape(x, (3, 4))

    def test_assert_no_nans_numpy(self):
        import numpy as np

        x = np.ones((3,))
        assert_no_nans(x)

    def test_assert_no_nans_numpy_with_nan(self):
        import numpy as np

        x = np.array([1.0, float("nan")])
        with pytest.raises(ValueError, match="NaN"):
            assert_no_nans(x)

    def test_assert_finite_numpy(self):
        import numpy as np

        x = np.ones((3,))
        assert_finite(x)

    def test_assert_finite_numpy_with_inf(self):
        import numpy as np

        x = np.array([1.0, float("inf")])
        with pytest.raises(ValueError, match="Inf"):
            assert_finite(x)

    def test_log_tensor_stats_numpy(self):
        import numpy as np
        import structlog

        log = structlog.get_logger("test")
        x = np.array([1.0, 2.0, 3.0])
        stats = log_tensor_stats(log, "numpy_tensor", x)
        assert stats["shape"] == (3,)
        assert abs(stats["mean"] - 2.0) < 1e-5

    def test_check_gradient_health_numpy(self):
        import numpy as np

        grads = {"layer1": np.ones(5)}
        report = check_gradient_health(grads)
        assert report["healthy"] is True

    def test_log_tensor_stats_scalar(self):
        import structlog

        log = structlog.get_logger("test")
        stats = log_tensor_stats(log, "scalar", 42)
        assert stats["shape"] == "unknown"
