"""Tests for backend-aware structured logging.

Covers BackendLogger, create_backend_logger factory, tensor inspection,
and computation profiling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.backend.logging import BackendLogger, create_backend_logger


class TestBackendLoggerInit:
    """Test BackendLogger initialization with various context combos."""

    def test_minimal_init(self) -> None:
        logger = BackendLogger("test_component")
        assert logger is not None

    def test_full_context(self) -> None:
        logger = BackendLogger(
            "encoder",
            backend_type="torch",
            device="cuda:0",
            precision="float32",
            run_id="run-123",
        )
        assert logger is not None

    def test_partial_context(self) -> None:
        logger = BackendLogger("decoder", backend_type="jax")
        assert logger is not None

    def test_extra_kwargs(self) -> None:
        logger = BackendLogger("layer", custom_key="custom_value")
        assert logger is not None


class TestBackendLoggerTensor:
    """Test log_tensor method."""

    @pytest.fixture
    def logger(self) -> BackendLogger:
        return BackendLogger("test", backend_type="torch")

    def test_log_tensor_numpy(self, logger: BackendLogger) -> None:
        arr = np.random.randn(3, 4).astype(np.float32)
        logger.log_tensor("weights", arr)

    def test_log_tensor_no_shape(self, logger: BackendLogger) -> None:
        logger.log_tensor("scalar", 42)  # type: ignore[arg-type]

    def test_log_tensor_no_dtype(self, logger: BackendLogger) -> None:
        obj = MagicMock(spec=[])
        obj.shape = (3, 4)
        del obj.dtype  # type: ignore[attr-defined]
        logger.log_tensor("weird", obj)

    def test_log_tensor_no_stats(self, logger: BackendLogger) -> None:
        obj = MagicMock(spec=["shape", "dtype"])
        obj.shape = (2,)
        obj.dtype = "float32"
        # min/max/mean will raise AttributeError
        logger.log_tensor("no_stats", obj)

    @pytest.mark.parametrize("level", ["debug", "info", "warning"])
    def test_log_tensor_levels(
        self, logger: BackendLogger, level: str
    ) -> None:
        arr = np.ones((2, 3), dtype=np.float32)
        logger.log_tensor("test_arr", arr, level=level)


class TestBackendLoggerComputation:
    """Test log_computation method."""

    @pytest.fixture
    def logger(self) -> BackendLogger:
        return BackendLogger("test", backend_type="torch")

    def test_log_computation_basic(self, logger: BackendLogger) -> None:
        result = logger.log_computation("add", lambda a, b: a + b, 1, 2)
        assert result == 3

    def test_log_computation_with_arrays(
        self, logger: BackendLogger
    ) -> None:
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((2, 3), dtype=np.float32)
        result = logger.log_computation("matmul", np.add, a, b)
        assert result.shape == (2, 3)

    def test_log_computation_no_shape_result(
        self, logger: BackendLogger
    ) -> None:
        result = logger.log_computation("scalar_fn", lambda: 42)
        assert result == 42

    def test_log_computation_mixed_args(
        self, logger: BackendLogger
    ) -> None:
        arr = np.ones((2,), dtype=np.float32)
        result = logger.log_computation(
            "scale", lambda a, s: a * s, arr, 2.0
        )
        np.testing.assert_allclose(result, [2.0, 2.0])


class TestCreateBackendLogger:
    """Test create_backend_logger factory function."""

    def test_no_config(self) -> None:
        logger = create_backend_logger("test")
        assert isinstance(logger, BackendLogger)

    def test_with_config_enum_attrs(self) -> None:
        config = MagicMock()
        config.backend.value = "torch"
        config.device.value = "cpu"
        config.precision.value = "float32"
        logger = create_backend_logger("test", config=config)
        assert isinstance(logger, BackendLogger)

    def test_with_config_string_attrs(self) -> None:
        """Config with plain string attrs (no .value)."""

        class SimpleConfig:
            backend = "torch"
            device = "cpu"
            precision = "float32"

        logger = create_backend_logger("test", config=SimpleConfig())
        assert isinstance(logger, BackendLogger)

    def test_with_config_no_attrs(self) -> None:
        config = object()
        logger = create_backend_logger("test", config=config)
        assert isinstance(logger, BackendLogger)

    def test_extra_context(self) -> None:
        logger = create_backend_logger(
            "test", experiment="exp1", epoch=10
        )
        assert isinstance(logger, BackendLogger)
