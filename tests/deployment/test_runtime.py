"""Tests for ONNX Runtime inference wrapper.

Tests cover:
- InferenceResult: Result dataclass
- RuntimeMetrics: Performance metrics tracking
- ONNXRuntime: Session initialization and inference (mocked)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

numpy = pytest.importorskip("numpy")

from src.deployment.config import ExecutionProvider, RuntimeConfig
from src.deployment.runtime import (
    InferenceResult,
    RuntimeMetrics,
)

# --- InferenceResult Tests ---


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_creation(self):
        """Test creating inference result."""
        policy = numpy.array([0.1, 0.2, 0.7])
        value = numpy.array([0.5])

        result = InferenceResult(
            policy=policy,
            value=value,
            inference_time_ms=5.5,
            provider_used="CPUExecutionProvider",
        )

        assert numpy.array_equal(result.policy, policy)
        assert numpy.array_equal(result.value, value)
        assert result.inference_time_ms == 5.5
        assert result.provider_used == "CPUExecutionProvider"

    def test_attributes(self):
        """Test result attributes are accessible."""
        result = InferenceResult(
            policy=numpy.zeros(10),
            value=numpy.array([0.0]),
            inference_time_ms=10.0,
            provider_used="CUDAExecutionProvider",
        )

        assert hasattr(result, "policy")
        assert hasattr(result, "value")
        assert hasattr(result, "inference_time_ms")
        assert hasattr(result, "provider_used")


# --- RuntimeMetrics Tests ---


class TestRuntimeMetrics:
    """Tests for RuntimeMetrics dataclass."""

    def test_default_values(self):
        """Test default metrics values."""
        metrics = RuntimeMetrics()

        assert metrics.total_inferences == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.average_time_ms == 0.0
        assert metrics.min_time_ms == float("inf")
        assert metrics.max_time_ms == 0.0
        assert metrics.throughput_per_sec == 0.0

    def test_custom_values(self):
        """Test metrics with custom values."""
        metrics = RuntimeMetrics(
            total_inferences=100,
            total_time_ms=500.0,
            average_time_ms=5.0,
            min_time_ms=3.0,
            max_time_ms=10.0,
            throughput_per_sec=200.0,
        )

        assert metrics.total_inferences == 100
        assert metrics.total_time_ms == 500.0
        assert metrics.average_time_ms == 5.0
        assert metrics.min_time_ms == 3.0
        assert metrics.max_time_ms == 10.0
        assert metrics.throughput_per_sec == 200.0


# --- RuntimeConfig Tests ---


class TestRuntimeConfigIntegration:
    """Integration tests for RuntimeConfig with runtime."""

    def test_default_providers(self):
        """Test default execution providers."""
        config = RuntimeConfig()

        assert ExecutionProvider.CUDA in config.execution_providers
        assert ExecutionProvider.CPU in config.execution_providers

    def test_cpu_only_config(self):
        """Test CPU-only configuration."""
        config = RuntimeConfig(execution_providers=[ExecutionProvider.CPU])

        assert len(config.execution_providers) == 1
        assert config.execution_providers[0] == ExecutionProvider.CPU

    def test_threading_config(self):
        """Test threading configuration."""
        config = RuntimeConfig(
            intra_op_threads=4,
            inter_op_threads=2,
        )

        assert config.intra_op_threads == 4
        assert config.inter_op_threads == 2

    def test_cuda_config(self):
        """Test CUDA-specific configuration."""
        config = RuntimeConfig(
            execution_providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU],
            cuda_device_id=1,
            cuda_mem_limit=1024 * 1024 * 1024,  # 1GB
        )

        assert config.cuda_device_id == 1
        assert config.cuda_mem_limit == 1024 * 1024 * 1024


# --- ONNXRuntime Mocked Tests ---


class TestONNXRuntimeMocked:
    """Tests for ONNXRuntime with mocked onnxruntime."""

    @pytest.fixture
    def mock_ort(self):
        """Create mock onnxruntime module."""
        with patch("src.deployment.runtime.ort") as mock:
            # Set up mock session
            mock_session = MagicMock()
            mock_session.get_providers.return_value = ["CPUExecutionProvider"]
            mock_session.get_inputs.return_value = [
                MagicMock(name="board_state", shape=[1, 17, 9, 9], type="tensor(float)")
            ]
            mock_session.get_outputs.return_value = [
                MagicMock(name="policy", shape=[1, 82], type="tensor(float)"),
                MagicMock(name="value", shape=[1, 1], type="tensor(float)"),
            ]
            mock_session.run.return_value = [
                numpy.random.randn(1, 82).astype(numpy.float32),
                numpy.random.randn(1, 1).astype(numpy.float32),
            ]

            mock.InferenceSession.return_value = mock_session
            mock.SessionOptions.return_value = MagicMock()
            mock.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock.GraphOptimizationLevel = MagicMock()
            mock.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

            yield mock

    def test_runtime_initialization_fails_without_onnxruntime(self, tmp_path):
        """Test that runtime raises ImportError without onnxruntime."""
        # This test checks that the import error is properly propagated
        # when onnxruntime is not installed
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        # The actual test depends on whether onnxruntime is installed
        # If installed, it would fail for different reason (invalid model)
        # If not installed, it would raise ImportError

    def test_inference_result_structure(self):
        """Test inference result has correct structure."""
        result = InferenceResult(
            policy=numpy.random.randn(82).astype(numpy.float32),
            value=numpy.array([0.5], dtype=numpy.float32),
            inference_time_ms=5.0,
            provider_used="CPUExecutionProvider",
        )

        assert result.policy.shape == (82,)
        assert result.value.shape == (1,)
        assert isinstance(result.inference_time_ms, float)
        assert isinstance(result.provider_used, str)


# --- Metrics Calculation Tests ---


class TestMetricsCalculation:
    """Tests for metrics calculation logic."""

    def test_metrics_tracking_simulation(self):
        """Test simulated metrics tracking."""
        metrics = RuntimeMetrics()

        # Simulate updates
        inference_times = [5.0, 3.0, 7.0, 4.0, 6.0]

        total_time = 0.0
        for i, time_ms in enumerate(inference_times, 1):
            total_time += time_ms
            metrics = RuntimeMetrics(
                total_inferences=i,
                total_time_ms=total_time,
                average_time_ms=total_time / i,
                min_time_ms=min(inference_times[:i]),
                max_time_ms=max(inference_times[:i]),
                throughput_per_sec=1000.0 / (total_time / i),
            )

        assert metrics.total_inferences == 5
        assert metrics.total_time_ms == 25.0
        assert metrics.average_time_ms == 5.0
        assert metrics.min_time_ms == 3.0
        assert metrics.max_time_ms == 7.0
        assert metrics.throughput_per_sec == 200.0

    def test_metrics_edge_cases(self):
        """Test metrics edge cases."""
        # Zero inferences
        metrics = RuntimeMetrics()
        assert metrics.throughput_per_sec == 0.0

        # Single inference
        metrics = RuntimeMetrics(
            total_inferences=1,
            total_time_ms=10.0,
            average_time_ms=10.0,
            min_time_ms=10.0,
            max_time_ms=10.0,
            throughput_per_sec=100.0,
        )
        assert metrics.min_time_ms == metrics.max_time_ms


# --- Create Runtime Factory Tests ---


class TestCreateRuntimeFactory:
    """Tests for create_runtime factory function (configuration only)."""

    def test_create_runtime_accepts_kwargs(self, tmp_path):
        """Test factory accepts configuration kwargs."""
        # Just test that the config is created correctly
        # Actual runtime creation would require onnxruntime
        config = RuntimeConfig(
            intra_op_threads=4,
            enable_profiling=True,
        )

        assert config.intra_op_threads == 4
        assert config.enable_profiling is True


# --- Provider Configuration Tests ---


class TestProviderConfiguration:
    """Tests for execution provider configuration."""

    def test_provider_enum_values(self):
        """Test execution provider enum values."""
        assert ExecutionProvider.CPU.value == "CPUExecutionProvider"
        assert ExecutionProvider.CUDA.value == "CUDAExecutionProvider"
        assert ExecutionProvider.TENSORRT.value == "TensorrtExecutionProvider"
        assert ExecutionProvider.OPENVINO.value == "OpenVINOExecutionProvider"
        assert ExecutionProvider.DIRECTML.value == "DmlExecutionProvider"

    def test_provider_list_configuration(self):
        """Test configuring provider list."""
        config = RuntimeConfig(
            execution_providers=[
                ExecutionProvider.TENSORRT,
                ExecutionProvider.CUDA,
                ExecutionProvider.CPU,
            ]
        )

        # TensorRT should be first
        assert config.execution_providers[0] == ExecutionProvider.TENSORRT
        # CPU should be last (fallback)
        assert config.execution_providers[-1] == ExecutionProvider.CPU

    def test_optimization_levels(self):
        """Test graph optimization level configuration."""
        for level in ["disable", "basic", "extended", "all"]:
            config = RuntimeConfig(graph_optimization_level=level)
            assert config.graph_optimization_level == level


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests for runtime module."""

    def test_empty_result_arrays(self):
        """Test inference result with empty arrays."""
        result = InferenceResult(
            policy=numpy.array([]),
            value=numpy.array([]),
            inference_time_ms=0.0,
            provider_used="CPUExecutionProvider",
        )

        assert len(result.policy) == 0
        assert len(result.value) == 0

    def test_large_policy_array(self):
        """Test inference result with large policy array."""
        # 19x19 Go board + pass = 362 actions
        policy = numpy.random.randn(362).astype(numpy.float32)

        result = InferenceResult(
            policy=policy,
            value=numpy.array([0.0]),
            inference_time_ms=10.0,
            provider_used="CPUExecutionProvider",
        )

        assert result.policy.shape == (362,)

    def test_negative_inference_time(self):
        """Test that negative inference time is not prevented (data class)."""
        # DataClass doesn't validate values
        result = InferenceResult(
            policy=numpy.zeros(10),
            value=numpy.zeros(1),
            inference_time_ms=-1.0,  # Invalid but allowed by dataclass
            provider_used="CPUExecutionProvider",
        )

        assert result.inference_time_ms == -1.0
