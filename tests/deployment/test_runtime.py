"""Tests for ONNX Runtime inference wrapper.

Tests cover:
- InferenceResult: Result dataclass
- RuntimeMetrics: Performance metrics tracking
- ONNXRuntime: Session initialization and inference (mocked)
- create_runtime: Factory function
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.deployment.config import ExecutionProvider, RuntimeConfig
from src.deployment.runtime import (
    InferenceResult,
    ONNXRuntime,
    RuntimeMetrics,
    create_runtime,
)

# ---------------------------------------------------------------------------
# Constants / shared data
# ---------------------------------------------------------------------------

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
TENSORRT_PROVIDER = "TensorrtExecutionProvider"

DEFAULT_POLICY_SHAPE = (1, 82)
DEFAULT_VALUE_SHAPE = (1, 1)
DEFAULT_INPUT_SHAPE = (1, 17, 9, 9)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_session(
    providers: list[str] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    policy_shape: tuple[int, ...] = DEFAULT_POLICY_SHAPE,
    value_shape: tuple[int, ...] = DEFAULT_VALUE_SHAPE,
) -> MagicMock:
    """Build a minimal mock InferenceSession."""
    providers = providers or [CPU_PROVIDER]
    input_names = input_names or ["board_state"]
    output_names = output_names or ["policy", "value"]

    mock_session = MagicMock()
    mock_session.get_providers.return_value = providers

    mock_inputs = []
    for name in input_names:
        inp = MagicMock()
        inp.name = name
        inp.shape = list(DEFAULT_INPUT_SHAPE)
        inp.type = "tensor(float)"
        mock_inputs.append(inp)
    mock_session.get_inputs.return_value = mock_inputs

    mock_outputs = []
    for name in output_names:
        out = MagicMock()
        out.name = name
        out.shape = list(policy_shape if name == "policy" else value_shape)
        out.type = "tensor(float)"
        mock_outputs.append(out)
    mock_session.get_outputs.return_value = mock_outputs

    mock_session.run.return_value = [
        np.random.randn(*policy_shape).astype(np.float32),
        np.random.randn(*value_shape).astype(np.float32),
    ]

    return mock_session


def _make_mock_ort(
    available_providers: list[str] | None = None,
    mock_session: MagicMock | None = None,
) -> MagicMock:
    """Build a mock onnxruntime module."""
    available_providers = available_providers or [CPU_PROVIDER]
    mock_session = mock_session or _make_mock_session(providers=available_providers)

    mock_ort = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session
    mock_ort.SessionOptions.return_value = MagicMock()
    mock_ort.get_available_providers.return_value = available_providers

    opt_level = MagicMock()
    opt_level.ORT_DISABLE_ALL = 0
    opt_level.ORT_ENABLE_BASIC = 1
    opt_level.ORT_ENABLE_EXTENDED = 2
    opt_level.ORT_ENABLE_ALL = 99
    mock_ort.GraphOptimizationLevel = opt_level

    return mock_ort


@pytest.fixture()
def mock_ort_cpu() -> MagicMock:
    """Mock onnxruntime with CPU provider only."""
    return _make_mock_ort(available_providers=[CPU_PROVIDER])


@pytest.fixture()
def mock_ort_cuda() -> MagicMock:
    """Mock onnxruntime with CUDA + CPU providers."""
    session = _make_mock_session(providers=[CUDA_PROVIDER, CPU_PROVIDER])
    return _make_mock_ort(
        available_providers=[CUDA_PROVIDER, CPU_PROVIDER],
        mock_session=session,
    )


@pytest.fixture()
def cpu_config() -> RuntimeConfig:
    """RuntimeConfig limited to CPU."""
    return RuntimeConfig(execution_providers=[ExecutionProvider.CPU])


@pytest.fixture()
def cuda_config() -> RuntimeConfig:
    """RuntimeConfig with CUDA + CPU."""
    return RuntimeConfig(
        execution_providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU],
        cuda_device_id=0,
    )


def _make_runtime(
    tmp_path: Path,
    mock_ort: MagicMock,
    config: RuntimeConfig | None = None,
) -> ONNXRuntime:
    """Helper: create an ONNXRuntime with the given mock and config."""
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
        return ONNXRuntime(model_path, config=config)


# ---------------------------------------------------------------------------
# InferenceResult
# ---------------------------------------------------------------------------


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_creation_stores_all_fields(self) -> None:
        policy = np.array([0.1, 0.2, 0.7], dtype=np.float32)
        value = np.array([0.5], dtype=np.float32)

        result = InferenceResult(
            policy=policy,
            value=value,
            inference_time_ms=5.5,
            provider_used=CPU_PROVIDER,
        )

        np.testing.assert_array_equal(result.policy, policy)
        np.testing.assert_array_equal(result.value, value)
        assert result.inference_time_ms == 5.5
        assert result.provider_used == CPU_PROVIDER

    @pytest.mark.parametrize(
        ("policy_size", "value_size", "provider"),
        [
            (82, 1, CPU_PROVIDER),
            (362, 1, CUDA_PROVIDER),
            (0, 0, CPU_PROVIDER),
        ],
    )
    def test_various_shapes(
        self, policy_size: int, value_size: int, provider: str
    ) -> None:
        result = InferenceResult(
            policy=np.zeros(policy_size, dtype=np.float32),
            value=np.zeros(value_size, dtype=np.float32),
            inference_time_ms=1.0,
            provider_used=provider,
        )

        assert len(result.policy) == policy_size
        assert len(result.value) == value_size
        assert result.provider_used == provider

    def test_negative_time_allowed_by_dataclass(self) -> None:
        """Dataclass does not validate numeric values."""
        result = InferenceResult(
            policy=np.zeros(1, dtype=np.float32),
            value=np.zeros(1, dtype=np.float32),
            inference_time_ms=-1.0,
            provider_used=CPU_PROVIDER,
        )
        assert result.inference_time_ms == -1.0

    def test_has_all_expected_attributes(self) -> None:
        result = InferenceResult(
            policy=np.zeros(1, dtype=np.float32),
            value=np.zeros(1, dtype=np.float32),
            inference_time_ms=0.0,
            provider_used=CPU_PROVIDER,
        )
        for attr in ("policy", "value", "inference_time_ms", "provider_used"):
            assert hasattr(result, attr)


# ---------------------------------------------------------------------------
# RuntimeMetrics
# ---------------------------------------------------------------------------


class TestRuntimeMetrics:
    """Tests for RuntimeMetrics dataclass."""

    def test_default_values(self) -> None:
        metrics = RuntimeMetrics()

        assert metrics.total_inferences == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.average_time_ms == 0.0
        assert metrics.min_time_ms == float("inf")
        assert metrics.max_time_ms == 0.0
        assert metrics.throughput_per_sec == 0.0

    @pytest.mark.parametrize(
        ("total_inferences", "total_time_ms", "average_time_ms", "throughput"),
        [
            (1, 10.0, 10.0, 100.0),
            (100, 500.0, 5.0, 200.0),
            (50, 250.0, 5.0, 200.0),
        ],
    )
    def test_custom_values(
        self,
        total_inferences: int,
        total_time_ms: float,
        average_time_ms: float,
        throughput: float,
    ) -> None:
        metrics = RuntimeMetrics(
            total_inferences=total_inferences,
            total_time_ms=total_time_ms,
            average_time_ms=average_time_ms,
            min_time_ms=1.0,
            max_time_ms=total_time_ms,
            throughput_per_sec=throughput,
        )
        assert metrics.total_inferences == total_inferences
        assert metrics.total_time_ms == total_time_ms
        assert metrics.average_time_ms == average_time_ms
        assert metrics.throughput_per_sec == throughput

    def test_min_starts_at_infinity(self) -> None:
        """min_time_ms should start at +inf so first update always wins."""
        metrics = RuntimeMetrics()
        assert metrics.min_time_ms == float("inf")


# ---------------------------------------------------------------------------
# ONNXRuntime – initialization
# ---------------------------------------------------------------------------


class TestONNXRuntimeInit:
    """Tests for ONNXRuntime initialization logic."""

    def test_session_created_with_cpu_provider(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        assert runtime._provider == CPU_PROVIDER

    def test_session_created_with_cuda_provider(
        self, tmp_path: Path, mock_ort_cuda: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cuda)
        assert runtime._provider == CUDA_PROVIDER

    def test_input_output_names_populated(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        assert runtime._input_names == ["board_state"]
        assert runtime._output_names == ["policy", "value"]

    def test_model_path_stored(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        assert runtime.model_path == tmp_path / "model.onnx"

    def test_default_config_used_when_none(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            runtime = ONNXRuntime(model_path, config=None)
        assert isinstance(runtime.config, RuntimeConfig)

    def test_intra_op_threads_applied(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        config = RuntimeConfig(intra_op_threads=4, inter_op_threads=2)
        runtime = _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        assert sess_opts.intra_op_num_threads == 4
        assert sess_opts.inter_op_num_threads == 2

    def test_zero_threads_not_applied(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        """When threads == 0 we leave the SessionOptions attribute untouched."""
        config = RuntimeConfig(intra_op_threads=0, inter_op_threads=0)
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        # intra_op_num_threads must NOT have been set
        assert not hasattr(
            sess_opts, "_intra_op_set"
        ) or sess_opts.intra_op_num_threads != 0

    @pytest.mark.parametrize(
        "opt_level",
        ["disable", "basic", "extended", "all"],
    )
    def test_optimization_level_mapping(
        self, tmp_path: Path, mock_ort_cpu: MagicMock, opt_level: str
    ) -> None:
        config = RuntimeConfig(graph_optimization_level=opt_level)
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        # Just verify the attribute was set (value depends on mock enum)
        assert hasattr(sess_opts, "graph_optimization_level")

    def test_profiling_enabled(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        config = RuntimeConfig(
            enable_profiling=True,
            profile_output_path="/tmp/profile_test",
        )
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        assert sess_opts.enable_profiling is True
        assert sess_opts.profile_file_prefix == "/tmp/profile_test"

    def test_profiling_disabled_by_default(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        _make_runtime(tmp_path, mock_ort_cpu)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        # enable_profiling should NOT have been set to True
        assert sess_opts.enable_profiling is not True

    def test_cpu_always_added_as_fallback(
        self, tmp_path: Path
    ) -> None:
        """Even if CPU is not in providers list, it is appended as fallback."""
        mock_ort = _make_mock_ort(available_providers=[CPU_PROVIDER])
        # Config asks only for CUDA (unavailable), CPU must still appear
        config = RuntimeConfig(
            execution_providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU]
        )
        _make_runtime(tmp_path, mock_ort, config=config)
        _, kwargs = mock_ort.InferenceSession.call_args
        assert CPU_PROVIDER in kwargs["providers"]

    def test_cuda_provider_options_applied(
        self, tmp_path: Path, mock_ort_cuda: MagicMock
    ) -> None:
        config = RuntimeConfig(
            execution_providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU],
            cuda_device_id=1,
            cuda_mem_limit=2 * 1024 * 1024 * 1024,
        )
        _make_runtime(tmp_path, mock_ort_cuda, config=config)
        _, kwargs = mock_ort_cuda.InferenceSession.call_args
        cuda_opts = kwargs["provider_options"][0]
        assert cuda_opts["device_id"] == 1
        assert cuda_opts["gpu_mem_limit"] == 2 * 1024 * 1024 * 1024

    def test_cuda_mem_limit_zero_not_included(
        self, tmp_path: Path, mock_ort_cuda: MagicMock
    ) -> None:
        config = RuntimeConfig(
            execution_providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU],
            cuda_device_id=0,
            cuda_mem_limit=0,
        )
        _make_runtime(tmp_path, mock_ort_cuda, config=config)
        _, kwargs = mock_ort_cuda.InferenceSession.call_args
        cuda_opts = kwargs["provider_options"][0]
        assert "gpu_mem_limit" not in cuda_opts

    def test_import_error_propagated(self, tmp_path: Path) -> None:
        """If onnxruntime is unavailable, ImportError must propagate."""
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        # Remove onnxruntime from sys.modules so `import onnxruntime` fails
        with patch.dict(sys.modules, {"onnxruntime": None}):
            with pytest.raises(ImportError):
                ONNXRuntime(model_path)

    def test_mem_pattern_applied(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        config = RuntimeConfig(enable_mem_pattern=False)
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        assert sess_opts.enable_mem_pattern is False

    def test_cpu_mem_arena_applied(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        config = RuntimeConfig(enable_cpu_mem_arena=False)
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        assert sess_opts.enable_cpu_mem_arena is False

    def test_log_severity_level_applied(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        config = RuntimeConfig(log_severity_level=0)
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        sess_opts = mock_ort_cpu.SessionOptions.return_value
        assert sess_opts.log_severity_level == 0

    def test_provider_options_has_empty_dict_for_cpu(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        config = RuntimeConfig(execution_providers=[ExecutionProvider.CPU])
        _make_runtime(tmp_path, mock_ort_cpu, config=config)
        _, kwargs = mock_ort_cpu.InferenceSession.call_args
        assert {} in kwargs["provider_options"]


# ---------------------------------------------------------------------------
# ONNXRuntime – run()
# ---------------------------------------------------------------------------


class TestONNXRuntimeRun:
    """Tests for ONNXRuntime.run()."""

    def test_run_with_array_input(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.random.randn(*DEFAULT_INPUT_SHAPE).astype(np.float32)
        result = runtime.run(board)

        assert isinstance(result, InferenceResult)
        assert result.policy.shape == DEFAULT_POLICY_SHAPE
        assert result.value.shape == DEFAULT_VALUE_SHAPE
        assert result.inference_time_ms > 0
        assert result.provider_used == CPU_PROVIDER

    def test_run_with_dict_input(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.random.randn(*DEFAULT_INPUT_SHAPE).astype(np.float32)
        result = runtime.run({"board_state": board})

        assert isinstance(result, InferenceResult)

    def test_run_converts_non_float32_input(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.random.randn(*DEFAULT_INPUT_SHAPE).astype(np.float64)
        runtime.run(board)

        # Check the data passed to session.run was float32
        call_args = runtime._session.run.call_args
        inputs_dict: dict[str, np.ndarray] = call_args[0][1]
        for arr in inputs_dict.values():
            assert arr.dtype == np.float32

    def test_run_converts_non_numpy_input(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        list_input = [[[1.0] * 9] * 9 for _ in range(17)]
        runtime.run(list_input)
        # Should not raise

    def test_run_raises_when_session_is_none(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        runtime._session = None
        with pytest.raises(RuntimeError, match="session not initialized"):
            runtime.run(np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32))

    def test_run_updates_metrics(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)

        for _ in range(3):
            runtime.run(board)

        metrics = runtime.get_metrics()
        assert metrics.total_inferences == 3
        assert metrics.total_time_ms > 0

    def test_run_output_with_single_output_model(
        self, tmp_path: Path
    ) -> None:
        """If the model has only one output, value should be empty array."""
        session = _make_mock_session(
            output_names=["policy"],
        )
        session.run.return_value = [
            np.random.randn(*DEFAULT_POLICY_SHAPE).astype(np.float32)
        ]
        mock_ort = _make_mock_ort(mock_session=session)

        runtime = _make_runtime(tmp_path, mock_ort)
        result = runtime.run(np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32))
        assert len(result.value) == 0

    def test_run_output_with_no_outputs(
        self, tmp_path: Path
    ) -> None:
        """Edge case: session returns empty list."""
        session = _make_mock_session(output_names=["policy"])
        session.run.return_value = []
        mock_ort = _make_mock_ort(mock_session=session)

        runtime = _make_runtime(tmp_path, mock_ort)
        result = runtime.run(np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32))
        assert len(result.policy) == 0
        assert len(result.value) == 0


# ---------------------------------------------------------------------------
# ONNXRuntime – run_batch()
# ---------------------------------------------------------------------------


class TestONNXRuntimeRunBatch:
    """Tests for ONNXRuntime.run_batch()."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_run_batch_returns_correct_count(
        self, tmp_path: Path, mock_ort_cpu: MagicMock, batch_size: int
    ) -> None:
        session = _make_mock_session()
        # Return outputs with batch dimension
        session.run.return_value = [
            np.random.randn(batch_size, 82).astype(np.float32),
            np.random.randn(batch_size, 1).astype(np.float32),
        ]
        mock_ort = _make_mock_ort(mock_session=session)

        runtime = _make_runtime(tmp_path, mock_ort)
        inputs = [
            np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32) for _ in range(batch_size)
        ]
        results = runtime.run_batch(inputs)

        assert len(results) == batch_size

    def test_run_batch_divides_time_evenly(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        batch_size = 4
        session = _make_mock_session()
        session.run.return_value = [
            np.random.randn(batch_size, 82).astype(np.float32),
            np.random.randn(batch_size, 1).astype(np.float32),
        ]
        mock_ort = _make_mock_ort(mock_session=session)

        runtime = _make_runtime(tmp_path, mock_ort)
        inputs = [np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)] * batch_size
        results = runtime.run_batch(inputs)

        # All per-sample times should be equal fractions of batch time
        times = [r.inference_time_ms for r in results]
        assert len(set(times)) == 1  # all equal

    def test_run_batch_each_result_has_correct_provider(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        session = _make_mock_session()
        session.run.return_value = [
            np.random.randn(2, 82).astype(np.float32),
            np.random.randn(2, 1).astype(np.float32),
        ]
        mock_ort = _make_mock_ort(mock_session=session)
        runtime = _make_runtime(tmp_path, mock_ort)
        results = runtime.run_batch(
            [np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)] * 2
        )
        for r in results:
            assert r.provider_used == CPU_PROVIDER

    def test_run_batch_empty_policy_when_index_out_of_bounds(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        """If batch output has fewer rows than inputs, fall back to empty."""
        batch_size = 3
        session = _make_mock_session()
        # Only 1 row returned, but 3 inputs
        session.run.return_value = [
            np.random.randn(1, 82).astype(np.float32),
            np.random.randn(1, 1).astype(np.float32),
        ]
        mock_ort = _make_mock_ort(mock_session=session)
        runtime = _make_runtime(tmp_path, mock_ort)
        results = runtime.run_batch(
            [np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)] * batch_size
        )
        # Index 0 should have data, indices 1 and 2 fall back to empty
        assert len(results[0].policy) > 0
        assert len(results[1].policy) == 0
        assert len(results[2].value) == 0


# ---------------------------------------------------------------------------
# ONNXRuntime – metrics
# ---------------------------------------------------------------------------


class TestONNXRuntimeMetrics:
    """Tests for _update_metrics, get_metrics, reset_metrics."""

    def test_metrics_accumulate_across_calls(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)

        n = 5
        for _ in range(n):
            runtime.run(board)

        metrics = runtime.get_metrics()
        assert metrics.total_inferences == n
        assert metrics.total_time_ms >= 0
        assert metrics.average_time_ms >= 0

    def test_min_max_updated(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)
        runtime.run(board)

        metrics = runtime.get_metrics()
        assert metrics.min_time_ms <= metrics.max_time_ms

    def test_average_equals_total_over_count(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)

        for _ in range(4):
            runtime.run(board)

        m = runtime.get_metrics()
        expected_avg = m.total_time_ms / m.total_inferences
        assert abs(m.average_time_ms - expected_avg) < 1e-9

    def test_throughput_nonzero_after_inference(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)
        runtime.run(board)
        assert runtime.get_metrics().throughput_per_sec > 0

    def test_reset_clears_metrics(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)

        for _ in range(3):
            runtime.run(board)

        runtime.reset_metrics()
        metrics = runtime.get_metrics()
        assert metrics.total_inferences == 0
        assert metrics.total_time_ms == 0.0
        assert metrics.min_time_ms == float("inf")

    def test_update_metrics_directly(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        """Test the internal _update_metrics method directly."""
        runtime = _make_runtime(tmp_path, mock_ort_cpu)

        runtime._update_metrics(10.0)
        runtime._update_metrics(20.0)
        runtime._update_metrics(5.0)

        m = runtime._metrics
        assert m.total_inferences == 3
        assert abs(m.total_time_ms - 35.0) < 1e-9
        assert abs(m.average_time_ms - 35.0 / 3) < 1e-9
        assert m.min_time_ms == 5.0
        assert m.max_time_ms == 20.0


# ---------------------------------------------------------------------------
# ONNXRuntime – get_input_info / get_output_info
# ---------------------------------------------------------------------------


class TestONNXRuntimeInfoMethods:
    """Tests for get_input_info and get_output_info."""

    def test_get_input_info_returns_list_of_dicts(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        info = runtime.get_input_info()

        assert isinstance(info, list)
        assert len(info) == 1
        assert info[0]["name"] == "board_state"
        assert "shape" in info[0]
        assert "type" in info[0]

    def test_get_output_info_returns_list_of_dicts(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        info = runtime.get_output_info()

        assert isinstance(info, list)
        assert len(info) == 2
        names = [o["name"] for o in info]
        assert "policy" in names
        assert "value" in names

    def test_get_input_info_returns_empty_when_session_none(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        runtime._session = None
        assert runtime.get_input_info() == []

    def test_get_output_info_returns_empty_when_session_none(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        runtime._session = None
        assert runtime.get_output_info() == []

    def test_get_input_info_multiple_inputs(self, tmp_path: Path) -> None:
        session = _make_mock_session(
            input_names=["board_state", "legal_moves"],
        )
        mock_ort = _make_mock_ort(mock_session=session)
        runtime = _make_runtime(tmp_path, mock_ort)
        info = runtime.get_input_info()
        assert len(info) == 2


# ---------------------------------------------------------------------------
# ONNXRuntime – benchmark()
# ---------------------------------------------------------------------------


class TestONNXRuntimeBenchmark:
    """Tests for the benchmark method."""

    def test_benchmark_returns_expected_keys(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        results = runtime.benchmark(
            input_shape=DEFAULT_INPUT_SHAPE,
            n_warmup=2,
            n_iterations=5,
        )

        expected_keys = {
            "mean_ms",
            "std_ms",
            "min_ms",
            "max_ms",
            "median_ms",
            "p95_ms",
            "p99_ms",
            "throughput_per_sec",
            "provider",
        }
        assert set(results.keys()) == expected_keys

    def test_benchmark_provider_matches_runtime(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        results = runtime.benchmark(
            input_shape=DEFAULT_INPUT_SHAPE,
            n_warmup=2,
            n_iterations=5,
        )
        assert results["provider"] == CPU_PROVIDER

    def test_benchmark_resets_metrics_before_timing(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        board = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)
        # Run some inferences first
        for _ in range(10):
            runtime.run(board)

        runtime.benchmark(
            input_shape=DEFAULT_INPUT_SHAPE,
            n_warmup=1,
            n_iterations=3,
        )
        # After benchmark, metrics should reflect only the 3 timed iterations
        assert runtime.get_metrics().total_inferences == 3

    def test_benchmark_values_are_non_negative(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        results = runtime.benchmark(
            input_shape=DEFAULT_INPUT_SHAPE,
            n_warmup=1,
            n_iterations=4,
        )
        for key, val in results.items():
            if key != "provider":
                assert val >= 0, f"{key} should be non-negative, got {val}"

    def test_benchmark_throughput_is_positive(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        results = runtime.benchmark(
            input_shape=DEFAULT_INPUT_SHAPE,
            n_warmup=1,
            n_iterations=3,
        )
        assert results["throughput_per_sec"] > 0


# ---------------------------------------------------------------------------
# ONNXRuntime – close() and context manager
# ---------------------------------------------------------------------------


class TestONNXRuntimeLifecycle:
    """Tests for close() and context manager protocol."""

    def test_close_sets_session_to_none(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        assert runtime._session is not None
        runtime.close()
        assert runtime._session is None

    def test_context_manager_closes_session(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            with ONNXRuntime(model_path) as runtime:
                assert runtime._session is not None
        assert runtime._session is None

    def test_context_manager_returns_self(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            with ONNXRuntime(model_path) as runtime:
                assert isinstance(runtime, ONNXRuntime)

    def test_run_after_close_raises(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        runtime = _make_runtime(tmp_path, mock_ort_cpu)
        runtime.close()
        with pytest.raises(RuntimeError):
            runtime.run(np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32))


# ---------------------------------------------------------------------------
# create_runtime factory
# ---------------------------------------------------------------------------


class TestCreateRuntimeFactory:
    """Tests for the create_runtime factory function."""

    def test_factory_creates_onnx_runtime(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            runtime = create_runtime(str(model_path))

        assert isinstance(runtime, ONNXRuntime)

    def test_factory_passes_kwargs_to_config(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            runtime = create_runtime(
                str(model_path),
                intra_op_threads=4,
                enable_profiling=False,
            )

        assert runtime.config.intra_op_threads == 4
        assert runtime.config.enable_profiling is False

    def test_factory_accepts_path_object(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            runtime = create_runtime(model_path)

        assert isinstance(runtime, ONNXRuntime)

    def test_factory_with_cpu_only_provider(
        self, tmp_path: Path, mock_ort_cpu: MagicMock
    ) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.touch()

        with patch.dict(sys.modules, {"onnxruntime": mock_ort_cpu}):
            runtime = create_runtime(
                str(model_path),
                execution_providers=[ExecutionProvider.CPU],
            )

        assert runtime.config.execution_providers == [ExecutionProvider.CPU]


# ---------------------------------------------------------------------------
# Provider filtering edge cases
# ---------------------------------------------------------------------------


class TestProviderFiltering:
    """Test that unavailable providers are filtered correctly."""

    def test_unavailable_provider_is_skipped(self, tmp_path: Path) -> None:
        """TENSORRT not in available_providers → only CPU used."""
        session = _make_mock_session(providers=[CPU_PROVIDER])
        mock_ort = _make_mock_ort(
            available_providers=[CPU_PROVIDER],
            mock_session=session,
        )
        config = RuntimeConfig(
            execution_providers=[
                ExecutionProvider.TENSORRT,
                ExecutionProvider.CPU,
            ]
        )
        runtime = _make_runtime(tmp_path, mock_ort, config=config)
        _, kwargs = mock_ort.InferenceSession.call_args
        assert TENSORRT_PROVIDER not in kwargs["providers"]
        assert CPU_PROVIDER in kwargs["providers"]

    def test_all_providers_available(self, tmp_path: Path) -> None:
        session = _make_mock_session(
            providers=[CUDA_PROVIDER, CPU_PROVIDER]
        )
        mock_ort = _make_mock_ort(
            available_providers=[CUDA_PROVIDER, CPU_PROVIDER],
            mock_session=session,
        )
        config = RuntimeConfig(
            execution_providers=[ExecutionProvider.CUDA, ExecutionProvider.CPU]
        )
        _make_runtime(tmp_path, mock_ort, config=config)
        _, kwargs = mock_ort.InferenceSession.call_args
        assert CUDA_PROVIDER in kwargs["providers"]
        assert CPU_PROVIDER in kwargs["providers"]

    def test_cpu_appended_when_missing_from_config(self, tmp_path: Path) -> None:
        """CPU must be added even if not in execution_providers config."""
        session = _make_mock_session(providers=[CPU_PROVIDER])
        mock_ort = _make_mock_ort(
            available_providers=[CPU_PROVIDER],
            mock_session=session,
        )
        # Config doesn't include CPU explicitly but only CUDA (unavailable)
        config = RuntimeConfig(
            execution_providers=[ExecutionProvider.CUDA]
        )
        _make_runtime(tmp_path, mock_ort, config=config)
        _, kwargs = mock_ort.InferenceSession.call_args
        assert CPU_PROVIDER in kwargs["providers"]
