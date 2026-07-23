"""ONNX Runtime inference wrapper for AlphaGalerkin.

This module provides a unified interface for running inference using
ONNX Runtime with various execution providers.

Features:
    - Multiple execution provider support (CPU, CUDA, TensorRT)
    - Batched inference
    - Performance profiling
    - Automatic provider fallback
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from src.deployment.config import ExecutionProvider, RuntimeConfig

if TYPE_CHECKING:
    import onnxruntime as ort

logger = structlog.get_logger(__name__)


@dataclass
class InferenceResult:
    """Result from ONNX Runtime inference."""

    policy: np.ndarray
    value: np.ndarray
    inference_time_ms: float
    provider_used: str


@dataclass
class RuntimeMetrics:
    """Performance metrics from inference."""

    total_inferences: int = 0
    total_time_ms: float = 0.0
    average_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    throughput_per_sec: float = 0.0


class ONNXRuntime:
    """ONNX Runtime inference wrapper.

    Provides a simple interface for running inference on ONNX models
    with automatic execution provider selection and fallback.

    Attributes:
        config: Runtime configuration.
        session: ONNX Runtime inference session.
        provider: Active execution provider.

    """

    def __init__(
        self,
        model_path: str | Path,
        config: RuntimeConfig | None = None,
    ) -> None:
        """Initialize ONNX Runtime.

        Args:
            model_path: Path to ONNX model.
            config: Runtime configuration.

        """
        self.model_path = Path(model_path)
        self.config = config or RuntimeConfig()

        self._session: ort.InferenceSession | None = None
        self._provider: str = ""
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._metrics = RuntimeMetrics()

        self._logger = structlog.get_logger(__name__).bind(
            model_path=str(model_path),
        )

        # Initialize session
        self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize ONNX Runtime inference session."""
        try:
            import onnxruntime as ort

            # Create session options
            sess_options = ort.SessionOptions()

            # Set threading
            if self.config.intra_op_threads > 0:
                sess_options.intra_op_num_threads = self.config.intra_op_threads
            if self.config.inter_op_threads > 0:
                sess_options.inter_op_num_threads = self.config.inter_op_threads

            # Set optimization level
            opt_level_map = {
                "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            }
            sess_options.graph_optimization_level = opt_level_map.get(
                self.config.graph_optimization_level,
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
            )

            # Memory settings
            sess_options.enable_mem_pattern = self.config.enable_mem_pattern
            sess_options.enable_cpu_mem_arena = self.config.enable_cpu_mem_arena

            # Profiling
            if self.config.enable_profiling:
                sess_options.enable_profiling = True
                if self.config.profile_output_path:
                    sess_options.profile_file_prefix = self.config.profile_output_path

            # Log level
            sess_options.log_severity_level = self.config.log_severity_level

            # Try execution providers in order
            providers_to_try = [p.value for p in self.config.execution_providers]
            available_providers = ort.get_available_providers()

            # Filter to available providers
            providers = []
            for provider in providers_to_try:
                if provider in available_providers:
                    providers.append(provider)

            # Add CPU as fallback
            if ExecutionProvider.CPU.value not in providers:
                providers.append(ExecutionProvider.CPU.value)

            # Provider-specific options
            provider_options = []
            for provider in providers:
                if provider == ExecutionProvider.CUDA.value:
                    provider_options.append(
                        {
                            "device_id": self.config.cuda_device_id,
                            "arena_extend_strategy": self.config.cuda_arena_extend_strategy,
                        }
                    )
                    if self.config.cuda_mem_limit > 0:
                        provider_options[-1]["gpu_mem_limit"] = self.config.cuda_mem_limit
                else:
                    provider_options.append({})

            # Create session with provider options
            # ONNX Runtime accepts providers as list of (name, options) tuples
            # or as separate providers and provider_options arguments
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
            )

            # Get active provider
            self._provider = self._session.get_providers()[0]

            # Get input/output names
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]

            self._logger.info(
                "session_initialized",
                provider=self._provider,
                inputs=self._input_names,
                outputs=self._output_names,
            )

        except ImportError as e:
            self._logger.error(
                "onnxruntime_not_available",
                error=str(e),
            )
            raise

    def run(
        self,
        input_data: np.ndarray | dict[str, np.ndarray],
    ) -> InferenceResult:
        """Run inference on input data.

        Args:
            input_data: Input array or dictionary of inputs.

        Returns:
            InferenceResult with outputs and timing.

        """
        if self._session is None:
            raise RuntimeError("ONNX Runtime session not initialized")

        # Prepare inputs
        if isinstance(input_data, dict):
            inputs = input_data
        else:
            inputs = {self._input_names[0]: input_data}

        # Ensure inputs are numpy arrays with correct dtype
        for name in inputs:
            if not isinstance(inputs[name], np.ndarray):
                inputs[name] = np.array(inputs[name])
            if inputs[name].dtype != np.float32:
                inputs[name] = inputs[name].astype(np.float32)

        # Run inference
        start_time = time.perf_counter()
        outputs = self._session.run(self._output_names, inputs)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Update metrics
        self._update_metrics(inference_time)

        # Parse outputs (assuming standard AlphaGalerkin output order)
        policy = outputs[0] if len(outputs) > 0 else np.array([])
        value = outputs[1] if len(outputs) > 1 else np.array([])

        return InferenceResult(
            policy=policy,
            value=value,
            inference_time_ms=inference_time,
            provider_used=self._provider,
        )

    def run_batch(
        self,
        inputs: list[np.ndarray],
    ) -> list[InferenceResult]:
        """Run inference on a batch of inputs.

        Args:
            inputs: List of input arrays.

        Returns:
            List of InferenceResults.

        """
        # Stack inputs into single batch
        batch = np.stack(inputs, axis=0)
        result = self.run(batch)

        # Split results back to individual samples
        results = []
        batch_size = len(inputs)

        for i in range(batch_size):
            results.append(
                InferenceResult(
                    policy=result.policy[i] if len(result.policy) > i else np.array([]),
                    value=result.value[i] if len(result.value) > i else np.array([]),
                    inference_time_ms=result.inference_time_ms / batch_size,
                    provider_used=result.provider_used,
                )
            )

        return results

    def _update_metrics(self, inference_time: float) -> None:
        """Update runtime metrics.

        Args:
            inference_time: Time for this inference in ms.

        """
        self._metrics.total_inferences += 1
        self._metrics.total_time_ms += inference_time
        self._metrics.min_time_ms = min(self._metrics.min_time_ms, inference_time)
        self._metrics.max_time_ms = max(self._metrics.max_time_ms, inference_time)
        self._metrics.average_time_ms = self._metrics.total_time_ms / self._metrics.total_inferences
        self._metrics.throughput_per_sec = (
            1000.0 / self._metrics.average_time_ms if self._metrics.average_time_ms > 0 else 0
        )

    def get_metrics(self) -> RuntimeMetrics:
        """Get runtime metrics.

        Returns:
            Current runtime metrics.

        """
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset runtime metrics."""
        self._metrics = RuntimeMetrics()

    def get_input_info(self) -> list[dict[str, Any]]:
        """Get information about model inputs.

        Returns:
            List of input info dictionaries.

        """
        if self._session is None:
            return []

        info = []
        for inp in self._session.get_inputs():
            info.append(
                {
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type,
                }
            )
        return info

    def get_output_info(self) -> list[dict[str, Any]]:
        """Get information about model outputs.

        Returns:
            List of output info dictionaries.

        """
        if self._session is None:
            return []

        info = []
        for out in self._session.get_outputs():
            info.append(
                {
                    "name": out.name,
                    "shape": out.shape,
                    "type": out.type,
                }
            )
        return info

    def benchmark(
        self,
        input_shape: tuple[int, ...],
        n_warmup: int = 10,
        n_iterations: int = 100,
    ) -> dict[str, float]:
        """Benchmark inference performance.

        Args:
            input_shape: Shape of test input.
            n_warmup: Number of warmup iterations.
            n_iterations: Number of timed iterations.

        Returns:
            Dictionary with benchmark results.

        """
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            self.run(test_input)

        # Reset metrics for benchmark
        self.reset_metrics()

        # Timed iterations
        times = []
        for _ in range(n_iterations):
            result = self.run(test_input)
            times.append(result.inference_time_ms)

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "median_ms": np.median(times),
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "throughput_per_sec": 1000.0 / np.mean(times),
            "provider": self._provider,
        }

    def close(self) -> None:
        """Close the runtime session."""
        self._session = None
        self._logger.info("session_closed")

    def __enter__(self) -> ONNXRuntime:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


def create_runtime(
    model_path: str | Path,
    **kwargs: Any,
) -> ONNXRuntime:
    """Factory function to create ONNX Runtime.

    Args:
        model_path: Path to ONNX model.
        **kwargs: Additional configuration options.

    Returns:
        Configured ONNXRuntime instance.

    """
    config = RuntimeConfig(**kwargs)
    return ONNXRuntime(model_path, config)
