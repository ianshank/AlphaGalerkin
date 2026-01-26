"""Computational complexity benchmark scenario.

This scenario validates the complexity claims:
    - O(N) for Galerkin attention
    - O(N log N) for FNet mixing
    - O(N²) for standard softmax attention (baseline)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
import torch

from src.poc.config import (
    ComplexityScenarioConfig,
    ScenarioResult,
    ScenarioStatus,
)
from src.poc.logging import ScenarioLogger
from src.poc.registry import BaseScenario, scenario

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    n_tokens: int
    mean_time_ms: float
    std_time_ms: float
    memory_mb: float


@scenario("complexity")
class ComplexityScenario(BaseScenario):
    """Computational complexity benchmark scenario.

    Validates O(N) Galerkin attention and O(N log N) FNet mixing
    by measuring execution time across different sequence lengths.

    Success Criteria:
        - FNet scaling exponent < 1.5 (sub-quadratic)
        - Softmax scaling exponent > 1.5 (quadratic baseline)
        - FNet speedup > 1.5x at largest size
    """

    config_class = ComplexityScenarioConfig

    def __init__(
        self,
        config: ComplexityScenarioConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize scenario."""
        super().__init__(config, **kwargs)
        self.config: ComplexityScenarioConfig  # Type hint

        self._device: torch.device | None = None
        self._scenario_logger: ScenarioLogger | None = None

    def setup(self) -> None:
        """Initialize resources."""
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._scenario_logger = ScenarioLogger(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
        )

        # Warn if not on GPU (timing will be less accurate)
        if not torch.cuda.is_available():
            self._scenario_logger.warning(
                "gpu_not_available",
                message="Timing results will be less accurate on CPU",
            )

    def teardown(self) -> None:
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self) -> ScenarioResult:
        """Execute the complexity benchmark.

        Returns:
            ScenarioResult with scaling analysis.
        """
        assert self._device is not None
        assert self._scenario_logger is not None

        torch.manual_seed(self.config.seed)

        # Benchmark FNet
        self._scenario_logger.info("benchmarking_fnet")
        fnet_results = self._benchmark_fnet()

        # Benchmark Softmax
        self._scenario_logger.info("benchmarking_softmax")
        softmax_results = self._benchmark_softmax()

        # Benchmark Galerkin
        self._scenario_logger.info("benchmarking_galerkin")
        galerkin_results = self._benchmark_galerkin()

        # Compute scaling exponents
        fnet_exponent = self._compute_scaling_exponent(fnet_results)
        softmax_exponent = self._compute_scaling_exponent(softmax_results)
        galerkin_exponent = self._compute_scaling_exponent(galerkin_results)

        # Compute speedup at largest size
        largest_n = max(r.n_tokens for r in fnet_results)
        fnet_time = next(r.mean_time_ms for r in fnet_results if r.n_tokens == largest_n)
        softmax_time = next(
            r.mean_time_ms for r in softmax_results if r.n_tokens == largest_n
        )
        speedup = softmax_time / fnet_time if fnet_time > 0 else 0

        # Record metrics
        self.record_metric("fnet_scaling_exponent", fnet_exponent)
        self.record_metric("softmax_scaling_exponent", softmax_exponent)
        self.record_metric("galerkin_scaling_exponent", galerkin_exponent)
        self.record_metric("fnet_speedup_at_largest", speedup)

        # Record detailed timings
        for result in fnet_results:
            self.record_metric(f"fnet_time_ms_n{result.n_tokens}", result.mean_time_ms)

        for result in softmax_results:
            self.record_metric(
                f"softmax_time_ms_n{result.n_tokens}", result.mean_time_ms
            )

        for result in galerkin_results:
            self.record_metric(
                f"galerkin_time_ms_n{result.n_tokens}", result.mean_time_ms
            )

        # Evaluate thresholds
        threshold_results = {
            "fnet_scaling_exponent": (
                fnet_exponent < self.config.fnet_scaling_exponent_max
            ),
            "softmax_scaling_exponent": (
                softmax_exponent > self.config.softmax_scaling_exponent_min
            ),
            "fnet_speedup": speedup > self.config.min_speedup_factor,
        }

        all_passed = all(threshold_results.values())
        status = ScenarioStatus.PASSED if all_passed else ScenarioStatus.FAILED

        # Create result
        end_time = datetime.now()
        assert self._start_time is not None
        duration = (end_time - self._start_time).total_seconds()

        return ScenarioResult(
            scenario_name=self.name,
            config_hash=self.config.compute_hash(),
            status=status,
            passed=all_passed,
            metrics=dict(self._metrics),
            threshold_results=threshold_results,
            artifacts={k: str(v) for k, v in self._artifacts.items()},
            start_time=self._start_time,
            end_time=end_time,
            duration_seconds=duration,
            device=str(self._device),
            python_version=sys.version,
            torch_version=torch.__version__,
            # Custom fields
            fnet_exponent=fnet_exponent,
            softmax_exponent=softmax_exponent,
            galerkin_exponent=galerkin_exponent,
            speedup=speedup,
        )

    def _benchmark_fnet(self) -> list[BenchmarkResult]:
        """Benchmark FNet mixing layer.

        Returns:
            List of benchmark results per grid size.
        """
        from src.modeling.fnet import FNetMixingLayer

        assert self._device is not None

        results = []

        for grid_size in self.config.grid_sizes:
            n_tokens = grid_size * grid_size
            layer = FNetMixingLayer(self.config.d_model).to(self._device)
            layer.eval()

            x = torch.randn(
                self.config.batch_size,
                n_tokens,
                self.config.d_model,
                device=self._device,
            )

            # Warmup
            for _ in range(self.config.n_warmup):
                with torch.no_grad():
                    _ = layer(x, grid_size=grid_size)

            # Synchronize before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timed runs
            times = []
            for _ in range(self.config.n_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = layer(x, grid_size=grid_size)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            # Memory usage
            memory_mb = 0.0
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            results.append(
                BenchmarkResult(
                    n_tokens=n_tokens,
                    mean_time_ms=float(np.mean(times)),
                    std_time_ms=float(np.std(times)),
                    memory_mb=memory_mb,
                )
            )

            if self._scenario_logger:
                self._scenario_logger.debug(
                    "fnet_benchmark_point",
                    n_tokens=n_tokens,
                    mean_time_ms=results[-1].mean_time_ms,
                )

        return results

    def _benchmark_softmax(self) -> list[BenchmarkResult]:
        """Benchmark standard softmax attention.

        Returns:
            List of benchmark results per grid size.
        """
        from src.modeling.attention import SoftmaxAttention

        assert self._device is not None

        results = []

        for grid_size in self.config.grid_sizes:
            n_tokens = grid_size * grid_size
            layer = SoftmaxAttention(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
            ).to(self._device)
            layer.eval()

            x = torch.randn(
                self.config.batch_size,
                n_tokens,
                self.config.d_model,
                device=self._device,
            )

            # Warmup
            for _ in range(self.config.n_warmup):
                with torch.no_grad():
                    _ = layer(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timed runs
            times = []
            for _ in range(self.config.n_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = layer(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            memory_mb = 0.0
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            results.append(
                BenchmarkResult(
                    n_tokens=n_tokens,
                    mean_time_ms=float(np.mean(times)),
                    std_time_ms=float(np.std(times)),
                    memory_mb=memory_mb,
                )
            )

        return results

    def _benchmark_galerkin(self) -> list[BenchmarkResult]:
        """Benchmark Galerkin linear attention.

        Returns:
            List of benchmark results per grid size.
        """
        from src.modeling.attention import GalerkinLinearAttention

        assert self._device is not None

        results = []

        for grid_size in self.config.grid_sizes:
            n_tokens = grid_size * grid_size
            layer = GalerkinLinearAttention(
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
            ).to(self._device)
            layer.eval()

            x = torch.randn(
                self.config.batch_size,
                n_tokens,
                self.config.d_model,
                device=self._device,
            )

            # Warmup
            for _ in range(self.config.n_warmup):
                with torch.no_grad():
                    _ = layer(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timed runs
            times = []
            for _ in range(self.config.n_iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = layer(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            memory_mb = 0.0
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            results.append(
                BenchmarkResult(
                    n_tokens=n_tokens,
                    mean_time_ms=float(np.mean(times)),
                    std_time_ms=float(np.std(times)),
                    memory_mb=memory_mb,
                )
            )

        return results

    def _compute_scaling_exponent(
        self,
        results: list[BenchmarkResult],
    ) -> float:
        """Compute scaling exponent from benchmark results.

        Fits log(time) = exponent * log(n) + constant

        Args:
            results: Benchmark results.

        Returns:
            Scaling exponent (1.0 = O(N), 2.0 = O(N²), etc.)
        """
        if len(results) < 2:
            return 0.0

        log_n = np.log([r.n_tokens for r in results])
        log_t = np.log([r.mean_time_ms for r in results])

        # Linear regression: log_t = exponent * log_n + intercept
        # Using np.polyfit for simplicity
        coeffs = np.polyfit(log_n, log_t, 1)
        exponent = coeffs[0]

        return float(exponent)
