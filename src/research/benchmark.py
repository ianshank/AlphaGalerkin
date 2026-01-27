"""Benchmarking utilities for research.

Provides:
- Timing and profiling utilities
- Memory measurement
- Throughput calculation
- Benchmark suites
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from src.research.config import BenchmarkConfig


@dataclass
class BenchmarkResult:
    """Result from a benchmark run.

    Contains timing, memory, and throughput metrics.
    """

    name: str
    size: int
    batch_size: int
    n_iterations: int

    # Timing
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    total_time_s: float

    # Throughput
    throughput: float = 0.0  # samples/sec
    tokens_per_second: float = 0.0

    # Memory
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    device: str = "cpu"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "size": self.size,
            "batch_size": self.batch_size,
            "n_iterations": self.n_iterations,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "total_time_s": self.total_time_s,
            "throughput": self.throughput,
            "tokens_per_second": self.tokens_per_second,
            "peak_memory_mb": self.peak_memory_mb,
            "allocated_memory_mb": self.allocated_memory_mb,
            "timestamp": self.timestamp,
            "device": self.device,
            "metadata": self.metadata,
        }


class Benchmark:
    """Executes benchmarks with timing and profiling.

    Provides utilities for accurate measurement of model performance.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize benchmark.

        Args:
            config: Benchmark configuration.
            logger: Optional structured logger.

        """
        self.config = config
        self._logger = logger or structlog.get_logger(__name__).bind(
            benchmark=config.name,
        )
        self._results: list[BenchmarkResult] = []

    @property
    def results(self) -> list[BenchmarkResult]:
        """Get benchmark results."""
        return self._results

    def benchmark_function(
        self,
        func: Callable[[], Any],
        name: str,
        size: int,
        batch_size: int | None = None,
    ) -> BenchmarkResult:
        """Benchmark a function.

        Args:
            func: Function to benchmark (no arguments).
            name: Benchmark name.
            size: Size parameter (for logging).
            batch_size: Batch size used.

        Returns:
            BenchmarkResult with timing metrics.

        """
        batch_size = batch_size or self.config.batch_size
        times = []

        # Try to import torch for GPU timing
        torch_available = False
        try:
            import torch
            torch_available = True
            device = "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
        except ImportError:
            device = "cpu"

        # Warmup
        gc.collect()
        if torch_available and device == "cuda":
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        for _ in range(self.config.n_warmup):
            func()

        # Sync if CUDA
        if torch_available and device == "cuda" and self.config.sync_cuda:
            import torch
            torch.cuda.synchronize()

        # Timed iterations
        start_total = time.perf_counter()

        for _ in range(self.config.n_iterations):
            if torch_available and device == "cuda" and self.config.sync_cuda:
                import torch
                torch.cuda.synchronize()

            start = time.perf_counter()
            func()

            if torch_available and device == "cuda" and self.config.sync_cuda:
                import torch
                torch.cuda.synchronize()

            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        total_time = time.perf_counter() - start_total

        # Memory stats
        peak_memory = 0.0
        allocated_memory = 0.0
        if torch_available and device == "cuda" and self.config.measure_memory:
            import torch
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024

        # Compute statistics
        import statistics
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0

        # Throughput
        throughput = batch_size / (mean_time / 1000) if mean_time > 0 else 0.0
        tokens = size * size * batch_size  # For grid-based models
        tokens_per_sec = tokens / (mean_time / 1000) if mean_time > 0 else 0.0

        result = BenchmarkResult(
            name=name,
            size=size,
            batch_size=batch_size,
            n_iterations=self.config.n_iterations,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            min_time_ms=min(times),
            max_time_ms=max(times),
            total_time_s=total_time,
            throughput=throughput,
            tokens_per_second=tokens_per_sec,
            peak_memory_mb=peak_memory,
            allocated_memory_mb=allocated_memory,
            device=device,
        )

        self._results.append(result)

        self._logger.info(
            "benchmark_complete",
            name=name,
            size=size,
            mean_time_ms=mean_time,
            throughput=throughput,
        )

        return result

    def benchmark_model(
        self,
        model: Any,
        input_generator: Callable[[int, int], Any],
        name: str,
    ) -> list[BenchmarkResult]:
        """Benchmark a model across multiple sizes.

        Args:
            model: Model to benchmark.
            input_generator: Function(size, batch_size) -> input.
            name: Benchmark name.

        Returns:
            List of results for each size.

        """
        results = []

        for size in self.config.sizes:
            inputs = input_generator(size, self.config.batch_size)

            def run_model() -> Any:
                return model(inputs) if callable(model) else model.forward(*inputs)

            result = self.benchmark_function(
                run_model,
                name=name,
                size=size,
                batch_size=self.config.batch_size,
            )
            results.append(result)

        return results

    def compute_scaling_exponent(
        self,
        results: list[BenchmarkResult] | None = None,
    ) -> tuple[float, float]:
        """Compute scaling exponent from results.

        Fits time = c * n^alpha to find alpha.

        Args:
            results: Results to analyze (uses stored if None).

        Returns:
            Tuple of (exponent, r_squared).

        """
        import math

        results = results or self._results
        if len(results) < 2:
            return 0.0, 0.0

        # Extract sizes and times
        n_values = [r.size * r.size for r in results]  # tokens
        times = [r.mean_time_ms for r in results]

        # Log-log fit
        log_n = [math.log(n) for n in n_values]
        log_t = [math.log(t) for t in times]

        # Linear regression
        n = len(log_n)
        sum_x = sum(log_n)
        sum_y = sum(log_t)
        sum_xy = sum(x * y for x, y in zip(log_n, log_t))
        sum_xx = sum(x * x for x in log_n)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        # R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in log_t)
        intercept = (sum_y - slope * sum_x) / n
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(log_n, log_t))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, r_squared

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark summary.

        Returns:
            Summary dictionary.

        """
        if not self._results:
            return {"name": self.config.name, "results": []}

        exponent, r_squared = self.compute_scaling_exponent()

        return {
            "name": self.config.name,
            "n_benchmarks": len(self._results),
            "sizes": [r.size for r in self._results],
            "scaling_exponent": exponent,
            "r_squared": r_squared,
            "mean_throughput": sum(r.throughput for r in self._results) / len(self._results),
            "results": [r.to_dict() for r in self._results],
        }

    def clear(self) -> None:
        """Clear stored results."""
        self._results.clear()


class BenchmarkSuite:
    """Collection of benchmarks.

    Manages multiple related benchmarks.
    """

    def __init__(
        self,
        name: str,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            name: Suite name.
            logger: Optional structured logger.

        """
        self.name = name
        self._logger = logger or structlog.get_logger(__name__).bind(
            suite=name,
        )
        self._benchmarks: dict[str, Benchmark] = {}
        self._results: dict[str, list[BenchmarkResult]] = {}

    @property
    def benchmarks(self) -> dict[str, Benchmark]:
        """Get all benchmarks."""
        return self._benchmarks

    def add_benchmark(
        self,
        name: str,
        config: BenchmarkConfig | None = None,
    ) -> Benchmark:
        """Add a benchmark to the suite.

        Args:
            name: Benchmark name.
            config: Optional configuration.

        Returns:
            Created Benchmark.

        """
        if config is None:
            config = BenchmarkConfig(name=name)
        else:
            # Update config name to match provided name
            config = config.model_copy(update={"name": name})

        benchmark = Benchmark(config=config)
        self._benchmarks[name] = benchmark
        return benchmark

    def run_all(
        self,
        model_factory: Callable[[str], Any],
        input_generator: Callable[[int, int], Any],
    ) -> dict[str, list[BenchmarkResult]]:
        """Run all benchmarks in the suite.

        Args:
            model_factory: Function(name) -> model.
            input_generator: Function(size, batch) -> input.

        Returns:
            Results by benchmark name.

        """
        self._results.clear()

        for name, benchmark in self._benchmarks.items():
            model = model_factory(name)
            results = benchmark.benchmark_model(
                model=model,
                input_generator=input_generator,
                name=name,
            )
            self._results[name] = results

        return self._results

    def compare(self) -> dict[str, Any]:
        """Compare benchmark results.

        Returns:
            Comparison data.

        """
        if not self._results:
            return {}

        # Get all sizes
        all_sizes = set()
        for results in self._results.values():
            for r in results:
                all_sizes.add(r.size)

        comparison: dict[str, Any] = {}
        for size in sorted(all_sizes):
            size_data: dict[str, dict[str, float]] = {}
            for name, results in self._results.items():
                matching = [r for r in results if r.size == size]
                if matching:
                    size_data[name] = {
                        "mean_time_ms": matching[0].mean_time_ms,
                        "throughput": matching[0].throughput,
                    }
            comparison[size] = size_data

        return comparison

    def get_speedup(self, baseline: str, target: str) -> dict[int, float]:
        """Calculate speedup of target over baseline.

        Args:
            baseline: Baseline benchmark name.
            target: Target benchmark name.

        Returns:
            Speedup by size.

        """
        if baseline not in self._results or target not in self._results:
            return {}

        speedups = {}
        baseline_results = {r.size: r for r in self._results[baseline]}
        target_results = {r.size: r for r in self._results[target]}

        for size in baseline_results:
            if size in target_results:
                baseline_time = baseline_results[size].mean_time_ms
                target_time = target_results[size].mean_time_ms
                speedups[size] = baseline_time / target_time if target_time > 0 else 0.0

        return speedups


def create_benchmark(
    name: str = "benchmark",
    sizes: list[int] | None = None,
    **kwargs: Any,
) -> Benchmark:
    """Factory function to create a benchmark.

    Args:
        name: Benchmark name.
        sizes: Sizes to benchmark.
        **kwargs: Additional configuration.

    Returns:
        Benchmark instance.

    """
    config = BenchmarkConfig(
        name=name,
        sizes=sizes or [9, 13, 19],
        **kwargs,
    )
    return Benchmark(config=config)
