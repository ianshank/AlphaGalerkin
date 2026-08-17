import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pytest


@dataclass
class BenchmarkStats:
    mean_time_s: float
    std_time_s: float
    min_time_s: float
    max_time_s: float
    throughput_per_s: float


class BenchmarkTimerProtocol(Protocol):
    def __call__(
        self,
        func: Callable[..., Any],
        *args: Any,
        warmup_rounds: int = 2,
        timing_rounds: int = 5,
        **kwargs: Any,
    ) -> BenchmarkStats: ...


@pytest.fixture
def benchmark_timer(request: pytest.FixtureRequest) -> BenchmarkTimerProtocol:
    """Fixture to execute a callable with warmup rounds and timing rounds."""
    # Try to get pytest-benchmark's fixture if available
    try:
        pytest_benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest_benchmark = None

    def timer(
        func: Callable[..., Any],
        *args: Any,
        warmup_rounds: int = 2,
        timing_rounds: int = 5,
        **kwargs: Any,
    ) -> BenchmarkStats:

        # If pytest-benchmark is available and not bypassed, use it.
        # Otherwise fallback to our benchmark timer.
        if pytest_benchmark is not None:
            # We wrap the func call
            def wrapped() -> Any:
                return func(*args, **kwargs)

            # pytest_benchmark runs the function many times and tracks it
            pytest_benchmark(wrapped)

            stats = pytest_benchmark.stats.stats
            return BenchmarkStats(
                mean_time_s=stats.mean,
                std_time_s=stats.stddev,
                min_time_s=stats.min,
                max_time_s=stats.max,
                throughput_per_s=1.0 / stats.mean if stats.mean > 0 else 0.0,
            )

        # Fallback timer implementation
        # Warmup
        for _ in range(warmup_rounds):
            func(*args, **kwargs)

        times = []
        for _ in range(timing_rounds):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)

        if not times:
            return BenchmarkStats(0.0, 0.0, 0.0, 0.0, 0.0)

        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = math.sqrt(variance)
        min_time = min(times)
        max_time = max(times)
        throughput = 1.0 / mean_time if mean_time > 0 else 0.0

        return BenchmarkStats(
            mean_time_s=mean_time,
            std_time_s=std_time,
            min_time_s=min_time,
            max_time_s=max_time,
            throughput_per_s=throughput,
        )

    return timer


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
