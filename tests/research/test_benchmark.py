"""Tests for benchmarking utilities."""

from __future__ import annotations

import time

import pytest

from src.research.config import BenchmarkConfig
from src.research.benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkSuite,
    create_benchmark,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_initialization(self, benchmark_result: BenchmarkResult) -> None:
        """Test result initialization."""
        assert benchmark_result.name == "test_benchmark"
        assert benchmark_result.size == 9
        assert benchmark_result.mean_time_ms == 10.0
        assert benchmark_result.throughput == 3200.0

    def test_to_dict(self, benchmark_result: BenchmarkResult) -> None:
        """Test serialization to dict."""
        data = benchmark_result.to_dict()

        assert data["name"] == "test_benchmark"
        assert data["size"] == 9
        assert data["mean_time_ms"] == 10.0


class TestBenchmark:
    """Tests for Benchmark."""

    def test_initialization(self, benchmark: Benchmark) -> None:
        """Test benchmark initialization."""
        assert benchmark.config.name == "test_benchmark"
        assert len(benchmark.results) == 0

    def test_benchmark_function(self, benchmark: Benchmark) -> None:
        """Test benchmarking a function."""

        def simple_func() -> int:
            return sum(range(1000))

        result = benchmark.benchmark_function(
            func=simple_func,
            name="simple",
            size=10,
            batch_size=1,
        )

        assert result.name == "simple"
        assert result.size == 10
        assert result.mean_time_ms > 0
        assert result.n_iterations == benchmark.config.n_iterations
        assert len(benchmark.results) == 1

    def test_benchmark_function_timing(self, benchmark: Benchmark) -> None:
        """Test that timing is accurate."""

        def slow_func() -> None:
            time.sleep(0.001)  # 1ms

        result = benchmark.benchmark_function(
            func=slow_func,
            name="slow",
            size=1,
        )

        # Should be at least 1ms per iteration
        assert result.mean_time_ms >= 0.5

    def test_benchmark_multiple_sizes(self, benchmark: Benchmark) -> None:
        """Test benchmarking across sizes."""
        calls = []

        def tracked_func() -> None:
            calls.append(1)

        for size in benchmark.config.sizes:
            benchmark.benchmark_function(
                func=tracked_func,
                name="tracked",
                size=size,
            )

        assert len(benchmark.results) == len(benchmark.config.sizes)

    def test_compute_scaling_exponent(self, benchmark: Benchmark) -> None:
        """Test scaling exponent computation."""
        # Add some synthetic results
        for size in [9, 13, 19, 25]:
            result = BenchmarkResult(
                name="test",
                size=size,
                batch_size=32,
                n_iterations=100,
                mean_time_ms=size * size * 0.1,  # O(N) scaling
                std_time_ms=0.1,
                min_time_ms=size * size * 0.09,
                max_time_ms=size * size * 0.11,
                total_time_s=1.0,
            )
            benchmark._results.append(result)

        exponent, r_squared = benchmark.compute_scaling_exponent()

        # Should be close to 1.0 for O(N) with N = size^2
        assert 0.8 <= exponent <= 1.5
        assert r_squared > 0.9

    def test_get_summary(self, benchmark: Benchmark) -> None:
        """Test getting summary."""
        benchmark._results.append(BenchmarkResult(
            name="test",
            size=9,
            batch_size=32,
            n_iterations=100,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=8.0,
            max_time_ms=12.0,
            total_time_s=1.0,
            throughput=3200.0,
        ))

        summary = benchmark.get_summary()

        assert summary["name"] == "test_benchmark"
        assert summary["n_benchmarks"] == 1

    def test_clear(self, benchmark: Benchmark) -> None:
        """Test clearing results."""
        benchmark._results.append(BenchmarkResult(
            name="test",
            size=9,
            batch_size=32,
            n_iterations=100,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=8.0,
            max_time_ms=12.0,
            total_time_s=1.0,
        ))

        benchmark.clear()
        assert len(benchmark.results) == 0


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_initialization(self) -> None:
        """Test suite initialization."""
        suite = BenchmarkSuite(name="test_suite")
        assert suite.name == "test_suite"
        assert len(suite.benchmarks) == 0

    def test_add_benchmark(self) -> None:
        """Test adding benchmark."""
        suite = BenchmarkSuite(name="test_suite")
        benchmark = suite.add_benchmark(
            name="bench1",
            config=BenchmarkConfig(sizes=[9, 13], n_warmup=1, n_iterations=10),
        )

        assert "bench1" in suite.benchmarks
        assert benchmark.config.name == "bench1"

    def test_compare(self) -> None:
        """Test comparing benchmarks."""
        suite = BenchmarkSuite(name="test_suite")

        # Add some results
        suite._results["fast"] = [
            BenchmarkResult(
                name="fast", size=9, batch_size=32, n_iterations=100,
                mean_time_ms=5.0, std_time_ms=0.5, min_time_ms=4.0, max_time_ms=6.0,
                total_time_s=0.5, throughput=6400.0,
            ),
        ]
        suite._results["slow"] = [
            BenchmarkResult(
                name="slow", size=9, batch_size=32, n_iterations=100,
                mean_time_ms=15.0, std_time_ms=1.5, min_time_ms=12.0, max_time_ms=18.0,
                total_time_s=1.5, throughput=2133.0,
            ),
        ]

        comparison = suite.compare()

        assert 9 in comparison
        assert "fast" in comparison[9]
        assert "slow" in comparison[9]

    def test_get_speedup(self) -> None:
        """Test getting speedup."""
        suite = BenchmarkSuite(name="test_suite")

        suite._results["baseline"] = [
            BenchmarkResult(
                name="baseline", size=9, batch_size=32, n_iterations=100,
                mean_time_ms=20.0, std_time_ms=2.0, min_time_ms=16.0, max_time_ms=24.0,
                total_time_s=2.0, throughput=1600.0,
            ),
        ]
        suite._results["optimized"] = [
            BenchmarkResult(
                name="optimized", size=9, batch_size=32, n_iterations=100,
                mean_time_ms=5.0, std_time_ms=0.5, min_time_ms=4.0, max_time_ms=6.0,
                total_time_s=0.5, throughput=6400.0,
            ),
        ]

        speedup = suite.get_speedup("baseline", "optimized")

        assert 9 in speedup
        assert speedup[9] == pytest.approx(4.0, rel=0.1)


class TestCreateBenchmark:
    """Tests for create_benchmark factory."""

    def test_create_default(self) -> None:
        """Test creating default benchmark."""
        benchmark = create_benchmark()
        assert benchmark.config.name == "benchmark"
        assert 19 in benchmark.config.sizes

    def test_create_with_sizes(self) -> None:
        """Test creating with custom sizes."""
        benchmark = create_benchmark(sizes=[5, 9])
        assert benchmark.config.sizes == [5, 9]
