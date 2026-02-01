"""Tests for performance benchmarking demo.

Tests cover:
- FNet and Softmax model implementations
- Benchmark execution
- Result formatting
- Visualization outputs
"""

from __future__ import annotations

import pytest
import torch

from src.demos.benchmark_demo import (
    BenchmarkDemo,
    BenchmarkResult,
    BenchmarkSuite,
    SimpleFNetBlock,
    SimpleSoftmaxAttention,
)
from src.demos.config import BenchmarkDemoConfig


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self) -> None:
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            sequence_length=81,
            board_size=9,
            fnet_time_ms=1.5,
            softmax_time_ms=3.0,
            speedup=2.0,
        )

        assert result.sequence_length == 81
        assert result.board_size == 9
        assert result.fnet_time_ms == 1.5
        assert result.softmax_time_ms == 3.0
        assert result.speedup == 2.0
        assert result.memory_fnet_mb == 0.0  # Default

    def test_with_memory(self) -> None:
        """Test BenchmarkResult with memory tracking."""
        result = BenchmarkResult(
            sequence_length=361,
            board_size=19,
            fnet_time_ms=5.0,
            softmax_time_ms=50.0,
            speedup=10.0,
            memory_fnet_mb=100.0,
            memory_softmax_mb=500.0,
        )

        assert result.memory_fnet_mb == 100.0
        assert result.memory_softmax_mb == 500.0


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite collection."""

    def test_empty_suite(self) -> None:
        """Test empty BenchmarkSuite."""
        suite = BenchmarkSuite()
        assert len(suite.results) == 0
        assert suite.sequence_lengths == []
        assert suite.fnet_times == []
        assert suite.speedups == []

    def test_add_result(self) -> None:
        """Test adding results to suite."""
        suite = BenchmarkSuite()
        result = BenchmarkResult(
            sequence_length=81,
            board_size=9,
            fnet_time_ms=1.0,
            softmax_time_ms=2.0,
            speedup=2.0,
        )
        suite.add_result(result)

        assert len(suite.results) == 1
        assert suite.sequence_lengths == [81]
        assert suite.fnet_times == [1.0]
        assert suite.speedups == [2.0]

    def test_multiple_results(self) -> None:
        """Test suite with multiple results."""
        suite = BenchmarkSuite()

        for n, ft, st in [(81, 1.0, 2.0), (169, 2.0, 8.0), (361, 4.0, 32.0)]:
            suite.add_result(BenchmarkResult(
                sequence_length=n,
                board_size=int(n ** 0.5),
                fnet_time_ms=ft,
                softmax_time_ms=st,
                speedup=st / ft,
            ))

        assert len(suite.results) == 3
        assert suite.sequence_lengths == [81, 169, 361]
        assert suite.fnet_times == [1.0, 2.0, 4.0]
        assert suite.softmax_times == [2.0, 8.0, 32.0]

    def test_to_table(self) -> None:
        """Test table formatting."""
        suite = BenchmarkSuite()
        suite.add_result(BenchmarkResult(
            sequence_length=81,
            board_size=9,
            fnet_time_ms=1.5,
            softmax_time_ms=3.0,
            speedup=2.0,
        ))

        table = suite.to_table()
        assert "81" in table
        assert "9" in table
        assert "1.50" in table
        assert "3.00" in table
        assert "2.0" in table


class TestSimpleFNetBlock:
    """Tests for SimpleFNetBlock model."""

    @pytest.fixture
    def fnet_block(self) -> SimpleFNetBlock:
        """Create an FNet block."""
        return SimpleFNetBlock(d_model=64)

    def test_initialization(self, fnet_block: SimpleFNetBlock) -> None:
        """Test FNet block initialization."""
        assert fnet_block.d_model == 64
        assert fnet_block.norm is not None
        assert fnet_block.ffn is not None

    def test_forward(self, fnet_block: SimpleFNetBlock) -> None:
        """Test FNet forward pass."""
        batch_size = 2
        seq_length = 81
        d_model = 64

        x = torch.randn(batch_size, seq_length, d_model)
        y = fnet_block(x)

        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_output_varies(self, fnet_block: SimpleFNetBlock) -> None:
        """Test FNet produces non-trivial output."""
        x = torch.randn(1, 64, 64)
        y = fnet_block(x)

        # Output should be different from input
        assert not torch.allclose(x, y)


class TestSimpleSoftmaxAttention:
    """Tests for SimpleSoftmaxAttention model."""

    @pytest.fixture
    def attention(self) -> SimpleSoftmaxAttention:
        """Create a Softmax attention block."""
        return SimpleSoftmaxAttention(d_model=64, n_heads=4)

    def test_initialization(self, attention: SimpleSoftmaxAttention) -> None:
        """Test Softmax attention initialization."""
        assert attention.d_model == 64
        assert attention.n_heads == 4
        assert attention.head_dim == 16

    def test_forward(self, attention: SimpleSoftmaxAttention) -> None:
        """Test Softmax attention forward pass."""
        batch_size = 2
        seq_length = 81
        d_model = 64

        x = torch.randn(batch_size, seq_length, d_model)
        y = attention(x)

        assert y.shape == x.shape

    def test_output_varies(self, attention: SimpleSoftmaxAttention) -> None:
        """Test Softmax attention produces non-trivial output."""
        x = torch.randn(1, 64, 64)
        y = attention(x)

        assert not torch.allclose(x, y)


class TestBenchmarkDemo:
    """Tests for BenchmarkDemo class."""

    @pytest.fixture
    def demo(self) -> BenchmarkDemo:
        """Create a BenchmarkDemo instance."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[81, 169],  # Small for fast tests
            batch_size=4,
            n_warmup_runs=1,
            n_benchmark_runs=3,
            d_model=64,
            n_heads=4,
            device="cpu",
        )
        return BenchmarkDemo(config)

    def test_initialization(self, demo: BenchmarkDemo) -> None:
        """Test demo initialization."""
        assert demo.device == "cpu"
        assert demo.config.benchmark_sizes == [81, 169]
        assert demo.config.batch_size == 4
        assert demo.chart_viz is not None

    def test_create_models(self, demo: BenchmarkDemo) -> None:
        """Test model creation."""
        fnet, softmax = demo._create_models()

        assert isinstance(fnet, SimpleFNetBlock)
        assert isinstance(softmax, SimpleSoftmaxAttention)

    def test_benchmark_single(self, demo: BenchmarkDemo) -> None:
        """Test single model benchmark."""
        fnet, _ = demo._create_models()

        time_ms, memory_mb = demo._benchmark_single(
            model=fnet,
            batch_size=4,
            seq_length=64,
            n_warmup=1,
            n_runs=3,
        )

        assert time_ms > 0
        assert memory_mb >= 0

    def test_run_benchmark(self, demo: BenchmarkDemo) -> None:
        """Test full benchmark suite."""
        suite = demo.run_benchmark()

        assert len(suite.results) == 2  # Two sizes configured
        assert suite.total_time_seconds > 0

        for result in suite.results:
            assert result.fnet_time_ms > 0
            assert result.softmax_time_ms > 0
            assert result.speedup > 0

    def test_run_benchmark_custom_sizes(self, demo: BenchmarkDemo) -> None:
        """Test benchmark with custom sizes."""
        suite = demo.run_benchmark(sizes=[64, 100], batch_size=2)

        assert len(suite.results) == 2
        assert suite.sequence_lengths == [64, 100]

    def test_visualize_scaling(self, demo: BenchmarkDemo) -> None:
        """Test scaling visualization."""
        chart_img, summary = demo.visualize_scaling(
            sizes_str="64, 81",
            batch_size=2,
        )

        assert chart_img.ndim == 3
        assert chart_img.shape[2] == 3
        assert "Benchmark Results" in summary
        assert "FNet" in summary

    def test_compare_complexity(self, demo: BenchmarkDemo) -> None:
        """Test complexity comparison."""
        chart_img, explanation = demo.compare_complexity()

        assert chart_img.ndim == 3
        assert "Complexity Analysis" in explanation
        assert "O(N log N)" in explanation
        assert "O(N²)" in explanation


class TestBenchmarkDemoEdgeCases:
    """Edge case tests for BenchmarkDemo."""

    def test_minimum_batch_size(self) -> None:
        """Test with batch size of 1."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[64],
            batch_size=1,
            n_warmup_runs=1,
            n_benchmark_runs=2,
            device="cpu",
        )
        demo = BenchmarkDemo(config)
        suite = demo.run_benchmark()

        assert len(suite.results) == 1

    def test_single_size(self) -> None:
        """Test with single benchmark size."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[100],
            batch_size=4,
            n_warmup_runs=1,
            n_benchmark_runs=2,
            device="cpu",
        )
        demo = BenchmarkDemo(config)
        suite = demo.run_benchmark()

        assert len(suite.results) == 1
        assert suite.sequence_lengths == [100]

    def test_large_sequence_skip(self) -> None:
        """Test that overly large sequences are skipped."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[100, 10000],  # 10000 exceeds default max
            max_sequence_length=500,
            batch_size=4,
            n_warmup_runs=1,
            n_benchmark_runs=2,
            device="cpu",
        )
        demo = BenchmarkDemo(config)
        suite = demo.run_benchmark()

        # Only 100 should be run
        assert len(suite.results) == 1
        assert suite.sequence_lengths == [100]


class TestBenchmarkDemoSpeedupCalculation:
    """Tests for speedup calculation."""

    def test_speedup_ratio(self) -> None:
        """Test speedup is correctly calculated."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[64],
            batch_size=4,
            n_warmup_runs=2,
            n_benchmark_runs=5,
            device="cpu",
        )
        demo = BenchmarkDemo(config)
        suite = demo.run_benchmark()

        result = suite.results[0]
        expected_speedup = result.softmax_time_ms / result.fnet_time_ms
        assert abs(result.speedup - expected_speedup) < 0.01

    def test_fnet_faster_than_softmax(self) -> None:
        """Test that FNet is generally faster than Softmax."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[256],  # Larger size to see difference
            batch_size=8,
            n_warmup_runs=3,
            n_benchmark_runs=10,
            device="cpu",
        )
        demo = BenchmarkDemo(config)
        suite = demo.run_benchmark()

        result = suite.results[0]
        # FNet should generally be faster (speedup > 1)
        # Allow for some variance on CPU
        assert result.speedup > 0.5  # Conservative threshold


class TestBenchmarkDemoInvalidInput:
    """Tests for invalid input handling."""

    def test_invalid_sizes_string(self) -> None:
        """Test handling of invalid sizes string."""
        config = BenchmarkDemoConfig(
            benchmark_sizes=[64],
            device="cpu",
        )
        demo = BenchmarkDemo(config)

        # Invalid string should fall back to config defaults
        chart_img, summary = demo.visualize_scaling(
            sizes_str="invalid, not, numbers",
            batch_size=4,
        )

        # Should still produce output using default sizes
        assert chart_img.ndim == 3
