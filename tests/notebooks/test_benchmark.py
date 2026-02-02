"""Tests for notebook benchmark utilities."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from notebooks.utils.benchmark import (
    BenchmarkResult,
    benchmark_attention,
    benchmark_model_throughput,
    benchmark_module,
    format_benchmark_table,
    format_throughput_table,
)


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_throughput_calculation(self) -> None:
        """Test throughput calculation."""
        result = BenchmarkResult(
            name="test",
            seq_length=100,
            time_ms=10.0,  # 10ms
            batch_size=32,
            n_runs=50,
        )
        # 32 items in 10ms = 32 * 1000 / 10 = 3200 items/sec
        assert result.throughput == 3200.0

    def test_throughput_zero_time(self) -> None:
        """Test throughput with zero time returns zero."""
        result = BenchmarkResult(
            name="test",
            seq_length=100,
            time_ms=0.0,
            batch_size=32,
            n_runs=50,
        )
        assert result.throughput == 0.0

    def test_throughput_negative_time(self) -> None:
        """Test throughput with negative time returns zero."""
        result = BenchmarkResult(
            name="test",
            seq_length=100,
            time_ms=-1.0,
            batch_size=32,
            n_runs=50,
        )
        assert result.throughput == 0.0


class TestBenchmarkModule:
    """Tests for benchmark_module function."""

    def test_benchmark_linear_layer(self) -> None:
        """Test benchmarking a simple linear layer."""
        module = nn.Linear(64, 64)
        x = torch.randn(4, 10, 64)

        time_ms = benchmark_module(module, x, n_warmup=2, n_runs=5)

        assert isinstance(time_ms, float)
        assert time_ms > 0

    def test_benchmark_returns_reasonable_time(self) -> None:
        """Test that benchmark returns reasonable timing."""
        module = nn.Linear(64, 64)
        x = torch.randn(4, 10, 64)

        time_ms = benchmark_module(module, x, n_warmup=1, n_runs=10)

        # Should be less than 1 second for a simple linear layer
        assert time_ms < 1000


class TestBenchmarkAttention:
    """Tests for benchmark_attention function."""

    def test_benchmark_attention_modules(self) -> None:
        """Test benchmarking attention modules."""
        from src.modeling.attention import GalerkinAttention, SoftmaxAttention

        d_model = 32
        n_heads = 4
        galerkin = GalerkinAttention(d_model, n_heads)
        softmax = SoftmaxAttention(d_model, n_heads)

        galerkin_results, softmax_results = benchmark_attention(
            galerkin_attn=galerkin,
            softmax_attn=softmax,
            seq_lengths=[16, 25],
            d_model=d_model,
            batch_size=2,
            n_warmup=1,
            n_runs=3,
        )

        assert len(galerkin_results) == 2
        assert len(softmax_results) == 2
        assert all(isinstance(r, BenchmarkResult) for r in galerkin_results)
        assert all(isinstance(r, BenchmarkResult) for r in softmax_results)

    def test_benchmark_attention_names(self) -> None:
        """Test that results have correct names."""
        from src.modeling.attention import GalerkinAttention, SoftmaxAttention

        galerkin = GalerkinAttention(32, 4)
        softmax = SoftmaxAttention(32, 4)

        galerkin_results, softmax_results = benchmark_attention(
            galerkin_attn=galerkin,
            softmax_attn=softmax,
            seq_lengths=[16],
            d_model=32,
            n_warmup=1,
            n_runs=2,
        )

        assert galerkin_results[0].name == "Galerkin"
        assert softmax_results[0].name == "Softmax"


class TestBenchmarkModelThroughput:
    """Tests for benchmark_model_throughput function."""

    def test_throughput_simple_model(self) -> None:
        """Test throughput calculation for simple model."""

        # Simple model that outputs policy and value
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(17, 32, 3, padding=1)
                self.fc = nn.Linear(32, 82)  # Policy
                self.value = nn.Linear(32, 1)

            def forward(self, x):
                x = self.conv(x)
                x = x.mean(dim=(2, 3))  # Global pool
                from collections import namedtuple

                Output = namedtuple("Output", ["policy_logits", "value"])
                return Output(self.fc(x), self.value(x))

        model = SimpleModel()

        throughput = benchmark_model_throughput(
            model=model,
            board_size=9,
            input_channels=17,
            batch_size=4,
            n_evals=5,
        )

        assert throughput > 0


class TestFormatBenchmarkTable:
    """Tests for format_benchmark_table function."""

    def test_format_table(self) -> None:
        """Test table formatting."""
        galerkin_results = [
            BenchmarkResult("Galerkin", 25, 1.0, 4, 50),
            BenchmarkResult("Galerkin", 81, 2.0, 4, 50),
        ]
        softmax_results = [
            BenchmarkResult("Softmax", 25, 2.0, 4, 50),
            BenchmarkResult("Softmax", 81, 4.0, 4, 50),
        ]

        table = format_benchmark_table(galerkin_results, softmax_results)

        assert "Galerkin" in table
        assert "Softmax" in table
        assert "Speedup" in table

    def test_format_table_with_labels(self) -> None:
        """Test table formatting with custom labels."""
        galerkin_results = [BenchmarkResult("Galerkin", 25, 1.0, 4, 50)]
        softmax_results = [BenchmarkResult("Softmax", 25, 2.0, 4, 50)]

        table = format_benchmark_table(galerkin_results, softmax_results, board_labels=["5×5"])

        assert "5×5" in table

    def test_format_table_empty_results(self) -> None:
        """Test formatting with empty results."""
        table = format_benchmark_table([], [])
        assert "No benchmark results" in table

    def test_format_table_mismatched_lengths(self) -> None:
        """Test that mismatched lengths raise error."""
        galerkin_results = [BenchmarkResult("Galerkin", 25, 1.0, 4, 50)]
        softmax_results = []

        with pytest.raises(ValueError, match="length mismatch"):
            format_benchmark_table(galerkin_results, softmax_results)

    def test_format_table_mismatched_labels(self) -> None:
        """Test that mismatched labels raise error."""
        galerkin_results = [BenchmarkResult("Galerkin", 25, 1.0, 4, 50)]
        softmax_results = [BenchmarkResult("Softmax", 25, 2.0, 4, 50)]

        with pytest.raises(ValueError, match="Label count mismatch"):
            format_benchmark_table(
                galerkin_results,
                softmax_results,
                board_labels=["5×5", "9×9"],  # Too many labels
            )


class TestFormatThroughputTable:
    """Tests for format_throughput_table function."""

    def test_format_throughput_table(self) -> None:
        """Test throughput table formatting."""
        results = [
            ("9×9", 9, 1000.0, 2000.0),
            ("19×19", 19, 500.0, 1500.0),
        ]

        table = format_throughput_table(results)

        assert "9×9" in table
        assert "19×19" in table
        assert "Full Model" in table
        assert "Fast Model" in table
        assert "Speedup" in table
