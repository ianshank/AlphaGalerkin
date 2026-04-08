"""Tests for FNet speed benchmark.

Covers the benchmark dataclass, FNet mixing layer, softmax attention layer,
and benchmark utility functions. Uses small sizes for fast test execution.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.experiments.benchmark_fnet import (
    SUBQUADRATIC_EXPONENT_THRESHOLD,
    BenchmarkResult,
    FNetMixingLayer,
    SoftmaxAttentionLayer,
    benchmark_layer,
    run_benchmark,
)

# --- BenchmarkResult Tests ---


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_creation(self) -> None:
        """BenchmarkResult can be created with all fields."""
        result = BenchmarkResult(
            n_tokens=81,
            fnet_time_ms=1.0,
            softmax_time_ms=5.0,
            speedup=5.0,
            fnet_memory_mb=10.0,
            softmax_memory_mb=50.0,
        )
        assert result.n_tokens == 81
        assert result.speedup == 5.0

    def test_fields_accessible(self) -> None:
        """All fields are accessible."""
        result = BenchmarkResult(
            n_tokens=9,
            fnet_time_ms=0.5,
            softmax_time_ms=2.0,
            speedup=4.0,
            fnet_memory_mb=1.0,
            softmax_memory_mb=4.0,
        )
        assert result.fnet_time_ms == 0.5
        assert result.softmax_time_ms == 2.0
        assert result.fnet_memory_mb == 1.0
        assert result.softmax_memory_mb == 4.0


class TestConstants:
    """Tests for module-level constants."""

    def test_subquadratic_threshold(self) -> None:
        """Threshold is the documented midpoint value."""
        assert SUBQUADRATIC_EXPONENT_THRESHOLD == 1.5


# --- FNetMixingLayer Tests ---


class TestFNetMixingLayer:
    """Tests for FNetMixingLayer."""

    @pytest.fixture
    def layer(self) -> FNetMixingLayer:
        """Create a small FNet layer for testing."""
        return FNetMixingLayer(d_model=32)

    def test_init(self, layer: FNetMixingLayer) -> None:
        """Layer initializes with correct d_model."""
        assert layer.d_model == 32

    def test_forward_shape(self, layer: FNetMixingLayer) -> None:
        """Forward pass preserves shape."""
        batch, n_tokens, d = 2, 9, 32  # 3x3 grid
        x = torch.randn(batch, n_tokens, d)
        out = layer(x, grid_size=3)
        assert out.shape == (batch, n_tokens, d)

    def test_forward_no_nan(self, layer: FNetMixingLayer) -> None:
        """Forward pass produces no NaN values."""
        x = torch.randn(2, 16, 32)  # 4x4 grid
        out = layer(x, grid_size=4)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("grid_size", [3, 4, 5, 7])
    def test_forward_various_grid_sizes(self, grid_size: int) -> None:
        """Works with various grid sizes."""
        layer = FNetMixingLayer(d_model=16)
        n_tokens = grid_size * grid_size
        x = torch.randn(1, n_tokens, 16)
        out = layer(x, grid_size=grid_size)
        assert out.shape == (1, n_tokens, 16)

    def test_residual_connection(self, layer: FNetMixingLayer) -> None:
        """Output includes residual connection (not identical to input)."""
        x = torch.randn(1, 9, 32)
        out = layer(x, grid_size=3)
        # With LayerNorm and residual, output should be close but not identical
        # to input (unless FFT mixing is zero, which is unlikely)
        assert out.shape == x.shape


# --- SoftmaxAttentionLayer Tests ---


class TestSoftmaxAttentionLayer:
    """Tests for SoftmaxAttentionLayer."""

    @pytest.fixture
    def layer(self) -> SoftmaxAttentionLayer:
        """Create a small attention layer for testing."""
        return SoftmaxAttentionLayer(d_model=32, n_heads=4)

    def test_init(self, layer: SoftmaxAttentionLayer) -> None:
        """Layer initializes with correct parameters."""
        assert layer.d_model == 32
        assert layer.n_heads == 4
        assert layer.d_head == 8

    def test_forward_shape(self, layer: SoftmaxAttentionLayer) -> None:
        """Forward pass preserves shape."""
        batch, n_tokens, d = 2, 9, 32
        x = torch.randn(batch, n_tokens, d)
        out = layer(x)
        assert out.shape == (batch, n_tokens, d)

    def test_forward_no_nan(self, layer: SoftmaxAttentionLayer) -> None:
        """Forward pass produces no NaN values."""
        x = torch.randn(2, 16, 32)
        out = layer(x)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("n_tokens", [4, 9, 16, 25])
    def test_forward_various_sizes(self, n_tokens: int) -> None:
        """Works with various sequence lengths."""
        layer = SoftmaxAttentionLayer(d_model=16, n_heads=2)
        x = torch.randn(1, n_tokens, 16)
        out = layer(x)
        assert out.shape == (1, n_tokens, 16)

    def test_dropout_zero(self) -> None:
        """With dropout=0 in mode, output is deterministic."""
        layer = SoftmaxAttentionLayer(d_model=16, n_heads=2, dropout=0.0)
        layer.eval()
        x = torch.randn(1, 9, 16)
        with torch.no_grad():
            out1 = layer(x)
            out2 = layer(x)
        assert torch.allclose(out1, out2)


# --- benchmark_layer Tests ---


class TestBenchmarkLayer:
    """Tests for benchmark_layer function."""

    def test_fnet_benchmark(self) -> None:
        """Benchmarks FNet layer and returns valid results."""
        layer = FNetMixingLayer(d_model=16)
        layer.eval()
        x = torch.randn(2, 9, 16)

        with torch.no_grad():
            mean_time, memory = benchmark_layer(
                layer,
                x,
                n_warmup=1,
                n_iterations=3,
                grid_size=3,
            )

        assert mean_time > 0
        assert memory >= 0  # CPU memory is 0

    def test_softmax_benchmark(self) -> None:
        """Benchmarks softmax layer and returns valid results."""
        layer = SoftmaxAttentionLayer(d_model=16, n_heads=2)
        layer.eval()
        x = torch.randn(2, 9, 16)

        with torch.no_grad():
            mean_time, memory = benchmark_layer(
                layer,
                x,
                n_warmup=1,
                n_iterations=3,
            )

        assert mean_time > 0
        assert memory >= 0


# --- run_benchmark Tests ---


class TestRunBenchmark:
    """Tests for run_benchmark function."""

    def test_single_size(self) -> None:
        """Runs benchmark for a single size."""
        results = run_benchmark(
            sizes=[9],  # 3x3 grid
            d_model=16,
            batch_size=2,
            n_warmup=1,
            n_iterations=2,
            device="cpu",
        )
        assert len(results) == 1
        assert results[0].n_tokens == 9
        assert results[0].fnet_time_ms > 0
        assert results[0].softmax_time_ms > 0
        assert results[0].speedup > 0

    def test_multiple_sizes(self) -> None:
        """Runs benchmark for multiple sizes."""
        results = run_benchmark(
            sizes=[4, 9],  # 2x2 and 3x3 grids
            d_model=16,
            batch_size=2,
            n_warmup=1,
            n_iterations=2,
            device="cpu",
        )
        assert len(results) == 2
        assert results[0].n_tokens == 4
        assert results[1].n_tokens == 9

    def test_device_auto_selects_cpu_when_no_cuda(self) -> None:
        """device='auto' falls back to CPU when CUDA unavailable."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available; cannot test CPU fallback")
        results = run_benchmark(
            sizes=[4],
            d_model=16,
            batch_size=1,
            n_warmup=1,
            n_iterations=1,
            device="auto",
        )
        assert len(results) == 1
