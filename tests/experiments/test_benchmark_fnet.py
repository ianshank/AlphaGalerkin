"""Tests for FNet speed benchmark.

Validates BenchmarkResult dataclass, FNetMixingLayer and
SoftmaxAttentionLayer forward passes, benchmark_layer timing,
and analyze_scaling regression fitting.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.experiments.benchmark_fnet import (
    SUBQUADRATIC_EXPONENT_THRESHOLD,
    BenchmarkResult,
    FNetMixingLayer,
    SoftmaxAttentionLayer,
    analyze_scaling,
    benchmark_layer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[32, 64])
def d_model(request: pytest.FixtureRequest) -> int:
    """Model dimension for layer tests."""
    return request.param


@pytest.fixture
def fnet_layer(d_model: int) -> FNetMixingLayer:
    """FNet mixing layer instance."""
    layer = FNetMixingLayer(d_model)
    layer.eval()
    return layer


@pytest.fixture
def softmax_layer(d_model: int) -> SoftmaxAttentionLayer:
    """Softmax attention layer instance."""
    layer = SoftmaxAttentionLayer(d_model, n_heads=2)
    layer.eval()
    return layer


@pytest.fixture
def sample_benchmark_results() -> list[BenchmarkResult]:
    """Synthetic benchmark results with known scaling behaviour.

    FNet times ~ N^1 (linear), Softmax times ~ N^2 (quadratic).
    """
    sizes = [25, 49, 81, 121, 169]
    results = []
    for n in sizes:
        fnet_time = 0.01 * n  # O(N)
        softmax_time = 0.0001 * n * n  # O(N^2)
        results.append(
            BenchmarkResult(
                n_tokens=n,
                fnet_time_ms=fnet_time,
                softmax_time_ms=softmax_time,
                speedup=softmax_time / fnet_time if fnet_time > 0 else float("inf"),
                fnet_memory_mb=0.0,
                softmax_memory_mb=0.0,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Tests: BenchmarkResult dataclass
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    """Tests for the BenchmarkResult dataclass."""

    def test_fields_stored(self) -> None:
        """All fields are stored and accessible."""
        r = BenchmarkResult(
            n_tokens=81,
            fnet_time_ms=1.5,
            softmax_time_ms=10.0,
            speedup=6.67,
            fnet_memory_mb=0.0,
            softmax_memory_mb=5.0,
        )
        assert r.n_tokens == 81
        assert r.fnet_time_ms == pytest.approx(1.5)
        assert r.softmax_time_ms == pytest.approx(10.0)
        assert r.speedup == pytest.approx(6.67)
        assert r.fnet_memory_mb == pytest.approx(0.0)
        assert r.softmax_memory_mb == pytest.approx(5.0)

    def test_speedup_calculation(self) -> None:
        """Speedup is softmax_time / fnet_time."""
        fnet_t, softmax_t = 2.0, 8.0
        r = BenchmarkResult(
            n_tokens=49,
            fnet_time_ms=fnet_t,
            softmax_time_ms=softmax_t,
            speedup=softmax_t / fnet_t,
            fnet_memory_mb=0.0,
            softmax_memory_mb=0.0,
        )
        assert r.speedup == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Tests: FNetMixingLayer forward pass
# ---------------------------------------------------------------------------


class TestFNetMixingLayer:
    """Tests for FNet FFT mixing layer."""

    @pytest.mark.parametrize("grid_size", [5, 7, 9])
    def test_output_shape(self, fnet_layer: FNetMixingLayer, grid_size: int) -> None:
        """Output shape matches input shape."""
        batch, d = 2, fnet_layer.d_model
        n_tokens = grid_size * grid_size
        x = torch.randn(batch, n_tokens, d)

        out = fnet_layer(x, grid_size)

        assert out.shape == (batch, n_tokens, d)

    def test_output_is_finite(self, fnet_layer: FNetMixingLayer) -> None:
        """Output contains no NaN or Inf values."""
        x = torch.randn(2, 25, fnet_layer.d_model)
        out = fnet_layer(x, grid_size=5)
        assert out.isfinite().all()

    def test_residual_connection(self, d_model: int) -> None:
        """Output differs from input (mixing happened) but is not pure noise."""
        layer = FNetMixingLayer(d_model)
        layer.eval()
        x = torch.randn(1, 9, d_model)
        out = layer(x, grid_size=3)
        # Not identical to input (mixing + norm changed it)
        assert not torch.allclose(x, out, atol=1e-6)

    def test_stores_d_model(self) -> None:
        """Layer stores d_model attribute."""
        layer = FNetMixingLayer(128)
        assert layer.d_model == 128


# ---------------------------------------------------------------------------
# Tests: SoftmaxAttentionLayer forward pass
# ---------------------------------------------------------------------------


class TestSoftmaxAttentionLayer:
    """Tests for standard softmax attention layer."""

    @pytest.mark.parametrize("n_tokens", [9, 16, 25])
    def test_output_shape(
        self, softmax_layer: SoftmaxAttentionLayer, n_tokens: int
    ) -> None:
        """Output shape matches input shape."""
        batch, d = 2, softmax_layer.d_model
        x = torch.randn(batch, n_tokens, d)

        out = softmax_layer(x)

        assert out.shape == (batch, n_tokens, d)

    def test_output_is_finite(self, softmax_layer: SoftmaxAttentionLayer) -> None:
        """Output contains no NaN or Inf values."""
        x = torch.randn(2, 16, softmax_layer.d_model)
        out = softmax_layer(x)
        assert out.isfinite().all()

    def test_n_heads_stored(self) -> None:
        """Layer stores n_heads and computes d_head."""
        layer = SoftmaxAttentionLayer(64, n_heads=4)
        assert layer.n_heads == 4
        assert layer.d_head == 16

    def test_scale_factor(self) -> None:
        """Attention scale is 1/sqrt(d_head)."""
        layer = SoftmaxAttentionLayer(64, n_heads=4)
        expected_scale = (64 // 4) ** -0.5
        assert layer.scale == pytest.approx(expected_scale)


# ---------------------------------------------------------------------------
# Tests: benchmark_layer timing
# ---------------------------------------------------------------------------


class TestBenchmarkLayer:
    """Tests for the benchmark_layer utility."""

    def test_returns_positive_time(self, fnet_layer: FNetMixingLayer) -> None:
        """Mean time is positive."""
        x = torch.randn(1, 9, fnet_layer.d_model)
        mean_time, _mem = benchmark_layer(
            fnet_layer, x, n_warmup=1, n_iterations=3, grid_size=3
        )
        assert mean_time > 0

    def test_returns_float_tuple(self, softmax_layer: SoftmaxAttentionLayer) -> None:
        """Returns a (float, float) tuple."""
        x = torch.randn(1, 9, softmax_layer.d_model)
        result = benchmark_layer(softmax_layer, x, n_warmup=1, n_iterations=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_cpu_memory_is_zero(self, fnet_layer: FNetMixingLayer) -> None:
        """On CPU, peak memory is reported as 0."""
        x = torch.randn(1, 9, fnet_layer.d_model)
        _time, mem = benchmark_layer(
            fnet_layer, x, n_warmup=1, n_iterations=2, grid_size=3
        )
        assert mem == pytest.approx(0.0)

    def test_more_iterations_still_works(
        self, softmax_layer: SoftmaxAttentionLayer
    ) -> None:
        """Running more iterations completes without error."""
        x = torch.randn(1, 16, softmax_layer.d_model)
        mean_time, _ = benchmark_layer(
            softmax_layer, x, n_warmup=2, n_iterations=5
        )
        assert mean_time > 0


# ---------------------------------------------------------------------------
# Tests: analyze_scaling regression
# ---------------------------------------------------------------------------


class TestAnalyzeScaling:
    """Tests for scaling analysis from benchmark results."""

    def test_fnet_subquadratic(
        self, sample_benchmark_results: list[BenchmarkResult]
    ) -> None:
        """Synthetic linear FNet times yield sub-quadratic exponent."""
        analysis = analyze_scaling(sample_benchmark_results)
        assert analysis["fnet_scales_subquadratic"] is True
        assert analysis["fnet_scaling_exponent"] < SUBQUADRATIC_EXPONENT_THRESHOLD

    def test_softmax_quadratic(
        self, sample_benchmark_results: list[BenchmarkResult]
    ) -> None:
        """Synthetic quadratic Softmax times yield quadratic exponent."""
        analysis = analyze_scaling(sample_benchmark_results)
        assert analysis["softmax_scales_quadratic"] is True
        assert analysis["softmax_scaling_exponent"] > SUBQUADRATIC_EXPONENT_THRESHOLD

    def test_speedup_stats(
        self, sample_benchmark_results: list[BenchmarkResult]
    ) -> None:
        """Speedup statistics are positive and consistent."""
        analysis = analyze_scaling(sample_benchmark_results)
        assert analysis["mean_speedup"] > 0
        assert analysis["max_speedup"] >= analysis["mean_speedup"]
        assert analysis["speedup_at_largest"] > 0

    def test_expected_exponents_documented(
        self, sample_benchmark_results: list[BenchmarkResult]
    ) -> None:
        """Analysis includes documented expected exponents."""
        analysis = analyze_scaling(sample_benchmark_results)
        assert analysis["fnet_expected_exponent"] == pytest.approx(1.0)
        assert analysis["softmax_expected_exponent"] == pytest.approx(2.0)

    def test_exponent_close_to_one_for_linear(self) -> None:
        """Pure linear data yields exponent close to 1.0."""
        results = [
            BenchmarkResult(
                n_tokens=n,
                fnet_time_ms=0.5 * n,
                softmax_time_ms=0.5 * n,
                speedup=1.0,
                fnet_memory_mb=0.0,
                softmax_memory_mb=0.0,
            )
            for n in [25, 64, 100, 225, 400]
        ]
        analysis = analyze_scaling(results)
        assert abs(analysis["fnet_scaling_exponent"] - 1.0) < 0.15

    def test_minimum_results_required(self) -> None:
        """At least 2 data points are needed for regression."""
        single = [
            BenchmarkResult(
                n_tokens=25,
                fnet_time_ms=1.0,
                softmax_time_ms=2.0,
                speedup=2.0,
                fnet_memory_mb=0.0,
                softmax_memory_mb=0.0,
            )
        ]
        # polyfit with 1 point should still return a result (degree 1 is degenerate)
        # We just verify it doesn't crash
        analysis = analyze_scaling(single)
        assert "fnet_scaling_exponent" in analysis
