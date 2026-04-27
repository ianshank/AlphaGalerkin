"""Unit tests for stat helpers used by the perf benchmark."""

from __future__ import annotations

import math

import pytest

from src.video_compression.perf.metrics import (
    DEFAULT_PERCENTILES,
    LatencyStats,
    percentile,
    regression_pct,
    summarize_latencies,
    throughput_fps,
)


# --------------------------------------------------------------- percentile


class TestPercentile:
    def test_single_sample_returns_value(self) -> None:
        assert percentile([42.0], 50) == 42.0
        assert percentile([42.0], 99) == 42.0

    def test_known_quartiles(self) -> None:
        # values 0..10, p50 should be 5
        vals = list(range(11))
        assert percentile(vals, 0) == 0
        assert percentile(vals, 50) == 5
        assert percentile(vals, 100) == 10

    def test_linear_interpolation(self) -> None:
        # values [0, 1, 2, 3], p50 is between 1 and 2 = 1.5
        assert percentile([0, 1, 2, 3], 50) == pytest.approx(1.5)

    def test_unsorted_input_handled(self) -> None:
        assert percentile([3, 1, 2, 0], 50) == pytest.approx(1.5)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            percentile([], 50)

    @pytest.mark.parametrize("p", [-1, 101])
    def test_invalid_percentile_raises(self, p: int) -> None:
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile([1.0], p)


# ------------------------------------------------------- summarize_latencies


class TestSummarizeLatencies:
    def test_known_values(self) -> None:
        stats = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.count == 5
        assert stats.mean_ms == 3.0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 5.0
        assert stats.std_ms == pytest.approx(math.sqrt(2.5))
        assert stats.percentile(50) == 3.0

    def test_default_percentiles_recorded(self) -> None:
        stats = summarize_latencies([1.0, 2.0, 3.0])
        for p in DEFAULT_PERCENTILES:
            stats.percentile(p)  # should not raise

    def test_unrecorded_percentile_raises(self) -> None:
        stats = summarize_latencies([1.0, 2.0])
        with pytest.raises(KeyError, match="p95"):
            stats.percentile(95)

    def test_single_sample_zero_std(self) -> None:
        stats = summarize_latencies([5.0])
        assert stats.std_ms == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_latencies([])

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            summarize_latencies([1.0, -1.0])

    def test_to_dict_round_trip(self) -> None:
        stats = summarize_latencies([1.0, 2.0, 3.0])
        as_dict = stats.to_dict()
        assert as_dict["count"] == 3
        assert "percentiles_ms" in as_dict


# ----------------------------------------------------------- throughput_fps


class TestThroughputFps:
    def test_basic(self) -> None:
        # 100 ms per iter, 1 frame per iter => 10 fps
        assert throughput_fps([100.0], frames_per_iter=1) == pytest.approx(10.0)

    def test_batched_frames(self) -> None:
        # 100 ms per iter, 4 frames per iter => 40 fps
        assert throughput_fps([100.0], frames_per_iter=4) == pytest.approx(40.0)

    def test_mean_over_iters(self) -> None:
        # 100ms and 200ms => mean 150ms => 6.667 fps
        assert throughput_fps([100.0, 200.0], frames_per_iter=1) == pytest.approx(
            1000.0 / 150.0
        )

    def test_zero_latency_returns_inf(self) -> None:
        assert throughput_fps([0.0], frames_per_iter=1) == math.inf

    def test_invalid_frames_per_iter(self) -> None:
        with pytest.raises(ValueError, match="frames_per_iter"):
            throughput_fps([100.0], frames_per_iter=0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            throughput_fps([], frames_per_iter=1)


# -------------------------------------------------------- regression_pct


class TestRegressionPct:
    def test_throughput_drop_is_positive(self) -> None:
        # Baseline 100 fps, observed 90 fps, higher_is_better
        # That is a 10% regression.
        assert regression_pct(100.0, 90.0, higher_is_better=True) == pytest.approx(10.0)

    def test_throughput_gain_is_negative(self) -> None:
        assert regression_pct(100.0, 110.0, higher_is_better=True) == pytest.approx(
            -10.0
        )

    def test_latency_increase_is_positive(self) -> None:
        # baseline 100 ms, observed 110 ms, lower_is_better
        # That is a 10% regression.
        assert regression_pct(100.0, 110.0, higher_is_better=False) == pytest.approx(
            10.0
        )

    def test_latency_decrease_is_negative(self) -> None:
        assert regression_pct(100.0, 90.0, higher_is_better=False) == pytest.approx(
            -10.0
        )

    def test_zero_baseline_returns_zero(self) -> None:
        # Avoids division by zero spuriously tripping a CI gate
        assert regression_pct(0.0, 50.0, higher_is_better=True) == 0.0

    def test_negative_baseline_returns_zero(self) -> None:
        assert regression_pct(-1.0, 50.0, higher_is_better=True) == 0.0


# ---------------------------------------------------- LatencyStats accessors


class TestLatencyStats:
    def test_to_dict_contains_all_fields(self) -> None:
        stats = LatencyStats(
            count=3,
            mean_ms=2.0,
            min_ms=1.0,
            max_ms=3.0,
            std_ms=0.5,
            percentiles_ms={50: 2.0},
        )
        d = stats.to_dict()
        assert {"count", "mean_ms", "min_ms", "max_ms", "std_ms", "percentiles_ms"} <= d.keys()

    def test_percentile_round_trip(self) -> None:
        stats = LatencyStats(
            count=1, mean_ms=1.0, min_ms=1.0, max_ms=1.0, std_ms=0.0,
            percentiles_ms={50: 1.0, 90: 1.0, 99: 1.0},
        )
        for p in (50, 90, 99):
            assert stats.percentile(p) == 1.0
