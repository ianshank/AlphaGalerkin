"""Tests for MetricCollector (training/metrics.py)."""
from __future__ import annotations

import pytest

from src.alphagalerkin.training.metrics import MetricCollector


class TestMetricCollectorRecord:
    """Recording metrics."""

    def test_record_stores_value(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 0.5)

        assert mc.get_latest("loss") == pytest.approx(0.5)

    def test_record_multiple_values(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 1.0)
        mc.record("loss", 0.5)
        mc.record("loss", 0.25)

        assert mc.get_latest("loss") == pytest.approx(0.25)

    def test_record_different_names(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 0.5)
        mc.record("accuracy", 0.9)

        assert mc.get_latest("loss") == pytest.approx(0.5)
        assert mc.get_latest("accuracy") == pytest.approx(0.9)


class TestMetricCollectorGetLatest:
    """get_latest queries."""

    def test_get_latest_returns_none_for_unknown(self) -> None:
        mc = MetricCollector()

        assert mc.get_latest("nonexistent") is None

    def test_get_latest_returns_last_recorded(self) -> None:
        mc = MetricCollector()
        mc.record("lr", 0.01)
        mc.record("lr", 0.001)

        assert mc.get_latest("lr") == pytest.approx(0.001)


class TestMetricCollectorGetAverage:
    """get_average queries."""

    def test_get_average_returns_none_for_unknown(self) -> None:
        mc = MetricCollector()

        assert mc.get_average("nonexistent") is None

    def test_get_average_full_window(self) -> None:
        mc = MetricCollector()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            mc.record("x", v)

        avg = mc.get_average("x", window=5)

        assert avg == pytest.approx(3.0)

    def test_get_average_partial_window(self) -> None:
        mc = MetricCollector()
        mc.record("x", 10.0)
        mc.record("x", 20.0)

        # window=5 but only 2 values -> average of those 2
        avg = mc.get_average("x", window=5)

        assert avg == pytest.approx(15.0)

    def test_get_average_truncated_window(self) -> None:
        mc = MetricCollector()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            mc.record("x", v)

        # window=3 -> average of last 3: (3, 4, 5)
        avg = mc.get_average("x", window=3)

        assert avg == pytest.approx(4.0)

    def test_get_average_default_window(self) -> None:
        mc = MetricCollector()
        for v in range(1, 21):
            mc.record("val", float(v))

        # Default window is 10 -> last 10 values: 11..20
        avg = mc.get_average("val")

        expected = sum(range(11, 21)) / 10
        assert avg == pytest.approx(expected)


class TestMetricCollectorIterationSummary:
    """get_iteration_summary drain behavior."""

    def test_iteration_summary_returns_latest_per_name(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 0.5)
        mc.record("loss", 0.3)
        mc.record("lr", 0.01)

        summary = mc.get_iteration_summary()

        assert summary["loss"] == pytest.approx(0.3)
        assert summary["lr"] == pytest.approx(0.01)

    def test_iteration_summary_drains(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 1.0)

        _ = mc.get_iteration_summary()
        second = mc.get_iteration_summary()

        assert second == {}

    def test_iteration_summary_does_not_affect_history(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 1.0)
        mc.record("loss", 2.0)

        _ = mc.get_iteration_summary()

        # Full history is still intact.
        assert mc.get_latest("loss") == pytest.approx(2.0)


class TestMetricCollectorGetFullHistory:
    """get_full_history returns the complete record."""

    def test_full_history_empty(self) -> None:
        mc = MetricCollector()

        assert mc.get_full_history() == {}

    def test_full_history_preserves_order(self) -> None:
        mc = MetricCollector()
        values = [0.9, 0.8, 0.7]
        for v in values:
            mc.record("loss", v)

        history = mc.get_full_history()

        assert history["loss"] == pytest.approx(values)


class TestMetricCollectorReset:
    """reset clears everything."""

    def test_reset_clears_metrics(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 1.0)
        mc.record("accuracy", 0.9)

        mc.reset()

        assert mc.get_latest("loss") is None
        assert mc.get_latest("accuracy") is None
        assert mc.get_full_history() == {}

    def test_reset_clears_iteration_metrics(self) -> None:
        mc = MetricCollector()
        mc.record("loss", 1.0)

        mc.reset()

        assert mc.get_iteration_summary() == {}
