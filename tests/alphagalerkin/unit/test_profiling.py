"""Tests for profiling utilities (utils/profiling.py)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.alphagalerkin.utils.profiling import (
    PerformanceTracker,
    cprofile_section,
    get_global_tracker,
    profile_function,
)


class TestPerformanceTrackerTrack:
    """PerformanceTracker.track accumulates timings."""

    def test_single_section(self) -> None:
        tracker = PerformanceTracker()

        with tracker.track("forward"):
            _ = sum(range(100))

        report = tracker.report()
        assert "forward" in report
        assert report["forward"]["call_count"] == 1.0
        assert report["forward"]["total_time"] >= 0.0

    def test_multiple_calls(self) -> None:
        tracker = PerformanceTracker()

        for _ in range(3):
            with tracker.track("step"):
                pass

        report = tracker.report()
        assert report["step"]["call_count"] == 3.0

    def test_multiple_sections(self) -> None:
        tracker = PerformanceTracker()

        with tracker.track("encode"):
            pass
        with tracker.track("decode"):
            pass

        report = tracker.report()
        assert "encode" in report
        assert "decode" in report

    def test_min_max_time(self) -> None:
        tracker = PerformanceTracker()

        with tracker.track("work"):
            time.sleep(0.01)
        with tracker.track("work"):
            time.sleep(0.02)

        report = tracker.report()
        assert report["work"]["min_time"] <= report["work"]["max_time"]

    def test_mean_time(self) -> None:
        tracker = PerformanceTracker()

        with tracker.track("op"):
            pass
        with tracker.track("op"):
            pass

        report = tracker.report()
        expected_mean = (
            report["op"]["total_time"] / report["op"]["call_count"]
        )
        assert report["op"]["mean_time"] == pytest.approx(
            expected_mean, abs=1e-5,
        )


class TestPerformanceTrackerReport:
    """PerformanceTracker.report returns structured results."""

    def test_empty_report(self) -> None:
        tracker = PerformanceTracker()
        report = tracker.report()

        assert report == {}

    def test_report_keys(self) -> None:
        tracker = PerformanceTracker()
        with tracker.track("x"):
            pass

        report = tracker.report()
        expected_keys = {
            "total_time",
            "call_count",
            "mean_time",
            "min_time",
            "max_time",
        }

        assert set(report["x"].keys()) == expected_keys


class TestPerformanceTrackerReset:
    """PerformanceTracker.reset clears state."""

    def test_reset_clears_all(self) -> None:
        tracker = PerformanceTracker()
        with tracker.track("a"):
            pass

        tracker.reset()

        assert tracker.report() == {}


class TestProfileFunction:
    """profile_function decorator."""

    def test_bare_decorator(self) -> None:
        tracker = PerformanceTracker()

        @profile_function(tracker=tracker)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)

        assert result == 5
        report = tracker.report()
        # section name is based on module.qualname
        assert len(report) == 1
        section = next(iter(report.values()))
        assert section["call_count"] == 1.0

    def test_custom_name(self) -> None:
        tracker = PerformanceTracker()

        @profile_function(tracker=tracker, name="my_op")
        def noop() -> None:
            pass

        noop()

        report = tracker.report()
        assert "my_op" in report

    def test_bare_decorator_no_parens(self) -> None:
        """@profile_function without parentheses uses global tracker."""
        # Save global tracker state.
        global_tracker = get_global_tracker()
        global_tracker.reset()

        @profile_function
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply(3, 4)

        assert result == 12
        report = global_tracker.report()
        assert len(report) >= 1

    def test_preserves_return_value(self) -> None:
        tracker = PerformanceTracker()

        @profile_function(tracker=tracker)
        def make_list() -> list[int]:
            return [1, 2, 3]

        assert make_list() == [1, 2, 3]

    def test_multiple_calls_accumulate(self) -> None:
        tracker = PerformanceTracker()

        @profile_function(tracker=tracker, name="inc")
        def inc(x: int) -> int:
            return x + 1

        for _ in range(5):
            inc(0)

        report = tracker.report()
        assert report["inc"]["call_count"] == 5.0


class TestGetGlobalTracker:
    """get_global_tracker returns the singleton."""

    def test_returns_performance_tracker(self) -> None:
        tracker = get_global_tracker()

        assert isinstance(tracker, PerformanceTracker)

    def test_returns_same_instance(self) -> None:
        a = get_global_tracker()
        b = get_global_tracker()

        assert a is b


class TestCprofileSection:
    """cprofile_section runs cProfile over a block."""

    def test_no_output_path(self) -> None:
        with cprofile_section() as prof:
            _ = sum(range(1000))

        # prof should be a cProfile.Profile
        import cProfile

        assert isinstance(prof, cProfile.Profile)

    def test_writes_output_file(self, tmp_path: Path) -> None:
        output = tmp_path / "profile.prof"

        with cprofile_section(output_path=output) as prof:
            _ = [i * 2 for i in range(100)]

        assert output.exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = tmp_path / "sub" / "dir" / "profile.prof"

        with cprofile_section(output_path=output):
            pass

        assert output.exists()

    def test_custom_sort_and_topn(self) -> None:
        # Should not raise with custom sort_by and top_n.
        with cprofile_section(sort_by="tottime", top_n=5):
            _ = sum(range(100))
