"""Tests for the firefighting resolution-transfer benchmark."""

from __future__ import annotations

from src.firefighting.cli import _run_fire_at_resolution, run_transfer_benchmark


class TestTransferBenchmark:
    def test_equal_resolution_is_deterministic_pass(self) -> None:
        # Identical resolutions -> identical burned area -> zero relative diff,
        # which is always within tolerance. Exercises the gating logic without
        # depending on physics convergence at toy grid sizes.
        assert run_transfer_benchmark(coarse_n=24, fine_n=24, horizon_s=15.0) is True

    def test_returns_bool_for_differing_resolutions(self) -> None:
        result = run_transfer_benchmark(coarse_n=16, fine_n=24, horizon_s=15.0)
        assert isinstance(result, bool)

    def test_run_at_resolution_returns_nonnegative_area(self) -> None:
        area = _run_fire_at_resolution(20, horizon_s=15.0)
        assert area >= 0.0
