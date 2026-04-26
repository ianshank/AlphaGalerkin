"""Tests for edge deployment modules: memory manager and profiler."""

from __future__ import annotations

import numpy as np
import pytest

from src.firefighting.config.edge import EdgeConfig
from src.firefighting.edge.memory import MemoryBudgetManager
from src.firefighting.edge.profiler import CycleTimer, EdgeProfiler, LatencyBreakdown


class TestMemoryBudgetManager:
    def test_allocation_within_budget(self) -> None:
        config = EdgeConfig(name="test", max_memory_mb=4096)
        mgr = MemoryBudgetManager(config)
        assert mgr.register_allocation("model", 500 * 1024 * 1024)  # 500 MB
        assert mgr.check_budget()

    def test_allocation_exceeds_budget(self) -> None:
        config = EdgeConfig(name="test", max_memory_mb=512)
        mgr = MemoryBudgetManager(config)
        result = mgr.register_allocation("model", 600 * 1024 * 1024)  # 600 MB > 512
        assert result is False

    def test_release_frees_space(self) -> None:
        config = EdgeConfig(name="test", max_memory_mb=512)
        mgr = MemoryBudgetManager(config)
        mgr.register_allocation("temp", 200 * 1024 * 1024)
        mgr.release_allocation("temp")
        snap = mgr.snapshot()
        assert snap.total_mb == pytest.approx(0.0)

    def test_snapshot(self) -> None:
        config = EdgeConfig(name="test", max_memory_mb=4096)
        mgr = MemoryBudgetManager(config)
        mgr.register_allocation("model", 500 * 1024 * 1024)
        mgr.register_allocation("working", 200 * 1024 * 1024)
        snap = mgr.snapshot()
        assert snap.model_mb == pytest.approx(500.0, rel=0.01)
        assert snap.working_mb == pytest.approx(200.0, rel=0.01)
        assert snap.headroom_mb > 3000

    def test_estimate_array_mb(self) -> None:
        config = EdgeConfig(name="test")
        mgr = MemoryBudgetManager(config)
        mb = mgr.estimate_array_mb((100, 100), dtype=np.float32)
        expected = 100 * 100 * 4 / (1024 * 1024)
        assert mb == pytest.approx(expected)


class TestLatencyBreakdown:
    def test_total(self) -> None:
        bd = LatencyBreakdown(
            sensor_ingest_ms=50,
            mcts_search_ms=300,
            pde_solve_ms=100,
            output_encode_ms=50,
        )
        assert bd.total_ms == 500.0

    def test_within_budget(self) -> None:
        config = EdgeConfig(name="test", max_latency_ms=500.0)
        bd = LatencyBreakdown(
            sensor_ingest_ms=50,
            mcts_search_ms=200,
            pde_solve_ms=100,
            output_encode_ms=50,
        )
        assert bd.within_budget(config)

    def test_over_budget(self) -> None:
        config = EdgeConfig(name="test", max_latency_ms=500.0)
        bd = LatencyBreakdown(
            sensor_ingest_ms=100,
            mcts_search_ms=400,
            pde_solve_ms=100,
            output_encode_ms=50,
        )
        assert not bd.within_budget(config)


class TestEdgeProfiler:
    def test_summarize_empty(self) -> None:
        config = EdgeConfig(name="test")
        profiler = EdgeProfiler(config)
        result = profiler.summarize()
        assert result.n_cycles == 0

    def test_record_and_summarize(self) -> None:
        config = EdgeConfig(name="test", max_latency_ms=500.0)
        profiler = EdgeProfiler(config)

        for _ in range(10):
            bd = LatencyBreakdown(
                sensor_ingest_ms=40,
                mcts_search_ms=200,
                pde_solve_ms=80,
                output_encode_ms=30,
            )
            profiler.record(bd)

        result = profiler.summarize()
        assert result.n_cycles == 10
        assert result.mean_latency_ms == pytest.approx(350.0)
        assert result.budget_violations == 0

    def test_cycle_timer(self) -> None:
        config = EdgeConfig(name="test")
        timer = CycleTimer(config)
        timer.start_stage("sensor_ingest")
        timer.end_stage("sensor_ingest")
        assert timer.breakdown.sensor_ingest_ms >= 0
