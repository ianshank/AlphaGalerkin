"""Benchmark smoke tests for firefighting module."""

from __future__ import annotations

import numpy as np

from src.firefighting.config.edge import EdgeConfig
from src.firefighting.config.fire import FireConfig
from src.firefighting.config.solver import FireSolverConfig
from src.firefighting.edge.profiler import EdgeProfiler, LatencyBreakdown
from src.firefighting.solver.coupled import CoupledFireSolver


class TestGrassFireBenchmarkSmoke:
    """Smoke test: grass fire runs and produces spread."""

    def test_50x50_grass_fire_runs(self) -> None:
        config = FireSolverConfig(
            name="bench",
            nx=30,
            ny=30,
            domain_size_x_m=300.0,
            domain_size_y_m=300.0,
            dt_s=0.5,
            prediction_horizon_s=30.0,
            max_steps=80,
        )
        fire_config = FireConfig(name="bench")
        solver = CoupledFireSolver(config, fire_config)
        state = solver.create_initial_state(
            ignition_center=(150.0, 150.0),
            ignition_radius_m=15.0,
        )
        wind_u = np.full((30, 30), 3.0)
        wind_v = np.zeros((30, 30))
        result = solver.run(state, wind_u, wind_v, t_final=30.0)

        assert result.total_steps > 0
        assert result.max_temperature_K > fire_config.ambient_temperature_K
        assert result.burned_area_m2 > 0


class TestEdgeLatencyBenchmarkSmoke:
    """Smoke test: edge profiling works."""

    def test_profiling_within_budget(self) -> None:
        config = EdgeConfig(name="bench", max_latency_ms=500.0)
        profiler = EdgeProfiler(config)

        for _ in range(20):
            bd = LatencyBreakdown(
                sensor_ingest_ms=40,
                mcts_search_ms=200,
                pde_solve_ms=80,
                output_encode_ms=30,
            )
            profiler.record(bd)

        result = profiler.summarize()
        assert result.n_cycles == 20
        assert result.budget_violations == 0
        assert result.mean_latency_ms < 500.0
