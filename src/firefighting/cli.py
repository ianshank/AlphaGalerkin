"""CLI entry point for firefighting drone benchmarks and edge profiling.

Usage:
    python -m src.firefighting.cli --benchmark grass_fire_50x50
    python -m src.firefighting.cli --benchmark transfer_50_to_500
    python -m src.firefighting.cli --profile --max-memory 4096
"""

from __future__ import annotations

import argparse

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

BENCHMARKS = [
    "grass_fire_50x50",
    "transfer_50_to_500",
    "fds_comparison",
]


def run_grass_fire_benchmark() -> bool:
    """Run canonical 50x50 grass fire benchmark."""
    from src.firefighting.config.fire import FireConfig
    from src.firefighting.config.solver import FireSolverConfig
    from src.firefighting.solver.coupled import CoupledFireSolver

    config = FireSolverConfig(
        name="grass_fire_bench",
        nx=50,
        ny=50,
        domain_size_x_m=500.0,
        domain_size_y_m=500.0,
        dt_s=0.5,
        prediction_horizon_s=120.0,
        max_steps=500,
    )
    fire_config = FireConfig(name="grass_fire")
    solver = CoupledFireSolver(config, fire_config)
    state = solver.create_initial_state(
        ignition_center=(250.0, 250.0),
        ignition_radius_m=20.0,
    )

    wind_u = np.full((50, 50), 3.0)
    wind_v = np.zeros((50, 50))
    result = solver.run(state, wind_u, wind_v, t_final=120.0)

    logger.info(
        "grass_fire_benchmark_complete",
        burned_area_m2=result.burned_area_m2,
        max_temperature_K=result.max_temperature_K,
        total_steps=result.total_steps,
    )
    return True


def run_edge_profile(max_memory_mb: int) -> None:
    """Run edge deployment profiling."""
    from src.firefighting.config.edge import EdgeConfig
    from src.firefighting.edge.profiler import EdgeProfiler, LatencyBreakdown

    config = EdgeConfig(name="profile", max_memory_mb=max_memory_mb)
    profiler = EdgeProfiler(config)

    # Simulate 10 prediction cycles
    for _ in range(10):
        bd = LatencyBreakdown(
            sensor_ingest_ms=40.0,
            mcts_search_ms=200.0,
            pde_solve_ms=80.0,
            output_encode_ms=30.0,
        )
        profiler.record(bd)

    result = profiler.summarize()
    logger.info(
        "edge_profile_complete",
        n_cycles=result.n_cycles,
        mean_latency_ms=result.mean_latency_ms,
        p95_latency_ms=result.p95_latency_ms,
        budget_violations=result.budget_violations,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AlphaGalerkin Firefighting Benchmarks")
    parser.add_argument("--benchmark", choices=BENCHMARKS, help="Run specific benchmark")
    parser.add_argument("--profile", action="store_true", help="Run edge profiling")
    parser.add_argument("--max-memory", type=int, default=4096, help="Max memory (MB)")
    parser.add_argument("--list", action="store_true", help="List benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for b in BENCHMARKS:
            print(f"  - {b}")
        return

    if args.profile:
        run_edge_profile(args.max_memory)
        return

    if args.benchmark == "grass_fire_50x50":
        run_grass_fire_benchmark()
    elif args.benchmark:
        logger.info("benchmark_placeholder", benchmark=args.benchmark)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
