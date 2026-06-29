"""CLI entry point for firefighting drone benchmarks and edge profiling.

Usage:
    python -m src.firefighting.cli --benchmark grass_fire_50x50
    python -m src.firefighting.cli --benchmark transfer_50_to_500
    python -m src.firefighting.cli --profile --max-memory 4096
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    from src.firefighting.solver.coupled import FireSolverResult

logger = structlog.get_logger(__name__)

BENCHMARKS = [
    "grass_fire_50x50",
    "transfer_50_to_500",
    "fds_comparison",
]

# Physical setup shared by the grid-resolution benchmarks. Surfaced as named
# constants (no buried literals) so resolutions/horizon are reproducible.
DOMAIN_SIZE_M: float = 500.0
IGNITION_CENTER_M: tuple[float, float] = (250.0, 250.0)
IGNITION_RADIUS_M: float = 20.0
WIND_SPEED_X_M_S: float = 3.0
DT_S: float = 0.5
TRANSFER_HORIZON_S: float = 120.0
TRANSFER_COARSE_N: int = 50
TRANSFER_FINE_N: int = 500
# Acceptance: the resolution-independent operator should predict a similar
# burned area at both resolutions. Relative burned-area difference floor.
TRANSFER_REL_TOLERANCE: float = 0.5

# Canonical grass-fire benchmark resolution.
GRASS_FIRE_N: int = 50

# Edge-profiling demo inputs (illustrative latency breakdown, named not buried).
EDGE_PROFILE_CYCLES: int = 10
EDGE_SENSOR_INGEST_MS: float = 40.0
EDGE_MCTS_SEARCH_MS: float = 200.0
EDGE_PDE_SOLVE_MS: float = 80.0
EDGE_OUTPUT_ENCODE_MS: float = 30.0


def _run_fire_at_resolution(
    n: int,
    *,
    horizon_s: float = TRANSFER_HORIZON_S,
) -> FireSolverResult:
    """Run a grass-fire spread scenario at an ``n`` x ``n`` resolution.

    Shared by both the grass-fire and resolution-transfer benchmarks (same
    physical scenario, parameterized resolution) so neither re-rolls the setup.

    Returns:
        The full :class:`FireSolverResult` (burned area, max temperature, steps).

    """
    from src.firefighting.config.fire import FireConfig
    from src.firefighting.config.solver import FireSolverConfig
    from src.firefighting.solver.coupled import CoupledFireSolver

    config = FireSolverConfig(
        name=f"grass_fire_{n}x{n}",
        nx=n,
        ny=n,
        domain_size_x_m=DOMAIN_SIZE_M,
        domain_size_y_m=DOMAIN_SIZE_M,
        dt_s=DT_S,
        prediction_horizon_s=horizon_s,
        max_steps=int(horizon_s / DT_S) + 1,
    )
    fire_config = FireConfig(name=f"grass_fire_{n}")
    solver = CoupledFireSolver(config, fire_config)
    state = solver.create_initial_state(
        ignition_center=IGNITION_CENTER_M,
        ignition_radius_m=IGNITION_RADIUS_M,
    )
    wind_u = np.full((n, n), WIND_SPEED_X_M_S)
    wind_v = np.zeros((n, n))
    return solver.run(state, wind_u, wind_v, t_final=horizon_s)


def run_transfer_benchmark(
    *,
    coarse_n: int = TRANSFER_COARSE_N,
    fine_n: int = TRANSFER_FINE_N,
    horizon_s: float = TRANSFER_HORIZON_S,
    rel_tolerance: float = TRANSFER_REL_TOLERANCE,
) -> bool:
    """Zero-shot resolution-transfer benchmark.

    Runs the same physical grass-fire scenario at a coarse and a fine grid
    resolution and compares the predicted burned area. A resolution-independent
    operator should agree across resolutions; the relative burned-area
    difference is reported and gated by ``rel_tolerance``.

    Returns:
        ``True`` if the relative burned-area difference is within tolerance.

    """
    coarse_area = _run_fire_at_resolution(coarse_n, horizon_s=horizon_s).burned_area_m2
    fine_area = _run_fire_at_resolution(fine_n, horizon_s=horizon_s).burned_area_m2

    denom = max(abs(coarse_area), 1e-9)
    rel_diff = abs(fine_area - coarse_area) / denom
    passed = rel_diff <= rel_tolerance

    logger.info(
        "transfer_benchmark_complete",
        coarse_n=coarse_n,
        fine_n=fine_n,
        coarse_burned_area_m2=coarse_area,
        fine_burned_area_m2=fine_area,
        relative_difference=rel_diff,
        rel_tolerance=rel_tolerance,
        passed=passed,
    )
    return passed


def run_grass_fire_benchmark(
    *,
    n: int = GRASS_FIRE_N,
    horizon_s: float = TRANSFER_HORIZON_S,
) -> bool:
    """Run the canonical grass-fire spread benchmark.

    Delegates to the shared :func:`_run_fire_at_resolution` (same physical
    scenario as the transfer benchmark); resolution and horizon are parameters,
    not buried literals.
    """
    result = _run_fire_at_resolution(n, horizon_s=horizon_s)
    logger.info(
        "grass_fire_benchmark_complete",
        n=n,
        burned_area_m2=result.burned_area_m2,
        max_temperature_K=result.max_temperature_K,
        total_steps=result.total_steps,
    )
    return True


def run_edge_profile(max_memory_mb: int, *, n_cycles: int = EDGE_PROFILE_CYCLES) -> None:
    """Run edge deployment profiling over ``n_cycles`` simulated prediction cycles."""
    from src.firefighting.config.edge import EdgeConfig
    from src.firefighting.edge.profiler import EdgeProfiler, LatencyBreakdown

    config = EdgeConfig(name="profile", max_memory_mb=max_memory_mb)
    profiler = EdgeProfiler(config)

    for _ in range(n_cycles):
        profiler.record(
            LatencyBreakdown(
                sensor_ingest_ms=EDGE_SENSOR_INGEST_MS,
                mcts_search_ms=EDGE_MCTS_SEARCH_MS,
                pde_solve_ms=EDGE_PDE_SOLVE_MS,
                output_encode_ms=EDGE_OUTPUT_ENCODE_MS,
            )
        )

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
    elif args.benchmark == "transfer_50_to_500":
        run_transfer_benchmark()
    elif args.benchmark == "fds_comparison":
        # Honest status: this needs an external FDS (Fire Dynamics Simulator)
        # reference dataset that is not bundled with the repo.
        raise NotImplementedError(
            "fds_comparison requires an external FDS reference dataset "
            "(not bundled); wire a reference fixture before enabling."
        )
    elif args.benchmark:
        raise NotImplementedError(f"unknown benchmark: {args.benchmark}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
