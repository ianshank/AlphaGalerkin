"""CLI entry point for reentry aerodynamics benchmarks.

Usage:
    python -m src.reentry.cli --benchmark sod_shock_tube
    python -m src.reentry.cli --benchmark mach6_cylinder
    python -m src.reentry.cli --benchmark fire2_1636s
    python -m src.reentry.cli --audit-conservation
"""

from __future__ import annotations

import argparse

import structlog

logger = structlog.get_logger(__name__)

BENCHMARKS = [
    "sod_shock_tube",
    "lax_shock_tube",
    "shu_osher",
    "mach6_cylinder",
    "fire2_1636s",
    "transfer_50_to_500",
]


def run_sod_benchmark() -> bool:
    """Run Sod shock tube validation."""
    from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
    from src.reentry.solver.euler_1d import Euler1DSolver, ShockTubeIC

    config = ReentrySolverConfig(name="sod_bench", flux_scheme=FluxScheme.ROE)
    solver = Euler1DSolver(config, n_cells=200, gamma=1.4)
    ic = ShockTubeIC.sod()
    result = solver.solve(ic, t_final=0.2)

    logger.info("sod_benchmark_complete", n_cells=200, n_steps=result.n_steps)
    return True


def run_audit_conservation() -> bool:
    """Audit conservation of mass/momentum/energy."""
    logger.info("conservation_audit", status="placeholder")
    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="AlphaGalerkin Reentry Benchmarks")
    parser.add_argument("--benchmark", choices=BENCHMARKS, help="Run specific benchmark")
    parser.add_argument("--audit-conservation", action="store_true", help="Run conservation audit")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for b in BENCHMARKS:
            print(f"  - {b}")
        return

    if args.audit_conservation:
        run_audit_conservation()
        return

    if args.benchmark == "sod_shock_tube":
        run_sod_benchmark()
    elif args.benchmark:
        logger.info("benchmark_placeholder", benchmark=args.benchmark)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
