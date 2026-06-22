"""CLI entry point for reentry aerodynamics benchmarks.

Usage:
    python -m src.reentry.cli --benchmark sod_shock_tube
    python -m src.reentry.cli --benchmark mach6_cylinder
    python -m src.reentry.cli --benchmark fire2_1636s
    python -m src.reentry.cli --audit-conservation
"""

from __future__ import annotations

import argparse

import numpy as np
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

# Default shock-tube discretization. Surfaced as named constants (not buried
# literals) so the benchmark/audit are reproducible and overridable.
DEFAULT_N_CELLS: int = 200
DEFAULT_GAMMA: float = 1.4  # ratio of specific heats for air
DEFAULT_T_FINAL: float = 0.2  # standard Sod evaluation time

# Shock-tube benchmarks whose initial conditions already exist on ShockTubeIC.
_SHOCK_TUBE_ICS = {
    "sod_shock_tube": "sod",
    "lax_shock_tube": "lax",
    "shu_osher": "shu_osher",
}

# Benchmarks that require infrastructure not yet implemented (2D blunt-body
# mesh + supersonic BCs, or an external reference dataset). Listed honestly so
# the CLI raises instead of silently logging a fake "placeholder" success.
_UNIMPLEMENTED_BENCHMARKS = {
    "mach6_cylinder": (
        "mach6_cylinder needs a 2D StructuredMesh2D blunt-body geometry with "
        "supersonic-inflow/wall BCs and a Mach-6 freestream IC (not yet wired)."
    ),
    "fire2_1636s": (
        "fire2_1636s needs the FIRE-II capsule geometry + trajectory point (not yet wired)."
    ),
    "transfer_50_to_500": (
        "transfer_50_to_500 is a firefighting-domain benchmark; run it via "
        "python -m src.firefighting.cli --benchmark transfer_50_to_500."
    ),
}


def run_shock_tube_benchmark(
    benchmark: str,
    *,
    n_cells: int = DEFAULT_N_CELLS,
    gamma: float = DEFAULT_GAMMA,
    t_final: float = DEFAULT_T_FINAL,
) -> bool:
    """Run a 1D shock-tube validation benchmark (sod / lax / shu_osher)."""
    from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
    from src.reentry.solver.euler_1d import Euler1DSolver, ShockTubeIC

    ic_name = _SHOCK_TUBE_ICS[benchmark]
    config = ReentrySolverConfig(name=f"{benchmark}_bench", flux_scheme=FluxScheme.ROE)
    solver = Euler1DSolver(config, n_cells=n_cells, gamma=gamma)
    ic = getattr(ShockTubeIC, ic_name)()
    result = solver.solve(ic, t_final=t_final)

    logger.info(
        "shock_tube_benchmark_complete",
        benchmark=benchmark,
        n_cells=n_cells,
        n_steps=result.n_steps,
    )
    return True


# Backwards-compatible alias for the original Sod-only entry point.
def run_sod_benchmark() -> bool:
    """Run Sod shock tube validation."""
    return run_shock_tube_benchmark("sod_shock_tube")


def run_audit_conservation(
    *,
    n_cells: int = DEFAULT_N_CELLS,
    gamma: float = DEFAULT_GAMMA,
    t_final: float = DEFAULT_T_FINAL,
    rtol: float | None = None,
) -> bool:
    """Audit mass/energy conservation of the 1D Euler solver on the Sod problem.

    Reconstructs the conserved integrals (mass ``∫ρ``, energy ``∫ρE``) at the
    initial and final times and checks that their relative drift is within
    tolerance. A finite-volume scheme conserves mass and energy to ~machine
    precision while the waves stay interior. Total momentum is reported but not
    gated: it legitimately changes via the net boundary pressure impulse
    (``p_left ≠ p_right``).

    Args:
        n_cells: Grid resolution.
        gamma: Ratio of specific heats.
        t_final: Final time (kept small enough that waves stay interior).
        rtol: Relative-drift tolerance; defaults to the solver config's
            ``conservation_rtol``.

    Returns:
        ``True`` if mass and energy relative drift are within tolerance.

    """
    from src.reentry.config.solver import FluxScheme, ReentrySolverConfig
    from src.reentry.conservation import conservation_drift, conserved_integrals_1d
    from src.reentry.solver.euler_1d import Euler1DSolver, ShockTubeIC

    config = ReentrySolverConfig(name="conservation_audit", flux_scheme=FluxScheme.ROE)
    tol = config.conservation_rtol if rtol is None else rtol

    solver = Euler1DSolver(config, n_cells=n_cells, gamma=gamma)
    ic = ShockTubeIC.sod()
    dx = (ic.x_max - ic.x_min) / n_cells

    # Reconstruct the initial primitive field on the solver grid.
    x = np.linspace(ic.x_min + 0.5 * dx, ic.x_max - 0.5 * dx, n_cells)
    left = x <= ic.x_diaphragm
    rho0 = np.where(left, ic.rho_l, ic.rho_r)
    u0 = np.where(left, ic.u_l, ic.u_r)
    p0 = np.where(left, ic.p_l, ic.p_r)
    initial = conserved_integrals_1d(rho0, u0, p0, dx, gamma)

    result = solver.solve(ic, t_final=t_final)
    final = conserved_integrals_1d(result.density, result.velocity, result.pressure, dx, gamma)

    drift = conservation_drift(initial, final)
    passed = drift["mass"]["relative"] <= tol and drift["energy"]["relative"] <= tol

    logger.info(
        "conservation_audit",
        passed=passed,
        rtol=tol,
        mass_rel_drift=drift["mass"]["relative"],
        energy_rel_drift=drift["energy"]["relative"],
        momentum_abs_drift=drift["momentum"]["absolute"],
        n_steps=result.n_steps,
    )
    return passed


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

    if args.benchmark in _SHOCK_TUBE_ICS:
        run_shock_tube_benchmark(args.benchmark)
    elif args.benchmark in _UNIMPLEMENTED_BENCHMARKS:
        raise NotImplementedError(_UNIMPLEMENTED_BENCHMARKS[args.benchmark])
    elif args.benchmark:
        raise NotImplementedError(f"unknown benchmark: {args.benchmark}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
