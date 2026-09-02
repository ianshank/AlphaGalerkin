"""Commit the artifact behind the charter's adaptive-vs-uniform claim.

The charter's evidence register carries the row *"L-shape adaptive Dörfler vs
uniform at matched DOF | Dörfler 5-9x worse"* and cites
``results/lshape_mcts_vs_dorfler.csv``. That file's ``method`` column contains
only ``{dorfler, mcts}``: **there is no uniform-refinement arm in any committed
artifact**, so the number traces to prose in
``docs/NEXT_STEPS_REVIEW_2026-08-18.md`` rather than to data. The claim is
correct -- it reproduces -- but its provenance was not, and the charter's own
guard could not tell, because it checks only that the cited file exists.

This script produces the missing artifact. Both arms share one solver
(``make_solve_fn``), one geometry predicate and one refinement primitive
(``DorflerAMRSolver._refine_grid``); only the marking differs -- Dörfler bulk
marking against "mark every element".

Run::

    python -m scripts.run_adaptive_vs_uniform

Writes ``results/lshape_adaptive_vs_uniform.csv`` and its ``.run.json`` sidecar.
"""

from __future__ import annotations

import argparse
import csv

# timezone.utc rather than datetime.UTC: the latter is 3.11+ and this repository
# supports 3.10 (requires-python = ">=3.10", and CI runs a 3.10 job).
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from numpy.typing import NDArray

from src.constants import DEFAULT_RATIO_FLOOR
from src.pde.config import PDEConfig, PDEType
from src.pde.operators import LShapedPoissonOperator
from src.research.baselines import DorflerAMRSolver
from src.research.lshape_amr_compare import (
    ComparisonParams,
    lshape_inside_predicate,
    make_solve_fn,
    run_dorfler_arm,
)
from src.research.run_manifest import (
    ArmProvenance,
    RunManifest,
    collect_git_provenance,
    collect_package_versions,
    manifest_path_for,
    write_run_manifest,
)

logger = structlog.get_logger(__name__)

#: Default artifact location, relative to the repository root.
DEFAULT_OUTPUT: str = "results/lshape_adaptive_vs_uniform.csv"
#: Hard ceiling on uniform refinement levels; uniform quadruples DOF each level,
#: so this is a runaway guard rather than a tuning knob.
MAX_UNIFORM_LEVELS: int = 12
#: Matched-DOF readings taken at this many log-spaced points.
N_MATCHED_READINGS: int = 4
#: Floor applied to any ratio denominator. Sourced from src.constants: this was
#: the FOURTH independent 1e-15 in the tree, found by a dead-code audit the day
#: after DEFAULT_RATIO_FLOOR's docstring promised a fourth could not drift.
RATIO_FLOOR: float = DEFAULT_RATIO_FLOOR


def build_operator(scale: float) -> LShapedPoissonOperator:
    """The standard L-shaped Poisson benchmark operator."""
    return LShapedPoissonOperator(
        PDEConfig(
            name="poisson_lshaped",
            pde_type=PDEType.POISSON,
            domain_dim=2,
            domain_min=[-scale, -scale],
            domain_max=[scale, scale],
        )
    )


def run_uniform_arm(solve_fn: Any, params: ComparisonParams) -> list[tuple[int, int, float]]:
    """Refine every element each level, on the same solver as the Dörfler arm.

    Returns:
        ``(level, n_dof, l2_error)`` rows.

    """
    xs = np.linspace(-params.scale, params.scale, params.initial_side + 1)
    ys = np.linspace(-params.scale, params.scale, params.initial_side + 1)
    rows: list[tuple[int, int, float]] = []
    for level in range(MAX_UNIFORM_LEVELS):
        solved = solve_fn(xs, ys)
        rows.append((level, int(solved.n_dof), float(solved.l2_error)))
        if solved.n_dof >= params.max_dof or solved.l2_error < params.error_tolerance:
            break
        all_marked_x = np.ones(len(xs) - 1, dtype=bool)
        all_marked_y = np.ones(len(ys) - 1, dtype=bool)
        xs = DorflerAMRSolver._refine_grid(xs, all_marked_x)
        ys = DorflerAMRSolver._refine_grid(ys, all_marked_y)
    return rows


def _log_interp(target: float, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> float:
    """Log-log interpolation, matching ``lshape_amr_compare._interp_log``."""
    return float(np.exp(np.interp(np.log(target), np.log(xs), np.log(ys))))


def _rate(dofs: NDArray[np.float64], errors: NDArray[np.float64]) -> float:
    """Least-squares log-log convergence exponent."""
    return float(np.polyfit(np.log(dofs), np.log(errors), 1)[0])


def compare(
    uniform: list[tuple[int, int, float]],
    dorfler_dofs: NDArray[np.float64],
    dorfler_errors: NDArray[np.float64],
) -> dict[str, float]:
    """Matched-DOF ratios and per-arm convergence rates.

    Ratios are ``dorfler / uniform``: above 1 means adaptive marking is *worse*,
    which is the claim being evidenced.
    """
    uniform_dofs = np.array([row[1] for row in uniform], dtype=np.float64)
    uniform_errors = np.array([row[2] for row in uniform], dtype=np.float64)
    low = max(uniform_dofs[0], dorfler_dofs[0])
    high = min(uniform_dofs[-1], dorfler_dofs[-1])

    metrics: dict[str, float] = {
        "matched_dof_min": float(low),
        "matched_dof_max": float(high),
        "uniform_convergence_exponent": _rate(uniform_dofs, uniform_errors),
        "dorfler_convergence_exponent": _rate(dorfler_dofs, dorfler_errors),
    }
    ratios: list[float] = []
    for target in np.geomspace(max(low, uniform_dofs[1]), high, N_MATCHED_READINGS):
        u = max(_log_interp(target, uniform_dofs, uniform_errors), RATIO_FLOOR)
        d = _log_interp(target, dorfler_dofs, dorfler_errors)
        ratios.append(d / u)
        metrics[f"dorfler_over_uniform_at_dof_{int(round(target))}"] = d / u
    metrics["dorfler_over_uniform_min"] = float(min(ratios))
    metrics["dorfler_over_uniform_max"] = float(max(ratios))
    return metrics


def export_csv(
    path: Path,
    uniform: list[tuple[int, int, float]],
    dorfler_rows: list[tuple[int, int, float]],
) -> Path:
    """Write both arms to one CSV, so the artifact contains the comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        # lineterminator="\n": csv.writer defaults to CRLF, which would make this
        # the only committed artifact with Windows line endings and produce a
        # spurious whole-file diff on every regeneration.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["problem", "method", "refinement_level", "n_dof", "l2_error", "error_per_dof"]
        )
        for method, rows in (("uniform", uniform), ("dorfler", dorfler_rows)):
            for level, n_dof, l2 in rows:
                writer.writerow(
                    [
                        "poisson_lshaped",
                        method,
                        level,
                        n_dof,
                        f"{l2:.8e}",
                        f"{l2 / max(n_dof, 1):.8e}",
                    ]
                )
    return path


def build_parser() -> argparse.ArgumentParser:
    """CLI parser (extracted so tests call the real one)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV artifact path.")
    parser.add_argument("--initial-side", type=int, default=4)
    parser.add_argument("--max-dof", type=int, default=2200)
    parser.add_argument("--marking-fraction", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run both arms, write the artifact and its provenance sidecar."""
    args = build_parser().parse_args(argv)
    params = ComparisonParams(
        scale=args.scale,
        initial_side=args.initial_side,
        max_dof=args.max_dof,
        marking_fraction=args.marking_fraction,
        max_refinements=40,
        error_tolerance=1e-6,
    )
    operator = build_operator(params.scale)
    solve_fn = make_solve_fn(operator, lshape_inside_predicate(params.scale))

    dorfler = run_dorfler_arm(operator, solve_fn, params)
    dorfler_rows = [(p.level, p.n_dof, p.l2_error) for p in dorfler.points]
    uniform = run_uniform_arm(solve_fn, params)

    metrics = compare(
        uniform,
        np.array([r[1] for r in dorfler_rows], dtype=np.float64),
        np.array([r[2] for r in dorfler_rows], dtype=np.float64),
    )

    output = Path(args.output)
    export_csv(output, uniform, dorfler_rows)
    write_run_manifest(
        RunManifest(
            run_id=f"adaptive-vs-uniform-{params.initial_side}-{params.max_dof}",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            harness="scripts.run_adaptive_vs_uniform",
            config=vars(args),
            git=collect_git_provenance(),
            packages=collect_package_versions(),
            arms=[
                ArmProvenance(
                    name="uniform",
                    parameters={"marking": "all elements"},
                    counters={"levels": float(len(uniform))},
                ),
                ArmProvenance(
                    name="dorfler",
                    parameters={"marking_fraction": params.marking_fraction},
                    counters={"levels": float(len(dorfler_rows))},
                ),
            ],
            metrics=metrics,
            artifacts={"csv": str(output)},
            notes=(
                "Evidences the charter row 'L-shape adaptive Doerfler vs uniform at "
                "matched DOF'. Both arms share one solver, one geometry predicate and "
                "one refinement primitive; only the marking differs. Ratios are "
                "dorfler/uniform, so above 1 means adaptive marking is WORSE -- which "
                "is the point: on a tensor-product substrate, refining one element "
                "inserts full grid lines, so the refinement budget is spent away from "
                "the singularity. This is why no marking-policy comparison on this "
                "substrate measures policy quality."
            ),
        ),
        manifest_path_for(output),
    )

    logger.info(
        "adaptive_vs_uniform_done",
        csv=str(output),
        manifest=str(manifest_path_for(output)),
        ratio_min=metrics["dorfler_over_uniform_min"],
        ratio_max=metrics["dorfler_over_uniform_max"],
    )
    print(f"wrote {output} and {manifest_path_for(output)}")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
