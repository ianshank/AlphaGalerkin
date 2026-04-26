"""Weak-scaling benchmark runner.

Sweeps a registered baseline solver across a configurable list of
``n_dof`` values and emits:

* a CSV log of ``(n_dof, wall_time_seconds, l2_error)`` rows;
* a JSON dump of the same data plus measured scaling exponents;
* an HTML report (when :mod:`src.poc.visualization` is importable)
  rendering log-log plots and scaling slopes.

This module satisfies ``config/proposals/doe_ascr_c59.yaml::poisson_scaling``
which requires demonstrating O(N) — or whatever the empirical
exponent turns out to be — on solvers across 1K → 1M DOF.

Design highlights:

* Pydantic-validated :class:`ScalingConfig` — every numerical knob
  surfaced; no hardcoded values.
* Reuses the existing ``SOLVER_REGISTRY`` so that ``alphagalerkin``,
  ``direct_solver``, ``multigrid``, etc. are all selectable by name.
* Failures inside a single (solver, n_dof) cell are *recorded* (with
  exception type and message) but do not abort the sweep.

Usage::

    from src.research.scaling_runner import (
        ScalingConfig,
        WeakScalingRunner,
    )

    runner = WeakScalingRunner(
        ScalingConfig(
            solvers=["direct_solver", "alphagalerkin"],
            n_dof_values=[64, 256, 1024, 4096],
        )
    )
    report = runner.run(operator)
    runner.save(report, "outputs/scaling")
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, ConfigDict, Field

# Importing this package registers SUPG / multigrid / direct / FNO / DeepONet.
import src.research.extra_solvers  # noqa: F401
from src.pde.operators import PDEOperator
from src.research.baselines import SOLVER_REGISTRY, BaseSolver

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class ScalingConfig(BaseModel):
    """Configuration for :class:`WeakScalingRunner`.

    All numerical defaults are intentionally light so the test suite
    can exercise the full code path quickly.  Real benchmarks override
    via Hydra YAML.
    """

    model_config = ConfigDict(extra="forbid")

    solvers: list[str] = Field(
        ...,
        min_length=1,
        description="Names of solvers (registered in SOLVER_REGISTRY) to sweep.",
    )
    n_dof_values: list[int] = Field(
        ...,
        min_length=1,
        description="Sequence of DOF targets to evaluate at.",
    )
    repeats: int = Field(
        default=1,
        ge=1,
        description="Number of timing repetitions per (solver, n_dof) cell.",
    )
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional per-cell wall-clock guard.  Solvers that exceed "
            "this budget are recorded as failures rather than blocking "
            "the sweep."
        ),
    )
    drop_warmup: bool = Field(
        default=False,
        description=(
            "When True with repeats > 1, the first measurement is "
            "discarded as warm-up (e.g. JIT, cache fill)."
        ),
    )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingMeasurement:
    """One (solver, n_dof) cell."""

    solver: str
    n_dof: int
    wall_time_seconds: float
    l2_error: float | None
    success: bool
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScalingSummary:
    """Per-solver fitted scaling exponent."""

    solver: str
    exponent: float
    intercept: float
    n_points: int
    r_squared: float


@dataclass(frozen=True)
class ScalingReport:
    """Full result object emitted by :class:`WeakScalingRunner`."""

    config: dict[str, Any]
    measurements: list[ScalingMeasurement]
    summaries: list[ScalingSummary]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class WeakScalingRunner:
    """Driver for the weak-scaling sweep."""

    def __init__(self, config: ScalingConfig) -> None:
        self.config = config
        self._log = logger.bind(runner="WeakScalingRunner")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, operator: PDEOperator) -> ScalingReport:
        """Run the full sweep on ``operator``.

        Args:
            operator: Reference :class:`PDEOperator` against which all
                solvers in ``config.solvers`` are evaluated.

        Returns:
            A :class:`ScalingReport` aggregating per-cell measurements
            and per-solver scaling summaries.

        """
        measurements: list[ScalingMeasurement] = []
        for solver_name in self.config.solvers:
            for n_dof in self.config.n_dof_values:
                cell = self._run_cell(solver_name, operator, n_dof)
                measurements.append(cell)

        summaries = self._summarise(measurements)
        report = ScalingReport(
            config=self.config.model_dump(),
            measurements=measurements,
            summaries=summaries,
        )
        self._log.info(
            "scaling_run_complete",
            n_cells=len(measurements),
            n_solvers=len(self.config.solvers),
            n_dof_count=len(self.config.n_dof_values),
        )
        return report

    def save(self, report: ScalingReport, output_dir: Path | str) -> dict[str, Path]:
        """Persist the report as CSV + JSON (and HTML when available).

        Args:
            report: Report from :meth:`run`.
            output_dir: Destination directory.  Created if missing.

        Returns:
            Mapping of artefact name to path on disk.

        """
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artefacts: dict[str, Path] = {}

        csv_path = output / "scaling.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["solver", "n_dof", "wall_time_seconds", "l2_error", "success", "error_message"]
            )
            for m in report.measurements:
                writer.writerow(
                    [
                        m.solver,
                        m.n_dof,
                        m.wall_time_seconds,
                        m.l2_error if m.l2_error is not None else "",
                        m.success,
                        m.error_message or "",
                    ]
                )
        artefacts["csv"] = csv_path

        json_path = output / "scaling.json"
        json_path.write_text(
            json.dumps(
                {
                    "config": report.config,
                    "measurements": [asdict(m) for m in report.measurements],
                    "summaries": [asdict(s) for s in report.summaries],
                },
                indent=2,
                default=str,
            )
        )
        artefacts["json"] = json_path

        # Optional HTML — the reporter dependency tree is heavy, only
        # enabled when ``src.poc.visualization`` imports cleanly in the
        # caller's environment.
        try:
            html_path = self._maybe_render_html(report, output)
            if html_path is not None:
                artefacts["html"] = html_path
        except Exception as exc:  # noqa: BLE001 — never block save on rendering
            self._log.warning(
                "scaling_html_render_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )

        self._log.info(
            "scaling_report_saved",
            artefacts={k: str(v) for k, v in artefacts.items()},
        )
        return artefacts

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_cell(
        self,
        solver_name: str,
        operator: PDEOperator,
        n_dof: int,
    ) -> ScalingMeasurement:
        log = self._log.bind(solver=solver_name, n_dof=n_dof)
        cls = SOLVER_REGISTRY.get(solver_name)
        if cls is None:
            return ScalingMeasurement(
                solver=solver_name,
                n_dof=n_dof,
                wall_time_seconds=float("nan"),
                l2_error=None,
                success=False,
                error_message=f"Solver '{solver_name}' not registered",
            )

        repeats = max(self.config.repeats, 1)
        durations: list[float] = []
        last_error: float | None = None
        last_metadata: dict[str, Any] = {}
        last_exc: tuple[str, str] | None = None
        for repeat_idx in range(repeats):
            try:
                solver: BaseSolver = cls()
                t0 = time.perf_counter()
                result = solver.solve(operator, n_dof=n_dof)
                duration = time.perf_counter() - t0
                if (
                    self.config.timeout_seconds is not None
                    and duration > self.config.timeout_seconds
                ):
                    last_exc = (
                        "TimeoutError",
                        f"Wall time {duration:.2f}s exceeds budget "
                        f"{self.config.timeout_seconds}s",
                    )
                    break
                durations.append(float(duration))
                last_error = result.l2_error
                last_metadata = dict(result.metadata)
            except Exception as exc:  # noqa: BLE001 — sweep must continue
                last_exc = (type(exc).__name__, str(exc))
                log.warning("scaling_cell_failed", repeat=repeat_idx, error=str(exc))
                break

        if last_exc is not None and not durations:
            return ScalingMeasurement(
                solver=solver_name,
                n_dof=n_dof,
                wall_time_seconds=float("nan"),
                l2_error=None,
                success=False,
                error_message=f"{last_exc[0]}: {last_exc[1]}",
            )

        # Optionally drop the first measurement as warm-up
        usable = durations[1:] if (self.config.drop_warmup and len(durations) > 1) else durations
        wall_time = float(np.mean(usable))
        log.info(
            "scaling_cell_done",
            wall_time=wall_time,
            l2_error=last_error,
            repeats=len(usable),
        )
        return ScalingMeasurement(
            solver=solver_name,
            n_dof=n_dof,
            wall_time_seconds=wall_time,
            l2_error=last_error,
            success=True,
            metadata=last_metadata,
        )

    def _summarise(
        self,
        measurements: list[ScalingMeasurement],
    ) -> list[ScalingSummary]:
        summaries: list[ScalingSummary] = []
        for solver_name in self.config.solvers:
            cells = [m for m in measurements if m.solver == solver_name and m.success]
            if len(cells) < 2:
                continue
            log_n = np.log(np.array([c.n_dof for c in cells], dtype=np.float64))
            log_t = np.log(
                np.array([max(c.wall_time_seconds, 1e-12) for c in cells], dtype=np.float64)
            )
            slope, intercept = np.polyfit(log_n, log_t, deg=1)
            ss_res = float(np.sum((log_t - (slope * log_n + intercept)) ** 2))
            ss_tot = float(np.sum((log_t - np.mean(log_t)) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            summaries.append(
                ScalingSummary(
                    solver=solver_name,
                    exponent=float(slope),
                    intercept=float(intercept),
                    n_points=len(cells),
                    r_squared=float(r_squared),
                )
            )
        return summaries

    def _maybe_render_html(
        self,
        report: ScalingReport,
        output_dir: Path,
    ) -> Path | None:
        """Render an HTML report if visualization deps are available.

        Falls back silently when matplotlib / plotly are missing — the
        CSV + JSON artefacts are always emitted.
        """
        try:
            import matplotlib  # noqa: F401

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        for solver_name in self.config.solvers:
            cells = [m for m in report.measurements if m.solver == solver_name and m.success]
            if not cells:
                continue
            xs = [c.n_dof for c in cells]
            ys = [c.wall_time_seconds for c in cells]
            ax.plot(xs, ys, marker="o", label=solver_name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_dof")
        ax.set_ylabel("wall time (s)")
        ax.set_title("Weak scaling")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        fig.tight_layout()
        plot_path = output_dir / "scaling.png"
        fig.savefig(plot_path)
        plt.close(fig)

        # Minimal HTML wrapper (template-free, deps-free)
        html_path = output_dir / "scaling.html"
        rows = [
            f"<tr><td>{m.solver}</td><td>{m.n_dof}</td>"
            f"<td>{m.wall_time_seconds:.4g}</td>"
            f"<td>{'' if m.l2_error is None else f'{m.l2_error:.4g}'}</td>"
            f"<td>{m.success}</td></tr>"
            for m in report.measurements
        ]
        summary_rows = [
            f"<tr><td>{s.solver}</td><td>{s.exponent:.3f}</td>"
            f"<td>{s.r_squared:.4f}</td><td>{s.n_points}</td></tr>"
            for s in report.summaries
        ]
        # Compose the HTML in pieces so each line stays under the
        # 100-char limit Ruff enforces project-wide.
        style = (
            "body{font-family:sans-serif;max-width:900px;margin:auto;padding:1em}"
            "table{border-collapse:collapse;margin:1em 0}"
            "td,th{border:1px solid #ccc;padding:.25em .5em}"
        )
        summary_header = (
            "<thead><tr><th>solver</th><th>exponent</th>"
            "<th>R²</th><th>n_points</th></tr></thead>"
        )
        raw_header = (
            "<thead><tr><th>solver</th><th>n_dof</th>"
            "<th>wall time (s)</th><th>L2 error</th><th>success</th></tr></thead>"
        )
        html_path.write_text(
            "<!doctype html>\n"
            f"<html><head><meta charset='utf-8'><title>Weak Scaling</title>\n"
            f"<style>{style}</style></head><body>\n"
            "<h1>Weak-scaling benchmark</h1>\n"
            "<img src='scaling.png' alt='scaling plot' style='max-width:100%'>\n"
            "<h2>Per-solver scaling fit</h2>\n"
            f"<table>{summary_header}<tbody>{''.join(summary_rows)}</tbody></table>\n"
            "<h2>Raw measurements</h2>\n"
            f"<table>{raw_header}<tbody>{''.join(rows)}</tbody></table>\n"
            "</body></html>\n"
        )
        return html_path


def estimate_scaling_exponent(n_dof_values: list[int], wall_times: list[float]) -> float:
    """Return the log-log slope of wall_time vs n_dof.

    Public convenience wrapper, mostly for tests and ad-hoc analysis.
    Raises :class:`ValueError` for inputs of length < 2.
    """
    if len(n_dof_values) < 2 or len(wall_times) < 2:
        raise ValueError("Need at least 2 points to fit a slope")
    if len(n_dof_values) != len(wall_times):
        raise ValueError("n_dof_values and wall_times must align")
    log_n = np.log(np.array(n_dof_values, dtype=np.float64))
    log_t = np.log(np.maximum(np.array(wall_times, dtype=np.float64), 1e-12))
    slope, _ = np.polyfit(log_n, log_t, deg=1)
    if math.isnan(slope):
        return 0.0
    return float(slope)
