"""PDE benchmark runner for SBIR demonstrations.

Runs AlphaGalerkin against classical baselines on standard
PDE benchmark problems and generates comparison reports.

Usage:
    runner = PDEBenchmarkRunner("config/benchmarks/sbir_suite.yaml")
    results = runner.run_all()
    runner.generate_report(results, Path("outputs/sbir_benchmarks"))
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.pde.config import PDEConfig, PDEType
from src.pde.registry import get_pde_operator, list_pde_operators
from src.research.baselines import BaseSolver, get_solver, list_solvers

logger = structlog.get_logger(__name__)

# Re-export metadata-key constants so existing callers (if any) can keep
# importing ``METADATA_KEY_*`` from this module without breaking.  The
# canonical home is :mod:`src.research.metadata_keys`.
from src.research.metadata_keys import (  # noqa: E402
    METADATA_KEY_METHOD,
    METADATA_KEY_REFINEMENT_LEVEL,
    METADATA_KEY_SEED,
)

# Log underflow guard for convergence rate computation: skip any step where
# the previous or current l2_error is below this threshold, since the log()
# result dominates numerical noise.  Kept module-private; if a caller needs
# to sweep it, expose it via a config field.
_CONVERGENCE_RATE_EPS = 1e-12


@dataclass
class PDEBenchmarkResult:
    """Result from a single benchmark problem + method combination.

    Attributes:
        benchmark_name: Name of the benchmark problem.
        method_name: Name of the solver method.
        n_dof: Degrees of freedom used.
        l2_error: L2 error versus exact solution.
        wall_time_seconds: Solve wall time.
        convergence_rate: Estimated convergence rate (if multi-level).
        metadata: Extra solver / problem metadata.

    """

    benchmark_name: str
    method_name: str
    n_dof: int
    l2_error: float
    wall_time_seconds: float
    convergence_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "benchmark_name": self.benchmark_name,
            "method_name": self.method_name,
            "n_dof": self.n_dof,
            "l2_error": self.l2_error,
            "wall_time_seconds": self.wall_time_seconds,
            "convergence_rate": self.convergence_rate,
            "metadata": self.metadata,
        }


class PDEBenchmarkRunner:
    """Runs PDE benchmarks and generates comparison reports.

    Loads a YAML configuration describing benchmark problems and
    baselines, executes each combination, and produces structured
    JSON and Markdown reports.
    """

    def __init__(self, config_path: str | Path) -> None:
        """Initialize from a YAML config file.

        Args:
            config_path: Path to the benchmark suite YAML.

        """
        self._config_path = Path(config_path)
        self._config = self._load_config(self._config_path)
        self._log = logger.bind(
            suite=self._config.get("suite_name", "unknown"),
        )
        self._log.info(
            "benchmark_runner_init",
            config_path=str(self._config_path),
            n_benchmarks=len(self._config.get("benchmarks", [])),
            n_baselines=len(self._config.get("baselines", [])),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> list[PDEBenchmarkResult]:
        """Run every benchmark problem with every applicable baseline.

        Returns:
            Flat list of results across all problems and methods.

        """
        all_results: list[PDEBenchmarkResult] = []

        for bench_cfg in self._config.get("benchmarks", []):
            try:
                results = self.run_benchmark(bench_cfg)
                all_results.extend(results)
            except Exception:
                self._log.exception(
                    "benchmark_failed",
                    benchmark=bench_cfg.get("name", "unknown"),
                )

        self._log.info("benchmark_run_all_done", total_results=len(all_results))
        return all_results

    def run_benchmark(self, benchmark_config: dict[str, Any]) -> list[PDEBenchmarkResult]:
        """Run a single benchmark problem across baselines.

        Args:
            benchmark_config: Dict from the YAML benchmarks list.

        Returns:
            List of results for each baseline on this problem.

        """
        name = benchmark_config["name"]
        pde_type = benchmark_config.get("pde_type", "poisson")
        refinement_levels = benchmark_config.get("refinement_levels", [16, 32, 64])

        self._log.info("benchmark_start", benchmark=name, pde_type=pde_type)

        operator = self._create_operator(benchmark_config)
        baselines = self._get_baselines()
        results: list[PDEBenchmarkResult] = []

        for solver in baselines:
            for n_dof in refinement_levels:
                try:
                    sr = solver.solve(operator, n_dof)
                    # Record the requested refinement_level hint alongside
                    # the solver-reported n_dof so CSV consumers can
                    # distinguish the input budget from the output DOF.
                    metadata_with_level = dict(sr.metadata)
                    metadata_with_level.setdefault(METADATA_KEY_REFINEMENT_LEVEL, n_dof)
                    result = PDEBenchmarkResult(
                        benchmark_name=name,
                        method_name=solver.name,
                        n_dof=sr.n_dof,
                        l2_error=sr.l2_error if sr.l2_error is not None else float("nan"),
                        wall_time_seconds=sr.wall_time_seconds,
                        metadata=metadata_with_level,
                    )
                    results.append(result)
                    self._log.info(
                        "solver_result",
                        benchmark=name,
                        solver=solver.name,
                        n_dof=sr.n_dof,
                        l2_error=sr.l2_error,
                        wall_time=sr.wall_time_seconds,
                    )
                except NotImplementedError:
                    self._log.warning(
                        "solver_not_implemented",
                        benchmark=name,
                        solver=solver.name,
                        n_dof=n_dof,
                    )
                except Exception:
                    self._log.exception(
                        "solver_error",
                        benchmark=name,
                        solver=solver.name,
                        n_dof=n_dof,
                    )

        # Compute convergence rates per method
        results = self._attach_convergence_rates(results)

        return results

    def generate_report(
        self,
        results: list[PDEBenchmarkResult],
        output_dir: Path,
    ) -> None:
        """Generate JSON, Markdown and CSV reports.

        Args:
            results: Benchmark results to report.
            output_dir: Directory for output files.

        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self._generate_json_report(results, output_dir / "results.json")
        md = self._generate_markdown_table(results)
        (output_dir / "results.md").write_text(md, encoding="utf-8")
        self.export_csv(results, output_dir / "results.csv")

        self._log.info("report_generated", output_dir=str(output_dir))

    def export_csv(
        self,
        results: list[PDEBenchmarkResult],
        output_path: Path,
    ) -> None:
        """Export benchmark results as CSV for Pareto plot + external analysis.

        Columns: problem, method, refinement_level, n_dof, l2_error,
        wall_time_seconds, convergence_rate, seed.

        Args:
            results: Benchmark results to export.
            output_path: Destination CSV file path.

        """
        # ``csv`` is already imported at module scope; no late import needed.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "problem",
                    METADATA_KEY_METHOD,
                    METADATA_KEY_REFINEMENT_LEVEL,
                    "n_dof",
                    "l2_error",
                    "wall_time_seconds",
                    "convergence_rate",
                    METADATA_KEY_SEED,
                ]
            )
            for r in results:
                seed = r.metadata.get(METADATA_KEY_SEED, "")
                refinement_level = r.metadata.get(METADATA_KEY_REFINEMENT_LEVEL, r.n_dof)
                writer.writerow(
                    [
                        r.benchmark_name,
                        r.method_name,
                        refinement_level,
                        r.n_dof,
                        r.l2_error if not math.isnan(r.l2_error) else "",
                        f"{r.wall_time_seconds:.6f}",
                        f"{r.convergence_rate:.4f}" if r.convergence_rate is not None else "",
                        seed,
                    ]
                )
        self._log.info("csv_export", path=str(output_path), rows=len(results))

    def build_pareto_plot_data(
        self,
        results: list[PDEBenchmarkResult],
    ) -> dict[str, dict[str, dict[str, list[Any]]]]:
        """Build per-problem ``methods`` payloads for :class:`ParetoFrontierPlot`.

        Results across different PDEs are incomparable on a shared
        error/time axis (e.g. an L-shaped Poisson error of 1e-3 is not
        directly comparable to a Burgers-shock error of 1e-3), so we
        partition by ``benchmark_name`` and return a nested
        ``{benchmark_name: {method_name: {wall_time, error, n_dof}}}``
        structure.  Callers render one plot per problem.

        Args:
            results: Benchmark results (may span multiple problems).

        Returns:
            Nested dict keyed by benchmark name, then by method name.
            Each inner dict has parallel ``wall_time``, ``error``,
            ``n_dof`` lists ready to feed into
            ``create_plot('pareto_frontier', {'methods': inner}, ...)``.

        """
        by_problem: dict[str, dict[str, dict[str, list[Any]]]] = {}
        for r in results:
            if math.isnan(r.l2_error):
                continue
            problem_bucket = by_problem.setdefault(r.benchmark_name, {})
            entry = problem_bucket.setdefault(
                r.method_name, {"wall_time": [], "error": [], "n_dof": []}
            )
            entry["wall_time"].append(r.wall_time_seconds)
            entry["error"].append(r.l2_error)
            entry["n_dof"].append(r.n_dof)
        return by_problem

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_markdown_table(self, results: list[PDEBenchmarkResult]) -> str:
        """Build a Markdown report with per-benchmark comparison tables."""
        lines: list[str] = []
        suite_name = self._config.get("suite_name", "PDE Benchmarks")
        lines.append(f"# {suite_name}\n")
        lines.append(f"Config: `{self._config_path}`\n")

        # Group by benchmark
        benchmarks: dict[str, list[PDEBenchmarkResult]] = {}
        for r in results:
            benchmarks.setdefault(r.benchmark_name, []).append(r)

        for bname, bresults in benchmarks.items():
            lines.append(f"\n## {bname}\n")
            lines.append("| Method | DOF | L2 Error | Wall Time (s) | Conv. Rate |")
            lines.append("|--------|-----|----------|---------------|------------|")

            # Sort by method then DOF
            bresults.sort(key=lambda r: (r.method_name, r.n_dof))

            for r in bresults:
                l2_str = f"{r.l2_error:.2e}" if not math.isnan(r.l2_error) else "N/A"
                cr_str = f"{r.convergence_rate:.2f}" if r.convergence_rate is not None else "-"
                lines.append(
                    f"| {r.method_name} | {r.n_dof} | {l2_str} "
                    f"| {r.wall_time_seconds:.4f} | {cr_str} |"
                )

        lines.append("")
        return "\n".join(lines)

    def _generate_json_report(
        self,
        results: list[PDEBenchmarkResult],
        output_path: Path,
    ) -> None:
        """Write results as JSON."""
        data = {
            "suite_name": self._config.get("suite_name", "unknown"),
            "config_path": str(self._config_path),
            "n_results": len(results),
            "results": [r.to_dict() for r in results],
        }
        output_path.write_text(
            json.dumps(data, indent=2, default=str),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        """Load and return the YAML config."""
        if not path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Benchmark config must be a YAML mapping, got {type(config)}")
        return config

    def _create_operator(self, bench_cfg: dict[str, Any]) -> Any:
        """Create a PDE operator from benchmark config.

        Falls back to a default PoissonOperator if the PDE type from
        the config is not registered.
        """
        pde_type_str = bench_cfg.get("pde_type", "poisson")
        domain_cfg = bench_cfg.get("domain", {})
        params = bench_cfg.get("parameters", {})

        # Map geometry string to domain bounds
        dim = domain_cfg.get("dim", 2)
        domain_min = domain_cfg.get("min", [0.0] * dim)
        domain_max = domain_cfg.get("max", [1.0] * dim)

        try:
            pde_enum = PDEType(pde_type_str)
        except ValueError:
            self._log.warning("unknown_pde_type", pde_type=pde_type_str)
            pde_enum = PDEType.POISSON

        # advection_coeff must match domain_dim (zero for non-advection PDEs)
        advection_coeff = params.get("advection_coeff", [0.0] * dim)
        if len(advection_coeff) != dim:
            advection_coeff = [0.0] * dim

        pde_config = PDEConfig(
            name=bench_cfg["name"],
            pde_type=pde_enum,
            domain_dim=dim,
            domain_min=list(domain_min),
            domain_max=list(domain_max),
            advection_coeff=advection_coeff,
        )

        # Try to get from registry
        try:
            operator_cls = get_pde_operator(pde_type_str)
            return operator_cls(pde_config)
        except KeyError:
            self._log.warning(
                "pde_operator_not_registered",
                pde_type=pde_type_str,
                available=list_pde_operators(),
            )
            # Fallback
            operator_cls = get_pde_operator("poisson")
            return operator_cls(pde_config)

    def _get_baselines(self) -> list[BaseSolver]:
        """Instantiate baseline solvers from config."""
        solvers: list[BaseSolver] = []
        available = list_solvers()

        for bl_cfg in self._config.get("baselines", []):
            name = bl_cfg.get("name", "")
            # Normalise to our solver registry keys
            solver_key = self._normalise_solver_name(name)

            if solver_key not in available:
                self._log.warning(
                    "baseline_not_available",
                    baseline=name,
                    solver_key=solver_key,
                    available=available,
                )
                continue

            kwargs: dict[str, Any] = {}
            if "marking_fraction" in bl_cfg:
                kwargs["marking_fraction"] = bl_cfg["marking_fraction"]

            try:
                solver = get_solver(solver_key, **kwargs)
                solvers.append(solver)
            except Exception:
                self._log.exception("baseline_init_failed", baseline=name)

        if not solvers:
            self._log.warning("no_baselines_available_using_defaults")
            # Provide defaults so the runner still works
            import contextlib

            with contextlib.suppress(Exception):
                solvers.append(get_solver("uniform_fdm"))

        return solvers

    @staticmethod
    def _normalise_solver_name(name: str) -> str:
        """Map YAML baseline names to solver registry keys."""
        mapping: dict[str, str] = {
            "uniform_fem": "uniform_fdm",
            "uniform_fdm": "uniform_fdm",
            "dorfler_amr": "dorfler_amr",
            "pinn": "pinn",
            "navier_stokes_fdm": "navier_stokes_fdm",
            "navier_stokes": "navier_stokes_fdm",
        }
        key = name.lower().replace("-", "_").replace(" ", "_")
        return mapping.get(key, key)

    @staticmethod
    def _attach_convergence_rates(
        results: list[PDEBenchmarkResult],
    ) -> list[PDEBenchmarkResult]:
        """Estimate convergence rates from multi-level results.

        Uses log-log slope between successive refinement levels for
        the same benchmark + method pair.
        """
        # Group by (benchmark, method)
        groups: dict[tuple[str, str], list[PDEBenchmarkResult]] = {}
        for r in results:
            key = (r.benchmark_name, r.method_name)
            groups.setdefault(key, []).append(r)

        for _key, group in groups.items():
            group.sort(key=lambda r: r.n_dof)
            for i in range(1, len(group)):
                prev, cur = group[i - 1], group[i]
                if (
                    prev.n_dof > 0
                    and cur.n_dof > 0
                    and prev.l2_error > 0
                    and cur.l2_error > 0
                    and not math.isnan(prev.l2_error)
                    and not math.isnan(cur.l2_error)
                    and cur.n_dof != prev.n_dof
                ):
                    log_dof_ratio = math.log(cur.n_dof / prev.n_dof)
                    log_err_ratio = math.log(prev.l2_error / cur.l2_error)
                    if abs(log_dof_ratio) > 1e-12:
                        cur.convergence_rate = log_err_ratio / log_dof_ratio

        return results
