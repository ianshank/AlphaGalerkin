"""SBIR Benchmark Demo for AlphaGalerkin.

End-to-end demo that runs all SBIR benchmark problems (L-Poisson, Burgers, NS),
compares AlphaGalerkin neural operator against FDM, AMR, and PINN baselines,
and generates an HTML report with convergence rate tables, error comparisons,
and timing data.

Usage:
    python -m src.demos.sbir_demo
    python -m src.demos.sbir_demo --config config/benchmarks/sbir_suite.yaml
    python -m src.demos.sbir_demo --output-dir outputs/sbir_demo --quick
"""

from __future__ import annotations

import html
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
import typer

from src.demos.config import SBIRDemoConfig
from src.research.pde_benchmarks import PDEBenchmarkResult, PDEBenchmarkRunner

logger = structlog.get_logger(__name__)

app = typer.Typer(name="sbir-demo", help="SBIR benchmark demonstration")


def _sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores.

    Used to derive per-benchmark plot file names from benchmark names
    that may contain whitespace, slashes, or other separators.
    """
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_") or "plot"


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------


class SBIRHTMLReportGenerator:
    """Generates HTML reports from SBIR benchmark results.

    Produces a self-contained HTML page with:
    - Executive summary
    - Convergence rate tables per benchmark
    - Error comparison across methods
    - Timing comparisons
    """

    def __init__(
        self,
        suite_name: str = "SBIR Benchmarks",
        config_path: str = "",
    ) -> None:
        self._suite_name = suite_name
        self._config_path = config_path
        self._log = logger.bind(component="html_report")

    def generate(
        self,
        results: list[PDEBenchmarkResult],
        total_time_seconds: float,
    ) -> str:
        """Generate full HTML report from benchmark results.

        Args:
            results: List of benchmark results across all problems and methods.
            total_time_seconds: Total wall-clock time for the benchmark suite.

        Returns:
            HTML string.

        """
        benchmarks = self._group_by_benchmark(results)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        sections: list[str] = []
        sections.append(self._render_header(timestamp, total_time_seconds, len(results)))
        sections.append(self._render_summary_table(benchmarks))

        for bench_name, bench_results in benchmarks.items():
            sections.append(self._render_benchmark_section(bench_name, bench_results))

        sections.append(self._render_footer())

        return self._wrap_html("\n".join(sections))

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_benchmark(
        results: list[PDEBenchmarkResult],
    ) -> dict[str, list[PDEBenchmarkResult]]:
        """Group results by benchmark name, preserving insertion order."""
        groups: dict[str, list[PDEBenchmarkResult]] = {}
        for r in results:
            groups.setdefault(r.benchmark_name, []).append(r)
        return groups

    @staticmethod
    def _group_by_method(
        results: list[PDEBenchmarkResult],
    ) -> dict[str, list[PDEBenchmarkResult]]:
        """Group results by method name."""
        groups: dict[str, list[PDEBenchmarkResult]] = {}
        for r in results:
            groups.setdefault(r.method_name, []).append(r)
        # Sort each group by n_dof
        for key in groups:
            groups[key].sort(key=lambda r: r.n_dof)
        return groups

    # ------------------------------------------------------------------
    # Rendering methods
    # ------------------------------------------------------------------

    def _render_header(
        self,
        timestamp: str,
        total_time: float,
        n_results: int,
    ) -> str:
        suite = html.escape(self._suite_name)
        config = html.escape(self._config_path)
        return f"""
        <div class="header">
            <h1>{suite}</h1>
            <p class="meta">Generated: {timestamp} | Config: <code>{config}</code></p>
            <p class="meta">Total results: {n_results} | Total time: {total_time:.1f}s</p>
        </div>
        """

    def _render_summary_table(
        self,
        benchmarks: dict[str, list[PDEBenchmarkResult]],
    ) -> str:
        """Render executive summary table."""
        rows: list[str] = []
        for bench_name, results in benchmarks.items():
            methods = self._group_by_method(results)
            for method_name, method_results in methods.items():
                best = min(
                    (r for r in method_results if not math.isnan(r.l2_error)),
                    key=lambda r: r.l2_error,
                    default=None,
                )
                if best is not None:
                    rows.append(
                        f"<tr>"
                        f"<td>{html.escape(bench_name)}</td>"
                        f"<td>{html.escape(method_name)}</td>"
                        f"<td>{best.n_dof}</td>"
                        f"<td>{best.l2_error:.2e}</td>"
                        f"<td>{best.wall_time_seconds:.4f}</td>"
                        f"<td>{best.convergence_rate:.2f}"
                        f"</td>"
                        f"</tr>"
                        if best.convergence_rate is not None
                        else f"<tr>"
                        f"<td>{html.escape(bench_name)}</td>"
                        f"<td>{html.escape(method_name)}</td>"
                        f"<td>{best.n_dof}</td>"
                        f"<td>{best.l2_error:.2e}</td>"
                        f"<td>{best.wall_time_seconds:.4f}</td>"
                        f"<td>-</td>"
                        f"</tr>"
                    )

        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Method</th>
                        <th>Best DOF</th>
                        <th>Best L2 Error</th>
                        <th>Time (s)</th>
                        <th>Conv. Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>
        """

    def _render_benchmark_section(
        self,
        bench_name: str,
        results: list[PDEBenchmarkResult],
    ) -> str:
        """Render a single benchmark section with convergence and timing tables."""
        methods = self._group_by_method(results)
        name_escaped = html.escape(bench_name)

        # Convergence table
        conv_rows: list[str] = []
        for method_name, method_results in methods.items():
            for r in method_results:
                l2_str = f"{r.l2_error:.2e}" if not math.isnan(r.l2_error) else "N/A"
                cr_str = f"{r.convergence_rate:.2f}" if r.convergence_rate is not None else "-"
                conv_rows.append(
                    f"<tr>"
                    f"<td>{html.escape(method_name)}</td>"
                    f"<td>{r.n_dof}</td>"
                    f"<td>{l2_str}</td>"
                    f"<td>{r.wall_time_seconds:.4f}</td>"
                    f"<td>{cr_str}</td>"
                    f"</tr>"
                )

        # Error comparison table (at largest DOF per method)
        comparison_rows: list[str] = []
        for method_name, method_results in methods.items():
            if not method_results:
                continue
            finest = method_results[-1]  # Already sorted by n_dof
            l2_str = f"{finest.l2_error:.2e}" if not math.isnan(finest.l2_error) else "N/A"
            comparison_rows.append(
                f"<tr>"
                f"<td>{html.escape(method_name)}</td>"
                f"<td>{finest.n_dof}</td>"
                f"<td>{l2_str}</td>"
                f"<td>{finest.wall_time_seconds:.4f}</td>"
                f"</tr>"
            )

        return f"""
        <div class="section">
            <h2>{name_escaped}</h2>

            <h3>Convergence Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>DOF</th>
                        <th>L2 Error</th>
                        <th>Wall Time (s)</th>
                        <th>Conv. Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(conv_rows)}
                </tbody>
            </table>

            <h3>Method Comparison (Finest Level)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>DOF</th>
                        <th>L2 Error</th>
                        <th>Wall Time (s)</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(comparison_rows)}
                </tbody>
            </table>
        </div>
        """

    @staticmethod
    def _render_footer() -> str:
        return """
        <div class="footer">
            <p>AlphaGalerkin SBIR Benchmark Suite</p>
        </div>
        """

    @staticmethod
    def _wrap_html(body: str) -> str:
        """Wrap body content in a complete HTML document."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaGalerkin SBIR Benchmark Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #fafafa;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header .meta {{ color: #aaa; margin: 4px 0; }}
        .header code {{ background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 3px; }}
        .section {{
            background: white;
            padding: 20px 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1a1a2e;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
        }}
        .section h3 {{ color: #444; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
            color: #555;
        }}
        tr:hover td {{ background: #f9f9f9; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #999;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


class SBIRDemo:
    """End-to-end SBIR benchmark demo.

    Orchestrates benchmark execution, report generation, and output.
    """

    def __init__(self, config: SBIRDemoConfig | None = None) -> None:
        self.config = config or SBIRDemoConfig()
        self._log = logger.bind(component="sbir_demo")

    def run(self) -> list[PDEBenchmarkResult]:
        """Execute the full SBIR benchmark suite.

        Returns:
            List of benchmark results.

        """
        self._log.info(
            "sbir_demo_start",
            config_path=self.config.suite_config_path,
            output_dir=self.config.output_dir,
        )
        t0 = time.perf_counter()

        # Run benchmarks
        runner = PDEBenchmarkRunner(self.config.suite_config_path)
        results = runner.run_all()

        total_time = time.perf_counter() - t0
        self._log.info("sbir_demo_benchmarks_done", n_results=len(results), total_time=total_time)

        # Generate outputs
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Always generate JSON
        if self.config.generate_json:
            self._write_json(results, output_dir / "results.json", total_time)

        # Generate Markdown via runner (also writes CSV)
        if self.config.generate_markdown:
            runner.generate_report(results, output_dir)

        # Generate HTML
        if self.config.generate_html:
            self._write_html(results, output_dir / "report.html", total_time)

        # Generate Pareto + convergence plots (best-effort; matplotlib optional)
        self._write_plots(runner, results, output_dir)

        self._log.info("sbir_demo_done", output_dir=str(output_dir))
        return results

    def _write_plots(
        self,
        runner: PDEBenchmarkRunner,
        results: list[PDEBenchmarkResult],
        output_dir: Path,
    ) -> None:
        """Render per-problem Pareto + convergence plots via the visualization registry.

        Each benchmark problem gets its own pair of plots (error axes are
        not cross-comparable between different PDEs).  File names are
        derived from the benchmark name with filesystem-unsafe characters
        replaced by underscores.
        """
        try:
            from src.poc.visualization.config import VisualizationConfig
            from src.poc.visualization.plots import create_plot
        except ImportError as exc:  # pragma: no cover - optional dep
            self._log.warning("plot_deps_missing", error=str(exc))
            return

        per_problem = runner.build_pareto_plot_data(results)
        if not per_problem:
            self._log.info("plot_skipped_no_valid_results")
            return

        viz_config = VisualizationConfig(name="sbir_demo_plots")

        for benchmark_name, methods_data in per_problem.items():
            safe_name = _sanitize_filename(benchmark_name)

            try:
                pareto_fig = create_plot(
                    "pareto_frontier",
                    {"methods": methods_data},
                    viz_config,
                )
                pareto_path = output_dir / f"pareto_{safe_name}.png"
                pareto_fig.savefig(pareto_path, dpi=viz_config.dpi, bbox_inches="tight")
                self._log.info(
                    "pareto_plot_written",
                    benchmark=benchmark_name,
                    path=str(pareto_path),
                )
            except Exception:  # pragma: no cover - plot failures are non-fatal
                self._log.exception("pareto_plot_failed", benchmark=benchmark_name)

            try:
                convergence_data: dict[str, dict[str, list[float]]] = {}
                for method_name, series in methods_data.items():
                    convergence_data[method_name] = {
                        "dof": [float(x) for x in series["n_dof"]],
                        "error": [float(x) for x in series["error"]],
                    }
                conv_fig = create_plot(
                    "convergence_rates",
                    {"methods": convergence_data},
                    viz_config,
                )
                conv_path = output_dir / f"convergence_{safe_name}.png"
                conv_fig.savefig(conv_path, dpi=viz_config.dpi, bbox_inches="tight")
                self._log.info(
                    "convergence_plot_written",
                    benchmark=benchmark_name,
                    path=str(conv_path),
                )
            except Exception:  # pragma: no cover - plot failures are non-fatal
                self._log.exception("convergence_plot_failed", benchmark=benchmark_name)

    def _write_json(
        self,
        results: list[PDEBenchmarkResult],
        path: Path,
        total_time: float,
    ) -> None:
        """Write JSON results file."""
        data: dict[str, Any] = {
            "suite_name": "sbir_benchmarks",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_time_seconds": total_time,
            "n_results": len(results),
            "results": [r.to_dict() for r in results],
        }
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        self._log.info("json_report_written", path=str(path))

    def _write_html(
        self,
        results: list[PDEBenchmarkResult],
        path: Path,
        total_time: float,
    ) -> None:
        """Write HTML report file."""
        generator = SBIRHTMLReportGenerator(
            suite_name="AlphaGalerkin SBIR Benchmarks",
            config_path=self.config.suite_config_path,
        )
        html_content = generator.generate(results, total_time)
        path.write_text(html_content, encoding="utf-8")
        self._log.info("html_report_written", path=str(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def run(
    config: str = typer.Option(
        "config/benchmarks/sbir_suite.yaml",
        "--config",
        help="Path to SBIR benchmark suite YAML configuration",
    ),
    output_dir: str = typer.Option(
        "outputs/sbir_demo",
        "--output-dir",
        help="Directory for output reports",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Quick mode with reduced refinement levels",
    ),
    no_html: bool = typer.Option(
        False,
        "--no-html",
        help="Skip HTML report generation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
    ),
) -> None:
    """Run the SBIR benchmark demonstration."""
    refinement_levels: list[int] | None = [4, 8, 16] if quick else None

    demo_config = SBIRDemoConfig(
        suite_config_path=config,
        output_dir=output_dir,
        generate_html=not no_html,
        verbose=verbose,
        refinement_levels=refinement_levels,
    )

    demo = SBIRDemo(config=demo_config)
    results = demo.run()

    n_benchmarks = len({r.benchmark_name for r in results})
    n_methods = len({r.method_name for r in results})
    typer.echo(
        f"Done: {len(results)} results across {n_benchmarks} benchmarks "
        f"and {n_methods} methods. Reports in {output_dir}/"
    )


def main() -> None:
    """Entry point for python -m src.demos.sbir_demo."""
    app()


if __name__ == "__main__":
    main()
