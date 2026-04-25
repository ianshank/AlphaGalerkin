"""End-to-end SBIR benchmark demonstration script.

Runs the SBIR benchmark suite (L-shaped Poisson, Burgers shock,
Navier-Stokes Taylor-Green) against classical baselines, generates
convergence plots, comparison tables, and a structured report.

Usage:
    python -m scripts.run_sbir_demo
    python -m scripts.run_sbir_demo --config config/benchmarks/sbir_suite.yaml
    python -m scripts.run_sbir_demo --output-dir outputs/sbir_demo --formats json markdown
    python -m scripts.run_sbir_demo --skip-baselines  # Run AlphaGalerkin only
    python -m scripts.run_sbir_demo --dry-run         # Skip computation, write placeholder JSON
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Default paths
DEFAULT_CONFIG = "config/benchmarks/sbir_suite.yaml"
DEFAULT_OUTPUT_DIR = "outputs/sbir_demo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run SBIR benchmark demonstration suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_sbir_demo
  python -m scripts.run_sbir_demo --config config/proposals/navy_n252_088.yaml
  python -m scripts.run_sbir_demo --output-dir outputs/navy_demo --formats json latex
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to benchmark suite YAML config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results and plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["json", "markdown"],
        choices=["json", "markdown", "latex"],
        help="Output report formats (default: json markdown)",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip baseline solvers (run AlphaGalerkin benchmarks only)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (text reports only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip actual benchmark computation and write a placeholder JSON report "
            "showing what a real run would produce. Completes in <5 seconds."
        ),
    )
    return parser.parse_args(argv)


def generate_convergence_plot(
    results: list[dict[str, object]],
    output_path: Path,
    x_key: str = "n_dof",
    y_key: str = "l2_error",
    title: str = "Convergence: Error vs DOF",
    xlabel: str = "Degrees of Freedom",
    ylabel: str = "L2 Error",
    log_log: bool = True,
) -> None:
    """Generate a convergence plot from benchmark results.

    Args:
        results: List of result dicts with x_key and y_key fields.
        output_path: Path to save PNG.
        x_key: Key for x-axis data.
        y_key: Key for y-axis data.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        log_log: Use log-log axes.

    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib_not_available", msg="Skipping plot generation")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Group by method
    methods: dict[str, list[tuple[float, float]]] = {}
    for r in results:
        method = str(r.get("method_name", "unknown"))
        x_val = float(r.get(x_key, 0))
        y_val = float(r.get(y_key, float("nan")))
        if x_val > 0 and math.isfinite(y_val) and y_val > 0:
            methods.setdefault(method, []).append((x_val, y_val))

    markers = ["o", "s", "^", "D", "v", "P", "*"]
    for idx, (method_name, data) in enumerate(sorted(methods.items())):
        data.sort(key=lambda t: t[0])
        xs, ys = zip(*data, strict=True)
        marker = markers[idx % len(markers)]
        if log_log:
            ax.loglog(xs, ys, f"-{marker}", label=method_name, markersize=6)
        else:
            ax.plot(xs, ys, f"-{marker}", label=method_name, markersize=6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("plot_saved", path=str(output_path))


def generate_latex_table(results: list[dict[str, object]], output_path: Path) -> None:
    """Generate a LaTeX comparison table from results.

    Args:
        results: List of result dicts.
        output_path: Path to save .tex file.

    """
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SBIR Benchmark Results: AlphaGalerkin vs. Baselines}",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Benchmark & Method & DOF & L2 Error & Wall Time (s) & Conv. Rate \\",
        r"\hline",
    ]

    def _sort_key(x: dict[str, object]) -> tuple[str, str, float]:
        return (str(x.get("benchmark_name")), str(x.get("method_name")), float(x.get("n_dof", 0)))

    for r in sorted(results, key=_sort_key):
        bname = str(r.get("benchmark_name", ""))
        mname = str(r.get("method_name", ""))
        dof = int(r.get("n_dof", 0))
        l2 = r.get("l2_error")
        l2_str = f"{float(l2):.2e}" if l2 is not None and math.isfinite(float(l2)) else "N/A"
        wt = float(r.get("wall_time_seconds", 0))
        cr = r.get("convergence_rate")
        cr_str = f"{float(cr):.2f}" if cr is not None else "-"
        lines.append(f"{bname} & {mname} & {dof} & {l2_str} & {wt:.4f} & {cr_str} \\\\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("latex_table_saved", path=str(output_path))


def run_dry_run(args: argparse.Namespace) -> int:
    """Execute the SBIR demo in dry-run mode.

    Skips all actual benchmark computation and writes a placeholder JSON
    report with realistic metrics that show the expected report structure.
    Completes in well under 5 seconds.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 on success).

    """
    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    logger.info(
        "sbir_demo_dry_run_start",
        config=str(config_path),
        output_dir=str(output_dir),
        formats=args.formats,
    )

    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder results representative of a real AlphaGalerkin run.
    # These values are illustrative (not computed) — see the actual run
    # for genuine benchmark numbers.
    placeholder_results = [
        # Poisson L-shaped: AlphaGalerkin vs classical baselines
        {
            "benchmark_name": "poisson_lshaped",
            "method_name": "alphagalerkin",
            "n_dof": 1024,
            "l2_error": 8.3e-4,
            "wall_time_seconds": 0.42,
            "convergence_rate": 1.33,
            "metadata": {"geometry": "l_shaped", "exact_solution": "r^(2/3)*sin(2θ/3)"},
        },
        {
            "benchmark_name": "poisson_lshaped",
            "method_name": "uniform_fdm",
            "n_dof": 1024,
            "l2_error": 6.7e-3,
            "wall_time_seconds": 0.18,
            "convergence_rate": 0.67,
            "metadata": {"geometry": "l_shaped"},
        },
        # Burgers shock: AlphaGalerkin vs PINN baseline
        {
            "benchmark_name": "burgers_shock",
            "method_name": "alphagalerkin",
            "n_dof": 512,
            "l2_error": 1.2e-4,
            "wall_time_seconds": 0.31,
            "convergence_rate": 2.01,
            "metadata": {"viscosity": 0.01, "exact_solution": "cole_hopf_transform"},
        },
        {
            "benchmark_name": "burgers_shock",
            "method_name": "pinn",
            "n_dof": 512,
            "l2_error": 3.4e-3,
            "wall_time_seconds": 2.75,
            "convergence_rate": 1.10,
            "metadata": {"viscosity": 0.01},
        },
        # Navier-Stokes Taylor-Green vortex
        {
            "benchmark_name": "navier_stokes_taylor_green",
            "method_name": "alphagalerkin",
            "n_dof": 2048,
            "l2_error": 4.1e-4,
            "wall_time_seconds": 0.87,
            "convergence_rate": 2.48,
            "metadata": {
                "reynolds_number": 100,
                "exact_solution": "taylor_green_analytical",
            },
        },
    ]

    # Top-level SBIR-specific summary metrics.
    # transfer_mse: zero-shot transfer from 9x9→19x19 (physics PoC milestone)
    # complexity_timing: O(N) FNet throughput (tokens/s at N=361)
    # lbb_sigma_min: minimum singular value of Key projection (LBB condition)
    sbir_summary = {
        "transfer_mse": 0.000209,
        "complexity_timing": {
            "fnet_tokens_per_second": 48320.0,
            "softmax_tokens_per_second": 5210.0,
            "speedup_factor": 9.27,
            "n_tokens_evaluated": 361,
        },
        "lbb_sigma_min": 0.142,
        "dry_run": True,
        "note": (
            "Placeholder report generated by --dry-run. "
            "Run without --dry-run for genuine benchmark numbers."
        ),
    }

    report: dict[str, object] = {
        "suite_name": "sbir_benchmarks",
        "config_path": str(config_path),
        "n_results": len(placeholder_results),
        "results": placeholder_results,
        **sbir_summary,
    }

    if "json" in args.formats:
        json_path = output_dir / "results.json"
        json_path.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("dry_run_json_written", path=str(json_path))

    if "markdown" in args.formats:
        md_lines = [
            "# sbir_benchmarks (DRY RUN)\n",
            f"Config: `{config_path}`\n",
            f"> **Note:** {sbir_summary['note']}\n",
            "## Summary Metrics\n",
            f"- **Transfer MSE (9x9→19x19):** {sbir_summary['transfer_mse']}",
            f"- **LBB σ_min:** {sbir_summary['lbb_sigma_min']}",
            (
                f"- **FNet speedup:** "
                f"{sbir_summary['complexity_timing']['speedup_factor']}x "  # type: ignore[index]
                f"vs softmax"
            ),
        ]
        md_path = output_dir / "results.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("dry_run_markdown_written", path=str(md_path))

    total_time = time.perf_counter() - t0

    n_benchmarks = len({r["benchmark_name"] for r in placeholder_results})
    n_methods = len({r["method_name"] for r in placeholder_results})

    logger.info(
        "sbir_demo_dry_run_complete",
        n_benchmarks=n_benchmarks,
        n_methods=n_methods,
        total_time_seconds=round(total_time, 3),
        output_dir=str(output_dir),
    )

    print(f"\n{'=' * 60}")
    print("SBIR DEMO COMPLETE (DRY RUN)")
    print(f"{'=' * 60}")
    print(f"  Config:     {config_path}")
    print(f"  Benchmarks: {n_benchmarks} (placeholder)")
    print(f"  Methods:    {n_methods} (placeholder)")
    print(f"  Time:       {total_time:.3f}s")
    print(f"  Output:     {output_dir}/")
    if "json" in args.formats:
        print("    - results.json  [placeholder]")
    if "markdown" in args.formats:
        print("    - results.md    [placeholder]")
    print(f"{'=' * 60}\n")

    return 0


def run_demo(args: argparse.Namespace) -> int:
    """Execute the SBIR demo pipeline.

    Returns:
        Exit code (0 on success).

    """
    if getattr(args, "dry_run", False):
        return run_dry_run(args)

    from src.research.pde_benchmarks import PDEBenchmarkRunner

    output_dir = Path(args.output_dir)
    config_path = Path(args.config)

    logger.info(
        "sbir_demo_start",
        config=str(config_path),
        output_dir=str(output_dir),
        formats=args.formats,
    )

    t0 = time.perf_counter()

    # Initialize runner
    try:
        runner = PDEBenchmarkRunner(config_path)
    except FileNotFoundError:
        logger.error("config_not_found", path=str(config_path))
        return 1

    # Run benchmarks
    results = runner.run_all()
    if not results:
        logger.warning("no_results", msg="Benchmark suite produced no results")
        return 1

    logger.info("benchmarks_complete", n_results=len(results))

    # Generate reports
    runner.generate_report(results, output_dir)

    # Convert results to dicts for plotting
    result_dicts = [r.to_dict() for r in results]

    # Generate LaTeX table if requested
    if "latex" in args.formats:
        generate_latex_table(result_dicts, output_dir / "results.tex")

    # Generate plots
    if not args.no_plots:
        # Plot 1: Error vs DOF (all benchmarks combined)
        generate_convergence_plot(
            result_dicts,
            output_dir / "error_vs_dof.png",
            x_key="n_dof",
            y_key="l2_error",
            title="Convergence: L2 Error vs Degrees of Freedom",
            xlabel="Degrees of Freedom (DOF)",
            ylabel="L2 Error",
        )

        # Plot 2: Error vs wall time
        generate_convergence_plot(
            result_dicts,
            output_dir / "error_vs_walltime.png",
            x_key="wall_time_seconds",
            y_key="l2_error",
            title="Efficiency: L2 Error vs Wall Time",
            xlabel="Wall Time (seconds)",
            ylabel="L2 Error",
            log_log=True,
        )

        # Plot 3: Per-benchmark convergence
        benchmarks = {str(r.get("benchmark_name")) for r in result_dicts}
        for bench_name in sorted(benchmarks):
            bench_results = [r for r in result_dicts if r.get("benchmark_name") == bench_name]
            safe_name = bench_name.replace(" ", "_").replace("/", "_")
            generate_convergence_plot(
                bench_results,
                output_dir / f"convergence_{safe_name}.png",
                title=f"Convergence: {bench_name}",
            )

    total_time = time.perf_counter() - t0

    # Print summary
    n_benchmarks = len({str(r.get("benchmark_name")) for r in result_dicts})
    n_methods = len({str(r.get("method_name")) for r in result_dicts})

    logger.info(
        "sbir_demo_complete",
        n_benchmarks=n_benchmarks,
        n_methods=n_methods,
        n_results=len(results),
        total_time_seconds=round(total_time, 2),
        output_dir=str(output_dir),
    )

    print(f"\n{'=' * 60}")
    print("SBIR DEMO COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Config:     {config_path}")
    print(f"  Benchmarks: {n_benchmarks}")
    print(f"  Methods:    {n_methods}")
    print(f"  Results:    {len(results)}")
    print(f"  Time:       {total_time:.1f}s")
    print(f"  Output:     {output_dir}/")
    print("    - results.json")
    print("    - results.md")
    if "latex" in args.formats:
        print("    - results.tex")
    if not args.no_plots:
        print("    - error_vs_dof.png")
        print("    - error_vs_walltime.png")
        print("    - convergence_*.png (per benchmark)")
    print(f"{'=' * 60}\n")

    return 0


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    sys.exit(run_demo(args))


if __name__ == "__main__":
    main()
