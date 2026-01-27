"""Results reporting and visualization.

Provides:
- Report generation in multiple formats
- Metrics tables and summaries
- Export utilities
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from src.research.experiment import Experiment, ExperimentRun
from src.research.benchmark import BenchmarkResult
from src.research.validator import TransferResult
from src.research.comparison import ComparisonResult


class ReportFormat(str, Enum):
    """Report output format."""

    JSON = "json"
    MARKDOWN = "markdown"
    CSV = "csv"
    TEXT = "text"


@dataclass
class ReportSection:
    """A section in a report."""

    title: str
    content: str
    data: dict[str, Any] | None = None


class Reporter:
    """Generates reports from research results.

    Creates formatted reports in various output formats.
    """

    def __init__(
        self,
        output_dir: Path | str = "outputs/reports",
        default_format: ReportFormat = ReportFormat.MARKDOWN,
        logger: structlog.stdlib.BoundLogger | None = None,
    ) -> None:
        """Initialize reporter.

        Args:
            output_dir: Output directory for reports.
            default_format: Default report format.
            logger: Optional structured logger.
        """
        self._output_dir = Path(output_dir)
        self._default_format = default_format
        self._logger = logger or structlog.get_logger(__name__)

    def report_experiment(
        self,
        experiment: Experiment,
        format: ReportFormat | None = None,
        save: bool = True,
    ) -> str:
        """Generate experiment report.

        Args:
            experiment: Experiment to report.
            format: Output format.
            save: Whether to save to file.

        Returns:
            Report content as string.
        """
        format = format or self._default_format
        sections = []

        # Summary section
        summary = experiment.get_summary()
        sections.append(ReportSection(
            title="Experiment Summary",
            content=self._format_summary(summary),
            data=summary,
        ))

        # Runs section
        if experiment.runs:
            runs_content = self._format_runs(experiment.runs)
            sections.append(ReportSection(
                title="Runs",
                content=runs_content,
            ))

            # Best run
            best = experiment.get_best_run("loss")
            if best:
                sections.append(ReportSection(
                    title="Best Run",
                    content=self._format_run(best),
                    data=best.to_dict(),
                ))

        report = self._build_report(
            title=f"Experiment: {experiment.config.name}",
            sections=sections,
            format=format,
        )

        if save:
            path = self._save_report(
                content=report,
                name=f"experiment_{experiment.config.name}",
                format=format,
            )
            self._logger.info("report_saved", path=str(path))

        return report

    def report_transfer(
        self,
        result: TransferResult,
        format: ReportFormat | None = None,
        save: bool = True,
    ) -> str:
        """Generate transfer validation report.

        Args:
            result: Transfer result to report.
            format: Output format.
            save: Whether to save to file.

        Returns:
            Report content as string.
        """
        format = format or self._default_format
        sections = []

        # Summary
        sections.append(ReportSection(
            title="Transfer Validation Summary",
            content=self._format_transfer_summary(result),
        ))

        # Per-target results
        if result.target_metrics:
            sections.append(ReportSection(
                title="Target Results",
                content=self._format_transfer_targets(result),
            ))

        # Training info
        if result.train_epochs > 0:
            sections.append(ReportSection(
                title="Training",
                content=self._format_training_info(result),
            ))

        report = self._build_report(
            title=f"Transfer Validation [{result.result_id}]",
            sections=sections,
            format=format,
        )

        if save:
            path = self._save_report(
                content=report,
                name=f"transfer_{result.result_id}",
                format=format,
            )
            self._logger.info("report_saved", path=str(path))

        return report

    def report_comparison(
        self,
        result: ComparisonResult,
        format: ReportFormat | None = None,
        save: bool = True,
    ) -> str:
        """Generate model comparison report.

        Args:
            result: Comparison result to report.
            format: Output format.
            save: Whether to save to file.

        Returns:
            Report content as string.
        """
        format = format or self._default_format
        sections = []

        # Summary
        sections.append(ReportSection(
            title="Comparison Summary",
            content=result.summary(),
        ))

        # Metrics table
        sections.append(ReportSection(
            title="Model Metrics",
            content=self._format_comparison_table(result),
        ))

        # Rankings
        if result.rankings:
            sections.append(ReportSection(
                title="Rankings",
                content=self._format_rankings(result.rankings),
            ))

        # Statistical tests
        if result.pairwise_tests:
            sections.append(ReportSection(
                title="Statistical Comparisons",
                content=self._format_pairwise_tests(result.pairwise_tests),
            ))

        report = self._build_report(
            title=f"Model Comparison [{result.comparison_id}]",
            sections=sections,
            format=format,
        )

        if save:
            path = self._save_report(
                content=report,
                name=f"comparison_{result.comparison_id}",
                format=format,
            )
            self._logger.info("report_saved", path=str(path))

        return report

    def report_benchmarks(
        self,
        results: list[BenchmarkResult],
        name: str = "benchmark",
        format: ReportFormat | None = None,
        save: bool = True,
    ) -> str:
        """Generate benchmark report.

        Args:
            results: Benchmark results to report.
            name: Report name.
            format: Output format.
            save: Whether to save to file.

        Returns:
            Report content as string.
        """
        format = format or self._default_format
        sections = []

        # Summary
        sections.append(ReportSection(
            title="Benchmark Summary",
            content=self._format_benchmark_summary(results),
        ))

        # Results table
        sections.append(ReportSection(
            title="Results",
            content=self._format_benchmark_table(results),
        ))

        report = self._build_report(
            title=f"Benchmark Report: {name}",
            sections=sections,
            format=format,
        )

        if save:
            path = self._save_report(
                content=report,
                name=f"benchmark_{name}",
                format=format,
            )
            self._logger.info("report_saved", path=str(path))

        return report

    def _build_report(
        self,
        title: str,
        sections: list[ReportSection],
        format: ReportFormat,
    ) -> str:
        """Build report from sections.

        Args:
            title: Report title.
            sections: Report sections.
            format: Output format.

        Returns:
            Formatted report.
        """
        if format == ReportFormat.MARKDOWN:
            return self._build_markdown(title, sections)
        elif format == ReportFormat.JSON:
            return self._build_json(title, sections)
        elif format == ReportFormat.TEXT:
            return self._build_text(title, sections)
        else:
            return self._build_text(title, sections)

    def _build_markdown(self, title: str, sections: list[ReportSection]) -> str:
        """Build markdown report."""
        lines = [
            f"# {title}",
            "",
            f"*Generated: {datetime.utcnow().isoformat()}*",
            "",
        ]

        for section in sections:
            lines.extend([
                f"## {section.title}",
                "",
                section.content,
                "",
            ])

        return "\n".join(lines)

    def _build_json(self, title: str, sections: list[ReportSection]) -> str:
        """Build JSON report."""
        data = {
            "title": title,
            "generated": datetime.utcnow().isoformat(),
            "sections": [
                {
                    "title": s.title,
                    "content": s.content,
                    "data": s.data,
                }
                for s in sections
            ],
        }
        return json.dumps(data, indent=2, default=str)

    def _build_text(self, title: str, sections: list[ReportSection]) -> str:
        """Build plain text report."""
        lines = [
            "=" * 60,
            title,
            "=" * 60,
            "",
        ]

        for section in sections:
            lines.extend([
                "-" * 40,
                section.title,
                "-" * 40,
                section.content,
                "",
            ])

        return "\n".join(lines)

    def _save_report(
        self,
        content: str,
        name: str,
        format: ReportFormat,
    ) -> Path:
        """Save report to file.

        Args:
            content: Report content.
            name: Report name.
            format: Output format.

        Returns:
            Path to saved file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        ext = {
            ReportFormat.MARKDOWN: ".md",
            ReportFormat.JSON: ".json",
            ReportFormat.CSV: ".csv",
            ReportFormat.TEXT: ".txt",
        }[format]

        path = self._output_dir / f"{name}{ext}"

        with open(path, "w") as f:
            f.write(content)

        return path

    def _format_summary(self, summary: dict[str, Any]) -> str:
        """Format summary dictionary."""
        lines = []
        for key, value in summary.items():
            lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)

    def _format_runs(self, runs: list[ExperimentRun]) -> str:
        """Format runs list."""
        lines = ["| Run ID | Status | Duration | Final Loss |", "| ------ | ------ | -------- | ---------- |"]
        for run in runs:
            duration = f"{run.duration_seconds:.1f}s" if run.duration_seconds else "-"
            loss = run.final_metrics.get("loss_final", "-")
            if isinstance(loss, float):
                loss = f"{loss:.6f}"
            lines.append(f"| {run.run_id} | {run.status} | {duration} | {loss} |")
        return "\n".join(lines)

    def _format_run(self, run: ExperimentRun) -> str:
        """Format single run."""
        lines = [
            f"- **Run ID**: {run.run_id}",
            f"- **Status**: {run.status}",
            f"- **Duration**: {run.duration_seconds:.1f}s" if run.duration_seconds else "",
            "",
            "**Final Metrics:**",
        ]
        for name, value in run.final_metrics.items():
            lines.append(f"- {name}: {value:.6f}")
        return "\n".join(lines)

    def _format_transfer_summary(self, result: TransferResult) -> str:
        """Format transfer result summary."""
        status = "PASS" if result.passed else "FAIL"
        lines = [
            f"- **Result ID**: {result.result_id}",
            f"- **Source Size**: {result.source_size}x{result.source_size}",
            f"- **Primary Target**: {result.primary_target}x{result.primary_target}",
            f"- **Status**: {status}",
            f"- **All Targets Passed**: {result.all_passed}",
        ]
        if result.primary_mse is not None:
            lines.append(f"- **Primary MSE**: {result.primary_mse:.6f}")
        return "\n".join(lines)

    def _format_transfer_targets(self, result: TransferResult) -> str:
        """Format transfer target results."""
        lines = ["| Target | MSE | MAE | Status |", "| ------ | --- | --- | ------ |"]
        for size, metrics in sorted(result.target_metrics.items()):
            status = "PASS" if metrics.passed else "FAIL"
            primary = " *" if size == result.primary_target else ""
            lines.append(f"| {size}x{size}{primary} | {metrics.mse:.6f} | {metrics.mae:.6f} | {status} |")
        return "\n".join(lines)

    def _format_training_info(self, result: TransferResult) -> str:
        """Format training info."""
        return "\n".join([
            f"- **Epochs**: {result.train_epochs}",
            f"- **Final Loss**: {result.train_loss:.6f}",
            f"- **Duration**: {result.train_duration_seconds:.1f}s",
        ])

    def _format_comparison_table(self, result: ComparisonResult) -> str:
        """Format comparison table."""
        metrics = ["mse_mean", "mae_mean", "throughput_mean"]
        lines = ["| Model | MSE | MAE | Throughput |", "| ----- | --- | --- | ---------- |"]

        for name, model_metrics in result.model_metrics.items():
            values = []
            for m in metrics:
                v = model_metrics.aggregate_metrics.get(m)
                values.append(f"{v:.6f}" if v is not None else "-")
            lines.append(f"| {name} | {' | '.join(values)} |")

        return "\n".join(lines)

    def _format_rankings(self, rankings: dict[str, list[str]]) -> str:
        """Format rankings."""
        lines = []
        for metric, ranking in rankings.items():
            lines.append(f"- **{metric}**: {' > '.join(ranking)}")
        return "\n".join(lines)

    def _format_pairwise_tests(self, tests: dict[str, dict[str, Any]]) -> str:
        """Format pairwise tests."""
        lines = []
        for pair, metrics in tests.items():
            lines.append(f"\n**{pair}:**")
            for metric, data in metrics.items():
                sig = "*" if data.get("is_significant") else ""
                p = data.get("p_value", 0)
                lines.append(f"- {metric}: p={p:.4f}{sig}")
        return "\n".join(lines)

    def _format_benchmark_summary(self, results: list[BenchmarkResult]) -> str:
        """Format benchmark summary."""
        if not results:
            return "No benchmark results."

        sizes = sorted(set(r.size for r in results))
        return "\n".join([
            f"- **Benchmarks**: {len(results)}",
            f"- **Sizes**: {sizes}",
            f"- **Device**: {results[0].device}",
        ])

    def _format_benchmark_table(self, results: list[BenchmarkResult]) -> str:
        """Format benchmark results table."""
        lines = [
            "| Size | Mean (ms) | Std (ms) | Throughput |",
            "| ---- | --------- | -------- | ---------- |",
        ]
        for r in sorted(results, key=lambda x: x.size):
            lines.append(
                f"| {r.size}x{r.size} | {r.mean_time_ms:.2f} | {r.std_time_ms:.2f} | {r.throughput:.0f} |"
            )
        return "\n".join(lines)


def create_reporter(
    output_dir: Path | str = "outputs/reports",
    format: str = "markdown",
) -> Reporter:
    """Factory function to create a reporter.

    Args:
        output_dir: Output directory.
        format: Default format.

    Returns:
        Reporter instance.
    """
    return Reporter(
        output_dir=Path(output_dir),
        default_format=ReportFormat(format),
    )
