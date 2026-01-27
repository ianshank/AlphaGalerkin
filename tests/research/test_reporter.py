"""Tests for results reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.research.config import ExperimentConfig
from src.research.experiment import Experiment, ExperimentRun
from src.research.benchmark import BenchmarkResult
from src.research.validator import TransferResult, TransferMetrics
from src.research.comparison import ComparisonResult, ModelMetrics
from src.research.reporter import Reporter, ReportFormat, create_reporter


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_all_formats_exist(self) -> None:
        """Test all formats exist."""
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.TEXT.value == "text"


class TestReporter:
    """Tests for Reporter."""

    def test_initialization(self, reporter: Reporter) -> None:
        """Test reporter initialization."""
        assert reporter._default_format == ReportFormat.MARKDOWN

    def test_report_experiment_markdown(
        self, reporter: Reporter
    ) -> None:
        """Test experiment report in markdown."""
        config = ExperimentConfig(name="test_exp")
        experiment = Experiment(config=config)
        run = experiment.start_run()
        run.log_metric("loss", 0.5)
        experiment.end_run()

        report = reporter.report_experiment(
            experiment,
            format=ReportFormat.MARKDOWN,
            save=False,
        )

        assert "# Experiment: test_exp" in report
        assert "Experiment Summary" in report
        assert "Runs" in report

    def test_report_experiment_json(
        self, reporter: Reporter
    ) -> None:
        """Test experiment report in JSON."""
        config = ExperimentConfig(name="test_exp")
        experiment = Experiment(config=config)

        report = reporter.report_experiment(
            experiment,
            format=ReportFormat.JSON,
            save=False,
        )

        import json
        data = json.loads(report)
        assert data["title"] == "Experiment: test_exp"

    def test_report_experiment_text(
        self, reporter: Reporter
    ) -> None:
        """Test experiment report in text."""
        config = ExperimentConfig(name="test_exp")
        experiment = Experiment(config=config)

        report = reporter.report_experiment(
            experiment,
            format=ReportFormat.TEXT,
            save=False,
        )

        assert "Experiment: test_exp" in report
        assert "=" in report  # Text dividers

    def test_report_transfer(
        self, reporter: Reporter, transfer_result: TransferResult
    ) -> None:
        """Test transfer report."""
        report = reporter.report_transfer(
            transfer_result,
            format=ReportFormat.MARKDOWN,
            save=False,
        )

        assert "Transfer Validation" in report
        assert "PASS" in report
        assert "9x9" in report

    def test_report_comparison(
        self, reporter: Reporter, comparison_result: ComparisonResult
    ) -> None:
        """Test comparison report."""
        report = reporter.report_comparison(
            comparison_result,
            format=ReportFormat.MARKDOWN,
            save=False,
        )

        assert "Model Comparison" in report
        assert "model_a" in report

    def test_report_benchmarks(
        self, reporter: Reporter, benchmark_result: BenchmarkResult
    ) -> None:
        """Test benchmark report."""
        report = reporter.report_benchmarks(
            results=[benchmark_result],
            name="speed_test",
            format=ReportFormat.MARKDOWN,
            save=False,
        )

        assert "Benchmark Report" in report
        assert "speed_test" in report

    def test_save_report(self, reporter: Reporter) -> None:
        """Test saving report to file."""
        config = ExperimentConfig(name="save_test")
        experiment = Experiment(config=config)

        report = reporter.report_experiment(
            experiment,
            format=ReportFormat.MARKDOWN,
            save=True,
        )

        # Check file was created
        expected_path = reporter._output_dir / "experiment_save_test.md"
        assert expected_path.exists()

    def test_save_json_format(self, reporter: Reporter) -> None:
        """Test saving JSON report."""
        config = ExperimentConfig(name="json_test")
        experiment = Experiment(config=config)

        reporter.report_experiment(
            experiment,
            format=ReportFormat.JSON,
            save=True,
        )

        expected_path = reporter._output_dir / "experiment_json_test.json"
        assert expected_path.exists()

    def test_format_runs_table(self, reporter: Reporter) -> None:
        """Test runs table formatting."""
        runs = [
            ExperimentRun(run_id="r1", status="completed"),
            ExperimentRun(run_id="r2", status="failed"),
        ]
        runs[0].start_time = "2024-01-01T00:00:00"
        runs[0].end_time = "2024-01-01T00:01:00"

        formatted = reporter._format_runs(runs)

        assert "r1" in formatted
        assert "r2" in formatted
        assert "completed" in formatted
        assert "failed" in formatted

    def test_format_transfer_targets(
        self, reporter: Reporter, transfer_result: TransferResult
    ) -> None:
        """Test transfer targets table formatting."""
        formatted = reporter._format_transfer_targets(transfer_result)

        assert "9x9" in formatted
        assert "19x19" in formatted
        assert "PASS" in formatted

    def test_format_comparison_table(
        self, reporter: Reporter, comparison_result: ComparisonResult
    ) -> None:
        """Test comparison table formatting."""
        formatted = reporter._format_comparison_table(comparison_result)

        assert "model_a" in formatted
        assert "model_b" in formatted

    def test_format_benchmark_table(
        self, reporter: Reporter, benchmark_result: BenchmarkResult
    ) -> None:
        """Test benchmark table formatting."""
        formatted = reporter._format_benchmark_table([benchmark_result])

        assert "9x9" in formatted
        assert "10.00" in formatted  # mean_time_ms


class TestCreateReporter:
    """Tests for create_reporter factory."""

    def test_create_default(self) -> None:
        """Test creating default reporter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = create_reporter(output_dir=tmpdir)
            assert reporter._default_format == ReportFormat.MARKDOWN

    def test_create_with_format(self) -> None:
        """Test creating with format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = create_reporter(output_dir=tmpdir, format="json")
            assert reporter._default_format == ReportFormat.JSON
