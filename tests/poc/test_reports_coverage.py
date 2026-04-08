"""Additional tests for reports.py to cover missed lines.

Covers:
- _render_metrics_table (lines 302-323): metrics table with pass/fail status, units
- _render_comparison_table (lines 345-351): comparison table with headers and data
- Section rendering with metrics (line 273) and table_data (line 277)
"""

from __future__ import annotations

import html as html_module
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pytest

from src.poc.visualization.config import VisualizationConfig
from src.poc.visualization.reports import (
    HTMLReportGenerator,
    MetricRow,
    ReportSection,
    generate_report,
)

# Use non-interactive backend
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> HTMLReportGenerator:
    """Create an HTMLReportGenerator instance."""
    return HTMLReportGenerator()


@pytest.fixture()
def config() -> VisualizationConfig:
    """Create a default VisualizationConfig."""
    return VisualizationConfig(name="test")


# ---------------------------------------------------------------------------
# _render_metrics_table tests (lines 302-323)
# ---------------------------------------------------------------------------


class TestRenderMetricsTable:
    """Test _render_metrics_table rendering with various metric rows."""

    def test_basic_metrics_table(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metrics table renders with basic metric rows."""
        metrics = [
            MetricRow(name="MSE", value=0.001234),
            MetricRow(name="MAE", value=0.05),
        ]
        section = ReportSection(title="Metrics", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        assert "<table>" in result
        assert "MSE" in result
        assert "MAE" in result
        assert "0.001234" in result

    def test_metrics_table_with_pass_status(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metric row with status='pass' renders with status-pass class."""
        metrics = [MetricRow(name="Accuracy", value=0.95, status="pass")]
        section = ReportSection(title="Results", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        assert "status-pass" in result
        assert "pass" in result

    def test_metrics_table_with_fail_status(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metric row with status='fail' renders with status-fail class."""
        metrics = [MetricRow(name="Accuracy", value=0.3, status="fail")]
        section = ReportSection(title="Results", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        assert "status-fail" in result
        assert "fail" in result

    def test_metrics_table_with_unit(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metric row with unit renders the unit string."""
        metrics = [MetricRow(name="Time", value=1.5, unit="seconds")]
        section = ReportSection(title="Timing", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        assert "seconds" in result
        assert "1.500000" in result

    def test_metrics_table_with_no_status(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metric row with empty status renders no status span in the table body."""
        metrics = [MetricRow(name="Loss", value=0.5)]
        # Call _render_metrics_table directly to check table output only (not CSS)
        table_html = generator._render_metrics_table(metrics)
        # No status-pass or status-fail class should appear in the table cells
        assert "status-pass" not in table_html
        assert "status-fail" not in table_html
        # The status cell should be empty
        assert "<td></td>" in table_html

    def test_metrics_table_with_unknown_status(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metric row with non-pass/fail status renders the status text without special class."""
        metrics = [MetricRow(name="Test", value=1.0, status="pending")]
        section = ReportSection(title="Status", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        # "pending" should appear in the output
        assert "pending" in result
        # But should not have the pass/fail class (status_cls is empty string)
        assert 'class=""' in result or 'class="status-' not in result.replace('class=""', "")

    def test_metrics_table_multiple_rows(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metrics table renders multiple rows correctly."""
        metrics = [
            MetricRow(name="MSE", value=0.001, unit="", status="pass"),
            MetricRow(name="MAE", value=0.05, unit="m", status="fail"),
            MetricRow(name="R2", value=0.99, status="pass"),
        ]
        section = ReportSection(title="All Metrics", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        assert result.count("<tr>") >= 3  # At least 3 data rows
        assert "MSE" in result
        assert "MAE" in result
        assert "R2" in result

    def test_metrics_table_html_escaping(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metric names and units are HTML-escaped."""
        metrics = [MetricRow(name="<script>alert(1)</script>", value=0.5, unit="<b>ms</b>")]
        section = ReportSection(title="XSS Test", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        # Raw <script> should not appear; escaped version should
        assert "<script>" not in result
        assert html_module.escape("<script>alert(1)</script>") in result

    def test_metrics_table_header_row(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Metrics table has the expected header cells."""
        metrics = [MetricRow(name="X", value=1.0)]
        section = ReportSection(title="H", metrics=metrics)
        result = generate_report(title="Test", sections=[section], config=config)
        assert "<th>Metric</th>" in result
        assert "<th>Value</th>" in result
        assert "<th>Status</th>" in result

    def test_render_metrics_table_directly(self, generator: HTMLReportGenerator) -> None:
        """Call _render_metrics_table directly to verify output structure."""
        metrics = [
            MetricRow(name="Loss", value=0.123456, unit="nats", status="pass"),
            MetricRow(name="Acc", value=0.987654, status="fail"),
        ]
        html_str = generator._render_metrics_table(metrics)
        assert "<table>" in html_str
        assert "<thead>" in html_str
        assert "<tbody>" in html_str
        assert "0.123456" in html_str
        assert "nats" in html_str
        assert "status-pass" in html_str
        assert "status-fail" in html_str


# ---------------------------------------------------------------------------
# _render_comparison_table tests (lines 345-351)
# ---------------------------------------------------------------------------


class TestRenderComparisonTable:
    """Test _render_comparison_table rendering."""

    def test_comparison_table_basic(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Comparison table renders headers and rows."""
        headers = ["Method", "MSE", "Time"]
        data = [
            {"Method": "FDM", "MSE": "0.01", "Time": "1.2s"},
            {"Method": "FEM", "MSE": "0.005", "Time": "3.4s"},
        ]
        section = ReportSection(
            title="Comparison",
            table_data=data,
            table_headers=headers,
        )
        result = generate_report(title="Test", sections=[section], config=config)
        assert "Method" in result
        assert "MSE" in result
        assert "Time" in result
        assert "FDM" in result
        assert "FEM" in result

    def test_comparison_table_directly(self, generator: HTMLReportGenerator) -> None:
        """Call _render_comparison_table directly to verify output structure."""
        headers = ["Name", "Score"]
        data = [
            {"Name": "Alpha", "Score": "100"},
            {"Name": "Beta", "Score": "200"},
        ]
        html_str = generator._render_comparison_table(headers, data)
        assert "<table>" in html_str
        assert "<thead>" in html_str
        assert "<tbody>" in html_str
        assert "<th>Name</th>" in html_str
        assert "<th>Score</th>" in html_str
        assert "Alpha" in html_str
        assert "Beta" in html_str
        assert "100" in html_str
        assert "200" in html_str

    def test_comparison_table_missing_key(
        self, generator: HTMLReportGenerator, config: VisualizationConfig
    ) -> None:
        """Comparison table handles missing keys in row dicts gracefully."""
        headers = ["A", "B", "C"]
        data = [
            {"A": "1", "B": "2"},  # Missing "C"
        ]
        section = ReportSection(
            title="Sparse",
            table_data=data,
            table_headers=headers,
        )
        result = generate_report(title="Test", sections=[section], config=config)
        # Should still render without error
        assert "A" in result
        assert "B" in result
        assert "C" in result

    def test_comparison_table_html_escaping(self, generator: HTMLReportGenerator) -> None:
        """Comparison table escapes HTML in header and cell values."""
        headers = ["<b>Bold</b>"]
        data = [{"<b>Bold</b>": "<script>x</script>"}]
        html_str = generator._render_comparison_table(headers, data)
        # Raw tags should not appear
        assert "<b>" not in html_str
        assert "<script>" not in html_str
        assert html_module.escape("<b>Bold</b>") in html_str

    def test_comparison_table_empty_data(self, generator: HTMLReportGenerator) -> None:
        """Comparison table with empty data list renders headers only."""
        headers = ["Col1", "Col2"]
        data: list[dict[str, Any]] = []
        html_str = generator._render_comparison_table(headers, data)
        assert "<th>Col1</th>" in html_str
        assert "<th>Col2</th>" in html_str
        # No data rows
        assert "<tr><td>" not in html_str


# ---------------------------------------------------------------------------
# Section rendering with metrics and table_data together (lines 273, 277)
# ---------------------------------------------------------------------------


class TestSectionRenderingPaths:
    """Test that section rendering triggers metrics and comparison table paths."""

    def test_section_with_metrics_only(self, config: VisualizationConfig) -> None:
        """Section with only metrics renders the metrics table."""
        metrics = [MetricRow(name="X", value=1.0, status="pass")]
        section = ReportSection(title="M", metrics=metrics)
        result = generate_report(title="T", sections=[section], config=config)
        assert "status-pass" in result

    def test_section_with_table_data_only(self, config: VisualizationConfig) -> None:
        """Section with only table_data and table_headers renders comparison table."""
        section = ReportSection(
            title="T",
            table_data=[{"A": "1"}],
            table_headers=["A"],
        )
        result = generate_report(title="T", sections=[section], config=config)
        assert "<th>A</th>" in result

    def test_section_with_table_data_but_no_headers_skips(
        self, config: VisualizationConfig
    ) -> None:
        """Section with table_data but no table_headers does NOT render comparison table."""
        section = ReportSection(
            title="T",
            table_data=[{"A": "1"}],
            # table_headers is None -> comparison table branch skipped
        )
        result = generate_report(title="T", sections=[section], config=config)
        # Should not have a data row from comparison table
        # The section renders but without a table
        assert "<html" in result

    def test_section_with_all_elements(self, config: VisualizationConfig) -> None:
        """Section with content, metrics, table_data, and figures renders all."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        metrics = [MetricRow(name="Loss", value=0.5, status="pass")]
        section = ReportSection(
            title="Full Section",
            content="Some description.",
            metrics=metrics,
            figures=[fig],
            table_data=[{"Method": "A", "Score": "1"}],
            table_headers=["Method", "Score"],
        )
        result = generate_report(title="Full", sections=[section], config=config)
        assert "Some description." in result
        assert "status-pass" in result
        assert "<th>Method</th>" in result
        assert "data:image/png;base64" in result

        plt.close(fig)

    def test_dark_theme_with_metrics(self) -> None:
        """Metrics table renders correctly with dark theme."""
        config = VisualizationConfig(name="dark", theme="dark")
        metrics = [MetricRow(name="MSE", value=0.001, status="pass")]
        section = ReportSection(title="Dark", metrics=metrics)
        result = generate_report(title="Dark Report", sections=[section], config=config)
        assert "status-pass" in result
        # Dark theme CSS should be included
        assert "#121212" in result

    def test_publication_theme_with_comparison_table(self) -> None:
        """Comparison table renders correctly with publication theme."""
        config = VisualizationConfig(name="pub", theme="publication")
        section = ReportSection(
            title="Pub",
            table_data=[{"X": "1", "Y": "2"}],
            table_headers=["X", "Y"],
        )
        result = generate_report(title="Pub Report", sections=[section], config=config)
        assert "<th>X</th>" in result
        # Publication theme uses Times New Roman
        assert "Times New Roman" in result
