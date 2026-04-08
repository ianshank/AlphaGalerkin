"""HTML report generator for PoC scenario results.

Generates self-contained, single-file HTML reports with embedded
base64-encoded plot images and styled metrics tables. Uses Python
string templates (no Jinja2 dependency) for minimal footprint.

Example:
    from src.poc.visualization.reports import HTMLReportGenerator, ReportSection
    from src.poc.visualization.config import VisualizationConfig

    config = VisualizationConfig(name="report")
    generator = HTMLReportGenerator()
    html = generator.generate_report(
        title="My Report",
        sections=[ReportSection(title="Results", content="All tests passed.")],
        config=config,
    )

"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Template
from typing import Any

import structlog
from matplotlib.figure import Figure

from src.poc.visualization.config import VisualizationConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricRow:
    """A single row in a metrics table.

    Attributes:
        name: Metric name.
        value: Metric value.
        unit: Optional unit string.
        status: Optional pass/fail indicator.

    """

    name: str
    value: float
    unit: str = ""
    status: str = ""


@dataclass
class ReportSection:
    """A section in an HTML report.

    Attributes:
        title: Section heading.
        content: Free-form HTML or plain text content.
        metrics: Optional list of metric rows for a table.
        figures: Optional list of matplotlib Figures to embed.
        table_data: Optional list-of-dicts for a comparison table.
        table_headers: Optional list of header names for table_data.

    """

    title: str
    content: str = ""
    metrics: list[MetricRow] | None = None
    figures: list[Figure] | None = None
    table_data: list[dict[str, Any]] | None = None
    table_headers: list[str] | None = None


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${title}</title>
<style>
${css}
</style>
</head>
<body>
<div class="container">
<header>
<h1>${title}</h1>
<p class="meta">Generated: ${timestamp}</p>
</header>
${body}
</div>
</body>
</html>
""")

_LIGHT_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       line-height: 1.6; color: #333; background: #f8f9fa; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
header { border-bottom: 2px solid #dee2e6; padding-bottom: 16px; margin-bottom: 32px; }
header h1 { font-size: 1.8rem; color: #212529; }
.meta { color: #6c757d; font-size: 0.9rem; margin-top: 4px; }
.section { margin-bottom: 40px; }
.section h2 { font-size: 1.4rem; color: #495057; margin-bottom: 16px;
              border-bottom: 1px solid #dee2e6; padding-bottom: 8px; }
.section-content { padding: 0 8px; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; }
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #dee2e6; }
th { background: #e9ecef; font-weight: 600; color: #495057; }
tr:hover { background: #f1f3f5; }
.status-pass { color: #28a745; font-weight: 600; }
.status-fail { color: #dc3545; font-weight: 600; }
.plot-container { margin: 16px 0; text-align: center; }
.plot-container img { max-width: 100%; height: auto; border: 1px solid #dee2e6; }
"""

_DARK_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       line-height: 1.6; color: #e0e0e0; background: #121212; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
header { border-bottom: 2px solid #333; padding-bottom: 16px; margin-bottom: 32px; }
header h1 { font-size: 1.8rem; color: #ffffff; }
.meta { color: #9e9e9e; font-size: 0.9rem; margin-top: 4px; }
.section { margin-bottom: 40px; }
.section h2 { font-size: 1.4rem; color: #bbbbbb; margin-bottom: 16px;
              border-bottom: 1px solid #333; padding-bottom: 8px; }
.section-content { padding: 0 8px; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; }
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #333; }
th { background: #1e1e1e; font-weight: 600; color: #bbbbbb; }
tr:hover { background: #1a1a2e; }
.status-pass { color: #66bb6a; font-weight: 600; }
.status-fail { color: #ef5350; font-weight: 600; }
.plot-container { margin: 16px 0; text-align: center; }
.plot-container img { max-width: 100%; height: auto; border: 1px solid #333; border-radius: 4px; }
"""

_PUBLICATION_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Times New Roman", Times, serif;
       line-height: 1.5; color: #000; background: #fff; }
.container { max-width: 900px; margin: 0 auto; padding: 32px; }
header { border-bottom: 2px solid #000; padding-bottom: 12px; margin-bottom: 28px; }
header h1 { font-size: 1.6rem; color: #000; }
.meta { color: #555; font-size: 0.85rem; margin-top: 4px; }
.section { margin-bottom: 36px; }
.section h2 { font-size: 1.2rem; color: #000; margin-bottom: 12px;
              border-bottom: 1px solid #000; padding-bottom: 6px; }
.section-content { padding: 0 4px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #999; }
th { background: #f0f0f0; font-weight: 700; color: #000; }
tr:hover { background: #fafafa; }
.status-pass { color: #006400; font-weight: 700; }
.status-fail { color: #8b0000; font-weight: 700; }
.plot-container { margin: 12px 0; text-align: center; }
.plot-container img { max-width: 100%; height: auto; border: 1px solid #999; }
"""

_THEME_CSS: dict[str, str] = {
    "light": _LIGHT_CSS,
    "dark": _DARK_CSS,
    "publication": _PUBLICATION_CSS,
}


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


class HTMLReportGenerator:
    """Generates self-contained HTML reports with embedded plots.

    Reports are single-file HTML documents that can be viewed offline.
    Plots are rendered as PNG images and embedded as base64 data URIs.
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="html_report_generator")

    def generate_report(
        self,
        title: str,
        sections: list[ReportSection],
        config: VisualizationConfig,
    ) -> str:
        """Generate a complete HTML report.

        Args:
            title: Report title displayed in the header.
            sections: Ordered list of report sections.
            config: Visualization configuration for theming and plot rendering.

        Returns:
            Complete HTML string suitable for writing to a file.

        """
        self._log.info(
            "generating_report",
            title=title,
            n_sections=len(sections),
            theme=config.theme,
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        css = _THEME_CSS[config.theme]

        body_parts: list[str] = []
        for section in sections:
            body_parts.append(self._render_section(section, config))

        body = "\n".join(body_parts)

        report_html = _PAGE_TEMPLATE.substitute(
            title=html.escape(title),
            timestamp=timestamp,
            css=css,
            body=body,
        )

        self._log.info("report_generated", size_bytes=len(report_html))
        return report_html

    # ------------------------------------------------------------------
    # Section rendering
    # ------------------------------------------------------------------

    def _render_section(
        self,
        section: ReportSection,
        config: VisualizationConfig,
    ) -> str:
        """Render a single report section to HTML.

        Args:
            section: Section to render.
            config: Visualization configuration.

        Returns:
            HTML fragment for the section.

        """
        parts: list[str] = [
            '<div class="section">',
            f"<h2>{html.escape(section.title)}</h2>",
            '<div class="section-content">',
        ]

        # Free-form content
        if section.content:
            parts.append(f"<p>{html.escape(section.content)}</p>")

        # Metrics table
        if section.metrics:
            parts.append(self._render_metrics_table(section.metrics))

        # Comparison table
        if section.table_data and section.table_headers:
            parts.append(self._render_comparison_table(section.table_headers, section.table_data))

        # Embedded plots
        if section.figures:
            for fig in section.figures:
                parts.append(self._render_figure(fig, config))

        parts.append("</div>")
        parts.append("</div>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Table rendering
    # ------------------------------------------------------------------

    def _render_metrics_table(self, metrics: list[MetricRow]) -> str:
        """Render a metrics summary table.

        Args:
            metrics: List of metric rows.

        Returns:
            HTML table string.

        """
        rows: list[str] = []
        for m in metrics:
            status_cls = ""
            status_text = ""
            if m.status:
                status_lower = m.status.lower()
                if status_lower == "pass":
                    status_cls = "status-pass"
                elif status_lower == "fail":
                    status_cls = "status-fail"
                status_text = f'<span class="{status_cls}">{html.escape(m.status)}</span>'

            unit_str = f" {html.escape(m.unit)}" if m.unit else ""
            rows.append(
                f"<tr>"
                f"<td>{html.escape(m.name)}</td>"
                f"<td>{m.value:.6f}{unit_str}</td>"
                f"<td>{status_text}</td>"
                f"</tr>"
            )

        return (
            "<table>"
            "<thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    def _render_comparison_table(
        self,
        headers: list[str],
        data: list[dict[str, Any]],
    ) -> str:
        """Render a comparison data table.

        Args:
            headers: Column header names.
            data: List of row dictionaries keyed by header names.

        Returns:
            HTML table string.

        """
        header_cells = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
        rows: list[str] = []
        for row in data:
            cells = "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
            rows.append(f"<tr>{cells}</tr>")

        return (
            f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )

    # ------------------------------------------------------------------
    # Figure embedding
    # ------------------------------------------------------------------

    @staticmethod
    def _render_figure(fig: Figure, config: VisualizationConfig) -> str:
        """Render a matplotlib Figure as an embedded base64 PNG image.

        Args:
            fig: Matplotlib Figure to embed.
            config: Visualization configuration for DPI.

        Returns:
            HTML fragment with base64-encoded <img> tag.

        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=config.dpi, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        buf.close()

        return (
            f'<div class="plot-container"><img src="data:image/png;base64,{b64}" alt="plot"></div>'
        )


def generate_report(
    title: str,
    sections: list[ReportSection],
    config: VisualizationConfig,
) -> str:
    """Convenience function to generate an HTML report.

    Args:
        title: Report title.
        sections: Ordered list of report sections.
        config: Visualization configuration.

    Returns:
        Complete HTML string.

    """
    generator = HTMLReportGenerator()
    return generator.generate_report(title, sections, config)
