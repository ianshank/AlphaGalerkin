"""Visualization utilities for rapid prototyping.

Provides simple plotting and visualization without
heavy dependencies (outputs text/ASCII or data for plotting).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from src.prototyping.evaluator import EvalResult
from src.prototyping.trainer import TrainResult

logger = structlog.get_logger(__name__)


class PlotType(Enum):
    """Types of plots."""

    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"


@dataclass
class PlotData:
    """Data for creating plots.

    Attributes:
        plot_id: Unique identifier.
        plot_type: Type of plot.
        title: Plot title.
        x_label: X-axis label.
        y_label: Y-axis label.
        x_data: X-axis data.
        y_data: Y-axis data.
        series_names: Names for multiple series.
        metadata: Additional metadata.

    """

    plot_id: str
    plot_type: PlotType
    title: str
    x_label: str
    y_label: str
    x_data: list[Any] = field(default_factory=list)
    y_data: list[Any] = field(default_factory=list)
    series_names: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plot_id": self.plot_id,
            "plot_type": self.plot_type.value,
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_data": self.x_data,
            "y_data": self.y_data,
            "series_names": self.series_names,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class Visualizer:
    """Visualizer for prototype results.

    Provides text-based visualization and data export
    for creating plots externally.

    Attributes:
        width: ASCII plot width.
        height: ASCII plot height.

    """

    def __init__(
        self,
        width: int = 60,
        height: int = 20,
    ) -> None:
        """Initialize visualizer.

        Args:
            width: ASCII plot width.
            height: ASCII plot height.

        """
        self.width = width
        self.height = height
        self._plots: list[PlotData] = []
        self._logger = logger.bind(visualizer="Visualizer")

    @property
    def plots(self) -> list[PlotData]:
        """Get all created plots."""
        return self._plots

    def plot_training_loss(
        self,
        result: TrainResult,
        title: str | None = None,
    ) -> str:
        """Plot training loss curve.

        Args:
            result: Training result.
            title: Optional title.

        Returns:
            ASCII plot string.

        """
        losses = result.metrics.get("loss", [])
        if not losses:
            return "No loss data available"

        plot_data = PlotData(
            plot_id=str(uuid.uuid4())[:8],
            plot_type=PlotType.LINE,
            title=title or f"Training Loss - {result.model_id}",
            x_label="Step",
            y_label="Loss",
            x_data=list(range(len(losses))),
            y_data=[losses],
            series_names=["loss"],
            metadata={"result_id": result.result_id},
        )
        self._plots.append(plot_data)

        return self._ascii_line_plot(
            x=list(range(len(losses))),
            y=losses,
            title=plot_data.title,
            x_label=plot_data.x_label,
            y_label=plot_data.y_label,
        )

    def plot_comparison(
        self,
        results: list[EvalResult],
        metric: str,
        title: str | None = None,
    ) -> str:
        """Plot metric comparison across models.

        Args:
            results: Evaluation results.
            metric: Metric to compare.
            title: Optional title.

        Returns:
            ASCII bar plot string.

        """
        labels = []
        values = []
        errors = []

        for result in results:
            if metric in result.metrics:
                labels.append(result.model_id)
                values.append(result.metrics[metric].value)
                std = result.metrics[metric].std
                errors.append(std if std else 0)

        if not values:
            return f"No {metric} data available"

        plot_data = PlotData(
            plot_id=str(uuid.uuid4())[:8],
            plot_type=PlotType.BAR,
            title=title or f"Model Comparison - {metric}",
            x_label="Model",
            y_label=metric,
            x_data=labels,
            y_data=[values],
            series_names=[metric],
            metadata={"metric": metric, "errors": errors},
        )
        self._plots.append(plot_data)

        return self._ascii_bar_plot(
            labels=labels,
            values=values,
            title=plot_data.title,
            x_label=plot_data.x_label,
            y_label=plot_data.y_label,
        )

    def plot_learning_curves(
        self,
        results: list[TrainResult],
        title: str | None = None,
    ) -> str:
        """Plot learning curves for multiple runs.

        Args:
            results: List of training results.
            title: Optional title.

        Returns:
            ASCII plot string.

        """
        if not results:
            return "No results to plot"

        # Find common length
        min_len = min(len(r.metrics.get("loss", [])) for r in results if r.metrics.get("loss"))
        if min_len == 0:
            return "No loss data available"

        plot_title = title or "Learning Curves"
        lines = [plot_title, "=" * len(plot_title), ""]

        for result in results:
            losses = result.metrics.get("loss", [])[:min_len]
            if losses:
                start_loss = losses[0]
                end_loss = losses[-1]
                improvement = ((start_loss - end_loss) / start_loss * 100) if start_loss > 0 else 0
                lines.append(
                    f"  {result.model_id}: {start_loss:.4f} -> {end_loss:.4f} "
                    f"({improvement:+.1f}%)"
                )

        plot_data = PlotData(
            plot_id=str(uuid.uuid4())[:8],
            plot_type=PlotType.LINE,
            title=plot_title,
            x_label="Step",
            y_label="Loss",
            x_data=list(range(min_len)),
            y_data=[r.metrics.get("loss", [])[:min_len] for r in results],
            series_names=[r.model_id for r in results],
        )
        self._plots.append(plot_data)

        return "\n".join(lines)

    def plot_metrics_table(
        self,
        results: list[EvalResult],
        metrics: list[str] | None = None,
    ) -> str:
        """Create metrics comparison table.

        Args:
            results: Evaluation results.
            metrics: Metrics to include.

        Returns:
            Formatted table string.

        """
        if not results:
            return "No results to display"

        # Collect all metrics
        all_metrics: set[str] = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        metrics_to_show = metrics or sorted(all_metrics)

        # Build table
        header = ["Model"] + metrics_to_show
        rows = []

        for result in results:
            row = [result.model_id]
            for metric in metrics_to_show:
                if metric in result.metrics:
                    value = result.metrics[metric].value
                    row.append(f"{value:.6f}")
                else:
                    row.append("-")
            rows.append(row)

        return self._format_table(header, rows)

    def _ascii_line_plot(
        self,
        x: list[float],
        y: list[float],
        title: str,
        x_label: str,
        y_label: str,
    ) -> str:
        """Create ASCII line plot.

        Args:
            x: X values.
            y: Y values.
            title: Plot title.
            x_label: X-axis label.
            y_label: Y-axis label.

        Returns:
            ASCII plot string.

        """
        if not y:
            return "No data"

        y_min, y_max = min(y), max(y)
        y_range = y_max - y_min if y_max != y_min else 1

        lines = [title, "=" * len(title), ""]

        # Create plot area
        plot = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Plot points
        for i, yi in enumerate(y):
            xi = int((i / max(len(y) - 1, 1)) * (self.width - 1))
            yi_norm = int(((yi - y_min) / y_range) * (self.height - 1))
            yi_inv = self.height - 1 - yi_norm
            if 0 <= xi < self.width and 0 <= yi_inv < self.height:
                plot[yi_inv][xi] = "*"

        # Add y-axis labels
        lines.append(f"{y_label}")
        lines.append(f"  {y_max:.4f} |")
        for row in plot:
            lines.append("           |" + "".join(row))
        lines.append(f"  {y_min:.4f} |" + "-" * self.width)
        lines.append(" " * 12 + x_label)
        lines.append("           0" + " " * (self.width - 10) + f"{len(y)}")

        return "\n".join(lines)

    def _ascii_bar_plot(
        self,
        labels: list[str],
        values: list[float],
        title: str,
        x_label: str,
        y_label: str,
    ) -> str:
        """Create ASCII bar plot.

        Args:
            labels: Bar labels.
            values: Bar values.
            title: Plot title.
            x_label: X-axis label.
            y_label: Y-axis label.

        Returns:
            ASCII plot string.

        """
        if not values:
            return "No data"

        max_val = max(values)
        max_bar_width = self.width - 20

        lines = [title, "=" * len(title), "", f"{y_label}"]

        for label, value in zip(labels, values):
            bar_width = int((value / max_val) * max_bar_width) if max_val > 0 else 0
            bar = "#" * bar_width
            lines.append(f"  {label:8s} |{bar} {value:.4f}")

        lines.append("")
        lines.append(f"{x_label}")

        return "\n".join(lines)

    def _format_table(
        self,
        header: list[str],
        rows: list[list[str]],
    ) -> str:
        """Format data as a table.

        Args:
            header: Table header.
            rows: Table rows.

        Returns:
            Formatted table string.

        """
        # Calculate column widths
        widths = [len(h) for h in header]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Format header
        header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(header))
        separator = "-+-".join("-" * w for w in widths)

        lines = [header_str, separator]

        # Format rows
        for row in rows:
            row_str = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            lines.append(row_str)

        return "\n".join(lines)

    def export_plot_data(self, plot_id: str | None = None) -> list[dict[str, Any]]:
        """Export plot data for external visualization.

        Args:
            plot_id: Optional specific plot to export.

        Returns:
            List of plot data dictionaries.

        """
        if plot_id:
            return [p.to_dict() for p in self._plots if p.plot_id == plot_id]
        return [p.to_dict() for p in self._plots]

    def clear(self) -> None:
        """Clear all plots."""
        self._plots.clear()


def create_visualizer(
    width: int = 60,
    height: int = 20,
) -> Visualizer:
    """Create a visualizer.

    Args:
        width: ASCII plot width.
        height: ASCII plot height.

    Returns:
        Configured Visualizer.

    """
    return Visualizer(width=width, height=height)
