"""Plot generation module for PoC scenario visualization.

Provides a registry-based system for creating matplotlib figures from
structured data. Each registered plot type accepts a data dictionary
and a VisualizationConfig, returning a matplotlib Figure.

Example:
    from src.poc.visualization.plots import create_plot
    from src.poc.visualization.config import VisualizationConfig

    config = VisualizationConfig(name="viz")
    data = {"steps": [1, 2, 3], "loss": [0.5, 0.3, 0.1]}
    fig = create_plot("training_curves", data, config)
    fig.savefig("training.png")

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.poc.visualization.config import VisualizationConfig
from src.templates.registry import create_registry

# Use non-interactive backend for headless rendering
matplotlib.use("Agg")


class BasePlot(ABC):
    """Base class for all registered plot types.

    Subclasses must implement ``create`` to produce a matplotlib Figure
    from the provided data dictionary and visualization config.
    """

    @abstractmethod
    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create a matplotlib Figure from structured data.

        Args:
            data: Plot-specific data dictionary.
            config: Visualization configuration.

        Returns:
            Matplotlib Figure instance.

        """
        ...


PlotRegistry, register_plot = create_registry("Plot", BasePlot)  # type: ignore[type-abstract]


# ---------------------------------------------------------------------------
# Theme helpers
# ---------------------------------------------------------------------------

_THEME_PARAMS: dict[str, dict[str, Any]] = {
    "light": {
        "facecolor": "white",
        "text_color": "#333333",
        "grid_color": "#cccccc",
        "grid_alpha": 0.7,
        "spine_color": "#666666",
    },
    "dark": {
        "facecolor": "#1e1e1e",
        "text_color": "#e0e0e0",
        "grid_color": "#444444",
        "grid_alpha": 0.5,
        "spine_color": "#888888",
    },
    "publication": {
        "facecolor": "white",
        "text_color": "black",
        "grid_color": "#aaaaaa",
        "grid_alpha": 0.4,
        "spine_color": "black",
    },
}


def _apply_theme(fig: Figure, config: VisualizationConfig) -> None:
    """Apply visual theme to a figure and all its axes.

    Args:
        fig: Matplotlib figure to style.
        config: Visualization configuration with theme selection.

    """
    params = _THEME_PARAMS[config.theme]
    fig.set_facecolor(params["facecolor"])

    for ax in fig.get_axes():
        ax.set_facecolor(params["facecolor"])
        ax.tick_params(colors=params["text_color"])
        ax.xaxis.label.set_color(params["text_color"])
        ax.yaxis.label.set_color(params["text_color"])
        ax.title.set_color(params["text_color"])
        ax.grid(True, color=params["grid_color"], alpha=params["grid_alpha"])
        for spine in ax.spines.values():
            spine.set_color(params["spine_color"])


def _new_figure(config: VisualizationConfig) -> tuple[Figure, Any]:
    """Create a new themed figure with a single axes.

    Args:
        config: Visualization configuration.

    Returns:
        Tuple of (Figure, Axes).

    """
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    return fig, ax


# ---------------------------------------------------------------------------
# Built-in plot types
# ---------------------------------------------------------------------------


@register_plot("training_curves")
class TrainingCurvesPlot(BasePlot):
    """Plot training loss and optional accuracy over steps.

    Expected data keys:
        steps: list[int | float] - training step numbers
        loss: list[float] - loss values per step
        accuracy: list[float] | None - optional accuracy values per step

    """

    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create training curves figure."""
        steps: list[float] = data["steps"]
        loss: list[float] = data["loss"]
        accuracy: list[float] | None = data.get("accuracy")

        fig, ax_loss = _new_figure(config)
        ax_loss.plot(steps, loss, color="#2196F3", linewidth=1.5, label="Loss")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Training Curves")

        if accuracy is not None:
            ax_acc = ax_loss.twinx()
            ax_acc.plot(steps, accuracy, color="#4CAF50", linewidth=1.5, label="Accuracy")
            ax_acc.set_ylabel("Accuracy")
            # Combine legends from both axes
            lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
            lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
            ax_loss.legend(lines_loss + lines_acc, labels_loss + labels_acc, loc="best")
        else:
            ax_loss.legend(loc="best")

        _apply_theme(fig, config)
        fig.tight_layout()
        return fig


@register_plot("convergence_rates")
class ConvergenceRatesPlot(BasePlot):
    """Plot error vs DOF on a log-log scale for convergence analysis.

    Expected data keys:
        methods: dict[str, dict] where each value has:
            dof: list[int] - degrees of freedom
            error: list[float] - error values

    """

    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create convergence rate figure."""
        methods: dict[str, dict[str, Any]] = data["methods"]
        colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(methods), 1)))

        fig, ax = _new_figure(config)

        for idx, (method_name, method_data) in enumerate(methods.items()):
            dof: list[float] = method_data["dof"]
            error: list[float] = method_data["error"]
            ax.loglog(
                dof,
                error,
                marker="o",
                color=colors[idx],
                linewidth=1.5,
                markersize=5,
                label=method_name,
            )

        ax.set_xlabel("Degrees of Freedom")
        ax.set_ylabel("Error")
        ax.set_title("Convergence Rates")
        ax.legend(loc="best")

        _apply_theme(fig, config)
        fig.tight_layout()
        return fig


@register_plot("hyperparameter_importance")
class HyperparameterImportancePlot(BasePlot):
    """Bar chart of hyperparameter importance scores.

    Expected data keys:
        parameters: list[str] - parameter names
        importance: list[float] - importance scores (0-1)

    """

    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create hyperparameter importance figure."""
        parameters: list[str] = data["parameters"]
        importance: list[float] = data["importance"]

        # Sort by importance descending
        sorted_pairs = sorted(zip(importance, parameters, strict=True), reverse=True)
        sorted_importance = [p[0] for p in sorted_pairs]
        sorted_params = [p[1] for p in sorted_pairs]

        fig, ax = _new_figure(config)
        y_pos = np.arange(len(sorted_params))
        ax.barh(y_pos, sorted_importance, color="#FF9800", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_params)
        ax.set_xlabel("Importance")
        ax.set_title("Hyperparameter Importance")
        ax.invert_yaxis()  # Highest importance at top

        _apply_theme(fig, config)
        fig.tight_layout()
        return fig


@register_plot("comparison_boxplot")
class ComparisonBoxPlot(BasePlot):
    """Box plots comparing distributions across methods.

    Expected data keys:
        methods: dict[str, list[float]] - method name -> list of metric values

    """

    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create comparison box plot figure."""
        methods: dict[str, list[float]] = data["methods"]
        labels = list(methods.keys())
        values = list(methods.values())

        fig, ax = _new_figure(config)
        try:
            bp = ax.boxplot(values, tick_labels=labels, patch_artist=True)
        except TypeError:
            # tick_labels= was added in matplotlib 3.9; fall back for older versions
            bp = ax.boxplot(values, labels=labels, patch_artist=True)

        colors = plt.colormaps["Set2"](np.linspace(0, 1, max(len(labels), 1)))
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)

        ax.set_ylabel("Metric Value")
        ax.set_title("Method Comparison")

        _apply_theme(fig, config)
        fig.tight_layout()
        return fig


@register_plot("timing_comparison")
class TimingComparisonPlot(BasePlot):
    """Bar chart comparing wall-clock times across methods.

    Expected data keys:
        methods: list[str] - method names
        times: list[float] - wall-clock times in seconds

    """

    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create timing comparison figure."""
        methods: list[str] = data["methods"]
        times: list[float] = data["times"]

        fig, ax = _new_figure(config)
        x_pos = np.arange(len(methods))
        ax.bar(x_pos, times, color="#9C27B0", width=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel("Wall Time (s)")
        ax.set_title("Timing Comparison")

        _apply_theme(fig, config)
        fig.tight_layout()
        return fig


@register_plot("pareto_frontier")
class ParetoFrontierPlot(BasePlot):
    """Plot L2 error vs wall-clock compute with Pareto frontier highlighted.

    Uses a log-log scale. This is the headline plot for the DOE Genesis
    Mission proposal:
    methods that sit on the lower-left frontier are strictly better
    (lower error at lower compute) than those above it.

    Expected data keys:
        methods: dict[str, dict] where each value has:
            wall_time: list[float] - wall-clock seconds
            error: list[float] - L2 error values
            n_dof: list[int] - optional DOF annotations

    """

    def create(self, data: dict[str, Any], config: VisualizationConfig) -> Figure:
        """Create Pareto frontier figure."""
        methods: dict[str, dict[str, Any]] = data["methods"]
        colors = plt.colormaps["tab10"](np.linspace(0, 1, max(len(methods), 1)))

        fig, ax = _new_figure(config)

        # Collect all (time, error) points across methods for the Pareto frontier
        all_points: list[tuple[float, float]] = []
        for method_data in methods.values():
            times: list[float] = method_data["wall_time"]
            errors: list[float] = method_data["error"]
            all_points.extend(zip(times, errors, strict=True))

        # Compute Pareto front: point (t, e) is on the front if no other
        # point has both strictly lower t and strictly lower e.
        pareto: list[tuple[float, float]] = []
        for t, e in all_points:
            dominated = any(
                (ot < t and oe <= e) or (ot <= t and oe < e)
                for ot, oe in all_points
                if (ot, oe) != (t, e)
            )
            if not dominated:
                pareto.append((t, e))
        pareto.sort()

        # Plot per-method series
        for idx, (method_name, method_data) in enumerate(methods.items()):
            times = method_data["wall_time"]
            errors = method_data["error"]
            ax.loglog(
                times,
                errors,
                marker="o",
                color=colors[idx],
                linewidth=1.5,
                markersize=6,
                label=method_name,
                alpha=0.85,
            )

        # Overlay the Pareto frontier as a dashed black line
        if len(pareto) >= 2:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.loglog(
                px,
                py,
                linestyle="--",
                color="black",
                linewidth=1.0,
                alpha=0.6,
                label="Pareto frontier",
            )

        ax.set_xlabel("Wall-clock time (s)")
        ax.set_ylabel("L2 error")
        ax.set_title("Error vs Compute (Pareto frontier)")
        ax.legend(loc="best")

        _apply_theme(fig, config)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_plot(
    plot_type: str,
    data: dict[str, Any],
    config: VisualizationConfig,
) -> Figure:
    """Create a plot by type name using the PlotRegistry.

    Args:
        plot_type: Registered plot type name.
        data: Plot-specific data dictionary.
        config: Visualization configuration.

    Returns:
        Matplotlib Figure.

    Raises:
        KeyError: If plot_type is not registered.

    """
    plot_cls = PlotRegistry().get_or_raise(plot_type)
    plot_instance = plot_cls()
    return plot_instance.create(data, config)
