"""Tests for the PoC visualization module.

Covers configuration validation, plot registry, plot generation,
and HTML report generation.
"""

from __future__ import annotations

import pytest

from src.poc.visualization.config import VisualizationConfig
from src.poc.visualization.plots import (
    PlotRegistry,
    create_plot,
)
from src.poc.visualization.reports import (
    HTMLReportGenerator,
    ReportSection,
    generate_report,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestVisualizationConfig:
    """Test VisualizationConfig validation."""

    def test_default_values(self) -> None:
        cfg = VisualizationConfig(name="test")
        assert cfg.output_format == "html"
        assert cfg.plot_width == 800
        assert cfg.plot_height == 500
        assert cfg.theme == "light"
        assert cfg.include_interactive is True
        assert cfg.dpi == 150

    def test_custom_values(self) -> None:
        cfg = VisualizationConfig(
            name="custom",
            output_format="png",
            plot_width=400,
            plot_height=300,
            theme="dark",
            include_interactive=False,
            dpi=300,
        )
        assert cfg.output_format == "png"
        assert cfg.plot_width == 400
        assert cfg.dpi == 300

    def test_width_too_small_rejected(self) -> None:
        with pytest.raises(Exception):
            VisualizationConfig(name="bad", plot_width=100)

    def test_width_too_large_rejected(self) -> None:
        with pytest.raises(Exception):
            VisualizationConfig(name="bad", plot_width=5000)

    def test_dpi_too_small_rejected(self) -> None:
        with pytest.raises(Exception):
            VisualizationConfig(name="bad", dpi=10)

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(Exception):
            VisualizationConfig(name="bad", output_format="svg")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plot registry tests
# ---------------------------------------------------------------------------


class TestPlotRegistry:
    """Test PlotRegistry discovery."""

    def test_registry_lists_builtin_plots(self) -> None:
        reg = PlotRegistry()
        items = reg.list_items()
        assert "training_curves" in items
        assert "convergence_rates" in items
        assert "hyperparameter_importance" in items
        assert "comparison_boxplot" in items
        assert "timing_comparison" in items

    def test_registry_get_returns_class(self) -> None:
        reg = PlotRegistry()
        cls = reg.get("training_curves")
        assert cls is not None

    def test_registry_get_unknown_returns_none(self) -> None:
        reg = PlotRegistry()
        cls = reg.get("nonexistent_plot")
        assert cls is None


# ---------------------------------------------------------------------------
# Plot generation tests
# ---------------------------------------------------------------------------


class TestPlotGeneration:
    """Test plot generation with sample data."""

    def test_training_curves(self) -> None:
        import matplotlib.pyplot as plt

        data = {"steps": [1, 2, 3, 4, 5], "loss": [1.0, 0.8, 0.6, 0.5, 0.4]}
        cfg = VisualizationConfig(name="test")
        fig = create_plot("training_curves", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_convergence_rates(self) -> None:
        import matplotlib.pyplot as plt

        data = {
            "methods": {
                "FDM": {"dof": [10, 100, 1000], "error": [0.1, 0.01, 0.001]},
            },
        }
        cfg = VisualizationConfig(name="test")
        fig = create_plot("convergence_rates", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_hyperparameter_importance(self) -> None:
        import matplotlib.pyplot as plt

        data = {
            "parameters": ["lr", "d_model", "n_layers"],
            "importance": [0.5, 0.3, 0.2],
        }
        cfg = VisualizationConfig(name="test")
        fig = create_plot("hyperparameter_importance", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_boxplot(self) -> None:
        import matplotlib.pyplot as plt

        data = {
            "methods": {
                "A": [0.1, 0.2, 0.15],
                "B": [0.3, 0.25, 0.35],
            },
        }
        cfg = VisualizationConfig(name="test")
        fig = create_plot("comparison_boxplot", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_timing_comparison(self) -> None:
        import matplotlib.pyplot as plt

        data = {"methods": ["FDM", "AMR", "PINN"], "times": [1.0, 0.5, 2.0]}
        cfg = VisualizationConfig(name="test")
        fig = create_plot("timing_comparison", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_unknown_plot_raises_error(self) -> None:
        cfg = VisualizationConfig(name="test")
        with pytest.raises(KeyError, match="nonexistent_plot"):
            create_plot("nonexistent_plot", {}, cfg)

    def test_training_curves_with_accuracy_combines_legends(self) -> None:
        """The ``accuracy is not None`` branch overlays a twin axis + legend.

        The plain-loss test above never sets ``accuracy``, so this branch
        (dual-axis plot + combined ``ax_loss.legend(lines_loss + lines_acc, ...)``)
        was previously untested.
        """
        import matplotlib.pyplot as plt

        data = {
            "steps": [1, 2, 3, 4, 5],
            "loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "accuracy": [0.2, 0.4, 0.6, 0.75, 0.9],
        }
        cfg = VisualizationConfig(name="test")
        fig = create_plot("training_curves", data, cfg)
        assert isinstance(fig, plt.Figure)
        # Two axes: the primary loss axis and the twinned accuracy axis.
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_pareto_frontier_plot(self) -> None:
        """The headline Pareto-frontier plot (previously entirely untested)."""
        import matplotlib.pyplot as plt

        data = {
            "methods": {
                "FDM": {"wall_time": [1.0, 2.0, 4.0], "error": [0.1, 0.05, 0.01]},
                "MCTS": {"wall_time": [0.5, 1.5, 3.0], "error": [0.2, 0.08, 0.02]},
            },
        }
        cfg = VisualizationConfig(name="test")
        fig = create_plot("pareto_frontier", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pareto_frontier_plot_with_single_dominant_point(self) -> None:
        """A single Pareto-minimal point skips the frontier overlay line.

        ``len(pareto) >= 2`` guards the dashed-line overlay; a method sweep
        that collapses to one non-dominated point exercises the False branch.
        """
        import matplotlib.pyplot as plt

        data = {
            "methods": {
                "Only": {"wall_time": [1.0], "error": [0.1]},
            },
        }
        cfg = VisualizationConfig(name="test")
        fig = create_plot("pareto_frontier", data, cfg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_boxplot_falls_back_on_older_matplotlib(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``tick_labels=`` (matplotlib>=3.9) falls back to ``labels=`` on TypeError.

        Simulates an older matplotlib whose ``Axes.boxplot`` does not accept
        ``tick_labels`` by rejecting that kwarg on the first call only.
        """
        import matplotlib.axes
        import matplotlib.pyplot as plt

        # The installed matplotlib (3.11+) accepts ``tick_labels=`` but has
        # fully removed the old ``labels=`` kwarg the except-branch falls
        # back to, so a real "older matplotlib" cannot be reproduced by
        # simply calling through. Instead: reject ``tick_labels=`` on the
        # first call (forcing the source's except-branch to run its
        # ``labels=`` call), then translate ``labels=`` back to
        # ``tick_labels=`` before delegating, so the retry still renders on
        # this matplotlib version. This asserts the source actually took the
        # except branch (via ``calls``) without depending on a removed API.
        original_boxplot = matplotlib.axes.Axes.boxplot
        calls: list[bool] = []

        def _boxplot(self: matplotlib.axes.Axes, x: object, **kwargs: object) -> object:
            calls.append("tick_labels" in kwargs)
            if "tick_labels" in kwargs:
                raise TypeError("tick_labels= not supported on this matplotlib")
            labels = kwargs.pop("labels", None)
            if labels is not None:
                kwargs["tick_labels"] = labels
            return original_boxplot(self, x, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "boxplot", _boxplot)

        data = {"methods": {"A": [0.1, 0.2, 0.15], "B": [0.3, 0.25, 0.35]}}
        cfg = VisualizationConfig(name="test")
        fig = create_plot("comparison_boxplot", data, cfg)

        assert isinstance(fig, plt.Figure)
        # First call rejected (tick_labels=), second call (labels=) succeeded.
        assert calls == [True, False]
        plt.close(fig)


# ---------------------------------------------------------------------------
# HTML report generation tests
# ---------------------------------------------------------------------------


class TestHTMLReportGenerator:
    """Test HTML report generation."""

    def test_generate_report_returns_html(self) -> None:
        sections = [
            ReportSection(title="Summary", content="<p>Test content</p>"),
        ]
        cfg = VisualizationConfig(name="test")
        result = generate_report(title="Test Report", sections=sections, config=cfg)
        assert "<html" in result
        assert "Test Report" in result

    def test_generate_with_multiple_sections(self) -> None:
        sections = [
            ReportSection(title="Section 1", content="<p>First</p>"),
            ReportSection(title="Section 2", content="<p>Second</p>"),
        ]
        cfg = VisualizationConfig(name="test")
        result = generate_report(title="Multi", sections=sections, config=cfg)
        assert "Section 1" in result
        assert "Section 2" in result

    def test_generate_empty_sections(self) -> None:
        cfg = VisualizationConfig(name="test")
        result = generate_report(title="Empty", sections=[], config=cfg)
        assert "<html" in result
        assert "Empty" in result

    def test_generate_with_figure(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        sections = [
            ReportSection(title="Plot", figures=[fig]),
        ]
        cfg = VisualizationConfig(name="test")
        result = generate_report(title="With Plot", sections=sections, config=cfg)
        assert "data:image/png;base64" in result
        plt.close(fig)

    def test_render_figure_base64(self) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        cfg = VisualizationConfig(name="test")
        result = HTMLReportGenerator._render_figure(fig, cfg)
        assert "base64" in result
        assert "<img" in result
        plt.close(fig)

    def test_themed_report(self) -> None:
        sections = [ReportSection(title="Test")]
        for theme in ("light", "dark", "publication"):
            cfg = VisualizationConfig(name="test", theme=theme)  # type: ignore[arg-type]
            result = generate_report(title="Themed", sections=sections, config=cfg)
            assert "<html" in result

    def test_instance_method_generate_report(self) -> None:
        gen = HTMLReportGenerator()
        sections = [ReportSection(title="Test", content="Hello")]
        cfg = VisualizationConfig(name="test")
        result = gen.generate_report(title="Instance", sections=sections, config=cfg)
        assert "Instance" in result
        assert "Hello" in result
