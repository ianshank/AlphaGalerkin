"""Tests for visualizer."""

from __future__ import annotations

from src.prototyping.evaluator import EvalResult, MetricResult
from src.prototyping.trainer import TrainResult
from src.prototyping.visualizer import (
    PlotData,
    PlotType,
    Visualizer,
    create_visualizer,
)


class TestPlotType:
    """Tests for PlotType enum."""

    def test_all_types_exist(self) -> None:
        """Test all plot types exist."""
        assert PlotType.LINE.value == "line"
        assert PlotType.BAR.value == "bar"
        assert PlotType.SCATTER.value == "scatter"
        assert PlotType.HEATMAP.value == "heatmap"
        assert PlotType.HISTOGRAM.value == "histogram"


class TestPlotData:
    """Tests for PlotData dataclass."""

    def test_initialization(self) -> None:
        """Test plot data initialization."""
        plot = PlotData(
            plot_id="plot123",
            plot_type=PlotType.LINE,
            title="Test Plot",
            x_label="X",
            y_label="Y",
            x_data=[1, 2, 3],
            y_data=[[1, 2, 3]],
        )

        assert plot.plot_id == "plot123"
        assert plot.plot_type == PlotType.LINE
        assert plot.title == "Test Plot"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        plot = PlotData(
            plot_id="plot123",
            plot_type=PlotType.LINE,
            title="Test",
            x_label="X",
            y_label="Y",
        )

        data = plot.to_dict()

        assert data["plot_id"] == "plot123"
        assert data["plot_type"] == "line"


class TestVisualizer:
    """Tests for Visualizer."""

    def test_initialization(self, visualizer: Visualizer) -> None:
        """Test visualizer initialization."""
        assert visualizer.width == 40
        assert visualizer.height == 10
        assert len(visualizer.plots) == 0

    def test_plot_training_loss(
        self,
        visualizer: Visualizer,
        train_result: TrainResult,
    ) -> None:
        """Test plotting training loss."""
        plot = visualizer.plot_training_loss(train_result)

        assert isinstance(plot, str)
        assert "Training Loss" in plot
        assert len(visualizer.plots) == 1

    def test_plot_training_loss_empty(self, visualizer: Visualizer) -> None:
        """Test plotting with no loss data."""
        result = TrainResult(
            result_id="test",
            model_id="model",
            n_epochs=1,
            n_steps=1,
            final_loss=0.0,
            best_loss=0.0,
            metrics={},  # No loss
        )

        plot = visualizer.plot_training_loss(result)
        assert "No loss data" in plot

    def test_plot_comparison(self, visualizer: Visualizer) -> None:
        """Test plotting comparison."""
        results = [
            EvalResult(
                result_id="r1",
                model_id="model1",
                n_samples=10,
                metrics={"mse": MetricResult(name="mse", value=0.1)},
            ),
            EvalResult(
                result_id="r2",
                model_id="model2",
                n_samples=10,
                metrics={"mse": MetricResult(name="mse", value=0.05)},
            ),
        ]

        plot = visualizer.plot_comparison(results, "mse")

        assert isinstance(plot, str)
        assert "Comparison" in plot
        assert "model1" in plot
        assert "model2" in plot

    def test_plot_learning_curves(self, visualizer: Visualizer) -> None:
        """Test plotting learning curves."""
        results = [
            TrainResult(
                result_id="r1",
                model_id="model1",
                n_epochs=5,
                n_steps=50,
                final_loss=0.1,
                best_loss=0.05,
                metrics={"loss": [0.5, 0.3, 0.2, 0.15, 0.1]},
            ),
            TrainResult(
                result_id="r2",
                model_id="model2",
                n_epochs=5,
                n_steps=50,
                final_loss=0.2,
                best_loss=0.15,
                metrics={"loss": [0.6, 0.4, 0.3, 0.25, 0.2]},
            ),
        ]

        plot = visualizer.plot_learning_curves(results)

        assert isinstance(plot, str)
        assert "Learning Curves" in plot
        assert "model1" in plot
        assert "model2" in plot

    def test_plot_metrics_table(self, visualizer: Visualizer) -> None:
        """Test metrics table."""
        results = [
            EvalResult(
                result_id="r1",
                model_id="model1",
                n_samples=10,
                metrics={
                    "mse": MetricResult(name="mse", value=0.1),
                    "mae": MetricResult(name="mae", value=0.2),
                },
            ),
            EvalResult(
                result_id="r2",
                model_id="model2",
                n_samples=10,
                metrics={
                    "mse": MetricResult(name="mse", value=0.05),
                },
            ),
        ]

        table = visualizer.plot_metrics_table(results)

        assert isinstance(table, str)
        assert "model1" in table
        assert "model2" in table
        assert "mse" in table

    def test_export_plot_data(self, visualizer: Visualizer) -> None:
        """Test exporting plot data."""
        result = TrainResult(
            result_id="test",
            model_id="model",
            n_epochs=1,
            n_steps=1,
            final_loss=0.1,
            best_loss=0.1,
            metrics={"loss": [0.5, 0.3, 0.1]},
        )

        visualizer.plot_training_loss(result)
        exported = visualizer.export_plot_data()

        assert len(exported) == 1
        assert exported[0]["plot_type"] == "line"

    def test_export_specific_plot(self, visualizer: Visualizer) -> None:
        """Test exporting specific plot."""
        result = TrainResult(
            result_id="test",
            model_id="model",
            n_epochs=1,
            n_steps=1,
            final_loss=0.1,
            best_loss=0.1,
            metrics={"loss": [0.5, 0.3, 0.1]},
        )

        visualizer.plot_training_loss(result)
        plot_id = visualizer.plots[0].plot_id

        exported = visualizer.export_plot_data(plot_id=plot_id)
        assert len(exported) == 1

    def test_clear(self, visualizer: Visualizer) -> None:
        """Test clearing plots."""
        result = TrainResult(
            result_id="test",
            model_id="model",
            n_epochs=1,
            n_steps=1,
            final_loss=0.1,
            best_loss=0.1,
            metrics={"loss": [0.5]},
        )

        visualizer.plot_training_loss(result)
        assert len(visualizer.plots) == 1

        visualizer.clear()
        assert len(visualizer.plots) == 0


class TestCreateVisualizer:
    """Tests for create_visualizer factory."""

    def test_create_default(self) -> None:
        """Test creating default visualizer."""
        visualizer = create_visualizer()
        assert visualizer.width == 60
        assert visualizer.height == 20

    def test_create_with_dimensions(self) -> None:
        """Test creating with custom dimensions."""
        visualizer = create_visualizer(width=80, height=30)
        assert visualizer.width == 80
        assert visualizer.height == 30
