"""Reusable visualization components for AlphaGalerkin demos.

Provides configurable, reusable visualization utilities for:
- Go board rendering with annotations
- Physics field heatmaps
- Performance benchmark charts
- Attention pattern visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from src.demos.config import ColorScheme, VisualizationConfig

logger = structlog.get_logger(__name__)


@dataclass
class PlotResult:
    """Result from a visualization operation.

    Attributes:
        image: Rendered image as numpy array (H, W, 3).
        figure: Matplotlib figure (for further customization).
        metadata: Additional data about the visualization.

    """

    image: NDArray[np.uint8]
    figure: Figure | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        """Close the matplotlib figure to free memory."""
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None


def figure_to_image(fig: Figure) -> NDArray[np.uint8]:
    """Convert matplotlib figure to numpy image array.

    Args:
        fig: Matplotlib figure.

    Returns:
        RGB image array (H, W, 3).

    """
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[:, :, :3]
    return image.astype(np.uint8)


def get_colormap(scheme: ColorScheme) -> Colormap:
    """Get matplotlib colormap from color scheme enum.

    Args:
        scheme: Color scheme enum value.

    Returns:
        Matplotlib colormap.

    """
    return plt.get_cmap(scheme.value)


class BoardVisualizer:
    """Visualizer for Go board states with annotations.

    Supports:
    - Standard board rendering with stones
    - Policy probability heatmaps
    - Move suggestions overlay
    - Territory visualization
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize board visualizer.

        Args:
            config: Visualization configuration (uses defaults if None).

        """
        self.config = config or VisualizationConfig()
        logger.debug("board_visualizer_initialized", dpi=self.config.dpi)

    def render_board(
        self,
        board: NDArray[np.int8],
        last_move: tuple[int, int] | None = None,
        policy: NDArray[np.float32] | None = None,
        suggested_moves: list[tuple[int, int, float]] | None = None,
        title: str | None = None,
    ) -> PlotResult:
        """Render a Go board state.

        Args:
            board: Board array where -1=empty, 0=black, 1=white.
            last_move: (row, col) of last move to highlight.
            policy: Policy probability array to overlay as heatmap.
            suggested_moves: List of (row, col, probability) for top moves.
            title: Optional title for the plot.

        Returns:
            PlotResult with rendered board image.

        """
        size = board.shape[0]
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )
        ax.set_aspect("equal")
        ax.set_facecolor(self.config.board_wood_color)

        # Draw grid
        for i in range(size):
            ax.plot(
                [i, i],
                [0, size - 1],
                color=self.config.grid_color,
                linewidth=1,
                zorder=1,
            )
            ax.plot(
                [0, size - 1],
                [i, i],
                color=self.config.grid_color,
                linewidth=1,
                zorder=1,
            )

        # Draw star points (hoshi)
        stars = self._get_star_points(size)
        for x, y in stars:
            ax.scatter(x, y, s=20, color=self.config.grid_color, zorder=2)

        # Draw policy heatmap if provided
        if policy is not None:
            policy_2d = policy[: size * size].reshape(size, size)
            # Normalize and apply transparency
            policy_norm = policy_2d / (policy_2d.max() + 1e-8)
            cmap = get_colormap(self.config.color_scheme)
            for r in range(size):
                for c in range(size):
                    if policy_norm[r, c] > 0.01:  # Only show significant probabilities
                        rect = plt.Rectangle(
                            (c - 0.4, r - 0.4),
                            0.8,
                            0.8,
                            color=cmap(policy_norm[r, c]),
                            alpha=0.3,
                            zorder=1,
                        )
                        ax.add_patch(rect)

        # Draw stones
        for r in range(size):
            for c in range(size):
                if board[r, c] == 0:  # Black stone
                    circle = plt.Circle(
                        (c, r),
                        0.45,
                        color=self.config.black_stone_color,
                        zorder=3,
                    )
                    ax.add_patch(circle)
                elif board[r, c] == 1:  # White stone
                    circle = plt.Circle(
                        (c, r),
                        0.45,
                        color=self.config.white_stone_color,
                        ec=self.config.stone_border_color,
                        zorder=3,
                    )
                    ax.add_patch(circle)

        # Mark last move
        if last_move is not None:
            lr, lc = last_move
            ax.scatter(
                lc,
                lr,
                s=30,
                color=self.config.last_move_marker_color,
                marker="x",
                linewidth=2,
                zorder=4,
            )

        # Mark suggested moves
        if suggested_moves:
            for rank, (r, c, _prob) in enumerate(suggested_moves[:5], 1):
                ax.annotate(
                    f"{rank}",
                    (c, r),
                    fontsize=self.config.font_size - 2,
                    color="blue" if rank == 1 else "gray",
                    ha="center",
                    va="center",
                    zorder=5,
                )

        ax.set_xlim(-0.5, size - 0.5)
        ax.set_ylim(-0.5, size - 0.5)
        ax.invert_yaxis()
        ax.axis("off")

        if title:
            ax.set_title(title, fontsize=self.config.font_size + 2)

        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "board_size": size,
                "has_policy": policy is not None,
                "has_suggestions": suggested_moves is not None,
            },
        )

    def _get_star_points(self, size: int) -> list[tuple[int, int]]:
        """Get star point (hoshi) coordinates for board size.

        Args:
            size: Board size.

        Returns:
            List of (x, y) star point coordinates.

        """
        star_points: dict[int, list[tuple[int, int]]] = {
            19: [
                (3, 3),
                (3, 9),
                (3, 15),
                (9, 3),
                (9, 9),
                (9, 15),
                (15, 3),
                (15, 9),
                (15, 15),
            ],
            13: [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)],
            9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
            7: [(2, 2), (2, 4), (3, 3), (4, 2), (4, 4)],
        }
        return star_points.get(size, [])


class FieldVisualizer:
    """Visualizer for physics fields (Poisson, heat, etc.).

    Supports:
    - 2D field heatmaps
    - Side-by-side comparison plots
    - Difference maps
    - Animation frames
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize field visualizer.

        Args:
            config: Visualization configuration (uses defaults if None).

        """
        self.config = config or VisualizationConfig()
        logger.debug("field_visualizer_initialized", dpi=self.config.dpi)

    def render_field(
        self,
        field: NDArray[np.float32],
        title: str = "Field",
        vmin: float | None = None,
        vmax: float | None = None,
        show_colorbar: bool = True,
    ) -> PlotResult:
        """Render a 2D scalar field as heatmap.

        Args:
            field: 2D array of field values.
            title: Plot title.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            show_colorbar: Whether to show colorbar.

        Returns:
            PlotResult with rendered field image.

        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        cmap = get_colormap(self.config.color_scheme)
        im = ax.imshow(
            field,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
        )

        if show_colorbar:
            plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title(title, fontsize=self.config.font_size + 2)
        ax.set_xlabel("x", fontsize=self.config.font_size)
        ax.set_ylabel("y", fontsize=self.config.font_size)

        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "field_shape": field.shape,
                "field_min": float(field.min()),
                "field_max": float(field.max()),
                "field_mean": float(field.mean()),
            },
        )

    def render_comparison(
        self,
        ground_truth: NDArray[np.float32],
        prediction: NDArray[np.float32],
        title: str = "Comparison",
        show_difference: bool = True,
    ) -> PlotResult:
        """Render ground truth, prediction, and difference.

        Args:
            ground_truth: Ground truth field.
            prediction: Predicted field.
            title: Overall title.
            show_difference: Whether to show difference map.

        Returns:
            PlotResult with comparison visualization.

        """
        n_plots = 3 if show_difference else 2
        fig, axes = plt.subplots(
            1,
            n_plots,
            figsize=(self.config.figure_width * n_plots / 2, self.config.figure_height),
            dpi=self.config.dpi,
        )

        cmap = get_colormap(self.config.color_scheme)

        # Shared colorbar limits
        vmin = min(ground_truth.min(), prediction.min())
        vmax = max(ground_truth.max(), prediction.max())

        # Ground truth
        im0 = axes[0].imshow(ground_truth, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axes[0].set_title("Ground Truth", fontsize=self.config.font_size)
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        # Prediction
        im1 = axes[1].imshow(prediction, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axes[1].set_title("Prediction", fontsize=self.config.font_size)
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        # Difference
        if show_difference:
            diff = ground_truth - prediction
            diff_cmap = get_colormap(ColorScheme.SEISMIC)
            diff_max = max(abs(diff.min()), abs(diff.max()))
            im2 = axes[2].imshow(
                diff,
                cmap=diff_cmap,
                vmin=-diff_max,
                vmax=diff_max,
                origin="lower",
            )
            axes[2].set_title("Difference (GT - Pred)", fontsize=self.config.font_size)
            plt.colorbar(im2, ax=axes[2], shrink=0.8)

        fig.suptitle(title, fontsize=self.config.font_size + 4)
        plt.tight_layout()
        image = figure_to_image(fig)

        # Calculate metrics
        mse = float(np.mean((ground_truth - prediction) ** 2))
        mae = float(np.mean(np.abs(ground_truth - prediction)))

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "mse": mse,
                "mae": mae,
                "max_error": float(np.max(np.abs(ground_truth - prediction))),
                "gt_shape": ground_truth.shape,
                "pred_shape": prediction.shape,
            },
        )

    def render_transfer_comparison(
        self,
        results: dict[int, dict[str, Any]],
        title: str = "Zero-Shot Transfer Results",
    ) -> PlotResult:
        """Render multi-resolution transfer results.

        Args:
            results: Dict mapping grid_size to {gt, pred, mse} dicts.
            title: Overall title.

        Returns:
            PlotResult with multi-resolution comparison.

        """
        n_sizes = len(results)
        fig, axes = plt.subplots(
            2,
            n_sizes,
            figsize=(self.config.figure_width * n_sizes / 3, self.config.figure_height * 1.5),
            dpi=self.config.dpi,
        )

        if n_sizes == 1:
            axes = axes.reshape(2, 1)

        cmap = get_colormap(self.config.color_scheme)

        for i, (grid_size, data) in enumerate(sorted(results.items())):
            gt = data["ground_truth"]
            pred = data["prediction"]

            # Top row: Predictions
            axes[0, i].imshow(pred, cmap=cmap, origin="lower")
            axes[0, i].set_title(f"{grid_size}×{grid_size}\nMSE: {data['mse']:.6f}")
            axes[0, i].axis("off")

            # Bottom row: Ground truth
            axes[1, i].imshow(gt, cmap=cmap, origin="lower")
            axes[1, i].set_title("Ground Truth")
            axes[1, i].axis("off")

        fig.suptitle(title, fontsize=self.config.font_size + 4)
        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "grid_sizes": list(results.keys()),
                "mse_values": {k: v["mse"] for k, v in results.items()},
            },
        )


class ChartVisualizer:
    """Visualizer for performance charts and metrics.

    Supports:
    - Bar charts for comparisons
    - Line plots for scaling analysis
    - Box plots for timing distributions
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize chart visualizer.

        Args:
            config: Visualization configuration (uses defaults if None).

        """
        self.config = config or VisualizationConfig()
        logger.debug("chart_visualizer_initialized", dpi=self.config.dpi)

    def render_scaling_comparison(
        self,
        sizes: list[int],
        fnet_times: list[float],
        softmax_times: list[float],
        title: str = "FNet vs Softmax Scaling",
    ) -> PlotResult:
        """Render FNet vs Softmax scaling comparison.

        Args:
            sizes: Sequence lengths (N values).
            fnet_times: FNet execution times.
            softmax_times: Softmax execution times.
            title: Plot title.

        Returns:
            PlotResult with scaling comparison chart.

        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        ax.plot(
            sizes,
            fnet_times,
            "o-",
            label="FNet O(N log N)",
            color="#2ecc71",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            sizes,
            softmax_times,
            "s-",
            label="Softmax O(N²)",
            color="#e74c3c",
            linewidth=2,
            markersize=8,
        )

        ax.set_xlabel("Sequence Length (N)", fontsize=self.config.font_size)
        ax.set_ylabel("Time (ms)", fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2)
        ax.legend(fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)

        # Add speedup annotations
        for n, ft, st in zip(sizes, fnet_times, softmax_times, strict=True):
            if st > 0:
                speedup = st / ft if ft > 0 else 0
                ax.annotate(
                    f"{speedup:.1f}×",
                    (n, (ft + st) / 2),
                    fontsize=self.config.font_size - 2,
                    ha="center",
                )

        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "sizes": sizes,
                "fnet_times_ms": fnet_times,
                "softmax_times_ms": softmax_times,
                "speedups": [
                    st / ft if ft > 0 else 0
                    for ft, st in zip(fnet_times, softmax_times, strict=True)
                ],
            },
        )

    def render_mse_bar_chart(
        self,
        labels: list[str],
        mse_values: list[float],
        threshold: float | None = None,
        title: str = "MSE by Resolution",
    ) -> PlotResult:
        """Render MSE bar chart for transfer results.

        Args:
            labels: Labels for each bar.
            mse_values: MSE values.
            threshold: Optional threshold line to show.
            title: Plot title.

        Returns:
            PlotResult with MSE bar chart.

        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        colors = [
            "#27ae60" if mse < (threshold or float("inf")) else "#e74c3c" for mse in mse_values
        ]

        bars = ax.bar(labels, mse_values, color=colors, edgecolor="black")

        # Add value labels on bars
        for bar, mse in zip(bars, mse_values, strict=True):
            ax.annotate(
                f"{mse:.6f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=self.config.font_size - 1,
            )

        if threshold is not None:
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                label=f"Threshold: {threshold}",
                linewidth=2,
            )
            ax.legend(fontsize=self.config.font_size)

        ax.set_ylabel("MSE", fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2)
        ax.set_yscale("log")  # Log scale for MSE visualization

        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "labels": labels,
                "mse_values": mse_values,
                "threshold": threshold,
                "all_below_threshold": all(m < (threshold or float("inf")) for m in mse_values),
            },
        )

    def render_attention_heatmap(
        self,
        attention_weights: NDArray[np.float32],
        head_idx: int = 0,
        title: str = "Attention Weights",
    ) -> PlotResult:
        """Render attention weight heatmap.

        Args:
            attention_weights: Attention weight matrix (heads, seq, seq).
            head_idx: Which attention head to visualize.
            title: Plot title.

        Returns:
            PlotResult with attention heatmap.

        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        weights = attention_weights[head_idx]
        cmap = get_colormap(self.config.color_scheme)

        im = ax.imshow(weights, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_title(f"{title} (Head {head_idx})", fontsize=self.config.font_size + 2)
        ax.set_xlabel("Key Position", fontsize=self.config.font_size)
        ax.set_ylabel("Query Position", fontsize=self.config.font_size)

        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "head_idx": head_idx,
                "seq_length": weights.shape[0],
                "max_attention": float(weights.max()),
                "entropy": float(-np.sum(weights * np.log(weights + 1e-10), axis=-1).mean()),
            },
        )

    def render_fourier_spectrum(
        self,
        frequencies: NDArray[np.float32],
        amplitudes: NDArray[np.float32],
        title: str = "Fourier Feature Spectrum",
    ) -> PlotResult:
        """Render Fourier feature frequency spectrum.

        Args:
            frequencies: Frequency values.
            amplitudes: Amplitude values.
            title: Plot title.

        Returns:
            PlotResult with spectrum visualization.

        """
        fig, ax = plt.subplots(
            figsize=(self.config.figure_width, self.config.figure_height),
            dpi=self.config.dpi,
        )

        ax.stem(
            frequencies,
            amplitudes,
            linefmt="C0-",
            markerfmt="C0o",
            basefmt="k-",
        )

        ax.set_xlabel("Frequency", fontsize=self.config.font_size)
        ax.set_ylabel("Amplitude", fontsize=self.config.font_size)
        ax.set_title(title, fontsize=self.config.font_size + 2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "n_frequencies": len(frequencies),
                "max_frequency": float(frequencies.max()),
                "dominant_frequency": float(frequencies[np.argmax(amplitudes)]),
            },
        )


class AttentionVisualizer:
    """Specialized visualizer for attention pattern analysis.

    Compares Galerkin vs Softmax attention patterns.
    """

    def __init__(self, config: VisualizationConfig | None = None) -> None:
        """Initialize attention visualizer.

        Args:
            config: Visualization configuration.

        """
        self.config = config or VisualizationConfig()
        self.chart_viz = ChartVisualizer(config)
        logger.debug("attention_visualizer_initialized", dpi=self.config.dpi)

    def render_galerkin_vs_softmax(
        self,
        galerkin_attn: NDArray[np.float32],
        softmax_attn: NDArray[np.float32],
        title: str = "Galerkin vs Softmax Attention",
    ) -> PlotResult:
        """Render side-by-side comparison of attention types.

        Args:
            galerkin_attn: Galerkin attention weights (heads, seq, seq).
            softmax_attn: Softmax attention weights (heads, seq, seq).
            title: Plot title.

        Returns:
            PlotResult with comparison visualization.

        """
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(self.config.figure_width * 1.5, self.config.figure_height),
            dpi=self.config.dpi,
        )

        cmap = get_colormap(self.config.color_scheme)

        # Average over heads for visualization
        galerkin_avg = galerkin_attn.mean(axis=0)
        softmax_avg = softmax_attn.mean(axis=0)

        im0 = axes[0].imshow(galerkin_avg, cmap=cmap, aspect="auto")
        axes[0].set_title("Galerkin (O(N))", fontsize=self.config.font_size)
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(softmax_avg, cmap=cmap, aspect="auto")
        axes[1].set_title("Softmax (O(N²))", fontsize=self.config.font_size)
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        for ax in axes:
            ax.set_xlabel("Key Position", fontsize=self.config.font_size - 1)
            ax.set_ylabel("Query Position", fontsize=self.config.font_size - 1)

        fig.suptitle(title, fontsize=self.config.font_size + 4)
        plt.tight_layout()
        image = figure_to_image(fig)

        return PlotResult(
            image=image,
            figure=fig,
            metadata={
                "galerkin_sparsity": float((galerkin_avg < 0.01).mean()),
                "softmax_sparsity": float((softmax_avg < 0.01).mean()),
                "galerkin_entropy": float(
                    -np.sum(galerkin_avg * np.log(galerkin_avg + 1e-10), axis=-1).mean()
                ),
                "softmax_entropy": float(
                    -np.sum(softmax_avg * np.log(softmax_avg + 1e-10), axis=-1).mean()
                ),
            },
        )
