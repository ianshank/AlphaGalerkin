"""Go board renderer with coordinate labels.

This module provides board rendering functionality with:
- Configurable coordinate labels (letters/numbers on perimeter)
- Star point (hoshi) rendering for all standard board sizes
- Last move highlighting
- Resolution-independent scaling
- Structured logging for debugging
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import structlog
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Circle

# Ensure config imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.board import (
    STAR_POINTS_BY_SIZE,
    BoardRenderConfig,
    CoordinateLabelConfig,
    get_column_letter,
    get_default_space_config,
)

if TYPE_CHECKING:
    from src.tools.gtp import SimpleGoGame

logger = structlog.get_logger(__name__)


class BoardRenderer:
    """Renderer for Go boards with coordinate labels.

    Renders Go boards using matplotlib with support for:
    - Variable board sizes (9x9, 13x13, 19x19)
    - Configurable coordinate labels on the perimeter
    - Star points (hoshi) following traditional patterns
    - Last move highlighting
    - Customizable colors and styling

    Attributes:
        config: Rendering configuration.

    Example:
        >>> renderer = BoardRenderer()
        >>> image = renderer.render(game, last_move=40)

    """

    def __init__(self, config: BoardRenderConfig | None = None) -> None:
        """Initialize renderer.

        Args:
            config: Rendering configuration. Uses defaults if None.

        """
        self.config = config or get_default_space_config().render
        self._logger = logger.bind(component="BoardRenderer")

    def render(
        self,
        game: SimpleGoGame,
        last_move: int | None = None,
    ) -> np.ndarray:
        """Render the board to an image array.

        Args:
            game: Game state to render.
            last_move: Index of last move for highlighting.
                Pass move index calculated as row * board_size + col.

        Returns:
            RGB image as numpy array of shape (H, W, 3).

        """
        size = game.board_size
        label_config = self.config.coordinate_labels

        # Calculate figure dimensions with label padding
        padding = label_config.padding if label_config.show_labels else 0.3
        fig_width, fig_height = self.config.figure_size

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect("equal")
        ax.set_facecolor(self.config.board_color)

        # Draw grid lines
        self._draw_grid(ax, size)

        # Draw star points (hoshi)
        self._draw_star_points(ax, size)

        # Draw stones and last move marker
        self._draw_stones(ax, game, last_move)

        # Draw coordinate labels on perimeter
        if label_config.show_labels:
            self._draw_coordinate_labels(ax, size, label_config)

        # Set axis limits with padding for labels
        margin = padding if label_config.show_labels else 0.5
        ax.set_xlim(-margin, size - 1 + margin)
        ax.set_ylim(-margin, size - 1 + margin)
        ax.invert_yaxis()
        ax.axis("off")

        # Convert to image array
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        self._logger.debug(
            "board_rendered",
            board_size=size,
            has_last_move=last_move is not None,
            image_shape=image.shape,
        )

        return image

    def _draw_grid(self, ax: plt.Axes, size: int) -> None:
        """Draw the board grid lines.

        Args:
            ax: Matplotlib axes to draw on.
            size: Board size.

        """
        for i in range(size):
            # Vertical lines
            ax.plot(
                [i, i],
                [0, size - 1],
                color=self.config.line_color,
                linewidth=self.config.line_width,
                zorder=1,
            )
            # Horizontal lines
            ax.plot(
                [0, size - 1],
                [i, i],
                color=self.config.line_color,
                linewidth=self.config.line_width,
                zorder=1,
            )

    def _draw_star_points(self, ax: plt.Axes, size: int) -> None:
        """Draw hoshi (star points) on the board.

        Args:
            ax: Matplotlib axes to draw on.
            size: Board size.

        """
        star_points = STAR_POINTS_BY_SIZE.get(size, [])

        for row, col in star_points:
            ax.scatter(
                col,
                row,
                s=self.config.star_point_size,
                color=self.config.line_color,
                zorder=2,
            )

    def _draw_stones(
        self,
        ax: plt.Axes,
        game: SimpleGoGame,
        last_move: int | None,
    ) -> None:
        """Draw stones and last move marker.

        Args:
            ax: Matplotlib axes to draw on.
            game: Game state with board positions.
            last_move: Index of last move (if any).

        """
        # Import here to avoid circular imports
        from src.tools.gtp import SimpleGoGame

        size = game.board_size

        for row in range(size):
            for col in range(size):
                stone = game.board[row, col]

                if stone == SimpleGoGame.BLACK:
                    circle = Circle(
                        (col, row),
                        self.config.stone_radius,
                        color="black",
                        zorder=3,
                    )
                    ax.add_patch(circle)
                elif stone == SimpleGoGame.WHITE:
                    circle = Circle(
                        (col, row),
                        self.config.stone_radius,
                        color="white",
                        edgecolor="black",
                        linewidth=1,
                        zorder=3,
                    )
                    ax.add_patch(circle)

        # Draw last move marker
        if last_move is not None:
            last_row = last_move // size
            last_col = last_move % size
            ax.scatter(
                last_col,
                last_row,
                s=self.config.last_move_marker_size,
                color=self.config.last_move_marker_color,
                marker="x",
                linewidths=2,
                zorder=4,
            )

    def _draw_coordinate_labels(
        self,
        ax: plt.Axes,
        size: int,
        config: CoordinateLabelConfig,
    ) -> None:
        """Draw coordinate labels around the board perimeter.

        Labels are drawn on all four sides:
        - Top and bottom: Column labels (A-T or 1-19)
        - Left and right: Row labels (1-19, with 1 at bottom per Go convention)

        Args:
            ax: Matplotlib axes to draw on.
            size: Board size.
            config: Coordinate label configuration.

        """
        # Column labels (letters or numbers)
        for col in range(size):
            if config.label_style == "letters":
                label = get_column_letter(col, config.skip_i)
            else:
                label = str(col + 1)

            # Top labels (above board)
            ax.text(
                col,
                -config.padding * 0.7,
                label,
                ha="center",
                va="center",
                fontsize=config.font_size,
                color=config.font_color,
                fontweight="bold",
            )
            # Bottom labels (below board)
            ax.text(
                col,
                size - 1 + config.padding * 0.7,
                label,
                ha="center",
                va="center",
                fontsize=config.font_size,
                color=config.font_color,
                fontweight="bold",
            )

        # Row labels (numbers, 1-indexed from bottom per Go convention)
        for row in range(size):
            # GTP convention: row 1 is at the bottom (highest row index)
            row_label = str(size - row)

            # Left labels
            ax.text(
                -config.padding * 0.7,
                row,
                row_label,
                ha="center",
                va="center",
                fontsize=config.font_size,
                color=config.font_color,
                fontweight="bold",
            )
            # Right labels
            ax.text(
                size - 1 + config.padding * 0.7,
                row,
                row_label,
                ha="center",
                va="center",
                fontsize=config.font_size,
                color=config.font_color,
                fontweight="bold",
            )


def render_board(
    game: SimpleGoGame,
    last_move: int | None = None,
    config: BoardRenderConfig | None = None,
) -> np.ndarray:
    """Render a board to an image array.

    Args:
        game: Game state to render.
        last_move: Index of last move.
        config: Rendering configuration.

    Returns:
        RGB image as numpy array.

    Example:
        >>> from src.tools.gtp import SimpleGoGame
        >>> game = SimpleGoGame(9)
        >>> image = render_board(game)
        >>> image.shape
        (H, W, 3)

    """
    renderer = BoardRenderer(config)
    return renderer.render(game, last_move)
