"""Board configuration for HuggingFace Space.

Defines board sizes, komi values, coordinate labels, and rendering parameters.
All values are configurable via Pydantic schemas with validation.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BoardSize(IntEnum):
    """Supported board sizes."""

    SIZE_9 = 9
    SIZE_13 = 13
    SIZE_19 = 19


# Standard komi values per board size (AGA/Chinese rules)
# Smaller boards use less komi due to reduced first-move advantage
KOMI_BY_SIZE: dict[int, float] = {
    9: 5.5,  # Smaller board = less first-move advantage
    13: 6.5,  # Medium board
    19: 7.5,  # Standard komi for full-size board
}

# Star points (hoshi) per board size following traditional Go patterns
STAR_POINTS_BY_SIZE: dict[int, list[tuple[int, int]]] = {
    9: [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)],
    13: [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)],
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
}

# GTP column letters (A-T, skipping I to avoid confusion with 1)
GTP_LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


class CoordinateLabelConfig(BaseModel):
    """Configuration for board coordinate labels.

    Attributes:
        show_labels: Whether to show coordinate labels on the board perimeter.
        label_style: Style of column labels ('letters' for A-T, 'numbers' for 1-19).
        font_size: Font size for coordinate labels.
        font_color: Color for coordinate label text.
        padding: Padding around board for coordinate labels.
        skip_i: Whether to skip 'I' in column letters (Go convention).

    """

    model_config = ConfigDict(extra="forbid")

    show_labels: bool = Field(default=True, description="Show coordinate labels")
    label_style: Literal["letters", "numbers"] = Field(
        default="letters",
        description="Style for column labels (A-T or 1-19)",
    )
    font_size: int = Field(default=10, ge=6, le=20, description="Label font size")
    font_color: str = Field(default="black", description="Label text color")
    padding: float = Field(default=0.8, ge=0.3, le=1.5, description="Padding for labels")
    skip_i: bool = Field(
        default=True,
        description="Skip 'I' in column labels (Go convention)",
    )


class BoardRenderConfig(BaseModel):
    """Configuration for board rendering.

    Attributes:
        board_color: Background color for the board (wood-like).
        line_color: Color for grid lines.
        line_width: Width of grid lines in points.
        stone_radius: Radius of stones relative to intersection spacing.
        star_point_size: Size of star point (hoshi) markers.
        last_move_marker_color: Color for last move indicator.
        last_move_marker_size: Size of last move marker.
        figure_size: Size of the rendered figure in inches (width, height).
        coordinate_labels: Coordinate label configuration.

    """

    model_config = ConfigDict(extra="forbid")

    board_color: str = Field(default="#e3c586", description="Board background color")
    line_color: str = Field(default="black", description="Grid line color")
    line_width: float = Field(default=1.0, ge=0.5, le=3.0, description="Grid line width")
    stone_radius: float = Field(default=0.45, ge=0.3, le=0.5, description="Stone radius")
    star_point_size: int = Field(default=20, ge=10, le=50, description="Star point marker size")
    last_move_marker_color: str = Field(default="red", description="Last move marker color")
    last_move_marker_size: int = Field(
        default=20, ge=10, le=40, description="Last move marker size"
    )
    figure_size: tuple[float, float] = Field(
        default=(6.5, 6.0),
        description="Figure size (width, height) in inches",
    )
    coordinate_labels: CoordinateLabelConfig = Field(
        default_factory=CoordinateLabelConfig,
        description="Coordinate label settings",
    )


class EndgameConfig(BaseModel):
    """Configuration for endgame detection.

    Controls when the AI should automatically pass to end the game
    when the human player passes in an endgame situation.

    Attributes:
        fill_threshold: Board fill ratio above which AI should pass if human passed.
        enable_auto_pass: Whether to enable automatic AI pass in endgame.
        pass_on_consecutive: Always pass if human just passed and game.passes >= 1.

    """

    model_config = ConfigDict(extra="forbid")

    fill_threshold: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="Board fill ratio triggering auto-pass",
    )
    enable_auto_pass: bool = Field(
        default=True,
        description="Enable automatic AI pass in endgame",
    )
    pass_on_consecutive: bool = Field(
        default=True,
        description="Always pass if game already has a pass pending",
    )


class SpaceConfig(BaseModel):
    """Root configuration for HuggingFace Space.

    Attributes:
        default_board_size: Default board size for new games.
        training_board_size: Board size used during model training (for zero-shot detection).
        supported_sizes: List of supported board sizes.
        render: Board rendering configuration.
        mcts_simulations: Number of MCTS simulations for AI moves.
        show_zero_shot_info: Show zero-shot transfer information in UI.
        fallback_komi: Default komi value for unsupported board sizes.

    """

    model_config = ConfigDict(extra="forbid")

    default_board_size: int = Field(default=9, description="Default board size")
    training_board_size: int = Field(
        default=9,
        description="Board size used during model training",
    )
    supported_sizes: list[int] = Field(
        default_factory=lambda: [9, 13, 19],
        description="Supported board sizes",
    )
    render: BoardRenderConfig = Field(
        default_factory=BoardRenderConfig,
        description="Board rendering configuration",
    )
    mcts_simulations: int = Field(
        default=60,
        ge=10,
        le=500,
        description="MCTS simulations per move",
    )
    show_zero_shot_info: bool = Field(
        default=True,
        description="Show zero-shot transfer information in UI",
    )
    fallback_komi: float = Field(
        default=7.5,
        ge=0.0,
        le=15.0,
        description="Default komi for unsupported board sizes",
    )
    endgame: EndgameConfig = Field(
        default_factory=EndgameConfig,
        description="Endgame detection configuration",
    )

    @field_validator("supported_sizes")
    @classmethod
    def validate_sizes(cls, v: list[int]) -> list[int]:
        """Validate and normalize supported sizes."""
        if not v:
            raise ValueError("At least one board size must be supported")
        valid_sizes = [size.value for size in BoardSize]
        for size in v:
            if size not in valid_sizes:
                raise ValueError(f"Unsupported board size: {size}. Valid sizes: {valid_sizes}")
        return sorted(set(v))

    @model_validator(mode="after")
    def validate_default_in_supported(self) -> SpaceConfig:
        """Ensure default size is in supported sizes."""
        if self.default_board_size not in self.supported_sizes:
            raise ValueError(
                f"default_board_size ({self.default_board_size}) "
                f"not in supported_sizes ({self.supported_sizes})"
            )
        return self

    def get_komi(self, board_size: int) -> float:
        """Get komi value for a board size.

        Args:
            board_size: Board dimension.

        Returns:
            Komi value for the board size.

        """
        return KOMI_BY_SIZE.get(board_size, self.fallback_komi)

    def is_zero_shot_size(self, board_size: int) -> bool:
        """Check if a board size is a zero-shot transfer size.

        A board size is considered zero-shot if it differs from the training size.

        Args:
            board_size: Board dimension.

        Returns:
            True if the size requires zero-shot transfer.

        """
        return board_size != self.training_board_size

    def get_star_points(self, board_size: int) -> list[tuple[int, int]]:
        """Get star point coordinates for a board size.

        Args:
            board_size: Board dimension.

        Returns:
            List of (row, col) star point coordinates.

        """
        return STAR_POINTS_BY_SIZE.get(board_size, [])


def get_default_space_config() -> SpaceConfig:
    """Get default space configuration.

    Returns:
        SpaceConfig with default values.

    """
    return SpaceConfig()


def get_column_letter(col: int, skip_i: bool = True) -> str:
    """Get column letter for a column index.

    Args:
        col: Column index (0-based).
        skip_i: Whether to skip 'I' (Go convention).

    Returns:
        Column letter (A-T, skipping I if requested).

    """
    if skip_i:
        return GTP_LETTERS[col]
    return chr(ord("A") + col)
