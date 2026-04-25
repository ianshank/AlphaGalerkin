"""Configuration schemas for PoC visualization.

Provides Pydantic-validated configuration for plot generation and
HTML report output, ensuring all rendering parameters are explicit
and constrained.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from src.templates.config import BaseModuleConfig


class VisualizationConfig(BaseModuleConfig):
    """Configuration for visualization and report generation.

    Attributes:
        output_format: Output format for generated artifacts.
        plot_width: Width of plots in pixels.
        plot_height: Height of plots in pixels.
        theme: Visual theme for plots.
        include_interactive: Whether to include interactive elements in HTML.
        dpi: Dots per inch for rasterized output.

    """

    output_format: Literal["html", "png", "both"] = Field(
        default="html",
        description="Output format for generated artifacts",
    )
    plot_width: int = Field(
        default=800,
        ge=200,
        le=4000,
        description="Width of plots in pixels",
    )
    plot_height: int = Field(
        default=500,
        ge=200,
        le=3000,
        description="Height of plots in pixels",
    )
    theme: Literal["light", "dark", "publication"] = Field(
        default="light",
        description="Visual theme for plots",
    )
    include_interactive: bool = Field(
        default=True,
        description="Whether to include interactive elements in HTML reports",
    )
    dpi: int = Field(
        default=150,
        ge=72,
        le=600,
        description="Dots per inch for rasterized output",
    )

    @property
    def figsize(self) -> tuple[float, float]:
        """Compute matplotlib figsize from pixel dimensions and DPI.

        Returns:
            Tuple of (width_inches, height_inches).

        """
        return (self.plot_width / self.dpi, self.plot_height / self.dpi)
