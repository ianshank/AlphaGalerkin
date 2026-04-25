"""Mesh configuration for reentry simulations.

Supports structured grids with block-structured AMR for
shock layer and boundary layer resolution.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from src.templates.config import BaseModuleConfig


class MeshType(str, Enum):
    """Supported mesh types."""

    STRUCTURED = "structured"
    BLOCK_STRUCTURED = "block_structured"


class ReentryMeshConfig(BaseModuleConfig):
    """Configuration for the computational mesh."""

    mesh_type: MeshType = Field(
        default=MeshType.STRUCTURED,
        description="Type of computational mesh.",
    )
    nx: int = Field(
        default=100,
        ge=4,
        description="Number of cells in x (streamwise) direction.",
    )
    ny: int = Field(
        default=50,
        ge=4,
        description="Number of cells in y (wall-normal) direction.",
    )
    # Domain bounds
    x_min: float = Field(default=-1.0, description="Domain x minimum.")
    x_max: float = Field(default=2.0, description="Domain x maximum.")
    y_min: float = Field(default=0.0, description="Domain y minimum.")
    y_max: float = Field(default=2.0, description="Domain y maximum.")

    # AMR settings
    enable_amr: bool = Field(
        default=False,
        description="Enable adaptive mesh refinement.",
    )
    max_amr_levels: int = Field(
        default=3,
        ge=0,
        le=8,
        description="Maximum number of AMR refinement levels.",
    )
    amr_refinement_ratio: int = Field(
        default=2,
        ge=2,
        le=4,
        description="Refinement ratio between AMR levels.",
    )
    amr_error_threshold: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Error indicator threshold for refinement.",
    )
    amr_coarsen_threshold: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Error indicator threshold for coarsening.",
    )
    # Wall clustering
    wall_clustering: bool = Field(
        default=True,
        description="Enable geometric clustering near walls for BL resolution.",
    )
    wall_first_cell_height: float = Field(
        default=1e-5,
        gt=0.0,
        description="First cell height at wall in meters.",
    )
    wall_growth_rate: float = Field(
        default=1.2,
        gt=1.0,
        le=3.0,
        description="Cell growth rate away from wall.",
    )

    @model_validator(mode="after")
    def _validate_domain(self) -> ReentryMeshConfig:
        if self.x_max <= self.x_min:
            msg = "x_max must be greater than x_min"
            raise ValueError(msg)
        if self.y_max <= self.y_min:
            msg = "y_max must be greater than y_min"
            raise ValueError(msg)
        if self.amr_coarsen_threshold >= self.amr_error_threshold:
            msg = "amr_coarsen_threshold must be less than amr_error_threshold"
            raise ValueError(msg)
        return self
