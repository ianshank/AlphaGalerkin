"""2D block-structured mesh for compressible flow simulations.

Provides a structured quadrilateral mesh with:
- Ghost cell layers for boundary condition implementation
- Geometric wall clustering for boundary layer resolution
- Cell volumes, face areas, and face normals
- AMR refinement hooks for future integration
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog
from numpy.typing import NDArray

from src.reentry.config.mesh import ReentryMeshConfig

logger = structlog.get_logger(__name__)


@dataclass
class MeshMetrics:
    """Precomputed geometric quantities for the mesh.

    All arrays have shape (ny, nx) for cell-centered quantities
    or (ny+1, nx+1) for node-centered quantities.
    """

    x_cell: NDArray[np.float64]  # Cell center x-coordinates (ny, nx)
    y_cell: NDArray[np.float64]  # Cell center y-coordinates (ny, nx)
    x_node: NDArray[np.float64]  # Node x-coordinates (ny+1, nx+1)
    y_node: NDArray[np.float64]  # Node y-coordinates (ny+1, nx+1)
    dx: NDArray[np.float64]  # Cell widths (ny, nx)
    dy: NDArray[np.float64]  # Cell heights (ny, nx)
    volume: NDArray[np.float64]  # Cell volumes (ny, nx)


class StructuredMesh2D:
    """2D structured mesh with ghost cells and wall clustering.

    The mesh covers the physical domain [x_min, x_max] x [y_min, y_max]
    with nx x ny interior cells plus n_ghost ghost cell layers on each side.

    Array layout (with 1 ghost cell):
        Total shape: (ny + 2*n_ghost, nx + 2*n_ghost)
        Interior cells: [n_ghost:-n_ghost, n_ghost:-n_ghost]
        Ghost cells: padding around the interior

    Wall clustering uses geometric stretching to concentrate cells
    near the y_min boundary (vehicle surface) for boundary layer resolution.
    """

    def __init__(self, config: ReentryMeshConfig, n_ghost: int = 2) -> None:
        self.config = config
        self.nx = config.nx
        self.ny = config.ny
        self.n_ghost = n_ghost
        self.total_nx = config.nx + 2 * n_ghost
        self.total_ny = config.ny + 2 * n_ghost

        self._metrics = self._build_mesh()

        logger.info(
            "structured_mesh_created",
            nx=self.nx,
            ny=self.ny,
            n_ghost=n_ghost,
            total_cells=self.nx * self.ny,
            min_dy=float(self._metrics.dy.min()),
        )

    @property
    def metrics(self) -> MeshMetrics:
        return self._metrics

    @property
    def interior_slice(self) -> tuple[slice, slice]:
        """Slice for accessing interior cells (excludes ghost cells)."""
        g = self.n_ghost
        return (slice(g, -g), slice(g, -g))

    def _build_mesh(self) -> MeshMetrics:
        """Build mesh coordinates with optional wall clustering."""
        config = self.config

        # X-direction: uniform spacing
        x_nodes = np.linspace(config.x_min, config.x_max, config.nx + 1)

        # Y-direction: wall clustering or uniform
        if config.wall_clustering:
            y_nodes = self._wall_clustered_nodes(
                config.y_min,
                config.y_max,
                config.ny,
                config.wall_first_cell_height,
                config.wall_growth_rate,
            )
        else:
            y_nodes = np.linspace(config.y_min, config.y_max, config.ny + 1)

        # Cell centers
        x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])

        # 2D grids for cell centers
        xx_cell, yy_cell = np.meshgrid(x_cell, y_cell)

        # Node grid
        xx_node, yy_node = np.meshgrid(x_nodes, y_nodes)

        # Cell sizes
        dx_1d = x_nodes[1:] - x_nodes[:-1]
        dy_1d = y_nodes[1:] - y_nodes[:-1]
        dx_2d, dy_2d = np.meshgrid(dx_1d, dy_1d)

        # Cell volumes (areas in 2D)
        volume = dx_2d * dy_2d

        return MeshMetrics(
            x_cell=xx_cell,
            y_cell=yy_cell,
            x_node=xx_node,
            y_node=yy_node,
            dx=dx_2d,
            dy=dy_2d,
            volume=volume,
        )

    @staticmethod
    def _wall_clustered_nodes(
        y_min: float,
        y_max: float,
        ny: int,
        first_cell_height: float,
        growth_rate: float,
    ) -> NDArray[np.float64]:
        """Generate wall-clustered node distribution using geometric stretching.

        The first cell at y_min has height `first_cell_height`, and
        subsequent cells grow by `growth_rate` until the domain is filled.

        Args:
            y_min: Wall location.
            y_max: Far-field boundary.
            ny: Number of cells.
            first_cell_height: Height of first cell at wall.
            growth_rate: Geometric growth ratio.

        Returns:
            Array of ny+1 node y-coordinates.

        """
        # Generate geometric series of cell heights
        heights = np.zeros(ny, dtype=np.float64)
        heights[0] = first_cell_height
        for i in range(1, ny):
            heights[i] = heights[i - 1] * growth_rate

        # Scale to fit domain
        total = heights.sum()
        domain_height = y_max - y_min
        heights *= domain_height / total

        # Accumulate to get node positions
        nodes = np.zeros(ny + 1, dtype=np.float64)
        nodes[0] = y_min
        for i in range(ny):
            nodes[i + 1] = nodes[i] + heights[i]
        nodes[-1] = y_max  # Exact far-field boundary

        return nodes

    def allocate_field(self, n_vars: int = 1) -> NDArray[np.float64]:
        """Allocate a cell-centered field array including ghost cells.

        Args:
            n_vars: Number of variables (1 for scalar, >1 for vector).

        Returns:
            Zero-initialized array of shape (total_ny, total_nx) or
            (total_ny, total_nx, n_vars).

        """
        if n_vars == 1:
            return np.zeros((self.total_ny, self.total_nx), dtype=np.float64)
        return np.zeros((self.total_ny, self.total_nx, n_vars), dtype=np.float64)

    def cell_centers(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return cell center coordinates for interior cells."""
        return self._metrics.x_cell, self._metrics.y_cell

    def min_cell_size(self) -> float:
        """Minimum cell dimension (for CFL computation)."""
        return float(min(self._metrics.dx.min(), self._metrics.dy.min()))
