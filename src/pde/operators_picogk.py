"""SDF-aware PDE operators for Leap 71 / Noyron geometries.

The classical operators in :mod:`src.pde.operators` work on rectangular
domains by default. For complex Leap 71 parts (helical heat exchangers,
lattices, rocket nozzles) the domain is given by a signed distance field
and we must override the collocation/boundary samplers to delegate to the
SDF-backed geometry.

This module follows the exact pattern established by
``LShapedPoissonOperator`` (``src/pde/operators.py:1262``): hold a
``DomainGeometry`` reference and override ``generate_collocation_points``
and ``generate_boundary_points`` to call ``geometry.sample_interior`` /
``geometry.sample_boundary``. Everything else (residual, autodiff Laplacian,
boundary value, source term) is inherited unchanged.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import PDEConfig
from src.pde.geometry import DomainGeometry, GeometryType, create_geometry
from src.pde.operators import HeatOperator

logger = structlog.get_logger(__name__)


class HelicalHeatOperator(HeatOperator):
    """Steady-state heat equation on an SDF-bounded helical tube.

    Geometry is supplied via the ``GeometryConfig.geometry_type=PICOGK``
    branch (typically ``sdf_kind='analytical_helix'`` for CI and
    ``'picogk'`` for production runs against a real Leap 71 STL).

    Boundary conditions:

    - **inner_dirichlet** (default): a single Dirichlet temperature applied
      uniformly to the tube surface — useful for an analytical harmonic
      reference test.
    - **hot_cold**: temperature ``hot_value`` for points whose centerline
      parameter is in the lower half of the helix, ``cold_value`` for the
      upper half. Models a co/counter-flow heat exchanger end-to-end
      gradient.

    The source term is zero by default (matching ``HeatOperator``); a
    user-supplied ``source_function`` is honored if provided.
    """

    name = "helical_heat"
    description = "Heat equation on a helical SDF domain (Leap 71 Noyron HX)."
    is_time_dependent = False  # Steady state for the v1 PoC.

    def __init__(
        self,
        config: PDEConfig,
        diffusivity: float | None = None,
        boundary_mode: Literal["inner_dirichlet", "hot_cold"] = "inner_dirichlet",
        hot_value: float = 1.0,
        cold_value: float = 0.0,
    ) -> None:
        if config.geometry.geometry_type != GeometryType.PICOGK:
            raise ValueError(
                "HelicalHeatOperator requires geometry_type=PICOGK; "
                f"got {config.geometry.geometry_type}"
            )
        if config.domain_dim != 3:
            raise ValueError(
                f"HelicalHeatOperator requires domain_dim=3; "
                f"got {config.domain_dim}"
            )
        super().__init__(config, diffusivity=diffusivity)

        self.geometry: DomainGeometry = create_geometry(config.geometry)
        self.boundary_mode = boundary_mode
        self.hot_value = float(hot_value)
        self.cold_value = float(cold_value)

        logger.info(
            "helical_heat_operator_created",
            sdf_kind=config.geometry.sdf_kind,
            diffusivity=self.diffusivity,
            boundary_mode=boundary_mode,
        )

    # ------------------------------------------------------------------
    # Geometry-aware sampling (the LShapedPoissonOperator pattern)
    # ------------------------------------------------------------------

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = 1e-5,
    ) -> NDArray[np.bool_] | Tensor:
        """Determine which points lie on the helical tube surface."""
        if isinstance(coords, Tensor):
            return self.geometry.is_boundary(coords, tol=tolerance)
        coords_t = torch.from_numpy(coords)
        return self.geometry.is_boundary(coords_t, tol=tolerance).numpy()

    def generate_collocation_points(
        self,
        n_points: int,
        method: str = "random",
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Sample interior points from the SDF-bounded helical domain."""
        if seed is not None:
            torch.manual_seed(seed)
        points = self.geometry.sample_interior(n_points)
        return points.numpy().astype(np.float32)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Sample boundary points on the helical tube surface.

        The helix has a single closed surface; ``n_points_per_face`` is
        treated as a target total count for compatibility with the base
        class API used by ``BasisSelectionGame``.
        """
        if seed is not None:
            torch.manual_seed(seed)
        total = max(n_points_per_face, 1)
        points = self.geometry.sample_boundary(total)
        return points.numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Boundary values: hot inner / cold outer Dirichlet
    # ------------------------------------------------------------------

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Dirichlet boundary temperature.

        The ``hot_cold`` mode assigns ``hot_value`` to boundary points whose
        z-coordinate is below the midpoint of the helical extent and
        ``cold_value`` above. ``inner_dirichlet`` returns a constant
        ``config.boundary_value`` everywhere on the surface.
        """
        if self.boundary_mode == "inner_dirichlet":
            return super().boundary_value(coords, time=time)

        # hot_cold mode: split along z midplane.
        (mins, maxs) = self.geometry.bounding_box()
        z_min = float(mins[2])
        z_max = float(maxs[2])
        z_mid = 0.5 * (z_min + z_max)

        if isinstance(coords, Tensor):
            z = coords[:, 2]
            hot_mask = z <= z_mid
            return torch.where(
                hot_mask,
                torch.full_like(z, self.hot_value),
                torch.full_like(z, self.cold_value),
            )
        z_np = coords[:, 2]
        hot_mask_np = z_np <= z_mid
        return np.where(
            hot_mask_np,
            np.full(z_np.shape, self.hot_value, dtype=np.float32),
            np.full(z_np.shape, self.cold_value, dtype=np.float32),
        ).astype(np.float32)
