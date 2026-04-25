"""SDF-aware PDE operators for Leap 71 / Noyron geometries.

The classical operators in :mod:`src.pde.operators` work on rectangular
domains by default. For complex Leap 71 parts (helical heat exchangers,
lattices, rocket nozzles, electromagnetic actuators) the domain is given
by a signed distance field and we must override the collocation/boundary
samplers to delegate to the SDF-backed geometry.

This module follows the exact pattern established by
``LShapedPoissonOperator`` (``src/pde/operators.py:1262``): hold a
``DomainGeometry`` reference and override ``generate_collocation_points``
and ``generate_boundary_points`` to call ``geometry.sample_interior`` /
``geometry.sample_boundary``. Everything else (residual, autodiff Laplacian,
boundary value, source term) is inherited unchanged.

The three concrete operators provided are:

- ``HelicalHeatOperator`` — steady heat equation for Noyron HX (the v1
  headline scenario).
- ``HelicalStokesOperator`` — steady incompressible Stokes flow for
  Noyron RP coolant channels (v2.3 expansion). Linear, low-Reynolds; the
  natural starting point before adding convective nonlinearity.
- ``HelicalMagnetostaticsOperator`` — vector-potential magnetostatics on a
  helical core for Noyron EA actuators (v3.1 expansion).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import DomainGeometry, GeometryType, create_geometry
from src.pde.operators import HeatOperator, PDEOperator, PDEResidual

logger = structlog.get_logger(__name__)


def _require_picogk_3d(config: PDEConfig, operator_name: str) -> None:
    """Validate that ``config`` describes a 3D PicoGK-backed domain.

    Centralises the geometry/dim assertions shared by every SDF-aware
    helical operator so individual operator constructors stay tiny.
    """
    if config.geometry.geometry_type != GeometryType.PICOGK:
        raise ValueError(
            f"{operator_name} requires geometry_type=PICOGK; "
            f"got {config.geometry.geometry_type}"
        )
    if config.domain_dim != 3:
        raise ValueError(
            f"{operator_name} requires domain_dim=3; got {config.domain_dim}"
        )


def _sample_with_geometry_interior(
    geometry: DomainGeometry, n_points: int, seed: int | None
) -> NDArray[np.float32]:
    """Shared interior-sampling helper used by every SDF-aware operator."""
    if seed is not None:
        torch.manual_seed(seed)
    points = geometry.sample_interior(n_points)
    return points.numpy().astype(np.float32)


def _sample_with_geometry_boundary(
    geometry: DomainGeometry, n_points: int, seed: int | None
) -> NDArray[np.float32]:
    """Shared boundary-sampling helper used by every SDF-aware operator."""
    if seed is not None:
        torch.manual_seed(seed)
    points = geometry.sample_boundary(max(n_points, 1))
    return points.numpy().astype(np.float32)


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
        _require_picogk_3d(config, "HelicalHeatOperator")
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
        return _sample_with_geometry_interior(self.geometry, n_points, seed)

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
        return _sample_with_geometry_boundary(
            self.geometry, n_points_per_face, seed
        )

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


# =============================================================================
# Noyron RP — Stokes flow on a helical channel (v2.3 expansion)
# =============================================================================


class HelicalStokesOperator(PDEOperator):
    """Steady incompressible Stokes flow on an SDF-bounded helical channel.

    Stokes flow is the linear, low-Reynolds limit of Navier-Stokes::

        - mu * Laplacian(u) + grad(p) = 0    (momentum, no convection)
        nabla . u                     = 0    (continuity / incompressibility)

    where ``u`` is the 3D velocity field and ``p`` is the pressure. We
    omit the convective term ``(u . nabla) u`` deliberately: rocket-engine
    coolant channels run at low to moderate Reynolds numbers where Stokes
    is a faithful first-order model, and a linear operator is a much
    cleaner training target for the AlphaGalerkin surrogate than the
    fully nonlinear Navier-Stokes residual.

    The residual returned is the per-component momentum imbalance for
    ``u_x``; callers using vector outputs can extend ``residual`` to
    return all three components if needed.
    """

    name = "helical_stokes"
    description = "Steady 3D Stokes flow on a Leap 71 helical channel SDF."
    pde_type = PDEType.NAVIER_STOKES
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        viscosity: float | None = None,
    ) -> None:
        _require_picogk_3d(config, "HelicalStokesOperator")
        super().__init__(config)
        # Reuse ``diffusion_coeff`` as the kinematic viscosity to keep
        # the existing PDEConfig schema unchanged.
        self.viscosity = (
            viscosity if viscosity is not None else config.diffusion_coeff
        )
        if self.viscosity <= 0:
            raise ValueError(
                f"viscosity must be > 0, got {self.viscosity}"
            )
        self.geometry: DomainGeometry = create_geometry(config.geometry)
        logger.info(
            "helical_stokes_operator_created",
            viscosity=self.viscosity,
            sdf_kind=config.geometry.sdf_kind,
        )

    # ------------------------------------------------------------------
    # Geometry-aware sampling
    # ------------------------------------------------------------------

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = 1e-5,
    ) -> NDArray[np.bool_] | Tensor:
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
        return _sample_with_geometry_interior(self.geometry, n_points, seed)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        return _sample_with_geometry_boundary(
            self.geometry, n_points_per_face, seed
        )

    # ------------------------------------------------------------------
    # Stokes physics
    # ------------------------------------------------------------------

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Stokes momentum residual ``-mu * Laplacian(u_x)`` (no convection).

        ``u`` is expected to be a single velocity component (the helical
        axial velocity). For multi-component runs, call this once per
        component and sum the residuals upstream.
        """
        derivatives = self.compute_derivatives(u, coords)
        laplacian = derivatives.get("laplacian", torch.zeros_like(u))
        residual_values = -self.viscosity * laplacian

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())
        # PDEResidual.derivatives accepts the union ``NDArray | Tensor``;
        # cast the all-Tensor dict so mypy doesn't flag the invariant
        # value-type mismatch.
        derivatives_typed: dict[str, Tensor | NDArray[np.float32]] = dict(
            derivatives if compute_derivatives else {}
        )
        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives_typed,
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Stokes flow has no body force in the v1 PoC."""
        if isinstance(coords, Tensor):
            return torch.zeros(
                coords.shape[0], dtype=coords.dtype, device=coords.device
            )
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """No-slip Dirichlet condition on the channel wall (u = 0)."""
        if isinstance(coords, Tensor):
            return torch.zeros(
                coords.shape[0], dtype=coords.dtype, device=coords.device
            )
        return np.zeros(coords.shape[0], dtype=np.float32)


# =============================================================================
# Noyron EA — magnetostatics on a helical actuator core (v3.1 expansion)
# =============================================================================


class HelicalMagnetostaticsOperator(PDEOperator):
    """Vector-potential magnetostatics on a helical SDF core.

    Solves the gauge-fixed magnetostatic equation::

        - (1/mu) * Laplacian(A) = J

    for the magnetic vector potential ``A`` given a coil current density
    ``J``. This is the magnetostatic analog of the Poisson equation
    component-wise; we model only the component along the helix axis
    (``A_z``) since that's what carries the dominant flux for a
    solenoidal helix and keeps the network output scalar-valued.

    ``mu`` is the magnetic permeability (taken constant inside the helix
    region — Leap 71 EA actuators use roughly soft-iron cores).
    """

    name = "helical_magnetostatics"
    description = (
        "Vector-potential magnetostatics on a Leap 71 helical actuator SDF."
    )
    # Reuse the Poisson PDEType — magnetostatics is a Poisson-type
    # equation per component.
    pde_type = PDEType.POISSON
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        permeability: float | None = None,
        current_density: float = 1.0,
    ) -> None:
        _require_picogk_3d(config, "HelicalMagnetostaticsOperator")
        super().__init__(config)
        self.permeability = (
            permeability if permeability is not None else config.diffusion_coeff
        )
        if self.permeability <= 0:
            raise ValueError(
                f"permeability must be > 0, got {self.permeability}"
            )
        self.current_density = float(current_density)
        self.geometry: DomainGeometry = create_geometry(config.geometry)
        logger.info(
            "helical_magnetostatics_operator_created",
            permeability=self.permeability,
            current_density=self.current_density,
            sdf_kind=config.geometry.sdf_kind,
        )

    # ------------------------------------------------------------------
    # Geometry-aware sampling
    # ------------------------------------------------------------------

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = 1e-5,
    ) -> NDArray[np.bool_] | Tensor:
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
        return _sample_with_geometry_interior(self.geometry, n_points, seed)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        return _sample_with_geometry_boundary(
            self.geometry, n_points_per_face, seed
        )

    # ------------------------------------------------------------------
    # Magnetostatic physics (axial component of vector potential)
    # ------------------------------------------------------------------

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Residual ``-1/mu * Laplacian(A_z) - J``."""
        derivatives = self.compute_derivatives(u, coords)
        laplacian = derivatives.get("laplacian", torch.zeros_like(u))
        source = self.source_term(coords)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)
        residual_values = -(1.0 / self.permeability) * laplacian - source

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())
        # PDEResidual.derivatives accepts the union ``NDArray | Tensor``;
        # cast the all-Tensor dict so mypy doesn't flag the invariant
        # value-type mismatch.
        derivatives_typed: dict[str, Tensor | NDArray[np.float32]] = dict(
            derivatives if compute_derivatives else {}
        )
        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives_typed,
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Constant axial current density in the core."""
        if isinstance(coords, Tensor):
            return torch.full(
                (coords.shape[0],),
                self.current_density,
                dtype=coords.dtype,
                device=coords.device,
            )
        return np.full(
            (coords.shape[0],), self.current_density, dtype=np.float32
        )

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Vector potential vanishes far from the core (Dirichlet A = 0)."""
        if isinstance(coords, Tensor):
            return torch.zeros(
                coords.shape[0], dtype=coords.dtype, device=coords.device
            )
        return np.zeros(coords.shape[0], dtype=np.float32)
