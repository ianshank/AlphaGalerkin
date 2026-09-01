"""Poisson equation on an L-shaped domain (reentrant-corner singularity benchmark)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.config import PDEConfig, PDEType
from src.pde.geometry import (
    DomainGeometry,
    GeometryType,
    LShapedDomain,
    create_geometry,
)
from src.pde.operators.base import PDEOperator, PDEResidual

logger = structlog.get_logger(__name__)


class LShapedPoissonOperator(PDEOperator):
    r"""Poisson equation on L-shaped domain.

    Solves -Delta u = f on the L-shaped domain [-1,1]^2 \\ [0,1]x[-1,0]
    with Dirichlet boundary conditions.

    Known singular solution near reentrant corner at the origin:
        u(r, theta) = r^(2/3) * sin(2*theta/3)

    where (r, theta) are polar coordinates centred at the origin.
    This is the standard benchmark for adaptive mesh refinement
    because the solution gradient is singular at the reentrant corner,
    requiring concentrated mesh refinement.

    The source term for the singular solution is f = 0 (harmonic).
    """

    name = "poisson_lshaped"
    description = "Poisson equation on L-shaped domain with corner singularity"
    pde_type = PDEType.POISSON
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        source_function: Callable[[NDArray | Tensor], NDArray | Tensor] | None = None,
    ) -> None:
        """Initialize L-shaped Poisson operator.

        Args:
            config: PDE configuration. The ``geometry`` field should have
                ``geometry_type=GeometryType.L_SHAPED``.
            source_function: Custom source term. Defaults to f=0 (for the
                singular benchmark solution).

        """
        super().__init__(config)
        self.diffusion = config.diffusion_coeff
        self._source_function = source_function

        # Build geometry from config or default to L-shaped
        if config.geometry.geometry_type == GeometryType.L_SHAPED:
            self.geometry: DomainGeometry = create_geometry(config.geometry)
        else:
            self.geometry = LShapedDomain(scale=config.geometry.scale)

        self._scale = (
            config.geometry.scale if config.geometry.geometry_type == GeometryType.L_SHAPED else 1.0
        )

        logger.info(
            "lshaped_poisson_operator_created",
            scale=self._scale,
            diffusion=self.diffusion,
        )

    # ------------------------------------------------------------------
    # Exact (singular) solution utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _polar_from_cartesian(
        x: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert Cartesian (x, y) to polar (r, theta).

        The angle theta is measured from the positive x-axis in the range
        [0, 2*pi) so that the reentrant-corner singular solution is
        well-defined on the full L-shaped domain.
        """
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        # Map to [0, 2*pi)
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        return r, theta

    @staticmethod
    def _singular_solution(r: Tensor, theta: Tensor) -> Tensor:
        """Evaluate the benchmark singular solution.

        u(r, theta) = r^(2/3) * sin(2*theta/3)
        """
        return r.pow(2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)

    @staticmethod
    def _singular_solution_np(
        x: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Numpy version of the singular solution."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        # Avoid 0^(2/3) producing nan
        result = np.where(
            r > 0,
            np.power(r, 2.0 / 3.0) * np.sin(2.0 * theta / 3.0),
            0.0,
        )
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # PDEOperator interface
    # ------------------------------------------------------------------

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute exact singular solution u = r^(2/3) sin(2*theta/3).

        Args:
            coords: Point coordinates (N, 2).
            time: Unused (steady-state problem).

        Returns:
            Solution values (N,).

        """
        if isinstance(coords, Tensor):
            x, y = coords[:, 0], coords[:, 1]
            r, theta = self._polar_from_cartesian(x, y)
            return self._singular_solution(r, theta)
        else:
            return self._singular_solution_np(coords[:, 0], coords[:, 1])

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute source/forcing term.

        The default singular solution is harmonic (f = 0). A custom
        source function can be provided via the constructor.

        Args:
            coords: Point coordinates (N, 2).
            time: Unused.

        Returns:
            Source term values (N,).

        """
        if self._source_function is not None:
            return self._source_function(coords)

        # Default: f = 0 (the singular benchmark is harmonic)
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute Dirichlet boundary values from the exact solution.

        Args:
            coords: Boundary point coordinates (N_b, 2).
            time: Unused.

        Returns:
            Boundary values (N_b,).

        """
        return self.exact_solution(coords, time)

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute Poisson residual: R = -nu * Laplacian(u) - f.

        Uses automatic differentiation to compute the Laplacian.

        Args:
            u: Solution values at collocation points (N,) or (N, 1).
            coords: Collocation point coordinates (N, 2).
            compute_derivatives: Whether to return derivative tensors.

        Returns:
            PDEResidual with values and norms.

        """
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u.squeeze()))
        source = self.source_term(coords)

        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        # Residual: -nu * Laplacian(u) - f = 0
        residual_values = -self.diffusion * laplacian - source

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> NDArray[np.bool_] | Tensor:
        """Determine which points are on the L-shaped boundary.

        Overrides the rectangular base implementation with geometry-aware
        boundary detection.

        Args:
            coords: Point coordinates (N, dim).
            tolerance: Distance tolerance for boundary detection.

        Returns:
            Boolean mask (N,) with True for boundary points.

        """
        if isinstance(coords, Tensor):
            return self.geometry.is_boundary(coords, tol=tolerance)
        else:
            coords_t = torch.from_numpy(coords)
            result = self.geometry.is_boundary(coords_t, tol=tolerance)
            return result.numpy()

    def generate_collocation_points(
        self,
        n_points: int,
        method: str = "random",
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate collocation points inside the L-shaped domain.

        Uses the geometry's interior sampling which handles rejection
        sampling automatically.

        Args:
            n_points: Number of interior points to generate.
            method: Sampling method (only 'random' supported for L-shaped).
            seed: Random seed for reproducibility.

        Returns:
            Collocation points (n_points, 2).

        """
        if seed is not None:
            torch.manual_seed(seed)
        points = self.geometry.sample_interior(n_points)
        return points.numpy().astype(np.float32)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate points on the L-shaped domain boundary.

        The L-shape has 6 boundary segments, so total points will be
        approximately 6 * n_points_per_face (distributed proportionally
        to segment length).

        Args:
            n_points_per_face: Approximate points per boundary segment.
            seed: Random seed.

        Returns:
            Boundary points (N_boundary, 2).

        """
        if seed is not None:
            torch.manual_seed(seed)
        # 6 segments total; distribute proportionally
        total = n_points_per_face * 6
        points = self.geometry.sample_boundary(total)
        return points.numpy().astype(np.float32)

    def compute_error(
        self,
        u_pred: Tensor,
        coords: Tensor,
    ) -> dict[str, float]:
        """Compute error metrics against the exact singular solution.

        Args:
            u_pred: Predicted solution values (N,).
            coords: Point coordinates (N, 2).

        Returns:
            Dictionary with 'l2_error', 'linf_error', and 'mse'.

        """
        u_exact = self.exact_solution(coords)
        assert isinstance(u_exact, Tensor)
        diff = u_pred - u_exact

        l2 = float(torch.sqrt(torch.mean(diff**2)).item())
        linf = float(torch.max(torch.abs(diff)).item())
        mse = float(torch.mean(diff**2).item())

        logger.debug(
            "error_computed",
            l2_error=l2,
            linf_error=linf,
            mse=mse,
        )

        return {"l2_error": l2, "linf_error": linf, "mse": mse}
