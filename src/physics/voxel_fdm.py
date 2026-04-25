"""Tiny 3D steady-state FDM heat solver for SDF-bounded domains.

Reference solver used by the Noyron HX scenario when the user requests
``ref_solver_kind='voxel_fdm'``. Discretizes the steady heat equation::

    -kappa * Laplacian(u) = f      inside the domain
    u                  = g(x)      on the boundary

on a uniform Cartesian voxel grid restricted to a signed-distance domain.
Interior voxels (``sdf <= 0``) are unknowns; voxels outside the domain are
masked, and any voxel adjacent to a masked voxel inherits the surface
Dirichlet value.

This is intentionally minimal — Jacobi/Gauss-Seidel iteration in NumPy on
a 64^3 grid converges in ~1k iterations and runs in seconds, which is
plenty for the v1 PoC headline number. It is **not** a CFD-grade solver.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import structlog
from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


def voxelize_sdf(
    sdf_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    resolution: int,
) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Sample an SDF on a uniform voxel grid.

    Args:
        sdf_fn: Callable mapping ``(N, 3)`` numpy points to ``(N,)``
            signed distances. Negative inside, positive outside.
        bbox_min: ``(x_min, y_min, z_min)``.
        bbox_max: ``(x_max, y_max, z_max)``.
        resolution: Number of voxels per axis (cubic grid).

    Returns:
        ``(interior_mask, voxel_coords)`` where ``interior_mask`` has
        shape ``(R, R, R)`` (True for interior voxels) and
        ``voxel_coords`` has shape ``(R, R, R, 3)`` with the world-space
        coordinate of each voxel center.

    """
    if resolution < 4:
        raise ValueError(f"resolution must be >= 4, got {resolution}")

    xs = np.linspace(bbox_min[0], bbox_max[0], resolution, dtype=np.float32)
    ys = np.linspace(bbox_min[1], bbox_max[1], resolution, dtype=np.float32)
    zs = np.linspace(bbox_min[2], bbox_max[2], resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)

    flat = grid.reshape(-1, 3)
    sdf_values = np.asarray(sdf_fn(flat), dtype=np.float32).reshape(
        resolution, resolution, resolution
    )
    interior = sdf_values <= 0.0
    return interior, grid


def solve_steady_heat_voxel(
    interior_mask: NDArray[np.bool_],
    voxel_coords: NDArray[np.float32],
    boundary_value_fn: Callable[
        [NDArray[np.float32]], NDArray[np.float32]
    ],
    source_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
    diffusivity: float = 1.0,
    n_iterations: int = 1500,
    tolerance: float = 1e-5,
    log_every_n_iters: int = 100,
) -> NDArray[np.float32]:
    """Solve the steady heat equation on a voxelized SDF domain.

    Uses Jacobi iteration with masked Laplacian: each interior voxel is
    updated from its six face-neighbors. Voxels neighboring an exterior
    voxel use the boundary Dirichlet value at the world-space midpoint
    between the interior cell and the exterior cell.

    Args:
        interior_mask: ``(R, R, R)`` boolean mask of interior voxels.
        voxel_coords: ``(R, R, R, 3)`` world-space voxel centers.
        boundary_value_fn: Callable mapping ``(N, 3)`` points to
            ``(N,)`` Dirichlet values.
        source_fn: Optional callable for the heat-equation source term;
            defaults to zero.
        diffusivity: Thermal diffusivity ``kappa``.
        n_iterations: Maximum Jacobi sweeps.
        tolerance: Convergence threshold on max-norm update.
        log_every_n_iters: Emit a debug log entry every Nth iteration.
            Set very large to disable; set to 1 for per-iteration tracing.

    Returns:
        Solution field ``u`` of shape ``(R, R, R)`` with NaN at exterior
        voxels (so callers can mask them out).

    """
    if log_every_n_iters < 1:
        raise ValueError(
            f"log_every_n_iters must be >= 1, got {log_every_n_iters}"
        )
    if voxel_coords.shape[:3] != interior_mask.shape:
        raise ValueError(
            f"voxel_coords shape {voxel_coords.shape[:3]} does not match "
            f"interior_mask shape {interior_mask.shape}"
        )
    if diffusivity <= 0:
        raise ValueError(f"diffusivity must be > 0, got {diffusivity}")

    R = interior_mask.shape[0]
    n_interior = int(interior_mask.sum())
    logger.info(
        "voxel_fdm_solve_start",
        resolution=R,
        n_interior_voxels=n_interior,
        diffusivity=diffusivity,
        max_iterations=n_iterations,
    )
    u = np.zeros((R, R, R), dtype=np.float32)

    # Voxel spacing assumed isotropic; recover from the first axis.
    h = float(voxel_coords[1, 0, 0, 0] - voxel_coords[0, 0, 0, 0])
    if h <= 0:
        raise ValueError(f"non-positive voxel spacing inferred: {h}")
    h2 = h * h

    # Precompute boundary values for *every* voxel center; only the values
    # at exterior cells adjacent to interior cells are actually used in
    # the update, but precomputing keeps the per-iteration code clean.
    flat_coords = voxel_coords.reshape(-1, 3)
    boundary_values = np.asarray(
        boundary_value_fn(flat_coords), dtype=np.float32
    ).reshape(R, R, R)
    if source_fn is None:
        source = np.zeros((R, R, R), dtype=np.float32)
    else:
        source = np.asarray(source_fn(flat_coords), dtype=np.float32).reshape(
            R, R, R
        )

    # Set u to boundary values on exterior cells so neighbour reads find
    # the right Dirichlet value.
    u[~interior_mask] = boundary_values[~interior_mask]

    # Pre-shifted neighbour arrays for vectorized Gauss-Seidel sweeps.
    # We use Jacobi iteration here (not in-place GS) so the implementation
    # stays vectorized; convergence is slower but still adequate.
    for it in range(n_iterations):
        u_left = np.roll(u, 1, axis=0)
        u_right = np.roll(u, -1, axis=0)
        u_front = np.roll(u, 1, axis=1)
        u_back = np.roll(u, -1, axis=1)
        u_down = np.roll(u, 1, axis=2)
        u_up = np.roll(u, -1, axis=2)

        # Apply Neumann reflection at the array edges (won't affect
        # interior cells in practice because the SDF puts the surface
        # well away from grid boundaries).
        u_left[0, :, :] = u[0, :, :]
        u_right[-1, :, :] = u[-1, :, :]
        u_front[:, 0, :] = u[:, 0, :]
        u_back[:, -1, :] = u[:, -1, :]
        u_down[:, :, 0] = u[:, :, 0]
        u_up[:, :, -1] = u[:, :, -1]

        u_new = (
            u_left + u_right + u_front + u_back + u_down + u_up
            + h2 * source / diffusivity
        ) / 6.0

        # Only update interior cells; exterior cells stay pinned to
        # boundary values.
        delta = np.where(interior_mask, u_new - u, 0.0)
        u = np.where(interior_mask, u_new, boundary_values)

        max_delta = float(np.max(np.abs(delta)))
        if it % log_every_n_iters == 0:
            logger.debug(
                "voxel_fdm_iteration",
                iteration=it,
                max_delta=max_delta,
            )
        if max_delta < tolerance:
            logger.info(
                "voxel_fdm_converged",
                iteration=it,
                max_delta=max_delta,
            )
            break

    # NaN-mask exterior voxels for downstream consumers.
    out = u.copy()
    out[~interior_mask] = np.nan
    return out
