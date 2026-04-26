"""Tests for the in-repo 3D steady-heat voxel-FDM reference solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.voxel_fdm import solve_steady_heat_voxel, voxelize_sdf


def _sphere_sdf(radius: float) -> object:
    """Closed-form SDF for a sphere centred at the origin."""

    def fn(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points, axis=-1) - radius

    return fn


class TestVoxelizeSDF:
    def test_returns_correct_shapes(self) -> None:
        mask, coords = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.5),
            bbox_min=(-1.0, -1.0, -1.0),
            bbox_max=(1.0, 1.0, 1.0),
            resolution=8,
        )
        assert mask.shape == (8, 8, 8)
        assert coords.shape == (8, 8, 8, 3)

    def test_origin_is_inside_sphere(self) -> None:
        mask, coords = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.5),
            bbox_min=(-1.0, -1.0, -1.0),
            bbox_max=(1.0, 1.0, 1.0),
            resolution=9,
        )
        # Center voxel (index 4 along each axis) is at the origin.
        assert bool(mask[4, 4, 4])

    def test_corner_is_outside_sphere(self) -> None:
        mask, _ = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.5),
            bbox_min=(-1.0, -1.0, -1.0),
            bbox_max=(1.0, 1.0, 1.0),
            resolution=8,
        )
        assert not bool(mask[0, 0, 0])
        assert not bool(mask[-1, -1, -1])

    def test_invalid_resolution_raises(self) -> None:
        with pytest.raises(ValueError, match="resolution"):
            voxelize_sdf(
                sdf_fn=_sphere_sdf(0.5),
                bbox_min=(-1.0, -1.0, -1.0),
                bbox_max=(1.0, 1.0, 1.0),
                resolution=2,
            )


class TestSolveSteadyHeatVoxel:
    def test_uniform_dirichlet_yields_constant_field(self) -> None:
        # A box with constant Dirichlet BC == 1.0 has the constant
        # u(x) = 1 as its steady-state heat solution.
        mask, coords = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.4),
            bbox_min=(-0.6, -0.6, -0.6),
            bbox_max=(0.6, 0.6, 0.6),
            resolution=12,
        )
        u = solve_steady_heat_voxel(
            interior_mask=mask,
            voxel_coords=coords,
            boundary_value_fn=lambda pts: np.ones(pts.shape[0], dtype=np.float32),
            n_iterations=300,
            tolerance=1e-5,
        )
        interior_values = u[mask]
        assert interior_values.shape[0] > 0
        np.testing.assert_allclose(interior_values, np.ones_like(interior_values), atol=5e-2)

    def test_exterior_voxels_are_nan(self) -> None:
        mask, coords = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.4),
            bbox_min=(-0.6, -0.6, -0.6),
            bbox_max=(0.6, 0.6, 0.6),
            resolution=8,
        )
        u = solve_steady_heat_voxel(
            interior_mask=mask,
            voxel_coords=coords,
            boundary_value_fn=lambda pts: np.zeros(pts.shape[0], dtype=np.float32),
            n_iterations=50,
        )
        assert np.all(np.isnan(u[~mask]))
        assert np.all(np.isfinite(u[mask]))

    def test_source_term_increases_solution(self) -> None:
        mask, coords = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.4),
            bbox_min=(-0.6, -0.6, -0.6),
            bbox_max=(0.6, 0.6, 0.6),
            resolution=10,
        )
        zero_bc = lambda pts: np.zeros(pts.shape[0], dtype=np.float32)  # noqa: E731

        u_no_source = solve_steady_heat_voxel(
            interior_mask=mask,
            voxel_coords=coords,
            boundary_value_fn=zero_bc,
            n_iterations=200,
            tolerance=1e-6,
        )
        u_with_source = solve_steady_heat_voxel(
            interior_mask=mask,
            voxel_coords=coords,
            boundary_value_fn=zero_bc,
            source_fn=lambda pts: np.ones(pts.shape[0], dtype=np.float32),
            n_iterations=200,
            tolerance=1e-6,
        )

        # With a positive source and zero Dirichlet boundary, the solution
        # is strictly positive in the interior; without a source, it must
        # equal the boundary value (zero) everywhere.
        no_src_interior = u_no_source[mask]
        src_interior = u_with_source[mask]
        np.testing.assert_allclose(no_src_interior, np.zeros_like(no_src_interior), atol=1e-3)
        assert float(src_interior.mean()) > 0

    def test_invalid_diffusivity_raises(self) -> None:
        mask, coords = voxelize_sdf(
            sdf_fn=_sphere_sdf(0.4),
            bbox_min=(-0.6, -0.6, -0.6),
            bbox_max=(0.6, 0.6, 0.6),
            resolution=6,
        )
        with pytest.raises(ValueError, match="diffusivity"):
            solve_steady_heat_voxel(
                interior_mask=mask,
                voxel_coords=coords,
                boundary_value_fn=lambda pts: np.zeros(pts.shape[0], dtype=np.float32),
                diffusivity=0.0,
            )

    def test_shape_mismatch_raises(self) -> None:
        mask = np.zeros((6, 6, 6), dtype=bool)
        coords = np.zeros((4, 4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            solve_steady_heat_voxel(
                interior_mask=mask,
                voxel_coords=coords,
                boundary_value_fn=lambda pts: np.zeros(pts.shape[0], dtype=np.float32),
            )

    def test_non_positive_voxel_spacing_raises(self) -> None:
        # Construct a degenerate coords array where two consecutive x-axis
        # voxel centres coincide (zero spacing).
        mask = np.ones((4, 4, 4), dtype=bool)
        coords = np.zeros((4, 4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="voxel spacing"):
            solve_steady_heat_voxel(
                interior_mask=mask,
                voxel_coords=coords,
                boundary_value_fn=lambda pts: np.zeros(pts.shape[0], dtype=np.float32),
            )
