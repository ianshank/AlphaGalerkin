"""Tests for src/pde/geometry.py.

Covers RectangularDomain, LShapedDomain, CylinderFlowDomain,
GeometryConfig, and create_geometry factory.
Uses pytest.approx for float assertions, hypothesis for property tests.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from src.pde.geometry import (
    CylinderFlowDomain,
    DomainGeometry,
    GeometryConfig,
    GeometryType,
    LShapedDomain,
    RectangularDomain,
    create_geometry,
)

# ---------------------------------------------------------------------------
# GeometryConfig
# ---------------------------------------------------------------------------

class TestGeometryConfig:
    def test_defaults(self):
        cfg = GeometryConfig()
        assert cfg.geometry_type == GeometryType.RECTANGULAR
        assert cfg.scale > 0

    def test_all_geometry_types(self):
        for gt in GeometryType:
            cfg = GeometryConfig(geometry_type=gt)
            assert cfg.geometry_type == gt

    def test_scale_must_be_positive(self):
        with pytest.raises(ValidationError):
            GeometryConfig(scale=0.0)
        with pytest.raises(ValidationError):
            GeometryConfig(scale=-1.0)

    def test_cylinder_radius_must_be_positive(self):
        with pytest.raises(ValidationError):
            GeometryConfig(cylinder_radius=0.0)

    def test_custom_values(self):
        cfg = GeometryConfig(
            geometry_type=GeometryType.L_SHAPED,
            scale=2.5,
            cylinder_cx=0.3,
            cylinder_cy=0.3,
            cylinder_radius=0.1,
        )
        assert cfg.scale == pytest.approx(2.5)
        assert cfg.cylinder_radius == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# create_geometry factory
# ---------------------------------------------------------------------------

class TestCreateGeometry:
    def test_rectangular(self):
        cfg = GeometryConfig(geometry_type=GeometryType.RECTANGULAR)
        geom = create_geometry(cfg)
        assert isinstance(geom, RectangularDomain)

    def test_l_shaped(self):
        cfg = GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=1.0)
        geom = create_geometry(cfg)
        assert isinstance(geom, LShapedDomain)

    def test_cylinder_flow(self):
        cfg = GeometryConfig(geometry_type=GeometryType.CYLINDER_FLOW)
        geom = create_geometry(cfg)
        assert isinstance(geom, CylinderFlowDomain)

    def test_all_types_produce_geometry(self):
        for gt in GeometryType:
            cfg = GeometryConfig(geometry_type=gt)
            geom = create_geometry(cfg)
            assert isinstance(geom, DomainGeometry)

    def test_rectangular_inherits_config_bounds(self):
        cfg = GeometryConfig(
            geometry_type=GeometryType.RECTANGULAR,
            x_min=-2.0, x_max=3.0, y_min=-1.0, y_max=5.0
        )
        geom = create_geometry(cfg)
        assert isinstance(geom, RectangularDomain)
        assert geom.x_min == pytest.approx(-2.0)
        assert geom.x_max == pytest.approx(3.0)

    def test_l_shaped_inherits_scale(self):
        cfg = GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=2.0)
        geom = create_geometry(cfg)
        assert isinstance(geom, LShapedDomain)
        assert geom.scale == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# RectangularDomain
# ---------------------------------------------------------------------------

class TestRectangularDomain:
    @pytest.fixture()
    def unit_square(self) -> RectangularDomain:
        return RectangularDomain(0.0, 1.0, 0.0, 1.0)

    @pytest.fixture()
    def general_rect(self) -> RectangularDomain:
        return RectangularDomain(-2.0, 3.0, -1.0, 5.0)

    def test_dim(self, unit_square: RectangularDomain):
        assert unit_square.dim == 2

    def test_area_unit_square(self, unit_square: RectangularDomain):
        assert unit_square.area == pytest.approx(1.0)

    def test_area_general(self, general_rect: RectangularDomain):
        expected = (3.0 - (-2.0)) * (5.0 - (-1.0))  # 5 * 6 = 30
        assert general_rect.area == pytest.approx(expected)

    def test_bounding_box(self, unit_square: RectangularDomain):
        lo, hi = unit_square.bounding_box()
        assert lo == (0.0, 0.0)
        assert hi == (1.0, 1.0)

    def test_contains_interior(self, unit_square: RectangularDomain):
        pts = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
        mask = unit_square.contains_point(pts)
        assert mask.all()

    def test_does_not_contain_outside(self, unit_square: RectangularDomain):
        pts = torch.tensor([[1.5, 0.5], [-0.1, 0.5], [0.5, 1.5]])
        mask = unit_square.contains_point(pts)
        assert not mask.any()

    def test_contains_boundary_points(self, unit_square: RectangularDomain):
        # Corners and edge midpoints are on boundary -> contained
        pts = torch.tensor([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
        mask = unit_square.contains_point(pts)
        assert mask.all()

    def test_is_boundary_detects_edges(self, unit_square: RectangularDomain):
        pts = torch.tensor([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]])
        bnd = unit_square.is_boundary(pts)
        assert bnd.all()

    def test_is_boundary_excludes_interior(self, unit_square: RectangularDomain):
        pts = torch.tensor([[0.5, 0.5]])
        bnd = unit_square.is_boundary(pts)
        assert not bnd.any()

    def test_sample_interior_shape(self, unit_square: RectangularDomain):
        pts = unit_square.sample_interior(100)
        assert pts.shape == (100, 2)

    def test_sample_interior_all_in_domain(self, unit_square: RectangularDomain):
        pts = unit_square.sample_interior(200)
        mask = unit_square.contains_point(pts)
        assert mask.all()

    def test_sample_boundary_shape(self, unit_square: RectangularDomain):
        pts = unit_square.sample_boundary(80)
        assert pts.shape[1] == 2
        assert pts.shape[0] > 0

    def test_sample_with_device(self, unit_square: RectangularDomain):
        device = torch.device("cpu")
        pts = unit_square.sample_interior(50, device=device)
        assert pts.device.type == "cpu"

    @pytest.mark.parametrize("n", [1, 10, 100, 500])
    def test_sample_interior_various_sizes(
        self, unit_square: RectangularDomain, n: int
    ):
        pts = unit_square.sample_interior(n)
        assert pts.shape == (n, 2)


# ---------------------------------------------------------------------------
# LShapedDomain
# ---------------------------------------------------------------------------

class TestLShapedDomain:
    @pytest.fixture()
    def unit_l(self) -> LShapedDomain:
        return LShapedDomain(scale=1.0)

    @pytest.fixture()
    def scaled_l(self) -> LShapedDomain:
        return LShapedDomain(scale=2.0)

    def test_dim(self, unit_l: LShapedDomain):
        assert unit_l.dim == 2

    def test_area_unit(self, unit_l: LShapedDomain):
        # Area = 3 * s^2 = 3.0 for s=1
        assert unit_l.area == pytest.approx(3.0)

    def test_area_scaled(self, scaled_l: LShapedDomain):
        # Area = 3 * 2^2 = 12.0
        assert scaled_l.area == pytest.approx(12.0)

    def test_bounding_box_unit(self, unit_l: LShapedDomain):
        lo, hi = unit_l.bounding_box()
        assert lo == (-1.0, -1.0)
        assert hi == (1.0, 1.0)

    def test_contains_upper_left_quadrant(self, unit_l: LShapedDomain):
        """(-0.5, 0.5) is in the upper-left region of the L."""
        pts = torch.tensor([[-0.5, 0.5]])
        assert unit_l.contains_point(pts).all()

    def test_contains_top_right_quadrant(self, unit_l: LShapedDomain):
        """(0.5, 0.5) is in the top-right region (part of L)."""
        pts = torch.tensor([[0.5, 0.5]])
        assert unit_l.contains_point(pts).all()

    def test_does_not_contain_removed_quadrant(self, unit_l: LShapedDomain):
        """(0.5, -0.5) is in the removed bottom-right quadrant."""
        pts = torch.tensor([[0.5, -0.5]])
        assert not unit_l.contains_point(pts).any()

    def test_does_not_contain_outside_bounding_box(self, unit_l: LShapedDomain):
        pts = torch.tensor([[2.0, 0.0], [0.0, -2.0]])
        assert not unit_l.contains_point(pts).any()

    def test_contains_origin(self, unit_l: LShapedDomain):
        """Origin (0,0) is a corner - strictly: (0 > 0) is False so it's in domain."""
        pts = torch.tensor([[0.0, 0.0]])
        # x > 0 is False, so not in removed quadrant => in domain
        assert unit_l.contains_point(pts).all()

    def test_is_boundary_bottom_edge(self, unit_l: LShapedDomain):
        """Y = -1, x in [-1, 0] is the bottom boundary segment."""
        pts = torch.tensor([[-0.5, -1.0], [-0.8, -1.0]])
        bnd = unit_l.is_boundary(pts)
        assert bnd.all()

    def test_is_boundary_left_edge(self, unit_l: LShapedDomain):
        pts = torch.tensor([[-1.0, 0.0], [-1.0, -0.5]])
        bnd = unit_l.is_boundary(pts)
        assert bnd.all()

    def test_is_boundary_top_edge(self, unit_l: LShapedDomain):
        pts = torch.tensor([[0.0, 1.0], [0.5, 1.0]])
        bnd = unit_l.is_boundary(pts)
        assert bnd.all()

    def test_is_boundary_reentrant_corner_segments(self, unit_l: LShapedDomain):
        """Reentrant horizontal y=0, x in [0,1] and vertical x=0, y in [-1,0]."""
        pts = torch.tensor([[0.5, 0.0], [0.0, -0.5]])
        bnd = unit_l.is_boundary(pts)
        assert bnd.all()

    def test_sample_interior_shape(self, unit_l: LShapedDomain):
        pts = unit_l.sample_interior(100)
        assert pts.shape == (100, 2)

    def test_sample_interior_all_in_domain(self, unit_l: LShapedDomain):
        pts = unit_l.sample_interior(200)
        mask = unit_l.contains_point(pts)
        assert mask.all()

    def test_sample_interior_none_in_removed_quadrant(self, unit_l: LShapedDomain):
        pts = unit_l.sample_interior(500)
        # No point should have x > 0 and y < 0
        in_removed = (pts[:, 0] > 0) & (pts[:, 1] < 0)
        assert not in_removed.any()

    def test_sample_boundary_returns_tensor(self, unit_l: LShapedDomain):
        pts = unit_l.sample_boundary(60)
        assert isinstance(pts, torch.Tensor)
        assert pts.shape[1] == 2

    def test_scaled_contains_point(self, scaled_l: LShapedDomain):
        """Scaled domain has correct extent."""
        pts = torch.tensor([[-1.5, 1.5]])  # In bounding box [-2,2]^2
        assert scaled_l.contains_point(pts).all()

    @pytest.mark.parametrize("n", [1, 50, 200])
    def test_sample_interior_sizes(self, unit_l: LShapedDomain, n: int):
        pts = unit_l.sample_interior(n)
        assert pts.shape == (n, 2)


# ---------------------------------------------------------------------------
# CylinderFlowDomain
# ---------------------------------------------------------------------------

class TestCylinderFlowDomain:
    @pytest.fixture()
    def dfg_domain(self) -> CylinderFlowDomain:
        """Standard DFG benchmark domain."""
        return CylinderFlowDomain(
            x_min=0.0, x_max=2.2, y_min=0.0, y_max=0.41,
            cx=0.2, cy=0.2, radius=0.05
        )

    def test_dim(self, dfg_domain: CylinderFlowDomain):
        assert dfg_domain.dim == 2

    def test_area_less_than_channel(self, dfg_domain: CylinderFlowDomain):
        """Domain area should be channel minus circle."""
        rect_area = 2.2 * 0.41
        circle_area = math.pi * 0.05**2
        expected = rect_area - circle_area
        assert dfg_domain.area == pytest.approx(expected, rel=1e-5)

    def test_bounding_box(self, dfg_domain: CylinderFlowDomain):
        lo, hi = dfg_domain.bounding_box()
        assert lo[0] == pytest.approx(0.0)
        assert hi[0] == pytest.approx(2.2)

    def test_contains_channel_interior(self, dfg_domain: CylinderFlowDomain):
        pts = torch.tensor([[1.0, 0.2], [1.5, 0.3]])
        assert dfg_domain.contains_point(pts).all()

    def test_does_not_contain_inside_cylinder(self, dfg_domain: CylinderFlowDomain):
        """Point at cylinder center is inside cylinder, not in domain."""
        pts = torch.tensor([[0.2, 0.2]])  # cylinder center
        assert not dfg_domain.contains_point(pts).any()

    def test_does_not_contain_outside_channel(self, dfg_domain: CylinderFlowDomain):
        pts = torch.tensor([[3.0, 0.2], [1.0, 0.5]])
        assert not dfg_domain.contains_point(pts).any()

    def test_is_boundary_channel_walls(self, dfg_domain: CylinderFlowDomain):
        pts = torch.tensor([[1.0, 0.0], [1.0, 0.41]])  # bottom and top wall
        bnd = dfg_domain.is_boundary(pts)
        assert bnd.all()

    def test_is_boundary_cylinder_surface(self, dfg_domain: CylinderFlowDomain):
        """A point exactly on the cylinder surface should be boundary."""
        cx, cy, r = 0.2, 0.2, 0.05
        pts = torch.tensor([[cx + r, cy]])
        bnd = dfg_domain.is_boundary(pts)
        assert bnd.all()

    def test_sample_interior_shape(self, dfg_domain: CylinderFlowDomain):
        pts = dfg_domain.sample_interior(100)
        assert pts.shape == (100, 2)

    def test_sample_interior_outside_cylinder(self, dfg_domain: CylinderFlowDomain):
        pts = dfg_domain.sample_interior(500)
        cx, cy, r = dfg_domain.cx, dfg_domain.cy, dfg_domain.radius
        dist = torch.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        assert (dist > r - 1e-6).all()

    def test_sample_boundary_returns_tensor(self, dfg_domain: CylinderFlowDomain):
        pts = dfg_domain.sample_boundary(80)
        assert isinstance(pts, torch.Tensor)
        assert pts.shape[1] == 2

    def test_custom_cylinder(self):
        geom = CylinderFlowDomain(
            x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
            cx=0.5, cy=0.5, radius=0.1
        )
        # Center is inside cylinder
        pts = torch.tensor([[0.5, 0.5]])
        assert not geom.contains_point(pts).any()
        # Far from center is in domain
        pts2 = torch.tensor([[0.9, 0.9]])
        assert geom.contains_point(pts2).all()


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

class TestGeometryProperties:
    @given(
        x=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
        y=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=60, deadline=5000)
    def test_rectangular_strict_interior_not_boundary(self, x: float, y: float):
        """Interior points should not be boundary (rectangular)."""
        geom = RectangularDomain(0.0, 1.0, 0.0, 1.0)
        pts = torch.tensor([[x, y]])
        assert geom.contains_point(pts).all()
        assert not geom.is_boundary(pts).any()

    @given(
        n=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30, deadline=10000)
    def test_sample_interior_count_exact(self, n: int):
        """sample_interior always returns exactly n points."""
        geom = RectangularDomain(0.0, 1.0, 0.0, 1.0)
        pts = geom.sample_interior(n)
        assert pts.shape == (n, 2)

    @given(
        n=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=30, deadline=10000)
    def test_l_shaped_sample_interior_count(self, n: int):
        geom = LShapedDomain(scale=1.0)
        pts = geom.sample_interior(n)
        assert pts.shape == (n, 2)

    @given(
        scale=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=5000)
    def test_l_shaped_area_formula(self, scale: float):
        geom = LShapedDomain(scale=scale)
        assert geom.area == pytest.approx(3.0 * scale**2, rel=1e-6)
