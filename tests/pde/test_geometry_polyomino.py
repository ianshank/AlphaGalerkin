"""Tests for ``PolyominoDomain``: exact quadrant coverage, corners, cut rays, sampling.

The multi-corner Poisson operator (``src/pde/operators/multi_corner_poisson.py``)
refuses a geometry its exact solution is not harmonic on, and it can only do
that if this module answers three questions exactly: which cells cover which
quadrant of a point, where the reentrant corners are, and whether a corner's
branch-cut ray touches the closed domain. Each is pinned here on synthetic
shapes where the right answer is known by construction.

No ``fem_required`` marker anywhere: this module is pure numpy/torch, so every
guard below runs on every CPU CI lane.

Mutation kills (harden-a-guard; each planted defect -> the named test that went red):

* T-junction check deleted from ``_edge_adjacent`` ->
  ``TestValidation::test_t_junction_is_rejected``.
* Pinch check deleted from ``_require_pinch_free`` ->
  ``TestValidation::test_pinch_vertex_is_rejected``.
* Connectivity check deleted from ``_require_edge_connected`` ->
  ``TestValidation::test_disconnected_union_is_rejected``.
* ``open_ray_meets_closure`` weakened from closure to open interior
  (``low <= high`` -> ``low < high``) ->
  ``TestOpenRayMeetsClosure::test_a_cut_that_grazes_a_vertex_meets_the_closure``.
* Quadrant half-open test made closed on both sides (``x0 <= x <= x1``) ->
  ``TestQuadrantCoverage::test_reentrant_vertex_has_exactly_three_covered_quadrants``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pde.geometry_polyomino import (
    LSHAPE_POLYOMINO_CELLS,
    POLYOMINO_REENTRANT_INTERIOR_ANGLE,
    QUADRANTS,
    ZSHAPE_POLYOMINO_CELLS,
    PolyominoCell,
    PolyominoDomain,
    PolyominoError,
    polyomino_domain_of,
)

C = PolyominoCell

#: A U whose arms are two cells tall: each inner corner's cut ray runs up the
#: gap diagonally and *crosses the interior* of the opposite arm's top cell.
TALL_U_CELLS = (
    C(0.0, 1.0, 0.0, 1.0),
    C(1.0, 2.0, 0.0, 1.0),
    C(2.0, 3.0, 0.0, 1.0),
    C(0.0, 1.0, 1.0, 2.0),
    C(0.0, 1.0, 2.0, 3.0),
    C(2.0, 3.0, 1.0, 2.0),
    C(2.0, 3.0, 2.0, 3.0),
)

#: A U whose arms are one cell tall: the cut rays only *graze* the opposite
#: arm's outer vertex ((2, 2) and (1, 2)) -- never entering an interior.
SHORT_U_CELLS = (
    C(0.0, 1.0, 0.0, 1.0),
    C(1.0, 2.0, 0.0, 1.0),
    C(2.0, 3.0, 0.0, 1.0),
    C(0.0, 1.0, 1.0, 2.0),
    C(2.0, 3.0, 1.0, 2.0),
)

#: A connected ring-like shape whose cells (0,1) and (1,2) meet only at the
#: vertex (1, 2): a pinch. Connected through the other cells, so only the
#: pinch check -- not the connectivity check -- can reject it.
PINCHED_CELLS = (
    C(0.0, 1.0, 0.0, 1.0),
    C(1.0, 2.0, 0.0, 1.0),
    C(2.0, 3.0, 0.0, 1.0),
    C(2.0, 3.0, 1.0, 2.0),
    C(2.0, 3.0, 2.0, 3.0),
    C(1.0, 2.0, 2.0, 3.0),
    C(0.0, 1.0, 1.0, 2.0),
)

#: A plus sign: four reentrant corners around the centre cell.
PLUS_CELLS = (
    C(0.0, 1.0, 0.0, 1.0),
    C(1.0, 2.0, 0.0, 1.0),
    C(-1.0, 0.0, 0.0, 1.0),
    C(0.0, 1.0, 1.0, 2.0),
    C(0.0, 1.0, -1.0, 0.0),
)


@pytest.fixture(scope="module")
def lshape() -> PolyominoDomain:
    return PolyominoDomain(LSHAPE_POLYOMINO_CELLS)


@pytest.fixture(scope="module")
def zshape() -> PolyominoDomain:
    return PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)


class TestPolyominoCell:
    def test_area_and_vertices(self) -> None:
        cell = C(-1.0, 0.5, 2.0, 4.0)
        assert cell.area == pytest.approx(3.0)
        assert cell.vertices == ((-1.0, 2.0), (0.5, 2.0), (0.5, 4.0), (-1.0, 4.0))

    @pytest.mark.parametrize(
        "coords",
        [(0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 1.0), (0.0, 1.0, 2.0, 1.0)],
    )
    def test_degenerate_or_inverted_cell_is_rejected(
        self, coords: tuple[float, float, float, float]
    ) -> None:
        with pytest.raises(PolyominoError, match="x0 < x1 and y0 < y1"):
            C(*coords)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_coordinate_is_rejected(self, bad: float) -> None:
        with pytest.raises(PolyominoError, match="finite"):
            C(0.0, bad, 0.0, 1.0)


class TestValidation:
    def test_empty_is_rejected(self) -> None:
        with pytest.raises(PolyominoError, match="at least one cell"):
            PolyominoDomain([])

    def test_overlapping_cells_are_rejected(self) -> None:
        with pytest.raises(PolyominoError, match="overlap"):
            PolyominoDomain([C(0.0, 1.0, 0.0, 1.0), C(0.5, 1.5, 0.0, 1.0)])

    @pytest.mark.parametrize(
        "cells",
        [
            (C(0.0, 1.0, 0.0, 1.0), C(1.0, 2.0, 0.0, 2.0)),  # vertical edge, partly shared
            (C(0.0, 1.0, 0.0, 1.0), C(0.0, 2.0, 1.0, 2.0)),  # horizontal edge, partly shared
        ],
    )
    def test_t_junction_is_rejected(self, cells: tuple[PolyominoCell, ...]) -> None:
        """A partly shared edge leaves a hanging node in a cell-by-cell mesh."""
        with pytest.raises(PolyominoError, match="T-junction"):
            PolyominoDomain(cells)

    def test_disconnected_union_is_rejected(self) -> None:
        with pytest.raises(PolyominoError, match="not edge-connected"):
            PolyominoDomain([C(0.0, 1.0, 0.0, 1.0), C(3.0, 4.0, 0.0, 1.0)])

    def test_pinch_vertex_is_rejected(self) -> None:
        with pytest.raises(PolyominoError, match="pinch"):
            PolyominoDomain(PINCHED_CELLS)

    def test_diagonal_contact_inside_a_filled_block_is_fine(self) -> None:
        """A 2x2 block has diagonal cells meeting at its centre -- not a pinch."""
        block = PolyominoDomain(
            [
                C(0.0, 1.0, 0.0, 1.0),
                C(1.0, 2.0, 0.0, 1.0),
                C(0.0, 1.0, 1.0, 2.0),
                C(1.0, 2.0, 1.0, 2.0),
            ]
        )
        assert block.reentrant_corners == ()
        assert block.area == pytest.approx(4.0)


class TestPresetsAndCorners:
    def test_lshape_structure(self, lshape: PolyominoDomain) -> None:
        assert lshape.dim == 2
        assert lshape.area == pytest.approx(3.0)
        assert lshape.perimeter == pytest.approx(8.0)
        assert lshape.bounding_box() == ((-1.0, -1.0), (1.0, 1.0))
        assert len(lshape.boundary_segments) == 6
        (corner,) = lshape.reentrant_corners
        assert corner.position == (0.0, 0.0)
        assert corner.exterior_quadrant == (1, -1)
        assert corner.exterior_bisector == -math.pi / 4
        assert corner.interior_angle == POLYOMINO_REENTRANT_INTERIOR_ANGLE

    def test_zshape_structure(self, zshape: PolyominoDomain) -> None:
        assert zshape.area == pytest.approx(4.0)
        assert zshape.perimeter == pytest.approx(10.0)
        assert zshape.bounding_box() == ((-1.0, -1.0), (2.0, 1.0))
        assert len(zshape.boundary_segments) == 8
        first, second = zshape.reentrant_corners
        assert (first.position, first.exterior_quadrant) == ((0.0, 0.0), (-1, -1))
        assert (second.position, second.exterior_quadrant) == ((1.0, 0.0), (1, 1))
        assert first.exterior_bisector == pytest.approx(-3 * math.pi / 4, abs=1e-15)
        assert second.exterior_bisector == pytest.approx(math.pi / 4, abs=1e-15)

    def test_a_rectangle_has_no_reentrant_corner(self) -> None:
        assert PolyominoDomain([C(0.0, 2.0, 0.0, 1.0)]).reentrant_corners == ()

    def test_a_plus_sign_has_four(self) -> None:
        corners = PolyominoDomain(PLUS_CELLS).reentrant_corners
        assert sorted(c.position for c in corners) == [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ]

    def test_corner_at(self, zshape: PolyominoDomain) -> None:
        assert zshape.corner_at((1.0, 0.0), atol=1e-12) is zshape.reentrant_corners[1]
        assert zshape.corner_at((1.0 + 1e-6, 0.0), atol=1e-12) is None
        assert zshape.corner_at((0.5, 0.0), atol=1e-12) is None


class TestQuadrantCoverage:
    def test_reentrant_vertex_has_exactly_three_covered_quadrants(
        self, zshape: PolyominoDomain
    ) -> None:
        coverage = zshape.quadrant_coverage(np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert coverage.sum(axis=1).tolist() == [3, 3]
        assert QUADRANTS[int(np.flatnonzero(~coverage[0])[0])] == (-1, -1)
        assert QUADRANTS[int(np.flatnonzero(~coverage[1])[0])] == (1, 1)

    @pytest.mark.parametrize(
        ("point", "expected"),
        [
            ((0.5, 0.5), [True, True, True, True]),  # inside cell B
            ((0.0, 0.5), [True, True, True, True]),  # on the A|B shared edge: interior
            ((-0.5, 1.0), [False, False, True, True]),  # top edge of A
            ((-1.0, 1.0), [False, False, False, True]),  # convex corner of A
            ((-0.5, -0.5), [False, False, False, False]),  # outside
        ],
    )
    def test_coverage_patterns(
        self, zshape: PolyominoDomain, point: tuple[float, float], expected: list[bool]
    ) -> None:
        assert zshape.quadrant_coverage(np.array([point])).tolist() == [expected]

    def test_negative_zero_is_classified_like_zero(self, zshape: PolyominoDomain) -> None:
        pts = np.array([[0.0, 0.5], [-0.0, 0.5], [0.5, 0.0], [0.5, -0.0]])
        assert zshape.interior_mask(pts).tolist() == [True, True, True, True]

    def test_interior_mask_excludes_the_boundary(self, zshape: PolyominoDomain) -> None:
        pts = np.array([[0.5, 0.5], [0.0, -0.5], [1.5, 0.0], [2.0, -0.5]])
        assert zshape.interior_mask(pts).tolist() == [True, False, False, False]


class TestContainsAndBoundary:
    def test_contains_point_is_the_closed_domain(self, zshape: PolyominoDomain) -> None:
        pts = torch.tensor([[0.5, 0.5], [2.0, -1.0], [-0.5, -0.5], [1.5, 0.5]])
        assert zshape.contains_point(pts).tolist() == [True, True, False, False]

    def test_is_boundary(self, zshape: PolyominoDomain) -> None:
        pts = torch.tensor(
            [[0.0, -0.5], [1.5, 0.0], [-1.0, 0.5], [0.5, 0.5], [0.0, 0.5], [1.5, 0.0 + 1e-3]]
        )
        assert zshape.is_boundary(pts).tolist() == [True, True, True, False, False, False]
        assert zshape.is_boundary(pts, tol=1e-2).tolist()[-1] is True

    def test_every_segment_endpoint_is_a_cell_vertex(self, zshape: PolyominoDomain) -> None:
        vertices = {v for cell in zshape.cells for v in cell.vertices}
        for segment in zshape.boundary_segments:
            assert segment.start in vertices
            assert segment.end in vertices
            assert segment.start <= segment.end


class TestOpenRayMeetsClosure:
    def test_preset_cut_rays_stay_outside(
        self, lshape: PolyominoDomain, zshape: PolyominoDomain
    ) -> None:
        for domain in (lshape, zshape):
            for corner in domain.reentrant_corners:
                assert not domain.open_ray_meets_closure(corner.position, corner.exterior_quadrant)

    def test_a_cut_through_the_interior_meets_the_closure(self) -> None:
        tall_u = PolyominoDomain(TALL_U_CELLS)
        assert [c.position for c in tall_u.reentrant_corners] == [(1.0, 1.0), (2.0, 1.0)]
        for corner in tall_u.reentrant_corners:
            assert tall_u.open_ray_meets_closure(corner.position, corner.exterior_quadrant)

    def test_a_cut_that_grazes_a_vertex_meets_the_closure(self) -> None:
        """Touching counts: the cut's kink would sit on a boundary node."""
        short_u = PolyominoDomain(SHORT_U_CELLS)
        for corner in short_u.reentrant_corners:
            assert short_u.open_ray_meets_closure(corner.position, corner.exterior_quadrant)

    def test_the_rays_own_origin_does_not_count(self, lshape: PolyominoDomain) -> None:
        """Every corner lies on three closed cells; ``t > 0`` must be strict."""
        assert not lshape.open_ray_meets_closure((0.0, 0.0), (1, -1))

    @pytest.mark.parametrize(
        ("origin", "direction", "expected"),
        [
            ((0.0, 0.0), (1, 0), True),  # runs along the L's reentrant edge y=0
            ((0.0, 0.0), (0, -1), True),  # runs along x=0, y<0
            ((2.0, 0.5), (1, 0), False),  # parallel, outside every x-slab
            ((0.5, 2.0), (0, 1), False),  # parallel, outside every y-slab
            ((1.5, 0.5), (-1, 0), True),  # enters the L from the right
        ],
    )
    def test_axis_aligned_rays(
        self,
        lshape: PolyominoDomain,
        origin: tuple[float, float],
        direction: tuple[float, float],
        expected: bool,
    ) -> None:
        assert lshape.open_ray_meets_closure(origin, direction) is expected


class TestSampling:
    def test_interior_samples_are_in_the_closed_domain(self, zshape: PolyominoDomain) -> None:
        pts = zshape.sample_interior(4000, generator=torch.Generator().manual_seed(7))
        assert pts.shape == (4000, 2)
        assert pts.dtype == torch.float32
        assert bool(zshape.contains_point(pts).all())

    def test_interior_samples_are_area_weighted(self, zshape: PolyominoDomain) -> None:
        """Four equal cells: each should receive about a quarter of the points."""
        pts = zshape.sample_interior(8000, generator=torch.Generator().manual_seed(3)).double()
        for cell in zshape.cells:
            inside = (
                (pts[:, 0] >= cell.x0)
                & (pts[:, 0] < cell.x1)
                & (pts[:, 1] >= cell.y0)
                & (pts[:, 1] < cell.y1)
            )
            assert float(inside.double().mean()) == pytest.approx(0.25, abs=0.03)

    def test_boundary_samples_are_on_the_boundary(self, zshape: PolyominoDomain) -> None:
        pts = zshape.sample_boundary(2000, generator=torch.Generator().manual_seed(11))
        assert bool(zshape.is_boundary(pts).all())
        assert bool(zshape.contains_point(pts).all())

    def test_a_generator_makes_draws_reproducible_without_touching_global_rng(
        self, zshape: PolyominoDomain
    ) -> None:
        state = torch.get_rng_state()
        first = zshape.sample_interior(64, generator=torch.Generator().manual_seed(5))
        second = zshape.sample_interior(64, generator=torch.Generator().manual_seed(5))
        assert torch.equal(first, second)
        assert torch.equal(torch.get_rng_state(), state)

    def test_device_argument_is_honoured(self, zshape: PolyominoDomain) -> None:
        cpu = torch.device("cpu")
        assert zshape.sample_interior(3, device=cpu).device == cpu
        assert zshape.sample_boundary(3, device=cpu).device == cpu

    @pytest.mark.parametrize("n_points", [0, -3])
    def test_non_positive_count_is_rejected(self, zshape: PolyominoDomain, n_points: int) -> None:
        with pytest.raises(ValueError, match="n_points"):
            zshape.sample_interior(n_points)
        with pytest.raises(ValueError, match="n_points"):
            zshape.sample_boundary(n_points)


class TestPolyominoDomainOf:
    def test_reads_a_polyomino_geometry(self, zshape: PolyominoDomain) -> None:
        class _Carrier:
            geometry = zshape

        assert polyomino_domain_of(_Carrier()) is zshape

    def test_anything_else_is_none(self) -> None:
        class _OtherGeometry:
            geometry = object()

        assert polyomino_domain_of(_OtherGeometry()) is None
        assert polyomino_domain_of(object()) is None


class TestExactClassificationProperties:
    """Invariants tying the exact tests to the closed-domain test."""

    @settings(max_examples=60, deadline=None)
    @given(
        cell_index=st.integers(min_value=0, max_value=len(ZSHAPE_POLYOMINO_CELLS) - 1),
        u=st.floats(min_value=0.001, max_value=0.999),
        v=st.floats(min_value=0.001, max_value=0.999),
    )
    def test_strictly_inside_a_cell_is_interior(self, cell_index: int, u: float, v: float) -> None:
        domain = PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)
        cell = domain.cells[cell_index]
        point = np.array([[cell.x0 + u * (cell.x1 - cell.x0), cell.y0 + v * (cell.y1 - cell.y0)]])
        assert domain.interior_mask(point).tolist() == [True]

    @settings(max_examples=80, deadline=None)
    @given(
        x=st.floats(min_value=-1.5, max_value=2.5),
        y=st.floats(min_value=-1.5, max_value=1.5),
    )
    def test_interior_implies_closed_membership(self, x: float, y: float) -> None:
        domain = PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)
        interior = bool(domain.interior_mask(np.array([[x, y]]))[0])
        closed = bool(domain.contains_point(torch.tensor([[x, y]], dtype=torch.float64))[0])
        assert not interior or closed
