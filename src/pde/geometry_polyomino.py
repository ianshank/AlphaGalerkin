"""Polyomino domains: unions of axis-aligned closed cells, with exact corner analysis.

A *polyomino domain* is the union of finitely many closed, axis-aligned
rectangles ("cells") ``[x0, x1] x [y0, y1]`` that meet edge-to-edge. It
generalises :class:`~src.pde.geometry.LShapedDomain` (three unit cells) to any
rectilinear shape assembled from cells. The motivating case is the Z-tetromino,
whose two reentrant corners make it the second AMR testbed the Gate 1 plan in
``docs/business/COMMERCIALIZATION_PEER_REVIEW.md`` (§6) asks for.

Three facts a multi-corner singular solution must *verify* rather than assume
are exact, sign-only computations on a union of cells -- no epsilon anywhere:

* **Quadrant coverage.** For a point ``v`` and a sign pair ``(sx, sy)``, the
  open quadrant ``v + (sx*d, sy*d)`` (all small ``d > 0``) lies inside some cell
  iff that cell satisfies half-open inequalities in ``v``
  (:meth:`PolyominoDomain.quadrant_coverage`). The answer is exact on the floats
  the cells are written in.
* **Reentrant corners** are the cell vertices with exactly three covered
  quadrants. The uncovered one *is* the exterior wedge, and its diagonal is the
  exterior bisector (:class:`PolyominoCorner`).
* **Branch-cut rays.** The ray from a corner along its exterior bisector moves
  in direction ``(sx, sy)`` with ``sx, sy`` in ``{-1, +1}``, so the slab test
  against a closed cell divides by +/-1 only
  (:meth:`PolyominoDomain.open_ray_meets_closure`). "Does the cut reach the
  closed domain" therefore has an exact answer -- including the grazing case, a
  ray passing exactly through another cell's vertex, where a floating-point
  direction ``(cos b, sin b)`` could round either way.

Validation is strict because the failure it prevents is silent. Overlapping
cells, a T-junction (two cells sharing only part of an edge, which leaves a
hanging node in any mesh built cell by cell), a pinch (cells meeting only at a
vertex, a non-Lipschitz point) and a disconnected union all construct finite,
plausible PDE problems that are not the domain the caller meant.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.geometry import DomainGeometry

logger = structlog.get_logger(__name__)

Point = tuple[float, float]
Quadrant = tuple[int, int]

#: The four open quadrants around a point as ``(sx, sy)`` sign pairs,
#: counter-clockwise from ``(+x, +y)``. Diagonal pairs sit two apart.
QUADRANTS: Final[tuple[Quadrant, ...]] = ((1, 1), (-1, 1), (-1, -1), (1, -1))

#: Covered quadrants at a 270-degree (reentrant) boundary vertex.
REENTRANT_COVERED_QUADRANTS: Final[int] = 3

#: Covered quadrants at a vertex where two cells meet only diagonally -- a
#: *pinch* when the two covered quadrants are opposite each other.
PINCH_COVERED_QUADRANTS: Final[int] = 2

#: Interior angle of every reentrant corner of a polyomino: 270 degrees.
POLYOMINO_REENTRANT_INTERIOR_ANGLE: Final[float] = 1.5 * math.pi

#: Coordinate columns of the ``(n_cells, 4)`` bounds array.
_X0, _X1, _Y0, _Y1 = 0, 1, 2, 3

#: Columns of the ``(n_segments, 4)`` boundary-segment array.
_SX, _SY, _EX, _EY = 0, 1, 2, 3


class PolyominoError(ValueError):
    """The cells do not describe a valid (edge-to-edge, pinch-free, connected) polyomino."""


@dataclass(frozen=True)
class PolyominoCell:
    """A closed axis-aligned rectangle ``[x0, x1] x [y0, y1]`` with positive area."""

    x0: float
    x1: float
    y0: float
    y1: float

    def __post_init__(self) -> None:
        """Reject non-finite coordinates and degenerate (zero-area) cells."""
        values = (self.x0, self.x1, self.y0, self.y1)
        if not all(math.isfinite(value) for value in values):
            raise PolyominoError(f"cell coordinates must be finite, got {values}")
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise PolyominoError(f"a cell needs x0 < x1 and y0 < y1, got {values}")

    @property
    def area(self) -> float:
        """Area of the cell."""
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def vertices(self) -> tuple[Point, Point, Point, Point]:
        """The four corners, counter-clockwise from ``(x0, y0)``."""
        return ((self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1))


@dataclass(frozen=True)
class PolyominoCorner:
    """A reentrant (270-degree) corner, located exactly by quadrant coverage.

    Attributes:
        position: The corner vertex.
        exterior_quadrant: ``(sx, sy)`` of the one quadrant the domain does not
            cover. It is also the corner's exact, unnormalised branch-cut direction.

    """

    position: Point
    exterior_quadrant: Quadrant

    @property
    def exterior_bisector(self) -> float:
        """Angle of the exterior wedge's bisector, in ``(-pi, pi]``."""
        sx, sy = self.exterior_quadrant
        return math.atan2(sy, sx)

    @property
    def interior_angle(self) -> float:
        """Always 270 degrees on a polyomino."""
        return POLYOMINO_REENTRANT_INTERIOR_ANGLE


@dataclass(frozen=True)
class BoundarySegment:
    """A maximal straight piece of the boundary, ``start <= end`` component-wise."""

    start: Point
    end: Point

    @property
    def length(self) -> float:
        """Euclidean length (axis-aligned, so one component is zero)."""
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])


def _overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    """Signed overlap of two closed intervals: > 0 overlap, 0 touch, < 0 gap."""
    return min(hi_a, hi_b) - max(lo_a, lo_b)


def _edge_adjacent(a: PolyominoCell, b: PolyominoCell) -> bool:
    """Whether two cells share a full edge; raise on overlap or a T-junction.

    Contact of positive length along a line must be a *whole* common edge: a
    cell edge touched along only part of its length puts a hanging node in any
    mesh assembled cell by cell, which silently breaks conformity.
    """
    x_overlap = _overlap(a.x0, a.x1, b.x0, b.x1)
    y_overlap = _overlap(a.y0, a.y1, b.y0, b.y1)
    if x_overlap > 0 and y_overlap > 0:
        raise PolyominoError(f"cells {a} and {b} overlap")
    vertical_contact = x_overlap == 0 and y_overlap > 0
    horizontal_contact = y_overlap == 0 and x_overlap > 0
    if vertical_contact and (a.y0, a.y1) != (b.y0, b.y1):
        raise PolyominoError(f"cells {a} and {b} share only part of a vertical edge (T-junction)")
    if horizontal_contact and (a.x0, a.x1) != (b.x0, b.x1):
        raise PolyominoError(f"cells {a} and {b} share only part of a horizontal edge (T-junction)")
    return vertical_contact or horizontal_contact


def _require_edge_connected(cells: tuple[PolyominoCell, ...]) -> None:
    """Validate every pair and require one edge-connected component."""
    neighbours: dict[int, set[int]] = {index: set() for index in range(len(cells))}
    for i, first in enumerate(cells):
        for j in range(i + 1, len(cells)):
            if _edge_adjacent(first, cells[j]):
                neighbours[i].add(j)
                neighbours[j].add(i)
    reached = {0}
    frontier = [0]
    while frontier:
        for other in neighbours[frontier.pop()] - reached:
            reached.add(other)
            frontier.append(other)
    if len(reached) != len(cells):
        stranded = sorted(set(neighbours) - reached)
        raise PolyominoError(
            f"cells {stranded} are not edge-connected to cell 0; a disconnected union is "
            f"almost always a typo in a coordinate, not the intended domain"
        )


def _merge_collinear(
    edges: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    """Merge ``(line, start, end)`` intervals on one line that abut end to start."""
    merged: list[tuple[float, float, float]] = []
    for line, start, end in sorted(edges):
        if merged and merged[-1][0] == line and merged[-1][2] == start:
            merged[-1] = (line, merged[-1][1], end)
        else:
            merged.append((line, start, end))
    return merged


class PolyominoDomain(DomainGeometry):
    """The union of edge-to-edge axis-aligned closed cells.

    Construction validates the union (see the module docstring) and computes
    its reentrant corners and maximal boundary segments once.
    """

    def __init__(self, cells: Sequence[PolyominoCell]) -> None:
        """Build and validate a polyomino.

        Args:
            cells: The closed cells. Order is preserved (``cells``) because mesh
                builders join per-cell meshes in this order.

        Raises:
            PolyominoError: On an empty cell list, overlapping cells, a
                T-junction, a pinch vertex, or a disconnected union.

        """
        self._cells = tuple(cells)
        if not self._cells:
            raise PolyominoError("a polyomino needs at least one cell")
        self._bounds = np.array(
            [[cell.x0, cell.x1, cell.y0, cell.y1] for cell in self._cells], dtype=np.float64
        )
        _require_edge_connected(self._cells)
        vertices = np.unique(
            np.array([v for cell in self._cells for v in cell.vertices], dtype=np.float64), axis=0
        )
        coverage = self.quadrant_coverage(vertices)
        self._require_pinch_free(vertices, coverage)
        self._corners = self._find_reentrant_corners(vertices, coverage)
        self._segments = self._find_boundary_segments()
        self._segment_array = np.array(
            [[*segment.start, *segment.end] for segment in self._segments], dtype=np.float64
        )
        logger.debug(
            "polyomino_domain_created",
            n_cells=len(self._cells),
            area=self.area,
            n_boundary_segments=len(self._segments),
            reentrant_corners=[corner.position for corner in self._corners],
        )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def cells(self) -> tuple[PolyominoCell, ...]:
        """The cells, in construction order."""
        return self._cells

    @property
    def reentrant_corners(self) -> tuple[PolyominoCorner, ...]:
        """Every 270-degree boundary vertex, sorted by ``(x, y)``."""
        return self._corners

    @property
    def boundary_segments(self) -> tuple[BoundarySegment, ...]:
        """Maximal straight boundary pieces (horizontal first, each sorted)."""
        return self._segments

    @property
    def perimeter(self) -> float:
        """Total boundary length."""
        return float(sum(segment.length for segment in self._segments))

    @property
    def dim(self) -> int:
        """Spatial dimension (always 2)."""
        return 2

    @property
    def area(self) -> float:
        """Sum of cell areas (interiors are disjoint by construction)."""
        return float(sum(cell.area for cell in self._cells))

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Axis-aligned bounding box as ``((x_min, y_min), (x_max, y_max))``."""
        return (
            (float(self._bounds[:, _X0].min()), float(self._bounds[:, _Y0].min())),
            (float(self._bounds[:, _X1].max()), float(self._bounds[:, _Y1].max())),
        )

    def corner_at(self, position: Point, *, atol: float) -> PolyominoCorner | None:
        """The reentrant corner within ``atol`` (max-norm) of ``position``, if any."""
        for corner in self._corners:
            dx = abs(corner.position[0] - position[0])
            dy = abs(corner.position[1] - position[1])
            if max(dx, dy) <= atol:
                return corner
        return None

    # ------------------------------------------------------------------
    # Exact point classification
    # ------------------------------------------------------------------

    def quadrant_coverage(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Which open quadrants around each point lie inside some cell.

        Args:
            points: ``(n, 2)`` coordinates.

        Returns:
            ``(n, 4)`` boolean array; column ``q`` refers to ``QUADRANTS[q]``.
            Quadrant ``(+1, sy)`` is covered by a cell with ``x0 <= x < x1``,
            ``(-1, sy)`` by one with ``x0 < x <= x1`` (and likewise in ``y``):
            half-open tests, so the result is exact.

        """
        pts = np.asarray(points, dtype=np.float64)
        x = pts[:, 0:1]
        y = pts[:, 1:2]
        x0, x1, y0, y1 = (self._bounds[:, column] for column in (_X0, _X1, _Y0, _Y1))
        along_x = {1: (x0 <= x) & (x < x1), -1: (x0 < x) & (x <= x1)}
        along_y = {1: (y0 <= y) & (y < y1), -1: (y0 < y) & (y <= y1)}
        return np.stack([np.any(along_x[sx] & along_y[sy], axis=1) for sx, sy in QUADRANTS], axis=1)

    def interior_mask(self, points: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Exact open-interior test: all four quadrants covered.

        Unlike :meth:`contains_point` (the closed domain), this is ``False`` on
        the boundary -- including the reentrant edges -- and ``True`` on an
        edge two cells share.
        """
        return np.asarray(self.quadrant_coverage(points).all(axis=1), dtype=bool)

    def contains_point(self, points: Tensor) -> Tensor:
        """Closed-domain membership: inside or on the boundary of some cell."""
        bounds = torch.as_tensor(self._bounds, dtype=points.dtype, device=points.device)
        x = points[:, 0:1]
        y = points[:, 1:2]
        inside = (
            (bounds[:, _X0] <= x)
            & (x <= bounds[:, _X1])
            & (bounds[:, _Y0] <= y)
            & (y <= bounds[:, _Y1])
        )
        return inside.any(dim=1)

    def is_boundary(self, points: Tensor, tol: float = DEFAULT_BOUNDARY_TOLERANCE) -> Tensor:
        """Points within ``tol`` (Euclidean) of a boundary segment.

        Each segment is axis-aligned, so clamping a point into the segment's
        (degenerate) bounding box is its exact nearest point.
        """
        segments = torch.as_tensor(self._segment_array, dtype=points.dtype, device=points.device)
        px = points[:, 0:1]
        py = points[:, 1:2]
        nearest_x = torch.clamp(px, segments[:, _SX], segments[:, _EX])
        nearest_y = torch.clamp(py, segments[:, _SY], segments[:, _EY])
        distance = torch.sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)
        return distance.min(dim=1).values < tol

    def open_ray_meets_closure(self, origin: Point, direction: tuple[float, float]) -> bool:
        """Whether ``{origin + t * direction : t > 0}`` touches any closed cell.

        Slab test per cell over ``t > 0`` (strict: the ray's own origin is on
        the boundary of the cells around a corner and must not count). Exact
        when ``direction`` components are in ``{-1, 0, +1}`` and the
        coordinates share a dyadic lattice -- which is how
        :class:`PolyominoCorner` cut rays are expressed. *Touching* counts: a
        cut that grazes a vertex of the closed domain puts a kink in the
        singular term arbitrarily close to that boundary point.
        """
        for cell in self._cells:
            x_low, x_high = _slab_interval(origin[0], direction[0], cell.x0, cell.x1)
            y_low, y_high = _slab_interval(origin[1], direction[1], cell.y0, cell.y1)
            low, high = max(0.0, x_low, y_low), min(x_high, y_high)
            if high > 0 and low <= high:
                return True
        return False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_interior(
        self,
        n_points: int,
        device: torch.device | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Uniform points in the closed domain: area-weighted cell, then uniform in it.

        Exact, with no rejection loop. ``generator`` makes a draw reproducible
        without touching torch's global RNG.
        """
        _require_positive_count(n_points)
        areas = torch.tensor([cell.area for cell in self._cells], dtype=torch.float64)
        chosen = torch.multinomial(areas, n_points, replacement=True, generator=generator)
        bounds = torch.as_tensor(self._bounds, dtype=torch.float64)[chosen]
        unit = torch.rand(n_points, 2, generator=generator, dtype=torch.float64)
        x = bounds[:, _X0] + unit[:, 0] * (bounds[:, _X1] - bounds[:, _X0])
        y = bounds[:, _Y0] + unit[:, 1] * (bounds[:, _Y1] - bounds[:, _Y0])
        return _on_device(torch.stack([x, y], dim=-1).to(torch.float32), device)

    def sample_boundary(
        self,
        n_points: int,
        device: torch.device | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Uniform points on the boundary: length-weighted segment, then uniform on it."""
        _require_positive_count(n_points)
        lengths = torch.tensor([segment.length for segment in self._segments], dtype=torch.float64)
        chosen = torch.multinomial(lengths, n_points, replacement=True, generator=generator)
        segments = torch.as_tensor(self._segment_array, dtype=torch.float64)[chosen]
        t = torch.rand(n_points, 1, generator=generator, dtype=torch.float64)
        starts = segments[:, [_SX, _SY]]
        points = starts + t * (segments[:, [_EX, _EY]] - starts)
        return _on_device(points.to(torch.float32), device)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_pinch_free(vertices: NDArray[np.float64], coverage: NDArray[np.bool_]) -> None:
        """Reject a vertex where exactly two *opposite* quadrants are covered."""
        n_covered = coverage.sum(axis=1)
        pinched = (n_covered == PINCH_COVERED_QUADRANTS) & (coverage[:, 0] == coverage[:, 2])
        if pinched.any():
            raise PolyominoError(
                f"cells meet only at vertex {vertices[pinched].tolist()} (a pinch): the domain "
                f"is not Lipschitz there and has no well-defined corner angle"
            )

    @staticmethod
    def _find_reentrant_corners(
        vertices: NDArray[np.float64], coverage: NDArray[np.bool_]
    ) -> tuple[PolyominoCorner, ...]:
        """Vertices with three covered quadrants; the fourth is the exterior wedge."""
        corners: list[PolyominoCorner] = []
        for vertex, covered in zip(vertices, coverage, strict=True):
            if int(covered.sum()) != REENTRANT_COVERED_QUADRANTS:
                continue
            exterior = QUADRANTS[int(np.flatnonzero(~covered)[0])]
            corners.append(
                PolyominoCorner(
                    position=(float(vertex[0]), float(vertex[1])), exterior_quadrant=exterior
                )
            )
        return tuple(corners)

    def _find_boundary_segments(self) -> tuple[BoundarySegment, ...]:
        """Cell edges whose midpoint is not interior, merged into maximal segments.

        Edge-to-edge contact (validated) means an edge is either shared whole,
        so its midpoint is interior, or not shared at all. Pinch-freedom means
        two abutting collinear boundary edges bound the domain on the same side,
        so merging them is geometrically a single straight edge.
        """
        horizontal: list[tuple[float, float, float]] = []
        vertical: list[tuple[float, float, float]] = []
        for cell in self._cells:
            horizontal.extend((y, cell.x0, cell.x1) for y in (cell.y0, cell.y1))
            vertical.extend((x, cell.y0, cell.y1) for x in (cell.x0, cell.x1))
        horizontal_mid = np.array([[(a + b) / 2, line] for line, a, b in horizontal])
        vertical_mid = np.array([[line, (a + b) / 2] for line, a, b in vertical])
        keep_h = ~self.interior_mask(horizontal_mid)
        keep_v = ~self.interior_mask(vertical_mid)
        segments = [
            BoundarySegment(start=(a, line), end=(b, line))
            for line, a, b in _merge_collinear(
                [e for e, k in zip(horizontal, keep_h, strict=True) if k]
            )
        ]
        segments.extend(
            BoundarySegment(start=(line, a), end=(line, b))
            for line, a, b in _merge_collinear(
                [e for e, k in zip(vertical, keep_v, strict=True) if k]
            )
        )
        return tuple(segments)


def _slab_interval(start: float, step: float, lo: float, hi: float) -> tuple[float, float]:
    """The ``t`` interval on which ``start + t * step`` lies in ``[lo, hi]``.

    An empty interval is returned as ``(inf, -inf)`` so intersecting it with
    anything stays empty.
    """
    if step == 0:
        return (-math.inf, math.inf) if lo <= start <= hi else (math.inf, -math.inf)
    t_a = (lo - start) / step
    t_b = (hi - start) / step
    return min(t_a, t_b), max(t_a, t_b)


def _require_positive_count(n_points: int) -> None:
    """Sampling zero or a negative number of points is a caller error, not an empty tensor."""
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")


def _on_device(points: Tensor, device: torch.device | None) -> Tensor:
    """Move ``points`` to ``device`` (``None`` keeps the CPU tensor the sampler built)."""
    return points if device is None else points.to(device)


def polyomino_domain_of(operator: object) -> PolyominoDomain | None:
    """The operator's :class:`PolyominoDomain`, or ``None`` if its geometry is anything else.

    The single dispatch rule both refinement substrates use to decide whether an
    operator needs the polyomino path, so the two cannot disagree about which
    operators carry a polyomino.
    """
    geometry = getattr(operator, "geometry", None)
    return geometry if isinstance(geometry, PolyominoDomain) else None


# ----------------------------------------------------------------------
# Presets. Cell order is load-bearing: mesh builders join cells in order.
# ----------------------------------------------------------------------

#: The canonical L-shape ``[-1,1]^2 \ [0,1]x[-1,0]`` as three unit cells, in the
#: order ``src.research.fem_baseline.build_lshaped_initial_mesh`` joins its
#: squares (``nw + ne + sw``). That order is what lets a polyomino mesh of this
#: preset reproduce the L-shape mesh byte for byte.
LSHAPE_CELL_NW: Final[PolyominoCell] = PolyominoCell(x0=-1.0, x1=0.0, y0=0.0, y1=1.0)
LSHAPE_CELL_NE: Final[PolyominoCell] = PolyominoCell(x0=0.0, x1=1.0, y0=0.0, y1=1.0)
LSHAPE_CELL_SW: Final[PolyominoCell] = PolyominoCell(x0=-1.0, x1=0.0, y0=-1.0, y1=0.0)
LSHAPE_POLYOMINO_CELLS: Final[tuple[PolyominoCell, ...]] = (
    LSHAPE_CELL_NW,
    LSHAPE_CELL_NE,
    LSHAPE_CELL_SW,
)

#: The Z-tetromino: A = [-1,0]x[0,1], B = [0,1]x[0,1], C = [0,1]x[-1,0],
#: D = [1,2]x[-1,0]. Two reentrant corners: (0, 0), exterior wedge x<0, y<0;
#: and (1, 0), exterior wedge x>1, y>0.
ZSHAPE_CELL_A: Final[PolyominoCell] = PolyominoCell(x0=-1.0, x1=0.0, y0=0.0, y1=1.0)
ZSHAPE_CELL_B: Final[PolyominoCell] = PolyominoCell(x0=0.0, x1=1.0, y0=0.0, y1=1.0)
ZSHAPE_CELL_C: Final[PolyominoCell] = PolyominoCell(x0=0.0, x1=1.0, y0=-1.0, y1=0.0)
ZSHAPE_CELL_D: Final[PolyominoCell] = PolyominoCell(x0=1.0, x1=2.0, y0=-1.0, y1=0.0)
ZSHAPE_POLYOMINO_CELLS: Final[tuple[PolyominoCell, ...]] = (
    ZSHAPE_CELL_A,
    ZSHAPE_CELL_B,
    ZSHAPE_CELL_C,
    ZSHAPE_CELL_D,
)

__all__ = [
    "LSHAPE_POLYOMINO_CELLS",
    "POLYOMINO_REENTRANT_INTERIOR_ANGLE",
    "QUADRANTS",
    "ZSHAPE_POLYOMINO_CELLS",
    "BoundarySegment",
    "PolyominoCell",
    "PolyominoCorner",
    "PolyominoDomain",
    "PolyominoError",
    "polyomino_domain_of",
]
