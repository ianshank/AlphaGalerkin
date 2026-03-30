"""Domain geometry abstractions for PDE solving.

Provides geometry primitives for rectangular and non-rectangular domains,
enabling MCTS-guided mesh refinement on complex geometries.

Supported geometries:
- Rectangular: Standard [x_min, x_max] x [y_min, y_max] domains.
- L-shaped: [-1,1]^2 minus [0,1]x[-1,0], the classic adaptive mesh benchmark.
- Cylinder flow: Rectangular domain with circular obstacle (DFG benchmark).

Each geometry supports:
- Interior/boundary point containment checks
- Efficient sampling (rejection sampling for non-convex domains)
- Bounding box queries for spatial data structures
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import structlog
import torch
from pydantic import BaseModel, Field
from torch import Tensor

logger = structlog.get_logger(__name__)


class GeometryType(str, Enum):
    """Supported domain geometry types."""

    RECTANGULAR = "rectangular"
    L_SHAPED = "l_shaped"
    CYLINDER_FLOW = "cylinder_flow"


class GeometryConfig(BaseModel):
    """Configuration for domain geometry.

    This config drives the ``create_geometry`` factory. Parameters that
    are irrelevant to the chosen ``geometry_type`` are simply ignored.
    """

    geometry_type: GeometryType = Field(
        default=GeometryType.RECTANGULAR,
        description="Type of domain geometry",
    )

    # Rectangular domain params
    x_min: float = Field(default=-1.0, description="Minimum x coordinate")
    x_max: float = Field(default=1.0, description="Maximum x coordinate")
    y_min: float = Field(default=-1.0, description="Minimum y coordinate")
    y_max: float = Field(default=1.0, description="Maximum y coordinate")

    # L-shaped domain params
    scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Scale factor for L-shaped domain",
    )

    # Cylinder flow domain params
    cylinder_cx: float = Field(default=0.2, description="Cylinder center x")
    cylinder_cy: float = Field(default=0.2, description="Cylinder center y")
    cylinder_radius: float = Field(
        default=0.05,
        gt=0.0,
        description="Cylinder radius",
    )


class DomainGeometry(ABC):
    """Abstract base class for domain geometries."""

    @abstractmethod
    def contains_point(self, points: Tensor) -> Tensor:
        """Check if points are inside the domain.

        Args:
            points: Tensor of shape (N, dim) with coordinates.

        Returns:
            Boolean tensor of shape (N,).

        """
        ...

    @abstractmethod
    def is_boundary(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        """Check if points are on the domain boundary.

        Args:
            points: Tensor of shape (N, dim).
            tol: Tolerance for boundary detection.

        Returns:
            Boolean tensor of shape (N,).

        """
        ...

    @abstractmethod
    def sample_interior(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample random points from the domain interior.

        Args:
            n_points: Number of points to sample.
            device: Torch device.

        Returns:
            Tensor of shape (n_points, dim).

        """
        ...

    @abstractmethod
    def sample_boundary(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample random points from the domain boundary.

        Args:
            n_points: Number of boundary points.
            device: Torch device.

        Returns:
            Tensor of shape (n_points, dim).

        """
        ...

    @abstractmethod
    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return axis-aligned bounding box as (min_coords, max_coords)."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension of the domain."""
        ...

    @property
    @abstractmethod
    def area(self) -> float:
        """Area/volume of the domain."""
        ...


class RectangularDomain(DomainGeometry):
    """Standard rectangular domain [x_min, x_max] x [y_min, y_max]."""

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        y_min: float = 0.0,
        y_max: float = 1.0,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        logger.debug(
            "rectangular_domain_created",
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )

    @property
    def dim(self) -> int:
        """Spatial dimension (always 2)."""
        return 2

    @property
    def area(self) -> float:
        """Area of the rectangle."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return axis-aligned bounding box."""
        return (self.x_min, self.y_min), (self.x_max, self.y_max)

    def contains_point(self, points: Tensor) -> Tensor:
        """Check if points lie inside the rectangle."""
        x, y = points[:, 0], points[:, 1]
        return (
            (x >= self.x_min)
            & (x <= self.x_max)
            & (y >= self.y_min)
            & (y <= self.y_max)
        )

    def is_boundary(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        """Check if points are on the rectangular boundary."""
        x, y = points[:, 0], points[:, 1]
        inside = self.contains_point(points)

        on_x_min = torch.abs(x - self.x_min) < tol
        on_x_max = torch.abs(x - self.x_max) < tol
        on_y_min = torch.abs(y - self.y_min) < tol
        on_y_max = torch.abs(y - self.y_max) < tol

        on_edge = on_x_min | on_x_max | on_y_min | on_y_max
        return inside & on_edge

    def sample_interior(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample uniform random points from the rectangle interior."""
        x = torch.rand(n_points, device=device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n_points, device=device) * (self.y_max - self.y_min) + self.y_min
        return torch.stack([x, y], dim=-1)

    def sample_boundary(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample points on the rectangular boundary, proportional to edge length."""
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        perimeter = 2 * (w + h)

        # Number of points per edge, proportional to length
        n_bottom = max(1, round(n_points * w / perimeter))
        n_top = max(1, round(n_points * w / perimeter))
        n_left = max(1, round(n_points * h / perimeter))
        n_right = n_points - n_bottom - n_top - n_left
        n_right = max(1, n_right)

        segments: list[Tensor] = []

        # Bottom edge: y = y_min
        t = torch.rand(n_bottom, device=device)
        segments.append(
            torch.stack([self.x_min + t * w, torch.full_like(t, self.y_min)], dim=-1)
        )

        # Top edge: y = y_max
        t = torch.rand(n_top, device=device)
        segments.append(
            torch.stack([self.x_min + t * w, torch.full_like(t, self.y_max)], dim=-1)
        )

        # Left edge: x = x_min
        t = torch.rand(n_left, device=device)
        segments.append(
            torch.stack([torch.full_like(t, self.x_min), self.y_min + t * h], dim=-1)
        )

        # Right edge: x = x_max
        t = torch.rand(n_right, device=device)
        segments.append(
            torch.stack([torch.full_like(t, self.x_max), self.y_min + t * h], dim=-1)
        )

        return torch.cat(segments, dim=0)[:n_points]


class LShapedDomain(DomainGeometry):
    """L-shaped domain: [-1,1]^2 minus [0,1]x[-1,0] (scaled).

    This is the standard benchmark domain for adaptive mesh refinement.
    The reentrant corner at the origin creates a singularity in the
    solution gradient: u ~ r^(2/3) * sin(2*theta/3).

    The L-shape consists of the full unit square [-s,s]^2 with the
    bottom-right quadrant [0,s]x[-s,0] removed, where s = scale.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self._s = scale
        # L-shape = [-s,s]^2 \ [0,s]x[-s,0]
        # Area = 4*s^2 - s^2 = 3*s^2
        logger.debug("l_shaped_domain_created", scale=scale)

    @property
    def dim(self) -> int:
        """Spatial dimension (always 2)."""
        return 2

    @property
    def area(self) -> float:
        """Area of the L-shaped domain (3/4 of the bounding square)."""
        return 3.0 * self._s * self._s

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return bounding box of the L-shape."""
        return (-self._s, -self._s), (self._s, self._s)

    def _in_removed_quadrant(self, x: Tensor, y: Tensor) -> Tensor:
        """Check if (x, y) falls in the removed bottom-right quadrant."""
        return (x > 0) & (y < 0)

    def contains_point(self, points: Tensor) -> Tensor:
        """Check if points are inside the L-shaped domain.

        A point is inside if it is in [-s,s]^2 AND NOT in (0,s]x[-s,0).
        """
        x, y = points[:, 0], points[:, 1]
        s = self._s
        in_square = (x >= -s) & (x <= s) & (y >= -s) & (y <= s)
        in_removed = (x > 0) & (y < 0)
        return in_square & ~in_removed

    def is_boundary(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        """Check if points lie on the L-shaped domain boundary.

        The boundary consists of 6 segments:
        1. Bottom:  y = -s, x in [-s, 0]
        2. Left:    x = -s, y in [-s, s]
        3. Top:     y = s,  x in [-s, s]
        4. Right:   x = s,  y in [0, s]
        5. Reentrant horizontal: y = 0, x in [0, s]
        6. Reentrant vertical:   x = 0, y in [-s, 0]
        """
        x, y = points[:, 0], points[:, 1]
        s = self._s

        # Segment 1: bottom edge y=-s, x in [-s, 0]
        seg1 = (torch.abs(y + s) < tol) & (x >= -s - tol) & (x <= tol)

        # Segment 2: left edge x=-s, y in [-s, s]
        seg2 = (torch.abs(x + s) < tol) & (y >= -s - tol) & (y <= s + tol)

        # Segment 3: top edge y=s, x in [-s, s]
        seg3 = (torch.abs(y - s) < tol) & (x >= -s - tol) & (x <= s + tol)

        # Segment 4: right edge x=s, y in [0, s]
        seg4 = (torch.abs(x - s) < tol) & (y >= -tol) & (y <= s + tol)

        # Segment 5: reentrant horizontal y=0, x in [0, s]
        seg5 = (torch.abs(y) < tol) & (x >= -tol) & (x <= s + tol)

        # Segment 6: reentrant vertical x=0, y in [-s, 0]
        seg6 = (torch.abs(x) < tol) & (y >= -s - tol) & (y <= tol)

        return seg1 | seg2 | seg3 | seg4 | seg5 | seg6

    def sample_interior(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample interior points via rejection sampling from bounding box.

        The acceptance rate is 3/4 (L-area / square-area), so we oversample
        by a factor of ~1.5 and iterate until we have enough points.
        """
        s = self._s
        collected: list[Tensor] = []
        n_remaining = n_points
        # Oversample factor accounts for 75% acceptance rate with margin
        oversample = 1.5

        while n_remaining > 0:
            n_candidates = int(n_remaining * oversample) + 64
            candidates = torch.rand(n_candidates, 2, device=device)
            # Map from [0,1] to [-s, s]
            candidates = candidates * (2 * s) - s

            mask = self.contains_point(candidates)
            # Also exclude boundary points for strict interior
            accepted = candidates[mask]

            if len(accepted) >= n_remaining:
                collected.append(accepted[:n_remaining])
                n_remaining = 0
            else:
                collected.append(accepted)
                n_remaining -= len(accepted)
                oversample *= 1.5  # increase if unlucky

        return torch.cat(collected, dim=0)

    def sample_boundary(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample boundary points proportionally to segment length.

        The 6 boundary segments have the following lengths (scale=1):
        1. Bottom:  s   (length s, from -s to 0)
        2. Left:    2s  (from -s to s)
        3. Top:     2s  (from -s to s)
        4. Right:   s   (from 0 to s)
        5. Reentrant horiz: s  (from 0 to s)
        6. Reentrant vert:  s  (from -s to 0)
        Total perimeter = 8s.
        """
        s = self._s
        lengths = [s, 2 * s, 2 * s, s, s, s]
        total = sum(lengths)  # 8s

        # Distribute points proportionally
        counts: list[int] = []
        for i, length in enumerate(lengths):
            if i < len(lengths) - 1:
                counts.append(max(1, round(n_points * length / total)))
            else:
                counts.append(max(1, n_points - sum(counts)))

        segments: list[Tensor] = []

        # Segment 1: bottom y=-s, x in [-s, 0]
        t = torch.rand(counts[0], device=device)
        segments.append(
            torch.stack([-s + t * s, torch.full_like(t, -s)], dim=-1)
        )

        # Segment 2: left x=-s, y in [-s, s]
        t = torch.rand(counts[1], device=device)
        segments.append(
            torch.stack([torch.full_like(t, -s), -s + t * 2 * s], dim=-1)
        )

        # Segment 3: top y=s, x in [-s, s]
        t = torch.rand(counts[2], device=device)
        segments.append(
            torch.stack([-s + t * 2 * s, torch.full_like(t, s)], dim=-1)
        )

        # Segment 4: right x=s, y in [0, s]
        t = torch.rand(counts[3], device=device)
        segments.append(
            torch.stack([torch.full_like(t, s), t * s], dim=-1)
        )

        # Segment 5: reentrant horizontal y=0, x in [0, s]
        t = torch.rand(counts[4], device=device)
        segments.append(
            torch.stack([t * s, torch.full_like(t, 0.0)], dim=-1)
        )

        # Segment 6: reentrant vertical x=0, y in [-s, 0]
        t = torch.rand(counts[5], device=device)
        segments.append(
            torch.stack([torch.full_like(t, 0.0), -s + t * s], dim=-1)
        )

        return torch.cat(segments, dim=0)[:n_points]


class CylinderFlowDomain(DomainGeometry):
    """Rectangular domain with circular hole for cylinder flow problems.

    Based on the DFG benchmark (Schafer & Turek):
    - Channel: [0, 2.2] x [0, 0.41]
    - Cylinder center: (0.2, 0.2), radius: 0.05
    """

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 2.2,
        y_min: float = 0.0,
        y_max: float = 0.41,
        cx: float = 0.2,
        cy: float = 0.2,
        radius: float = 0.05,
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cx = cx
        self.cy = cy
        self.radius = radius
        logger.debug(
            "cylinder_flow_domain_created",
            channel=(x_min, y_min, x_max, y_max),
            cylinder=(cx, cy, radius),
        )

    @property
    def dim(self) -> int:
        """Spatial dimension (always 2)."""
        return 2

    @property
    def area(self) -> float:
        """Area of the channel minus the cylinder."""
        rect = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        circle = np.pi * self.radius**2
        return rect - circle

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return bounding box of the rectangular channel."""
        return (self.x_min, self.y_min), (self.x_max, self.y_max)

    def _dist_to_cylinder(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute distance from points to cylinder center."""
        return torch.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)

    def contains_point(self, points: Tensor) -> Tensor:
        """Check if points are inside channel and outside cylinder."""
        x, y = points[:, 0], points[:, 1]
        in_rect = (
            (x >= self.x_min)
            & (x <= self.x_max)
            & (y >= self.y_min)
            & (y <= self.y_max)
        )
        dist = self._dist_to_cylinder(x, y)
        outside_cyl = dist > self.radius
        return in_rect & outside_cyl

    def is_boundary(self, points: Tensor, tol: float = 1e-6) -> Tensor:
        """Check if points are on channel walls or cylinder surface."""
        x, y = points[:, 0], points[:, 1]

        # Channel walls
        on_left = torch.abs(x - self.x_min) < tol
        on_right = torch.abs(x - self.x_max) < tol
        on_bottom = torch.abs(y - self.y_min) < tol
        on_top = torch.abs(y - self.y_max) < tol
        on_channel = on_left | on_right | on_bottom | on_top

        # Cylinder surface
        dist = self._dist_to_cylinder(x, y)
        on_cylinder = torch.abs(dist - self.radius) < tol

        # Must also be within bounding box for channel boundaries
        in_rect = (
            (x >= self.x_min - tol)
            & (x <= self.x_max + tol)
            & (y >= self.y_min - tol)
            & (y <= self.y_max + tol)
        )

        return (on_channel & in_rect) | on_cylinder

    def sample_interior(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample interior points via rejection sampling."""
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        # Acceptance rate is approximately (rect_area - circle_area) / rect_area
        # which is very close to 1 for the DFG benchmark
        oversample = 1.1
        collected: list[Tensor] = []
        n_remaining = n_points

        while n_remaining > 0:
            n_candidates = int(n_remaining * oversample) + 64
            x = torch.rand(n_candidates, device=device) * w + self.x_min
            y = torch.rand(n_candidates, device=device) * h + self.y_min
            candidates = torch.stack([x, y], dim=-1)

            mask = self.contains_point(candidates)
            accepted = candidates[mask]

            if len(accepted) >= n_remaining:
                collected.append(accepted[:n_remaining])
                n_remaining = 0
            else:
                collected.append(accepted)
                n_remaining -= len(accepted)
                oversample *= 1.2

        return torch.cat(collected, dim=0)

    def sample_boundary(
        self, n_points: int, device: torch.device | None = None
    ) -> Tensor:
        """Sample boundary points from channel walls and cylinder surface."""
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min
        cyl_perim = 2 * np.pi * self.radius
        rect_perim = 2 * (w + h)
        total = rect_perim + cyl_perim

        # Allocate points proportionally
        n_rect = max(4, round(n_points * rect_perim / total))
        n_cyl = max(1, n_points - n_rect)

        segments: list[Tensor] = []

        # Channel walls (distribute proportionally)
        n_bottom = max(1, round(n_rect * w / rect_perim))
        n_top = max(1, round(n_rect * w / rect_perim))
        n_left = max(1, round(n_rect * h / rect_perim))
        n_right = max(1, n_rect - n_bottom - n_top - n_left)

        # Bottom
        t = torch.rand(n_bottom, device=device)
        segments.append(
            torch.stack(
                [self.x_min + t * w, torch.full_like(t, self.y_min)], dim=-1
            )
        )

        # Top
        t = torch.rand(n_top, device=device)
        segments.append(
            torch.stack(
                [self.x_min + t * w, torch.full_like(t, self.y_max)], dim=-1
            )
        )

        # Left
        t = torch.rand(n_left, device=device)
        segments.append(
            torch.stack(
                [torch.full_like(t, self.x_min), self.y_min + t * h], dim=-1
            )
        )

        # Right
        t = torch.rand(n_right, device=device)
        segments.append(
            torch.stack(
                [torch.full_like(t, self.x_max), self.y_min + t * h], dim=-1
            )
        )

        # Cylinder surface
        theta = torch.rand(n_cyl, device=device) * 2 * np.pi
        cx = self.cx + self.radius * torch.cos(theta)
        cy = self.cy + self.radius * torch.sin(theta)
        segments.append(torch.stack([cx, cy], dim=-1))

        return torch.cat(segments, dim=0)[:n_points]


def create_geometry(config: GeometryConfig) -> DomainGeometry:
    """Factory to create domain geometry from config.

    Args:
        config: Geometry configuration specifying type and parameters.

    Returns:
        Concrete DomainGeometry instance.

    Raises:
        ValueError: If geometry_type is not supported.

    """
    if config.geometry_type == GeometryType.RECTANGULAR:
        return RectangularDomain(
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
        )
    elif config.geometry_type == GeometryType.L_SHAPED:
        return LShapedDomain(scale=config.scale)
    elif config.geometry_type == GeometryType.CYLINDER_FLOW:
        return CylinderFlowDomain(
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,
            y_max=config.y_max,
            cx=config.cylinder_cx,
            cy=config.cylinder_cy,
            radius=config.cylinder_radius,
        )
    else:
        raise ValueError(f"Unsupported geometry type: {config.geometry_type}")
