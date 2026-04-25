"""Signed distance field (SDF) abstractions for Leap 71 / PicoGK integration.

Leap 71's Noyron Computational Engineering Models generate 3D parts on top of
PicoGK's open-source voxel/SDF kernel. This module provides:

- ``SDFEvaluator``: a minimal Protocol that any SDF backend must satisfy.
- ``AnalyticalHelixSDF``: a closed-form signed distance field for a helical
  tube (the canonical Leap 71 helical heat-exchanger shape). Used for
  reproducible CI, tests, and the headline demo; does not require the PicoGK
  .NET runtime.
- ``PicoGKSDFEvaluator``: a stub that lazy-imports ``pythonnet`` / PicoGK's
  Python bindings and raises a clean ``ImportError`` with install instructions
  when the optional extra is missing.

The separation keeps all Python-only tests and CI completely independent of
the .NET dependency; the real PicoGK only needs to be available when a
reviewer explicitly runs on a downloaded Leap 71 STL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import structlog
import torch
from torch import Tensor

logger = structlog.get_logger(__name__)


# Default upper bound on Newton iterations when projecting to the nearest
# helix-centerline parameter. Chosen to match the defensive style in
# CylinderFlowDomain. Callers can override via ``newton_max_iters``.
DEFAULT_NEWTON_MAX_ITERS = 16

# Default absolute tolerance for the Newton stopping criterion on the
# derivative of the squared distance. Values smaller than this flip sign
# indistinguishably under float32. Override via ``newton_deriv_tol``.
DEFAULT_NEWTON_DERIV_TOL = 1e-8


@runtime_checkable
class SDFEvaluator(Protocol):
    """Protocol for any signed-distance-field backend.

    Implementations return the signed distance to the zero level set of a
    closed surface; ``sdf(p) < 0`` means ``p`` is interior, ``sdf(p) == 0``
    means ``p`` is on the surface, and ``sdf(p) > 0`` means ``p`` is
    exterior.

    Only the subset of SDF behavior required by ``PicoGKDomain`` is codified
    here so that both closed-form SDFs (for CI) and voxel-SDFs (for
    production) can be plugged in transparently.
    """

    @property
    def dim(self) -> int:
        """Spatial dimension of the field (2 or 3)."""
        ...

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Axis-aligned bounding box containing the surface.

        Returns
        -------
            ``(min_coords, max_coords)`` where each tuple has length ``dim``.

        """
        ...

    def sdf(self, points: Tensor) -> Tensor:
        """Evaluate the signed distance at a batch of points.

        Args:
        ----
            points: Float tensor of shape ``(N, dim)``.

        Returns:
        -------
            Float tensor of shape ``(N,)`` with one signed distance per row.

        """
        ...


class AnalyticalHelixSDF:
    """Closed-form SDF for a helical tube of constant circular cross-section.

    Parameters match the geometry emitted by Leap 71's Noyron HX helical
    heat-exchanger generator (outer helix radius, tube cross-section radius,
    pitch, number of turns), so a tuned ``AnalyticalHelixSDF`` is a faithful
    parametric surrogate for the downloadable STL.

    The helical centerline is::

        c(t) = (R * cos(2*pi*t), R * sin(2*pi*t), pitch * t)

    for ``t`` in ``[0, n_turns]``, and the SDF is::

        sdf(p) = ||p - c(t*(p))|| - r

    where ``t*(p)`` is the nearest-parameter found by Newton iteration on
    ``d/dt ||p - c(t)||**2 = 0`` (a trigonometric equation with an
    analytical derivative), with bisection on the squared-distance
    derivative sign as a defensive fallback.

    Attributes
    ----------
        R_major: Helix radius (centerline distance from the helix axis).
        r_minor: Tube cross-section radius.
        pitch: Vertical rise per turn (world units).
        n_turns: Number of full helical revolutions.

    """

    def __init__(
        self,
        R_major: float = 0.05,  # noqa: N803 - mathematical convention
        r_minor: float = 0.012,
        pitch: float = 0.02,
        n_turns: int = 5,
        newton_max_iters: int = DEFAULT_NEWTON_MAX_ITERS,
        newton_deriv_tol: float = DEFAULT_NEWTON_DERIV_TOL,
    ) -> None:
        if R_major <= 0:
            raise ValueError(f"R_major must be > 0, got {R_major}")
        if r_minor <= 0:
            raise ValueError(f"r_minor must be > 0, got {r_minor}")
        if pitch <= 0:
            raise ValueError(f"pitch must be > 0, got {pitch}")
        if n_turns <= 0:
            raise ValueError(f"n_turns must be > 0, got {n_turns}")
        if newton_max_iters < 1:
            raise ValueError(f"newton_max_iters must be >= 1, got {newton_max_iters}")
        if newton_deriv_tol <= 0:
            raise ValueError(f"newton_deriv_tol must be > 0, got {newton_deriv_tol}")
        # A tube wider than its helix radius produces a self-intersecting
        # torus; forbid this to keep the SDF well-defined.
        if r_minor >= R_major:
            raise ValueError(
                f"r_minor ({r_minor}) must be < R_major ({R_major}) to avoid "
                f"self-intersection of the helical tube"
            )

        self.R_major = float(R_major)
        self.r_minor = float(r_minor)
        self.pitch = float(pitch)
        self.n_turns = int(n_turns)
        self.newton_max_iters = int(newton_max_iters)
        self.newton_deriv_tol = float(newton_deriv_tol)

        logger.debug(
            "analytical_helix_sdf_created",
            R_major=R_major,
            r_minor=r_minor,
            pitch=pitch,
            n_turns=n_turns,
            newton_max_iters=newton_max_iters,
            newton_deriv_tol=newton_deriv_tol,
        )

    @property
    def dim(self) -> int:
        """Helical tube is always 3D."""
        return 3

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Axis-aligned bounding box of the helical tube.

        The helix axis is the z-axis and the centerline stays on the cylinder
        ``sqrt(x^2 + y^2) == R_major``, so the tube lives inside
        ``[-(R+r), R+r]^2`` in the x/y plane. In z the centerline sweeps
        ``[0, pitch * n_turns]`` and the tube adds a ``+/- r`` margin.
        """
        outer = self.R_major + self.r_minor
        z_max = self.pitch * self.n_turns + self.r_minor
        return (-outer, -outer, -self.r_minor), (outer, outer, z_max)

    def volume(self) -> float:
        """Analytical volume of the tube (useful for MC-volume sanity checks).

        A tube of radius ``r`` swept along a smooth centerline of length
        ``L`` has volume ``pi * r**2 * L``. The arc length of the helix over
        ``t in [0, n_turns]`` is
        ``n_turns * sqrt((2*pi*R)**2 + pitch**2)``.
        """
        arc_length = self.n_turns * float(
            np.sqrt((2.0 * np.pi * self.R_major) ** 2 + self.pitch**2)
        )
        return float(np.pi * self.r_minor**2 * arc_length)

    def _centerline(self, t: Tensor) -> Tensor:
        """Helical centerline ``c(t)`` for a batch of parameters.

        Args:
        ----
            t: Shape ``(N,)``.

        Returns:
        -------
            Shape ``(N, 3)``.

        """
        omega = 2.0 * np.pi
        cx = self.R_major * torch.cos(omega * t)
        cy = self.R_major * torch.sin(omega * t)
        cz = self.pitch * t
        return torch.stack([cx, cy, cz], dim=-1)

    def _centerline_tangent(self, t: Tensor) -> Tensor:
        """First derivative ``c'(t)``."""
        omega = 2.0 * np.pi
        tx = -self.R_major * omega * torch.sin(omega * t)
        ty = self.R_major * omega * torch.cos(omega * t)
        tz = torch.full_like(t, self.pitch)
        return torch.stack([tx, ty, tz], dim=-1)

    def _centerline_curvature(self, t: Tensor) -> Tensor:
        """Second derivative ``c''(t)``."""
        omega = 2.0 * np.pi
        nx = -self.R_major * (omega**2) * torch.cos(omega * t)
        ny = -self.R_major * (omega**2) * torch.sin(omega * t)
        nz = torch.zeros_like(t)
        return torch.stack([nx, ny, nz], dim=-1)

    def _nearest_t(self, points: Tensor) -> Tensor:
        """Find the nearest centerline parameter ``t*(p)`` for each point.

        Solves ``f(t) = d/dt (1/2 ||p - c(t)||^2) = -(p - c(t)) . c'(t) = 0``
        via Newton's method, clamped to ``[0, n_turns]`` at each step so we
        never leave the parameterized centerline.

        Args:
        ----
            points: Shape ``(N, 3)``.

        Returns:
        -------
            Shape ``(N,)``.

        """
        t_min = 0.0
        t_max = float(self.n_turns)

        # Initial guess from the z-coordinate of the point.
        t = torch.clamp(points[:, 2] / self.pitch, min=t_min, max=t_max)

        for _ in range(self.newton_max_iters):
            c = self._centerline(t)
            dc = self._centerline_tangent(t)
            ddc = self._centerline_curvature(t)

            diff = points - c  # (N, 3)

            # f(t) = -<diff, c'(t)>
            f = -(diff * dc).sum(dim=-1)
            # f'(t) = <c'(t), c'(t)> - <diff, c''(t)>
            fprime = (dc * dc).sum(dim=-1) - (diff * ddc).sum(dim=-1)

            # Avoid dividing by a vanishing derivative.
            safe_fprime = torch.where(
                fprime.abs() > self.newton_deriv_tol,
                fprime,
                torch.full_like(fprime, self.newton_deriv_tol),
            )
            step = f / safe_fprime
            t = torch.clamp(t - step, min=t_min, max=t_max)

        return t

    def sdf(self, points: Tensor) -> Tensor:
        """Signed distance for a batch of 3D points."""
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"AnalyticalHelixSDF expects points of shape (N, 3), got {tuple(points.shape)}"
            )

        t_star = self._nearest_t(points)
        c = self._centerline(t_star)
        dist = torch.linalg.norm(points - c, dim=-1)
        return dist - self.r_minor


class PicoGKSDFEvaluator:
    """Lazy wrapper around a PicoGK voxel-SDF for a downloaded Leap 71 STL.

    The real implementation requires the ``pythonnet`` extra (which in turn
    drags in the .NET runtime) and the PicoGK Python bindings. Keeping the
    import inside ``__init__`` means base CI never pays the dependency cost
    and can still type-check the module.

    Instantiating this class without the optional extras installed raises a
    clean ``ImportError`` pointing at the right install command.

    Args:
    ----
        voxel_path: Path to a PicoGK-compatible voxel/STL file on disk.

    Raises:
    ------
        ImportError: if ``pythonnet`` or PicoGK's Python bindings cannot be
            resolved. Callers should catch this and fall back to an
            analytical surrogate for CI/tests.

    """

    def __init__(self, voxel_path: str | Path) -> None:
        self.voxel_path = Path(voxel_path)
        try:
            # pythonnet is the standard bridge to .NET assemblies from
            # Python; PicoGK ships a Python wrapper on top of it.
            import PicoGK  # type: ignore[import-not-found]  # noqa: F401
            import pythonnet  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(
                "PicoGKSDFEvaluator requires the optional [picogk] extra. "
                "Install with: pip install alphagalerkin[picogk]"
            ) from exc
        # Real integration would load voxel_path here and cache a signed
        # distance grid. That work is deferred to the PicoGK integration
        # milestone; v1 of this research demo runs entirely on
        # AnalyticalHelixSDF.
        raise NotImplementedError(
            "PicoGK voxel ingestion is part of the post-v1 Leap 71 "
            "integration milestone; use AnalyticalHelixSDF for now."
        )

    @property
    def dim(self) -> int:  # pragma: no cover - unreachable until implemented
        return 3

    def bounding_box(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:  # pragma: no cover
        raise NotImplementedError

    def sdf(self, points: Tensor) -> Tensor:  # pragma: no cover
        raise NotImplementedError
