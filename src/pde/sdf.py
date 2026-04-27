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

# Default number of Newton refinement steps applied after the bisection /
# grid-search fallback selects a better initial parameter. Kept small —
# the grid bracket is already coarse enough that convergence is fast.
DEFAULT_FALLBACK_NEWTON_REFINE_ITERS = 4

# Default base coverage for the grid-search fallback (multiplier on
# ``n_turns`` plus a constant). Resolves to ``4 * n_turns + 1`` so each
# helical revolution gets at least four candidate parameters before
# Newton refines the winner.
DEFAULT_FALLBACK_GRID_PER_TURN = 4
DEFAULT_FALLBACK_GRID_BASELINE = 1


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
        enable_fallback: bool = True,
        fallback_grid_size: int | None = None,
        fallback_newton_refine_iters: int = DEFAULT_FALLBACK_NEWTON_REFINE_ITERS,
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
        if fallback_newton_refine_iters < 0:
            raise ValueError(
                f"fallback_newton_refine_iters must be >= 0, got {fallback_newton_refine_iters}"
            )
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
        self.enable_fallback = bool(enable_fallback)
        # Resolve the grid size: an explicit override wins; otherwise the
        # default scales linearly with n_turns so each revolution gets
        # enough candidate parameters to bracket the global optimum.
        resolved_grid = (
            fallback_grid_size
            if fallback_grid_size is not None
            else DEFAULT_FALLBACK_GRID_PER_TURN * self.n_turns + DEFAULT_FALLBACK_GRID_BASELINE
        )
        if resolved_grid < 2:
            raise ValueError(f"fallback_grid_size must be >= 2, got {resolved_grid}")
        self.fallback_grid_size = int(resolved_grid)
        self.fallback_newton_refine_iters = int(fallback_newton_refine_iters)

        logger.debug(
            "analytical_helix_sdf_created",
            R_major=R_major,
            r_minor=r_minor,
            pitch=pitch,
            n_turns=n_turns,
            newton_max_iters=newton_max_iters,
            newton_deriv_tol=newton_deriv_tol,
            enable_fallback=self.enable_fallback,
            fallback_grid_size=self.fallback_grid_size,
            fallback_newton_refine_iters=self.fallback_newton_refine_iters,
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

    def _newton_residual(self, points: Tensor, t: Tensor) -> Tensor:
        """Squared-distance derivative ``f(t) = -<p - c(t), c'(t)>``.

        At a true nearest-parameter ``t*(p)`` this residual is zero. Used
        both as the Newton stopping criterion and to identify points that
        need the bisection / grid-search fallback.
        """
        c = self._centerline(t)
        dc = self._centerline_tangent(t)
        return -((points - c) * dc).sum(dim=-1)

    def _newton_refine(self, points: Tensor, t: Tensor, n_iters: int) -> Tensor:
        """Run ``n_iters`` clamped Newton steps on the squared-distance derivative.

        Pure helper so both the primary loop and the fallback share one
        well-tested update rule.
        """
        t_min = 0.0
        t_max = float(self.n_turns)
        for _ in range(n_iters):
            c = self._centerline(t)
            dc = self._centerline_tangent(t)
            ddc = self._centerline_curvature(t)
            diff = points - c
            f = -(diff * dc).sum(dim=-1)
            fprime = (dc * dc).sum(dim=-1) - (diff * ddc).sum(dim=-1)
            safe_fprime = torch.where(
                fprime.abs() > self.newton_deriv_tol,
                fprime,
                torch.full_like(fprime, self.newton_deriv_tol),
            )
            step = f / safe_fprime
            t = torch.clamp(t - step, min=t_min, max=t_max)
        return t

    def _grid_fallback(self, points: Tensor, t_init: Tensor) -> Tensor:
        """Bisection-style fallback for points where Newton has not converged.

        The squared-distance ``||p - c(t)||^2`` is multimodal along the
        helix (one local minimum per revolution), so blind sign-change
        bisection on the derivative cannot guarantee the *global* minimum.
        Instead we evaluate the squared distance on a coarse uniform grid
        of ``[0, n_turns]``, pick the bracketing minimum per point, and
        run a few more Newton steps from there. This is the standard
        global-search-then-refine pattern and recovers the headline
        accuracy on adversarial points (thin tubes, pathological initial
        guesses) where pure Newton stalls.

        Only the subset of points whose Newton residual exceeds
        ``newton_deriv_tol`` participates so the cost is proportional to
        the failure rate rather than the batch size.
        """
        residual = self._newton_residual(points, t_init)
        needs_fallback = residual.abs() > self.newton_deriv_tol
        n_fallback = int(needs_fallback.sum().item())
        if n_fallback == 0:
            return t_init

        logger.debug(
            "analytical_helix_sdf_fallback",
            n_points=int(points.shape[0]),
            n_fallback=n_fallback,
            grid_size=self.fallback_grid_size,
        )

        pts_sub = points[needs_fallback]  # (M, 3)
        grid = torch.linspace(
            0.0,
            float(self.n_turns),
            steps=self.fallback_grid_size,
            device=points.device,
            dtype=points.dtype,
        )  # (G,)
        # ``_centerline`` accepts shape (G,) and returns (G, 3); broadcast
        # the squared-distance computation against the M failing points.
        centers = self._centerline(grid)  # (G, 3)
        dists_sq = ((pts_sub.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
        best = dists_sq.argmin(dim=-1)  # (M,)
        t_grid = grid[best]

        if self.fallback_newton_refine_iters > 0:
            t_grid = self._newton_refine(pts_sub, t_grid, self.fallback_newton_refine_iters)

        # Vectorized scatter-style update: indices where the mask is True
        # get the refined value, all others keep the original Newton result.
        t_out = t_init.clone()
        t_out[needs_fallback] = t_grid
        return t_out

    def _nearest_t(self, points: Tensor) -> Tensor:
        """Find the nearest centerline parameter ``t*(p)`` for each point.

        Solves ``f(t) = d/dt (1/2 ||p - c(t)||^2) = -(p - c(t)) . c'(t) = 0``
        via Newton's method, clamped to ``[0, n_turns]`` at each step so we
        never leave the parameterized centerline. Points that have not
        converged within ``newton_deriv_tol`` after ``newton_max_iters``
        are routed through ``_grid_fallback``.

        Args:
        ----
            points: Shape ``(N, 3)``.

        Returns:
        -------
            Shape ``(N,)``.

        """
        # Initial guess from the z-coordinate of the point.
        t = torch.clamp(points[:, 2] / self.pitch, min=0.0, max=float(self.n_turns))

        t = self._newton_refine(points, t, self.newton_max_iters)

        if self.enable_fallback:
            t = self._grid_fallback(points, t)

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
            import PicoGK  # noqa: F401  # pragma: no cover - optional dep
            import pythonnet  # noqa: F401  # pragma: no cover - optional dep
        except ImportError as exc:
            raise ImportError(
                "PicoGKSDFEvaluator requires the optional [picogk] extra. "
                "Install with: pip install alphagalerkin[picogk]"
            ) from exc
        # Real integration would load voxel_path here and cache a signed
        # distance grid. That work is deferred to the PicoGK integration
        # milestone; v1 of this research demo runs entirely on
        # AnalyticalHelixSDF.
        raise NotImplementedError(  # pragma: no cover - optional dep
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
