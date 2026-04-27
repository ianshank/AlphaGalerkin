"""PicoGK-backed domain geometry for Leap 71 Noyron integration.

This module wires an arbitrary ``SDFEvaluator`` (analytical or voxel-backed)
into the existing ``DomainGeometry`` ABC so every operator in
``src/pde/operators.py`` can solve on a PicoGK-generated part without code
changes.

Rejection sampling is used for ``sample_interior`` because Leap 71 parts are
typically non-convex (helical channels, branching lattices) with ill-shaped
bounding boxes. ``sample_boundary`` uses the same bounding-box draw followed
by a Newton projection along the SDF gradient, with the gradient itself
estimated by central differences so the adapter works with any SDF backend
that only exposes sign+distance (not analytical gradients).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.geometry import DomainGeometry

if TYPE_CHECKING:
    from src.pde.sdf import SDFEvaluator

logger = structlog.get_logger(__name__)


# Default step size for the central-difference SDF gradient. Chosen small
# enough to capture local curvature on typical Leap 71 parts (mm scale) but
# large enough to avoid float32 cancellation. Override via
# ``PicoGKDomain(grad_epsilon=...)``.
DEFAULT_GRAD_EPSILON = 1e-4

# Default upper bound on the oversample multiplier for rejection sampling
# before sampling fails loud. Prevents an unbounded loop on pathological
# SDFs. Override via ``PicoGKDomain(max_oversample=...)``.
DEFAULT_MAX_OVERSAMPLE = 256.0

# Default Newton projection iterations for sample_boundary.
DEFAULT_PROJECTION_MAX_ITERS = 16

# Default boundary tolerance used by ``is_boundary`` and the Newton
# projector. Mirrors the defensive style of CylinderFlowDomain.
DEFAULT_BOUNDARY_PROJECTION_TOL = 1e-5

# Default minimum gradient norm squared; below this the projector treats
# the gradient as numerically zero.
DEFAULT_MIN_GRAD_NORM_SQ = 1e-12

# Default number of bisection iterations used as a fallback after Newton.
# Each iteration halves the bracket, so 16 reduces a unit-scale bracket
# to ~1.5e-5 — well below the default boundary tolerance.
DEFAULT_BISECTION_MAX_ITERS = 16

# Default multiplier on |sdf(x)| used to stretch the initial bisection
# bracket beyond the Newton step length. Larger values give the bisection
# a better chance of straddling the zero level set on highly curved
# surfaces; smaller values keep the bracket close to the original point.
DEFAULT_BISECTION_BRACKET_FACTOR = 4.0


class PicoGKDomain(DomainGeometry):
    """Domain geometry backed by a signed distance field.

    Attributes
    ----------
        sdf_evaluator: any ``SDFEvaluator`` (analytical or PicoGK voxel).
        oversample_factor: initial multiplier for rejection sampling; grows
            adaptively if acceptance is low.
        boundary_tolerance: default tolerance used by ``is_boundary``.
        volume_samples: number of Monte-Carlo samples used to estimate the
            domain volume at construction time.

    """

    def __init__(
        self,
        sdf_evaluator: SDFEvaluator,
        oversample_factor: float = 50.0,
        boundary_tolerance: float = DEFAULT_BOUNDARY_PROJECTION_TOL,
        volume_samples: int = 8192,
        grad_epsilon: float = DEFAULT_GRAD_EPSILON,
        max_oversample: float = DEFAULT_MAX_OVERSAMPLE,
        projection_max_iters: int = DEFAULT_PROJECTION_MAX_ITERS,
        min_grad_norm_sq: float = DEFAULT_MIN_GRAD_NORM_SQ,
        enable_bisection_fallback: bool = True,
        bisection_max_iters: int = DEFAULT_BISECTION_MAX_ITERS,
        bisection_bracket_factor: float = DEFAULT_BISECTION_BRACKET_FACTOR,
    ) -> None:
        if oversample_factor <= 1.0:
            raise ValueError(f"oversample_factor must be > 1.0, got {oversample_factor}")
        if boundary_tolerance <= 0.0:
            raise ValueError(f"boundary_tolerance must be > 0, got {boundary_tolerance}")
        if volume_samples < 1:
            raise ValueError(f"volume_samples must be >= 1, got {volume_samples}")
        if grad_epsilon <= 0.0:
            raise ValueError(f"grad_epsilon must be > 0, got {grad_epsilon}")
        if max_oversample <= oversample_factor:
            raise ValueError(
                f"max_oversample ({max_oversample}) must be > "
                f"oversample_factor ({oversample_factor})"
            )
        if projection_max_iters < 1:
            raise ValueError(f"projection_max_iters must be >= 1, got {projection_max_iters}")
        if min_grad_norm_sq <= 0.0:
            raise ValueError(f"min_grad_norm_sq must be > 0, got {min_grad_norm_sq}")
        if bisection_max_iters < 1:
            raise ValueError(f"bisection_max_iters must be >= 1, got {bisection_max_iters}")
        if bisection_bracket_factor <= 0.0:
            raise ValueError(
                f"bisection_bracket_factor must be > 0, got {bisection_bracket_factor}"
            )

        self.sdf_evaluator = sdf_evaluator
        self.oversample_factor = float(oversample_factor)
        self.boundary_tolerance = float(boundary_tolerance)
        self.grad_epsilon = float(grad_epsilon)
        self.max_oversample = float(max_oversample)
        self.projection_max_iters = int(projection_max_iters)
        self.min_grad_norm_sq = float(min_grad_norm_sq)
        self.enable_bisection_fallback = bool(enable_bisection_fallback)
        self.bisection_max_iters = int(bisection_max_iters)
        self.bisection_bracket_factor = float(bisection_bracket_factor)

        (mins, maxs) = sdf_evaluator.bounding_box()
        if len(mins) != len(maxs):
            raise ValueError(f"bounding_box min/max length mismatch: {len(mins)} vs {len(maxs)}")
        for i, (lo, hi) in enumerate(zip(mins, maxs, strict=True)):
            if hi <= lo:
                raise ValueError(f"bounding_box dim {i}: max ({hi}) <= min ({lo})")

        self._bbox_min = tuple(float(m) for m in mins)
        self._bbox_max = tuple(float(m) for m in maxs)
        self._dim = sdf_evaluator.dim

        # Cache a Monte-Carlo volume estimate so .area is O(1) after init.
        # ``_estimate_volume`` populates ``_volume_accept_rate`` as a side
        # effect so callers can surface the rejection-sampling efficiency
        # as a metric without re-running the MC pass.
        self._volume_accept_rate: float = 0.0
        self._volume = self._estimate_volume(volume_samples)

        logger.debug(
            "picogk_domain_created",
            dim=self._dim,
            bbox_min=self._bbox_min,
            bbox_max=self._bbox_max,
            volume_estimate=self._volume,
            volume_accept_rate=self._volume_accept_rate,
            oversample_factor=self.oversample_factor,
        )

    @property
    def dim(self) -> int:
        """Spatial dimension (2 or 3) inherited from the SDF."""
        return self._dim

    @property
    def area(self) -> float:
        """Monte-Carlo estimate of the interior volume / area."""
        return self._volume

    @property
    def volume_accept_rate(self) -> float:
        """Empirical interior acceptance rate of bbox rejection sampling.

        Computed once at construction by ``_estimate_volume``; equals the
        fraction of uniform-bbox samples whose SDF reported them interior.
        Useful as a sanity check (very low rates flag thin or empty
        geometries) and as a scenario metric.
        """
        return self._volume_accept_rate

    def bounding_box(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the axis-aligned bounding box reported by the SDF."""
        return self._bbox_min, self._bbox_max

    def contains_point(self, points: Tensor) -> Tensor:
        """A point is interior iff its signed distance is <= 0."""
        values = self.sdf_evaluator.sdf(points)
        return values <= 0.0

    def is_boundary(
        self,
        points: Tensor,
        tol: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> Tensor:
        """A point is on the boundary iff ``|sdf(p)| < tol``."""
        values = self.sdf_evaluator.sdf(points)
        return values.abs() < tol

    def sample_interior(self, n_points: int, device: torch.device | None = None) -> Tensor:
        """Rejection-sample ``n_points`` interior points from the SDF bbox.

        Oversamples adaptively when the bbox is mostly empty (typical for
        thin helical or lattice geometries).
        """
        if n_points <= 0:
            raise ValueError(f"n_points must be > 0, got {n_points}")

        bbox_min = torch.tensor(self._bbox_min, device=device, dtype=torch.float32)
        bbox_extent = torch.tensor(
            [hi - lo for lo, hi in zip(self._bbox_min, self._bbox_max, strict=True)],
            device=device,
            dtype=torch.float32,
        )

        collected: list[Tensor] = []
        n_remaining = n_points
        oversample = self.oversample_factor

        while n_remaining > 0:
            n_candidates = max(int(n_remaining * oversample) + 64, 64)
            candidates = torch.rand(n_candidates, self._dim, device=device) * bbox_extent + bbox_min
            mask = self.contains_point(candidates)
            accepted = candidates[mask]

            if accepted.shape[0] >= n_remaining:
                collected.append(accepted[:n_remaining])
                n_remaining = 0
            else:
                collected.append(accepted)
                n_remaining -= int(accepted.shape[0])
                oversample = min(oversample * 2.0, self.max_oversample)
                if oversample >= self.max_oversample and accepted.shape[0] == 0:
                    raise RuntimeError(
                        "PicoGKDomain.sample_interior failed to find any "
                        "interior points; the SDF bounding box may be "
                        "empty or the SDF sign convention may be inverted."
                    )

        return torch.cat(collected, dim=0)

    def sample_boundary(self, n_points: int, device: torch.device | None = None) -> Tensor:
        """Sample ``n_points`` points on the zero level set of the SDF.

        Strategy: draw candidates uniformly in the bbox, then take up to
        ``_PROJECTION_MAX_ITERS`` Newton steps along ``grad(sdf)`` (estimated
        via central differences) to project each candidate onto the zero
        level set. Candidates that do not converge within
        ``boundary_tolerance`` are redrawn until enough have converged.

        This is the direct 3D-SDF analogue of the Newton-style boundary
        projection used by ``CylinderFlowDomain``.
        """
        if n_points <= 0:
            raise ValueError(f"n_points must be > 0, got {n_points}")

        bbox_min = torch.tensor(self._bbox_min, device=device, dtype=torch.float32)
        bbox_extent = torch.tensor(
            [hi - lo for lo, hi in zip(self._bbox_min, self._bbox_max, strict=True)],
            device=device,
            dtype=torch.float32,
        )

        collected: list[Tensor] = []
        n_remaining = n_points
        oversample = self.oversample_factor

        while n_remaining > 0:
            n_candidates = max(int(n_remaining * oversample) + 64, 64)
            candidates = torch.rand(n_candidates, self._dim, device=device) * bbox_extent + bbox_min
            projected = self._project_to_surface(candidates)
            residual = self.sdf_evaluator.sdf(projected).abs()
            mask = residual < self.boundary_tolerance
            accepted = projected[mask]

            if accepted.shape[0] >= n_remaining:
                collected.append(accepted[:n_remaining])
                n_remaining = 0
            else:
                collected.append(accepted)
                n_remaining -= int(accepted.shape[0])
                oversample = min(oversample * 2.0, self.max_oversample)
                if oversample >= self.max_oversample and accepted.shape[0] == 0:
                    raise RuntimeError(
                        "PicoGKDomain.sample_boundary could not project any "
                        "candidates onto the zero level set; the SDF may "
                        "have degenerate gradients or the geometry may be "
                        "too thin for the configured tolerance."
                    )

        return torch.cat(collected, dim=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_volume(self, n_samples: int) -> float:
        """Monte-Carlo volume estimate by rejection in the bbox.

        Records the empirical acceptance rate on ``_volume_accept_rate``
        so :pyattr:`volume_accept_rate` exposes it without recomputation.
        """
        candidates = torch.rand(n_samples, self._dim)
        bbox_min = torch.tensor(self._bbox_min, dtype=torch.float32)
        bbox_extent = torch.tensor(
            [hi - lo for lo, hi in zip(self._bbox_min, self._bbox_max, strict=True)],
            dtype=torch.float32,
        )
        candidates = candidates * bbox_extent + bbox_min
        mask = self.contains_point(candidates)
        bbox_volume = float(np.prod(bbox_extent.numpy()))
        accept_rate = float(mask.float().mean().item())
        self._volume_accept_rate = accept_rate
        return accept_rate * bbox_volume

    def _estimate_gradient(self, points: Tensor) -> Tensor:
        """Central-difference gradient of the SDF at each point.

        Args:
        ----
            points: shape ``(N, dim)``.

        Returns:
        -------
            Gradient tensor of shape ``(N, dim)``. Rows with vanishing
            magnitude are returned as-is; the caller guards against division
            by zero.

        """
        eps = self.grad_epsilon
        grads: list[Tensor] = []
        for d in range(self._dim):
            step = torch.zeros_like(points)
            step[:, d] = eps
            fwd = self.sdf_evaluator.sdf(points + step)
            bwd = self.sdf_evaluator.sdf(points - step)
            grads.append((fwd - bwd) / (2 * eps))
        return torch.stack(grads, dim=-1)

    def _project_to_surface(self, points: Tensor) -> Tensor:
        """Project points onto the zero level set via damped Newton steps.

        Two-phase strategy:
        1. Run damped Newton along the central-difference gradient until
           every point is within ``boundary_tolerance`` of the surface or
           ``projection_max_iters`` is exhausted.
        2. If ``enable_bisection_fallback`` is set, run a bracketed
           bisection along the gradient line for any point that did not
           converge. Bisection is robust where Newton can stall (high
           curvature, sharp edges) because each iteration halves the
           bracket length irrespective of local derivative behavior.
        """
        x = points.clone()
        for it in range(self.projection_max_iters):
            values = self.sdf_evaluator.sdf(x)
            if bool((values.abs() < self.boundary_tolerance).all()):
                logger.debug(
                    "picogk_projection_converged",
                    iteration=it,
                    n_points=int(x.shape[0]),
                )
                break
            grads = self._estimate_gradient(x)
            grad_norm_sq = (
                (grads * grads).sum(dim=-1, keepdim=True).clamp_min(self.min_grad_norm_sq)
            )
            step = (values.unsqueeze(-1) / grad_norm_sq) * grads
            x = x - step

        if self.enable_bisection_fallback:
            x = self._bisection_fallback(x)
        return x

    def _bisection_fallback(self, x: Tensor) -> Tensor:
        """Bracket-and-bisect any points still off the zero level set.

        For each non-converged point we walk along ``-grad/||grad||``
        scaled by ``bisection_bracket_factor * |sdf(x)|`` to construct a
        bracket whose endpoints have opposite SDF signs. Pure bisection
        on the bracketed interval is then guaranteed to halve the
        residual every iteration, so a small fixed budget (default 16)
        drives ``|sdf|`` well below the boundary tolerance for any
        reasonable SDF.

        Points that fail to bracket (e.g., when the SDF is locally
        constant — :class:`_FlatPositiveSDF` in tests) keep the Newton
        result; the caller still rejects them via
        ``residual < boundary_tolerance`` in the sampling loop, so the
        fail-loud RuntimeError stays intact.
        """
        values = self.sdf_evaluator.sdf(x)
        not_converged = values.abs() >= self.boundary_tolerance
        n_failing = int(not_converged.sum().item())
        if n_failing == 0:
            return x

        sub = x[not_converged]
        sub_values = values[not_converged]
        grads = self._estimate_gradient(sub)
        grad_norm = torch.linalg.norm(grads, dim=-1, keepdim=True).clamp_min(
            self.min_grad_norm_sq**0.5
        )
        # Walk in the direction that *decreases* |sdf|. For points where
        # sdf > 0 that is ``-grad/||grad||``; for sdf < 0 it is the
        # opposite, hence the ``sign(value)`` factor.
        direction = -grads / grad_norm * sub_values.sign().unsqueeze(-1)
        bracket_len = self.bisection_bracket_factor * sub_values.abs().unsqueeze(-1)

        near = sub.clone()
        far = sub + bracket_len * direction
        near_values = self.sdf_evaluator.sdf(near)
        far_values = self.sdf_evaluator.sdf(far)

        # Only bisect rows that actually bracket a sign change. Rows that
        # do not bracket (degenerate SDFs, points exactly on a flat
        # region) keep their Newton result so the caller's "not enough
        # accepted points" RuntimeError still fires for pathological SDFs.
        bracketed = (near_values * far_values) < 0
        n_bracketed = int(bracketed.sum().item())
        logger.debug(
            "picogk_projection_bisection_start",
            n_failing=n_failing,
            n_bracketed=n_bracketed,
            max_iters=self.bisection_max_iters,
        )
        if n_bracketed == 0:
            return x

        near_b = near[bracketed]
        far_b = far[bracketed]
        near_v = near_values[bracketed]

        for _ in range(self.bisection_max_iters):
            mid = 0.5 * (near_b + far_b)
            mid_v = self.sdf_evaluator.sdf(mid)
            same_sign = (mid_v * near_v) > 0
            near_b = torch.where(same_sign.unsqueeze(-1), mid, near_b)
            far_b = torch.where(same_sign.unsqueeze(-1), far_b, mid)
            near_v = torch.where(same_sign, mid_v, near_v)

        refined = 0.5 * (near_b + far_b)

        # Reassemble. ``not_converged`` selects the failing rows from x;
        # within those, ``bracketed`` selects the rows we just refined.
        # Use a helper-index trick to avoid double indexing.
        out = x.clone()
        failing_indices = torch.nonzero(not_converged, as_tuple=False).squeeze(-1)
        bracketed_indices = failing_indices[bracketed]
        out[bracketed_indices] = refined
        return out
