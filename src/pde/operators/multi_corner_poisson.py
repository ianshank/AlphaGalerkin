r"""Harmonic Poisson on a polyomino with several reentrant corners -- a second AMR testbed.

Exact solution
--------------
For each declared corner ``i`` at ``p_i`` with interior angle ``w_i`` and
exterior-wedge bisector angle ``b_i``::

    r_i     = |x - p_i|,          theta_i = atan2(x - p_i)
    psi_i   = (theta_i - b_i) mod 2*pi
    phi_i   = psi_i - (2*pi - w_i) / 2
    lam_i   = pi / w_i
    s_i     = r_i**lam_i * sin(lam_i * phi_i)

and ``u = sum_i c_i * s_i`` solves ``-Laplace(u) = 0`` in the domain with
Dirichlet data ``g = u`` on its boundary. Each ``s_i`` is harmonic off the
closed ray from ``p_i`` along ``b_i`` (its branch cut) and vanishes on both
edges of corner ``i`` (``phi_i = 0`` and ``phi_i = w_i``).

Why the angle wraps on the exterior bisector
--------------------------------------------
``psi_i`` jumps from ``2*pi`` back to ``0`` exactly on the cut ray, which lies
strictly inside the exterior wedge -- at least ``(2*pi - w_i) / 2`` away from
either edge. Rounding a boundary node's coordinates can therefore never move it
across the wrap. The ``[0, 2*pi)`` convention ``LShapedPoissonOperator`` uses
puts the wrap *on* its ``{y = 0, x > 0}`` edge instead: a node at
``(0.5, -1e-300)`` maps to ``theta ~ 2*pi`` and evaluates
``sin(4*pi/3) * 0.5**(2/3) ~ -0.55`` where the boundary value is 0. On the
Z-shape it is worse: the same convention would place corner 1's cut along the
*interior* cell interface ``{y = 0, 0 < x < 1}``.

Why the constructor verifies the geometry
-----------------------------------------
``u`` is harmonic in the domain iff no corner's cut ray meets it. A cut through
the domain yields a function with a kink inside -- still finite and plausible
everywhere, and not the solution of the Dirichlet problem it defines, so every
error measured against it is wrong without anything looking wrong. That is the
2026-08-16 L-shape retraction's signature, so it is refused at construction
(:class:`BranchCutError`, :class:`CornerDeclarationError`) rather than left to a
reviewer.

The canonical L preset (:func:`build_lshape_multi_corner_operator`) reproduces
:class:`~src.pde.operators.lshaped_poisson.LShapedPoissonOperator` on the closed
L-shape, which is what proves this is a generalisation rather than a new model.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.config import PDEConfig, PDEType
from src.pde.geometry_polyomino import (
    LSHAPE_POLYOMINO_CELLS,
    POLYOMINO_REENTRANT_INTERIOR_ANGLE,
    ZSHAPE_POLYOMINO_CELLS,
    PolyominoCorner,
    PolyominoDomain,
)
from src.pde.operators.base import PDEOperator, PDEResidual

logger = structlog.get_logger(__name__)

#: One full turn, the modulus of the angle wrap.
FULL_TURN: Final[float] = 2.0 * math.pi

#: This operator is planar: corners, cut rays and polyomino cells are 2-D.
PLANAR_DIM: Final[int] = 2

#: Default interior angle of a singular corner: the 270-degree reentrant corner
#: every polyomino has (exponent ``pi / w = 2/3``).
DEFAULT_CORNER_INTERIOR_ANGLE: Final[float] = POLYOMINO_REENTRANT_INTERIOR_ANGLE

#: Default singular strength ``c``: the canonical benchmark's unit coefficient.
DEFAULT_SINGULAR_COEFFICIENT: Final[float] = 1.0

#: How far a declared corner position may sit from the geometric corner.
CORNER_POSITION_ATOL: Final[float] = 1e-12

#: How far a declared bisector / interior angle may sit from the geometric one.
#: Tight on purpose: any other bisector stops ``s_i`` vanishing on the corner's
#: edges. ``-math.pi / 4`` and ``math.atan2(-1, 1)`` agree to the last bit.
CORNER_ANGLE_ATOL: Final[float] = 1e-12

#: Tolerance when comparing the config's bounding box with the polyomino's --
#: loose enough for a float round-trip through a Pydantic list, tight enough
#: that any real rescaling is rejected (same role as ``LSHAPE_DOMAIN_ATOL``).
DOMAIN_BOUNDS_ATOL: Final[float] = 1e-9

#: The only collocation distribution a polyomino supports (see
#: :meth:`MultiCornerPoissonOperator.generate_collocation_points`).
COLLOCATION_METHOD_RANDOM: Final[str] = "random"

# ----------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------

#: The canonical L-shape's single corner: the origin, exterior wedge x>0, y<0.
LSHAPE_CORNER_POSITION: Final[tuple[float, float]] = (0.0, 0.0)
LSHAPE_EXTERIOR_BISECTOR: Final[float] = -0.25 * math.pi

#: Z-tetromino corner 1: (0, 0), exterior wedge x<0, y<0.
ZSHAPE_PRIMARY_CORNER_POSITION: Final[tuple[float, float]] = (0.0, 0.0)
ZSHAPE_PRIMARY_EXTERIOR_BISECTOR: Final[float] = -0.75 * math.pi

#: Z-tetromino corner 2: (1, 0), exterior wedge x>1, y>0.
ZSHAPE_SECONDARY_CORNER_POSITION: Final[tuple[float, float]] = (1.0, 0.0)
ZSHAPE_SECONDARY_EXTERIOR_BISECTOR: Final[float] = 0.25 * math.pi

#: Default singular strengths of the Z preset. Unequal on purpose: with equal
#: strengths the budget-optimal split between the corners is the symmetric one;
#: with one far weaker (say 0.01) the second corner never matters inside a small
#: DOF budget. 4:1 makes both corners compete for the same budget.
DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT: Final[float] = 1.0
DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT: Final[float] = 0.25

#: ``PDEConfig.name`` of the two presets.
LSHAPE_PRESET_NAME: Final[str] = "poisson_multi_corner_lshape"
ZSHAPE_PRESET_NAME: Final[str] = "poisson_multi_corner_zshape"


class CornerDeclarationError(ValueError):
    """A declared singular corner does not match the domain's geometry."""


class BranchCutError(ValueError):
    """A singular term's branch-cut ray reaches the closed domain."""


class SingularCornerTerm(BaseModel):
    """One corner's singular harmonic ``c * r**lam * sin(lam * phi)``."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    x: float = Field(description="Corner x coordinate.")
    y: float = Field(description="Corner y coordinate.")
    exterior_bisector: float = Field(
        description=(
            "Angle (radians) bisecting the exterior wedge at the corner. The branch "
            "cut runs along it, so it must point out of the domain."
        ),
    )
    interior_angle: float = Field(
        default=DEFAULT_CORNER_INTERIOR_ANGLE,
        gt=math.pi,
        lt=FULL_TURN,
        description="Interior angle w of the reentrant corner; the exponent is pi / w.",
    )
    coefficient: float = Field(
        default=DEFAULT_SINGULAR_COEFFICIENT,
        description="Singular strength c of this corner's term.",
    )

    @property
    def position(self) -> tuple[float, float]:
        """The corner as an ``(x, y)`` pair."""
        return (self.x, self.y)

    @property
    def exponent(self) -> float:
        """Singular exponent ``lam = pi / w`` (2/3 for a 270-degree corner)."""
        return math.pi / self.interior_angle

    @property
    def wedge_offset(self) -> float:
        """``(2*pi - w) / 2``: the angle from the cut to either edge of the corner."""
        return (FULL_TURN - self.interior_angle) / 2.0


def singular_term_values(
    points: NDArray[np.float64], term: SingularCornerTerm
) -> NDArray[np.float64]:
    """``s_i`` (no coefficient) at ``(n, 2)`` points, in float64; exactly 0 at the corner."""
    dx = points[:, 0] - term.x
    dy = points[:, 1] - term.y
    r = np.hypot(dx, dy)
    phi = np.mod(np.arctan2(dy, dx) - term.exterior_bisector, FULL_TURN) - term.wedge_offset
    lam = term.exponent
    values = np.power(r, lam) * np.sin(lam * phi)
    return np.asarray(np.where(r > 0.0, values, 0.0), dtype=np.float64)


def singular_term_tensor(points: Tensor, term: SingularCornerTerm) -> Tensor:
    """Torch twin of :func:`singular_term_values`, in the input's dtype.

    Double-``where``: the corner itself is replaced by a harmless stand-in
    before ``atan2``/``hypot``/``pow`` see it, so autograd returns a finite
    (zero) gradient there instead of a NaN that would poison a whole batch.
    """
    dx = points[:, 0] - term.x
    dy = points[:, 1] - term.y
    at_corner = (dx == 0) & (dy == 0)
    safe_dx = torch.where(at_corner, torch.ones_like(dx), dx)
    safe_dy = torch.where(at_corner, torch.zeros_like(dy), dy)
    r = torch.hypot(safe_dx, safe_dy)
    theta = torch.atan2(safe_dy, safe_dx)
    phi = torch.remainder(theta - term.exterior_bisector, FULL_TURN) - term.wedge_offset
    lam = term.exponent
    values = r.pow(lam) * torch.sin(lam * phi)
    return torch.where(at_corner, torch.zeros_like(values), values)


def _wrapped_angle_difference(first: float, second: float) -> float:
    """``first - second`` reduced to ``[-pi, pi]``."""
    return math.remainder(first - second, FULL_TURN)


def _require_planar_matching_bounds(config: PDEConfig, domain: PolyominoDomain) -> None:
    """The config's box must *be* the polyomino's box (the D1 defect class).

    ``PDEOperator`` exposes ``domain_min``/``domain_max`` from the config, and
    other code sizes grids from them. A box that disagrees with the cells meshes
    one domain and evaluates the solution on another, silently.
    """
    if config.domain_dim != PLANAR_DIM:
        raise ValueError(f"a polyomino operator is planar; got domain_dim={config.domain_dim}")
    low, high = domain.bounding_box()
    if not (
        np.allclose(config.domain_min, low, atol=DOMAIN_BOUNDS_ATOL)
        and np.allclose(config.domain_max, high, atol=DOMAIN_BOUNDS_ATOL)
    ):
        raise ValueError(
            f"config box [{config.domain_min}, {config.domain_max}] is not the polyomino's "
            f"bounding box [{list(low)}, {list(high)}]; build the config from the domain"
        )


def _require_nondegenerate(corners: tuple[SingularCornerTerm, ...]) -> None:
    """At least one corner, and not every strength zero.

    All-zero strengths make ``u == 0``: every error is 0.0 at every DOF count
    and every ratio built on it is meaningless -- the defect that made every
    1-D Poisson AMR row in this repo degenerate (``tests/research/test_baselines.py``).
    A corner declared twice is refused once each declaration is matched to a
    geometric corner (:func:`_require_first_declaration`), not here.
    """
    if not corners:
        raise ValueError("at least one singular corner is required")
    if all(term.coefficient == 0.0 for term in corners):
        raise ValueError(
            "every singular coefficient is zero, so the exact solution is u == 0 and "
            "every measured error is 0.0 -- a degenerate substrate, not a testbed"
        )


def _matching_corner(
    domain: PolyominoDomain, term: SingularCornerTerm, index: int
) -> PolyominoCorner:
    """The geometric corner ``term`` declares; raise unless angle and bisector agree."""
    corner = domain.corner_at(term.position, atol=CORNER_POSITION_ATOL)
    if corner is None:
        raise CornerDeclarationError(
            f"corner {index} at {term.position} is not a reentrant corner of the domain; "
            f"its reentrant corners are {[c.position for c in domain.reentrant_corners]}"
        )
    if abs(term.interior_angle - corner.interior_angle) > CORNER_ANGLE_ATOL:
        raise CornerDeclarationError(
            f"corner {index} at {term.position} declares interior angle {term.interior_angle}, "
            f"but the domain's corner there is {corner.interior_angle}"
        )
    bisector_drift = abs(
        _wrapped_angle_difference(term.exterior_bisector, corner.exterior_bisector)
    )
    if bisector_drift > CORNER_ANGLE_ATOL:
        raise CornerDeclarationError(
            f"corner {index} at {term.position} declares exterior_bisector "
            f"{term.exterior_bisector}, but the exterior wedge there is bisected by "
            f"{corner.exterior_bisector}: a bisector pointing into the domain puts the branch "
            f"cut through it, and any other one stops s_i vanishing on the corner's edges"
        )
    return corner


def _require_first_declaration(
    matched: dict[tuple[float, float], int],
    corner: PolyominoCorner,
    corners: tuple[SingularCornerTerm, ...],
    index: int,
) -> None:
    """Refuse a second declaration of the geometric corner ``corner``, then record it.

    Keyed on the *matched* corner, not the declared position: matching allows
    ``CORNER_POSITION_ATOL``, so (0, 0) and (5e-13, 0) are one corner, and
    accepting both silently doubles its singular term.
    """
    first = matched.setdefault(corner.position, index)
    if first != index:
        raise ValueError(
            f"corners {first} at {corners[first].position} and {index} at "
            f"{corners[index].position} both match the domain's corner at {corner.position}: "
            f"a corner declared more than once would add its singular term twice"
        )


def _seeded_generator(seed: int | None) -> torch.Generator | None:
    """A private generator for ``seed`` -- never re-seeds torch's global RNG."""
    return None if seed is None else torch.Generator().manual_seed(seed)


class MultiCornerPoissonOperator(PDEOperator):
    """``-Laplace(u) = 0`` on a polyomino, exact solution a sum of corner singularities.

    See the module docstring for the solution, the wrap convention and why the
    constructor refuses a geometry the solution is not harmonic on.
    """

    name = "poisson_multi_corner"
    description = "Harmonic Poisson on a polyomino with several reentrant-corner singularities"
    pde_type = PDEType.POISSON
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        *,
        domain: PolyominoDomain,
        corners: Sequence[SingularCornerTerm],
    ) -> None:
        """Build the operator, verifying every corner against ``domain``.

        Args:
            config: PDE configuration. Its box must equal ``domain``'s bounding
                box; the presets build it from the domain.
            domain: The polyomino the problem lives on.
            corners: One singular term per reentrant corner that carries a
                singularity. A corner left out is simply smooth.

        Raises:
            ValueError: Non-planar config, box mismatch, no corners, a repeated
                corner, or every coefficient zero.
            CornerDeclarationError: A declared corner is not a reentrant corner
                of ``domain``, or its interior angle / bisector disagrees.
            BranchCutError: A corner's cut ray reaches the closed domain.

        """
        super().__init__(config)
        self.diffusion = config.diffusion_coeff
        self.geometry: PolyominoDomain = domain
        self._corners: tuple[SingularCornerTerm, ...] = tuple(corners)
        self._logged_method_substitution = False
        _require_planar_matching_bounds(config, domain)
        _require_nondegenerate(self._corners)
        matched: dict[tuple[float, float], int] = {}
        for index, term in enumerate(self._corners):
            corner = _matching_corner(domain, term, index)
            _require_first_declaration(matched, corner, self._corners, index)
            if domain.open_ray_meets_closure(corner.position, corner.exterior_quadrant):
                raise BranchCutError(
                    f"corner {index} at {term.position}: its branch cut (the ray along "
                    f"bisector {term.exterior_bisector:.6f}) reaches the closed domain, so "
                    f"s_{index} is not harmonic there and u is not the solution of the "
                    f"Dirichlet problem it defines"
                )
            logger.debug(
                "singular_corner_validated",
                index=index,
                position=term.position,
                exterior_bisector=term.exterior_bisector,
                exponent=term.exponent,
                coefficient=term.coefficient,
            )
        logger.info("branch_cut_validation_passed", n_corners=len(self._corners))
        logger.info(
            "multi_corner_poisson_operator_created",
            n_cells=len(domain.cells),
            corners=[(term.x, term.y, term.coefficient) for term in self._corners],
            bounding_box=domain.bounding_box(),
            diffusion=self.diffusion,
        )

    @property
    def corners(self) -> tuple[SingularCornerTerm, ...]:
        """The validated singular terms, in declaration order."""
        return self._corners

    # ------------------------------------------------------------------
    # PDEOperator interface
    # ------------------------------------------------------------------

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """``u = sum_i c_i * s_i``.

        Tensor in, tensor out in the input's dtype (float64 in, float64 out). The
        numpy path evaluates in float64 and returns float32, the dtype every
        operator's numpy path returns.
        """
        if isinstance(coords, Tensor):
            return self._exact_tensor(coords)
        points = np.asarray(coords, dtype=np.float64)
        values = np.zeros(points.shape[0], dtype=np.float64)
        for term in self._corners:
            values += term.coefficient * singular_term_values(points, term)
        return values.astype(np.float32)

    def _exact_tensor(self, coords: Tensor) -> Tensor:
        """The tensor branch of :meth:`exact_solution`, typed as a tensor."""
        total = torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        for term in self._corners:
            total = total + term.coefficient * singular_term_tensor(coords, term)
        return total

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """``f = 0``: the exact solution is harmonic."""
        if isinstance(coords, Tensor):
            return self._source_tensor(coords)
        return np.zeros(coords.shape[0], dtype=np.float32)

    @staticmethod
    def _source_tensor(coords: Tensor) -> Tensor:
        """The tensor branch of :meth:`source_term`, typed as a tensor."""
        return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Dirichlet data: the exact solution's trace."""
        return self.exact_solution(coords, time)

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """``R = -nu * Laplace(u) - f`` via automatic differentiation."""
        derivatives = self.compute_derivatives(u, coords)
        residual_values = -self.diffusion * derivatives["laplacian"] - self._source_tensor(coords)
        return PDEResidual(
            values=residual_values,
            l2_norm=float(torch.sqrt(torch.mean(residual_values**2)).item()),
            max_norm=float(torch.max(torch.abs(residual_values)).item()),
            derivatives=derivatives if compute_derivatives else {},
        )

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> NDArray[np.bool_] | Tensor:
        """Geometry-aware boundary test (the rectangular base test is wrong here)."""
        if isinstance(coords, Tensor):
            return self.geometry.is_boundary(coords, tol=tolerance)
        return self.geometry.is_boundary(torch.from_numpy(coords), tol=tolerance).numpy()

    def generate_collocation_points(
        self,
        n_points: int,
        method: str = COLLOCATION_METHOD_RANDOM,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Uniform random points in the polyomino, reproducible from ``seed``.

        ``method`` is accepted for interface compatibility -- basis-selection
        and PINN callers pass ``"lhs"``/``"uniform"`` -- and, like
        ``LShapedPoissonOperator``, every method samples uniformly at random:
        a bounding-box lattice or hypercube is not a design on a polyomino. The
        substitution is logged once per instance rather than silently absorbed.
        """
        if method != COLLOCATION_METHOD_RANDOM and not self._logged_method_substitution:
            self._logged_method_substitution = True
            logger.info(
                "collocation_method_substituted",
                requested=method,
                used=COLLOCATION_METHOD_RANDOM,
                note="logged once per operator instance",
            )
        points = self.geometry.sample_interior(n_points, generator=_seeded_generator(seed))
        return points.numpy().astype(np.float32)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """``n_points_per_face`` points per maximal boundary segment, length-weighted."""
        total = n_points_per_face * len(self.geometry.boundary_segments)
        points = self.geometry.sample_boundary(total, generator=_seeded_generator(seed))
        return points.numpy().astype(np.float32)

    def compute_error(
        self,
        u_pred: Tensor,
        coords: Tensor,
    ) -> dict[str, float]:
        """L2 (RMS), L-infinity and MSE against the exact solution."""
        diff = u_pred - self._exact_tensor(coords)
        errors = {
            "l2_error": float(torch.sqrt(torch.mean(diff**2)).item()),
            "linf_error": float(torch.max(torch.abs(diff)).item()),
            "mse": float(torch.mean(diff**2).item()),
        }
        logger.debug("error_computed", **errors)
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialise, including the corners and cells a config alone cannot carry."""
        data = super().to_dict()
        data["corners"] = [term.model_dump() for term in self._corners]
        data["cells"] = [dataclasses.asdict(cell) for cell in self.geometry.cells]
        return data


def _preset_config(name: str, domain: PolyominoDomain) -> PDEConfig:
    """A config whose box is, by construction, the domain's bounding box."""
    low, high = domain.bounding_box()
    return PDEConfig(
        name=name,
        pde_type=PDEType.POISSON,
        domain_dim=PLANAR_DIM,
        domain_min=list(low),
        domain_max=list(high),
    )


def build_lshape_multi_corner_operator(
    *, coefficient: float = DEFAULT_SINGULAR_COEFFICIENT
) -> MultiCornerPoissonOperator:
    """The canonical L-shape as a one-corner polyomino (``coefficient=1`` is the benchmark)."""
    domain = PolyominoDomain(LSHAPE_POLYOMINO_CELLS)
    corner = SingularCornerTerm(
        x=LSHAPE_CORNER_POSITION[0],
        y=LSHAPE_CORNER_POSITION[1],
        exterior_bisector=LSHAPE_EXTERIOR_BISECTOR,
        coefficient=coefficient,
    )
    return MultiCornerPoissonOperator(
        _preset_config(LSHAPE_PRESET_NAME, domain), domain=domain, corners=(corner,)
    )


def build_zshape_poisson_operator(
    *,
    primary_coefficient: float = DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT,
    secondary_coefficient: float = DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT,
) -> MultiCornerPoissonOperator:
    """The Z-tetromino with two 270-degree corners of configurable strength."""
    domain = PolyominoDomain(ZSHAPE_POLYOMINO_CELLS)
    corners = (
        SingularCornerTerm(
            x=ZSHAPE_PRIMARY_CORNER_POSITION[0],
            y=ZSHAPE_PRIMARY_CORNER_POSITION[1],
            exterior_bisector=ZSHAPE_PRIMARY_EXTERIOR_BISECTOR,
            coefficient=primary_coefficient,
        ),
        SingularCornerTerm(
            x=ZSHAPE_SECONDARY_CORNER_POSITION[0],
            y=ZSHAPE_SECONDARY_CORNER_POSITION[1],
            exterior_bisector=ZSHAPE_SECONDARY_EXTERIOR_BISECTOR,
            coefficient=secondary_coefficient,
        ),
    )
    return MultiCornerPoissonOperator(
        _preset_config(ZSHAPE_PRESET_NAME, domain), domain=domain, corners=corners
    )


__all__ = [
    "DEFAULT_ZSHAPE_PRIMARY_COEFFICIENT",
    "DEFAULT_ZSHAPE_SECONDARY_COEFFICIENT",
    "BranchCutError",
    "CornerDeclarationError",
    "MultiCornerPoissonOperator",
    "SingularCornerTerm",
    "build_lshape_multi_corner_operator",
    "build_zshape_poisson_operator",
    "singular_term_tensor",
    "singular_term_values",
]
