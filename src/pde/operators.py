"""PDE Operator definitions with automatic differentiation.

This module provides abstract and concrete PDE operators for:
- Defining PDE equations declaratively
- Computing residuals via automatic differentiation
- Supporting time-dependent and steady-state PDEs

Each operator implements:
- residual(): Computes PDE residual at collocation points
- exact_solution(): Optional analytical solution for testing
- source_term(): Source/forcing term
- boundary_condition(): Boundary value function

Supported PDEs:
- Poisson: ∇²u = f
- Burgers: u_t + u·∇u = ν∇²u
- Advection-Diffusion: u_t + a·∇u = ν∇²u + f
- Heat: u_t = κ∇²u + f
- Wave: u_tt = c²∇²u + f
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from scipy.special import ive
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.geometry import (
    DomainGeometry,
    GeometryType,
    LShapedDomain,
    create_geometry,
)

logger = structlog.get_logger(__name__)


@dataclass
class PDEResidual:
    """Container for PDE residual computation results.

    Attributes:
        values: Residual values at each point (N,).
        l2_norm: L2 norm of residual.
        max_norm: Maximum absolute residual.
        derivatives: Dictionary of computed derivatives.

    """

    values: NDArray[np.float32] | Tensor
    l2_norm: float
    max_norm: float
    derivatives: dict[str, NDArray[np.float32] | Tensor]

    def to_numpy(self) -> PDEResidual:
        """Convert tensors to numpy arrays."""
        values = (
            self.values.detach().cpu().numpy() if isinstance(self.values, Tensor) else self.values
        )
        derivatives = {
            k: (v.detach().cpu().numpy() if isinstance(v, Tensor) else v)
            for k, v in self.derivatives.items()
        }
        return PDEResidual(
            values=values,
            l2_norm=self.l2_norm,
            max_norm=self.max_norm,
            derivatives=derivatives,
        )


class PDEOperator(ABC):
    """Abstract base class for PDE operators.

    Defines the interface for PDE equations that can be solved
    using the AlphaGalerkin framework.

    Subclasses must implement:
    - residual(): PDE residual computation
    - source_term(): Forcing function
    - boundary_value(): Boundary condition values
    """

    # Class-level attributes
    name: str = "abstract_pde"
    description: str = "Abstract PDE operator"
    pde_type: PDEType = PDEType.POISSON

    # Properties
    is_time_dependent: bool = False
    is_linear: bool = True
    order: int = 2  # Order of highest derivative

    def __init__(self, config: PDEConfig) -> None:
        """Initialize PDE operator.

        Args:
            config: PDE configuration.

        """
        self.config = config
        self.dim = config.domain_dim
        self.domain_min = np.array(config.domain_min, dtype=np.float32)
        self.domain_max = np.array(config.domain_max, dtype=np.float32)
        self.domain_size = self.domain_max - self.domain_min
        # One-shot latch for the grad-disconnected debug log in
        # compute_derivatives, which sits on the MCTS per-node path.
        self._logged_disconnected_u = False

    @abstractmethod
    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute PDE residual at given points.

        The residual is R(u) = L(u) - f where L is the differential
        operator and f is the source term.

        Args:
            u: Solution values at collocation points (N,) or (N, 1).
            coords: Collocation point coordinates (N, dim).
            compute_derivatives: Whether to compute and return derivatives.

        Returns:
            PDEResidual with values and norms.

        """
        raise NotImplementedError

    @abstractmethod
    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute source/forcing term at given points.

        Args:
            coords: Point coordinates (N, dim).
            time: Time value for time-dependent PDEs.

        Returns:
            Source term values (N,).

        """
        raise NotImplementedError

    @abstractmethod
    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute boundary condition values.

        Args:
            coords: Boundary point coordinates (N_b, dim).
            time: Time value for time-dependent PDEs.

        Returns:
            Boundary values (N_b,).

        """
        raise NotImplementedError

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Compute exact solution if known analytically.

        Args:
            coords: Point coordinates (N, dim).
            time: Time value for time-dependent PDEs.

        Returns:
            Exact solution values (N,), or None if unknown.

        """
        return None

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Compute initial condition for time-dependent PDEs.

        Args:
            coords: Point coordinates (N, dim).

        Returns:
            Initial values (N,).

        """
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> NDArray[np.bool_] | Tensor:
        """Determine which points are on the boundary.

        Args:
            coords: Point coordinates (N, dim).
            tolerance: Distance tolerance for boundary detection.

        Returns:
            Boolean mask (N,) with True for boundary points.

        """
        if isinstance(coords, Tensor):
            on_boundary = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
            for d in range(self.dim):
                on_min = torch.abs(coords[:, d] - self.domain_min[d]) < tolerance
                on_max = torch.abs(coords[:, d] - self.domain_max[d]) < tolerance
                on_boundary = on_boundary | on_min | on_max
            return on_boundary
        else:
            on_boundary = np.zeros(coords.shape[0], dtype=bool)
            for d in range(self.dim):
                on_min = np.abs(coords[:, d] - self.domain_min[d]) < tolerance
                on_max = np.abs(coords[:, d] - self.domain_max[d]) < tolerance
                on_boundary = on_boundary | on_min | on_max
            return on_boundary

    def compute_derivatives(
        self,
        u: Tensor,
        coords: Tensor,
    ) -> dict[str, Tensor]:
        """Compute spatial derivatives using automatic differentiation.

        When ``u`` is disconnected from ``coords`` in the computational graph
        (e.g. computed via numpy and converted with ``torch.from_numpy``), all
        derivatives are returned as zero tensors because autograd cannot trace
        through the non-differentiable path.

        Args:
            u: Solution values as a function of coords.
            coords: Collocation point coordinates (N, dim).

        Returns:
            Dictionary with derivative tensors.

        """
        n_points = coords.shape[0]

        # If u has no grad_fn and doesn't require grad, it is disconnected
        # from coords in the computational graph — derivatives are undefined.
        # Return zeros so callers (e.g. PoissonOperator.residual) still work.
        if not u.requires_grad and u.grad_fn is None:
            # This branch is silent by design but has real consequences: every
            # derivative-bearing term vanishes, so a residual built from it
            # collapses to the source term alone (and to exactly 0.0 when the
            # source is zero). A caller that reads that as "converged" is
            # measuring nothing. Logged so the condition is diagnosable rather
            # than inferred. See the P0 entry in docs/CODE_HYGIENE_AUDIT.md §7.7.
            #
            # Emitted once per operator instance, not once per call: this runs on
            # the MCTS per-node path (once per apply_action, so ~n_simulations
            # times per move), where a per-call event would flood the log and pay
            # event-dict construction on every node. The condition is a property
            # of how the caller builds `u`, so the first occurrence carries all
            # the diagnostic value.
            if not self._logged_disconnected_u:
                self._logged_disconnected_u = True
                logger.debug(
                    "derivatives_skipped_u_disconnected",
                    operator=type(self).__name__,
                    n_points=n_points,
                    dim=self.dim,
                    consequence="all derivative terms are zero for this call",
                    note="logged once per operator instance; suppressing repeats",
                )
            derivatives: dict[str, Tensor] = {}
            for d in range(self.dim):
                derivatives[f"u_x{d}"] = torch.zeros(
                    n_points, dtype=coords.dtype, device=coords.device
                )
                derivatives[f"u_x{d}x{d}"] = torch.zeros(
                    n_points, dtype=coords.dtype, device=coords.device
                )
            derivatives["laplacian"] = torch.zeros(
                n_points, dtype=coords.dtype, device=coords.device
            )
            return derivatives

        coords = coords.requires_grad_(True)

        # First derivatives
        derivatives = {}

        if u.dim() == 1:
            u = u.unsqueeze(-1)

        # Gradient (first derivatives)
        grad_outputs = torch.ones_like(u)
        grad = torch.autograd.grad(
            u, coords, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
        )[0]

        if grad is not None:
            for d in range(self.dim):
                key = f"u_x{d}"
                derivatives[key] = grad[:, d]

            # Laplacian (second derivatives)
            laplacian = torch.zeros(n_points, dtype=coords.dtype, device=coords.device)
            for d in range(self.dim):
                grad_d = grad[:, d : d + 1]
                # Skip second derivative if first derivative is constant (no grad_fn)
                if grad_d.grad_fn is None and not grad_d.requires_grad:
                    derivatives[f"u_x{d}x{d}"] = torch.zeros(
                        n_points, dtype=coords.dtype, device=coords.device
                    )
                    continue
                grad2 = torch.autograd.grad(
                    grad_d,
                    coords,
                    grad_outputs=torch.ones_like(grad_d),
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if grad2 is not None:
                    derivatives[f"u_x{d}x{d}"] = grad2[:, d]
                    laplacian = laplacian + grad2[:, d]
                else:
                    derivatives[f"u_x{d}x{d}"] = torch.zeros(
                        n_points, dtype=coords.dtype, device=coords.device
                    )

            derivatives["laplacian"] = laplacian
        else:
            # u doesn't depend on coords — return zero derivatives
            for d in range(self.dim):
                derivatives[f"u_x{d}"] = torch.zeros(
                    n_points, dtype=coords.dtype, device=coords.device
                )
                derivatives[f"u_x{d}x{d}"] = torch.zeros(
                    n_points, dtype=coords.dtype, device=coords.device
                )
            derivatives["laplacian"] = torch.zeros(
                n_points, dtype=coords.dtype, device=coords.device
            )

        return derivatives

    def generate_collocation_points(
        self,
        n_points: int,
        method: str = "uniform",
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate collocation points in the domain.

        Args:
            n_points: Number of interior points to generate.
            method: Sampling method ('uniform', 'random', 'lhs').
            seed: Random seed for reproducibility.

        Returns:
            Collocation points (n_points, dim).

        """
        rng = np.random.default_rng(seed)

        if method == "uniform":
            # Uniform grid
            n_per_dim = int(np.ceil(n_points ** (1.0 / self.dim)))
            grids = [
                np.linspace(self.domain_min[d], self.domain_max[d], n_per_dim)
                for d in range(self.dim)
            ]
            mesh = np.meshgrid(*grids, indexing="ij")
            points = np.stack([m.flatten() for m in mesh], axis=-1)
            # Subsample if too many points
            if len(points) > n_points:
                indices = rng.choice(len(points), n_points, replace=False)
                points = points[indices]
        elif method == "random":
            # Random uniform sampling
            points = rng.uniform(self.domain_min, self.domain_max, size=(n_points, self.dim))
        elif method == "lhs":
            # Latin hypercube sampling
            try:
                from scipy.stats import qmc

                sampler = qmc.LatinHypercube(d=self.dim, seed=seed)
                samples = sampler.random(n=n_points)
                points = qmc.scale(samples, self.domain_min, self.domain_max)
            except ImportError:
                # Fallback to random
                points = rng.uniform(self.domain_min, self.domain_max, size=(n_points, self.dim))
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return points.astype(np.float32)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate points on the domain boundary.

        Args:
            n_points_per_face: Points per boundary face.
            seed: Random seed.

        Returns:
            Boundary points (N_boundary, dim).

        """
        rng = np.random.default_rng(seed)
        points = []

        for d in range(self.dim):
            for boundary_val in [self.domain_min[d], self.domain_max[d]]:
                # Generate random points on this face
                face_points = rng.uniform(
                    self.domain_min, self.domain_max, size=(n_points_per_face, self.dim)
                )
                face_points[:, d] = boundary_val
                points.append(face_points)

        return np.concatenate(points, axis=0).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Serialize operator to dictionary."""
        return {
            "name": self.name,
            "pde_type": self.pde_type.value,
            "config": self.config.to_yaml_dict(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.dim})"


class PoissonOperator(PDEOperator):
    """Poisson equation operator: -∇²u = f.

    The Poisson equation describes:
    - Electrostatic potential
    - Steady-state heat distribution
    - Gravitational potential

    This implementation supports:
    - Variable diffusion coefficient
    - Custom source terms
    - Dirichlet boundary conditions
    """

    name = "poisson"
    description = "Poisson equation: -∇²u = f"
    pde_type = PDEType.POISSON
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        source_function: Callable[[NDArray | Tensor], NDArray | Tensor] | None = None,
        exact_solution_function: Callable[[NDArray | Tensor], NDArray | Tensor] | None = None,
    ) -> None:
        """Initialize Poisson operator.

        Args:
            config: PDE configuration.
            source_function: Custom source term function.
            exact_solution_function: Known exact solution (for testing).

        """
        super().__init__(config)
        self.diffusion = config.diffusion_coeff
        self._source_function = source_function
        self._exact_solution_function = exact_solution_function

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute Poisson residual: R = -∇²u - f."""
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u))
        source = self.source_term(coords)

        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        # Residual: -∇²u - f = 0  =>  R = -∇²u - f
        residual_values = -self.diffusion * laplacian - source

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute source term."""
        if self._source_function is not None:
            return self._source_function(coords)

        # Source for the manufactured solution u = prod_d sin(pi*x_d), whose
        # Laplacian is -dim*pi^2*u, so f = -laplacian(u) = dim*pi^2*u.
        #
        # The product runs over every dimension rather than a hardcoded (x, y).
        # The previous form took ``y = 0`` when ``dim == 1``, so the ``sin(pi*y)``
        # factor made f -- and the exact solution below -- identically ZERO in 1D:
        # every 1D Poisson problem in the repo was the degenerate ``-u'' = 0``
        # with homogeneous Dirichlet data, whose solution is ``u == 0``. That is
        # what left the 1D Dorfler AMR baseline with ``max_indicator == 0.0`` at
        # every step (bulk marking could never fire) and ``l2_error == 0.0`` at
        # every DOF count. It also silently truncated ``dim >= 3`` to the 2D
        # expression. At ``dim == 2`` this is the old expression re-associated:
        # measured max deviation is 1 ULP (9.7e-8 relative to the amplitude at
        # float32, 1.8e-16 at float64), so no 2D caller changes meaningfully.
        return self.dim * (np.pi**2) * self._default_manufactured_solution(coords)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute boundary values (Dirichlet BC)."""
        if self.config.boundary_condition == BoundaryCondition.DIRICHLET:
            if isinstance(coords, Tensor):
                return torch.full(
                    (coords.shape[0],),
                    self.config.boundary_value,
                    dtype=coords.dtype,
                    device=coords.device,
                )
            return np.full(coords.shape[0], self.config.boundary_value, dtype=np.float32)

        # For exact solution test case
        if self._exact_solution_function is not None:
            return self._exact_solution_function(coords)

        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Compute exact solution for manufactured solution test."""
        if self._exact_solution_function is not None:
            return self._exact_solution_function(coords)

        return self._default_manufactured_solution(coords)

    def _default_manufactured_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Separable sinusoid ``u = prod_d sin(pi * x_d)`` over EVERY dimension.

        Shared by :meth:`source_term` and :meth:`exact_solution` so the two can
        never describe different problems. ``source_term`` deliberately calls
        *this* rather than ``exact_solution``: a caller-supplied
        ``exact_solution_function`` need not be a Laplacian eigenfunction, so
        scaling it by ``dim * pi^2`` would not be its source.

        See ``source_term`` for why the previous ``(x, y)``-only form was
        identically zero at ``dim == 1``. At ``dim == 2`` this is unchanged.
        """
        if isinstance(coords, Tensor):
            u = torch.ones_like(coords[:, 0])
            for d in range(self.dim):
                u = u * torch.sin(np.pi * coords[:, d])
            return u
        u_np: NDArray[np.float32] = np.ones_like(coords[:, 0])
        for d in range(self.dim):
            u_np = u_np * np.sin(np.pi * coords[:, d])
        return u_np


COLE_HOPF_N_TERMS: int = 50
"""Minimum number of Fourier-Bessel terms retained by
``BurgersOperator.exact_solution``'s Cole-Hopf series (shared verbatim by the
torch and numpy branches).

This is a floor, not the actual count: the coefficient envelope
``I_n(R)/I_0(R) ~ exp(-n^2/(2R))`` with ``R = 1/(2*pi*nu)`` widens as the
viscosity shrinks, so :func:`_cole_hopf_coefficients` adds terms as needed.
50 terms already truncate below float64 round-off for ``nu >= 0.01``."""

COLE_HOPF_MAX_TERMS: int = 4096
"""Hard cap on the adaptive Cole-Hopf term count.

Bounds both work and memory: the series is evaluated as an
``(n_points, n_terms)`` float64 array. The cap only binds for ``nu < 1e-6``,
three orders of magnitude below :data:`COLE_HOPF_MIN_RESOLVED_VISCOSITY`,
where the series has already lost all significance to cancellation."""

COLE_HOPF_TERM_TOLERANCE: float = 1e-17
"""Relative coefficient magnitude at which the Cole-Hopf series is truncated.

Terms are retained while ``I_n(R)/I_0(R) ~ exp(-n^2/(2R)) >= tolerance``,
i.e. up to ``n = sqrt(2*R*ln(1/tolerance))``. Chosen just below float64
machine epsilon so that truncation is never the dominant error term."""

COLE_HOPF_CLAMP_EPS: float = 1e-14
"""Strict-positivity floor applied to the Cole-Hopf denominator ``phi`` before
forming ``u = -2*nu*phi_x/phi`` in ``BurgersOperator.exact_solution``.

The exponentially-scaled coefficients satisfy ``c_0 + sum |c_n| == 1``
exactly (``e^-R * (I_0(R) + 2*sum I_n(R)) == 1``), so ``phi`` is bounded above
by 1 and this *absolute* floor is simultaneously a *relative* floor at ~45 ULP
of the series' own l1 norm -- it engages only where ``phi`` has been consumed
by cancellation and carries no significant digits.

Analytically ``phi > 0`` everywhere (it is ``exp`` of a real number smoothed by
the heat kernel); the floor exists purely to keep the *computed* value positive
once ``nu < COLE_HOPF_MIN_RESOLVED_VISCOSITY`` pushes ``phi`` into float64
round-off (measured raw minimum -4.9e-16 at ``nu=0.001``). Lowering it does not
buy accuracy, it only lets meaningless round-off reach the denominator:
measured over a 401-point grid at ``nu=0.001, t=0``, a floor of 1e-30 restores
the historical ~4e13 blow-up while 1e-14 keeps ``max|u| = 0.59``. Raising it
would clip physically meaningful values (``min phi = 1.5e-14`` at ``nu=0.01``).
Guarded by ``tests/pde/test_operators.py::TestBurgersColeHopf::
test_denominator_is_strictly_positive_across_viscosities``."""

COLE_HOPF_MIN_RESOLVED_VISCOSITY: float = 0.01
"""Smallest viscosity at which the float64 Cole-Hopf series resolves ``u``
across the *whole* domain.

``phi`` spans ``[exp(-2R), 1]`` with ``R = 1/(2*pi*nu)``, while the round-off of
the summation is ~machine epsilon of its unit l1 norm. Significance is
therefore lost wherever ``phi`` falls under that floor, which first happens at
``x = 0`` (where ``phi = exp(-2R)``) once ``exp(-2R) ~ 1e-16``, i.e.
``nu ~ 0.009``.

**The degraded region is not a neighbourhood of the origin -- it is a
left-hand interval whose width grows rapidly as the viscosity falls, and it
covers most of the domain well before ``nu`` reaches 1e-3.** Since
``phi(x, 0) = exp(-R*(1 + cos(pi*x)))`` (exponentially scaled), the value
drops below the :data:`COLE_HOPF_CLAMP_EPS` floor for every ``x`` left of::

    x_c(nu) = arccos(2*pi*nu*ln(1/COLE_HOPF_CLAMP_EPS) - 1) / pi

(and ``x_c = 0``, i.e. no degraded region, once the argument exceeds 1 at
``nu >= 1/(pi*ln(1/COLE_HOPF_CLAMP_EPS)) = 0.00987`` -- which is what sets
this constant, rounded up to 0.01). Left of ``x_c`` the clamp pins ``u`` to
~0 while the true solution is O(1), so the *absolute* error there is as large
as the solution itself; right of it the series is accurate to float64.

Measured ``max |u(x,0) + sin(pi*x)|`` on a 401-point grid (float64 internals,
before the float32 cast), with the region satisfying ``err > 1e-3``::

    nu      degraded region     fraction of domain   max err   x_c(nu)
    1.0     none                             0.0%    6e-16      0
    0.1     none                             0.0%    3e-15      0
    0.01    none                             0.0%    9.6e-4     0
    0.009   x <= 0.23                       20.2%    0.33       0.19
    0.005   x <= 0.52                       50.4%    0.98       0.50
    0.001   x <= 0.80                       79.6%    1.0        0.79

(``nu = 0.01`` sits exactly on the boundary: on a finer 4001-point grid its
max error is 1.2e-3, confined to ``x <= 0.10``.)

``t = 0`` is the worst case: diffusion lifts ``phi``'s minimum, so the
degraded interval shrinks with time (at ``nu = 0.005`` it spans ``x <~ 0.5``
at ``t = 0`` and has vanished by ``t = 1``, checked against a dps=300 mpmath
reference).

This is intrinsic to the Fourier-Bessel representation rather than to this
implementation -- ``ive`` itself is only accurate to float64 relative
precision, so no summation scheme can recover the lost digits. Below this
threshold :func:`_cole_hopf_coefficients` emits a ``cole_hopf_underresolved``
warning (once per distinct viscosity, via the cache)."""

COLE_HOPF_COEFFICIENT_CACHE_SIZE: int = 32
"""Number of distinct viscosities for which :func:`_cole_hopf_coefficients`
retains its (time-independent) Fourier-Bessel coefficients. Operators are
typically constructed with a handful of viscosities, so a small cache removes
the ``ive`` evaluation from every ``exact_solution`` call."""


@lru_cache(maxsize=COLE_HOPF_COEFFICIENT_CACHE_SIZE)
def _cole_hopf_coefficients(
    viscosity: float,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Fourier-Bessel coefficients of the Cole-Hopf potential for 1D Burgers.

    The Cole-Hopf transformation ``u = -2*nu*phi_x/phi`` maps
    ``u_t + u*u_x = nu*u_xx`` to the heat equation ``phi_t = nu*phi_xx``. For the
    standard benchmark initial condition ``u(x, 0) = -sin(pi*x)`` on ``[0, 1]``::

        phi(x, 0) = exp(-1/(2*nu) * int_0^x u(s, 0) ds)
                  = exp((1 - cos(pi*x)) / (2*pi*nu))
                  = e^R * exp(-R*cos(pi*x)),      R = 1 / (2*pi*nu)

    and the modified-Bessel generating function
    ``exp(z*cos(theta)) = I_0(z) + 2*sum_n I_n(z)*cos(n*theta)`` evaluated at
    ``z = -R`` (using ``I_n(-R) = (-1)^n * I_n(R)``) gives::

        phi(x, 0) / e^R = I_0(R) + 2*sum_n (-1)^n * I_n(R) * cos(n*pi*x)

    The Dirichlet condition ``u(0, t) = u(1, t) = 0`` is the Neumann condition
    ``phi_x = 0`` on the potential, under which every cosine mode decays as
    ``exp(-n^2*pi^2*nu*t)``. The coefficients are therefore time-independent,
    which is why they can be cached here and reused for every ``t``.

    The alternating ``(-1)^n`` is load-bearing and was verified numerically:
    dropping it produces the Cole-Hopf image of ``u(x, 0) = +sin(pi*x)``, and
    setting every coefficient to 1 -- the historical defect -- produces the
    image of a Dirac comb (a valid solution to the wrong problem, whose
    truncated denominator is negative over half the domain).

    ``scipy.special.ive`` (exponentially scaled, ``ive(n, R) = I_n(R)*e^-R``) is
    used rather than ``iv`` because ``iv`` overflows for small viscosity
    (``iv(0, R) = 4.2e+67`` already at ``nu = 0.001``). The common ``e^-R``
    factor cancels between ``phi_x`` and ``phi``, so ``u`` is unchanged, and the
    scaled coefficients gain the exact normalisation ``c_0 + sum |c_n| == 1``.

    Args:
        viscosity: Kinematic viscosity ``nu``; must be strictly positive.

    Returns:
        ``(c0, n, c)`` with ``c0 = ive(0, R)``, ``n = [1, ..., n_terms]`` and
        ``c[k] = 2*(-1)^(k+1)*ive(k+1, R)``, all float64. The arrays are cached
        and shared; callers must treat them as read-only.

    Raises:
        ValueError: If ``viscosity`` is not strictly positive.

    """
    if viscosity <= 0.0:
        raise ValueError(
            f"Cole-Hopf requires a strictly positive viscosity, got {viscosity}",
        )

    r = 1.0 / (2.0 * np.pi * viscosity)
    n_needed = int(np.ceil(np.sqrt(2.0 * r * np.log(1.0 / COLE_HOPF_TERM_TOLERANCE)))) + 1
    n_terms = min(max(n_needed, COLE_HOPF_N_TERMS), COLE_HOPF_MAX_TERMS)

    if viscosity < COLE_HOPF_MIN_RESOLVED_VISCOSITY:
        logger.warning(
            "cole_hopf_underresolved",
            viscosity=viscosity,
            min_resolved_viscosity=COLE_HOPF_MIN_RESOLVED_VISCOSITY,
            n_terms=n_terms,
            reason=(
                "phi spans [exp(-2R), 1] and loses float64 significance over "
                "the whole interval x < arccos(2*pi*nu*ln(1/clamp_eps) - 1)/pi "
                "(~50% of the domain at nu=0.005, ~80% at nu=0.001), not just "
                "near x=0"
            ),
        )

    n = np.arange(1, n_terms + 1, dtype=np.float64)
    coefficients: NDArray[np.float64] = 2.0 * ((-1.0) ** n) * ive(n, r)
    return float(ive(0.0, r)), n, coefficients


class BurgersOperator(PDEOperator):
    """Burgers equation operator: u_t + u·∇u = ν∇²u.

    The Burgers equation is a fundamental nonlinear PDE that:
    - Models fluid dynamics and shock formation
    - Serves as a simplified Navier-Stokes equation
    - Exhibits both advection and diffusion

    This implementation supports:
    - Time-dependent and steady-state cases
    - Variable viscosity
    - 1D and 2D domains

    The time-dependent case is pinned to the standard Basdevant / Cole-Hopf
    benchmark, and :meth:`initial_condition`, :meth:`boundary_value` and
    :meth:`exact_solution` all describe that one problem::

        u_t + u*u_x = nu*u_xx    on x in [0, 1]
        u(x, 0) = -sin(pi*x)
        u(0, t) = u(1, t) = 0    (homogeneous Dirichlet)

    ``exact_solution`` is the closed-form Cole-Hopf series for exactly this
    initial/boundary data, so ``exact_solution(coords, time=0.0)`` reproduces
    ``initial_condition(coords)`` to machine precision.
    """

    name = "burgers"
    description = "Burgers equation: u_t + u·∇u = ν∇²u"
    pde_type = PDEType.BURGERS
    is_time_dependent = True
    is_linear = False
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        viscosity: float | None = None,
    ) -> None:
        """Initialize Burgers operator.

        Args:
            config: PDE configuration.
            viscosity: Kinematic viscosity (overrides config if provided).

        """
        super().__init__(config)
        self.viscosity = viscosity if viscosity is not None else config.diffusion_coeff
        # PDEConfig.is_time_dependent defaults to False, and unconditionally
        # assigning it here silently downgraded every caller that never
        # mentioned the flag (e.g. _centaur_common.build_pde_operator) from
        # the class-level `is_time_dependent = True` default to False, which
        # made exact_solution() return None unconditionally (the flat-reward
        # defect tracked as P0-1 in docs/CODE_HYGIENE_AUDIT.md). Only honour
        # config.is_time_dependent when the caller actually set it — either
        # explicitly at construction (``PDEConfig(..., is_time_dependent=...)``)
        # or via a later mutation — so an explicit False still disables the
        # Cole-Hopf solution (see test_steady_returns_none) while an unset
        # config keeps Burgers' true default of being time-dependent.
        if "is_time_dependent" in config.model_fields_set:
            self.is_time_dependent = config.is_time_dependent

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
        time: float | None = None,
    ) -> PDEResidual:
        """Compute the *steady-state* Burgers residual: R = u·∇u - ν∇²u.

        The time derivative ``u_t`` is deliberately absent (see the "Steady
        state" comment below): this operator evaluates the residual of a single
        spatial configuration and is never handed a time history, so it cannot
        form ``u_t``. Consequently the usual "residual vanishes on the exact
        solution" manufactured-solution check does **not** apply to Burgers --
        :meth:`exact_solution` solves the *time-dependent* equation, whose
        steady residual is non-zero by construction. Correctness of
        :meth:`exact_solution` is pinned instead by its agreement with
        :meth:`initial_condition` at ``t = 0``.

        Args:
            u: Solution values at ``coords``.
            coords: Collocation points (N, dim); requires grad for autodiff.
            compute_derivatives: Whether to return the derivative dictionary.
            time: Unused; accepted for interface compatibility.

        Returns:
            The steady-state residual and its norms.

        """
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u))

        # Nonlinear advection term: u · ∇u
        #
        # ``u.squeeze()`` on the accumulator as well as the operand, matching
        # ``AdvectionDiffusionOperator`` (which already does this) and the
        # ``(N,)``/``(N, 1)`` contract in :meth:`PDEOperator.residual`'s
        # docstring. Accumulating into ``zeros_like(u)`` instead broadcast a
        # ``(N, 1)`` input against the ``(N,)`` derivative and produced an
        # ``(N, N)`` residual -- silently, because ``zeros_like(u)`` makes every
        # row identical, so ``l2_norm`` and ``max_norm`` still came out right.
        # What was wrong was ``PDEResidual.values``: N=500 yielded 250 000
        # entries where callers document and reshape ``(N,)``
        # (``basis_selection.py`` assigns it to ``PDEState.residuals`` and then
        # reshapes to a square grid), plus an O(N^2) allocation.
        advection = torch.zeros_like(u.squeeze())
        for d in range(self.dim):
            du_dx = derivatives.get(f"u_x{d}", torch.zeros_like(u.squeeze()))
            advection = advection + u.squeeze() * du_dx

        # Steady state: u·∇u = ν∇²u
        residual_values = advection - self.viscosity * laplacian

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Burgers equation has no explicit source term."""
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute boundary values: homogeneous Dirichlet ``u(0, t) = u(1, t) = 0``.

        This is the boundary data of the Cole-Hopf benchmark solved by
        :meth:`exact_solution` (whose ``-sin(pi*x)`` initial condition already
        vanishes at both endpoints, and whose Neumann potential condition
        ``phi_x = 0`` keeps it vanishing for all ``t``).

        Previously this returned a ``0.5*(1 - tanh(w*(x - x0)))`` shock profile,
        i.e. ``u(0, t) = 1``, ``u(1, t) = 0`` -- a *third* problem, inconsistent
        with both the initial condition and the exact solution.
        """
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Compute initial condition ``u(x, 0) = -sin(pi*x)``.

        This is the standard Basdevant / Cole-Hopf benchmark profile, and it is
        the initial datum that :meth:`exact_solution` propagates: the two agree
        to machine precision at ``t = 0``.

        Previously this returned ``sin(2*pi*x)``, which is neither what
        :meth:`exact_solution` propagates nor compatible with the homogeneous
        Dirichlet data of :meth:`boundary_value` at ``x = 0.5``.
        """
        if isinstance(coords, Tensor):
            x = coords[:, 0]
            return -torch.sin(np.pi * x)
        else:
            x = coords[:, 0]
            return (-np.sin(np.pi * x)).astype(np.float32)

    def cole_hopf_potential(
        self,
        coords: NDArray[np.float32] | NDArray[np.float64],
        time: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the Cole-Hopf potential ``phi`` and its ``x``-derivative.

        This is the numerator/denominator pair behind :meth:`exact_solution`,
        exposed (in full float64, *before* the :data:`COLE_HOPF_CLAMP_EPS`
        positivity floor) so that the denominator's sign and magnitude can be
        inspected directly -- the historical defect was a *negative* denominator
        that the floor silently converted into a finite but meaningless ``u``.

        Args:
            coords: Points (N, dim); only the first column (``x``) is used.
            time: Time at which to evaluate (default 0).

        Returns:
            ``(phi, dphi)``, both float64 of shape (N,). ``phi`` is scaled by
            ``e^-R`` relative to the true potential, which cancels in
            ``u = -2*nu*dphi/phi``.

        """
        c0, n, coefficients = _cole_hopf_coefficients(self.viscosity)
        t = time if time is not None else 0.0
        pi = float(np.pi)

        amplitude = coefficients * np.exp(-((n * pi) ** 2) * self.viscosity * t)
        angle = np.outer(np.asarray(coords[:, 0], dtype=np.float64), n) * pi
        phi: NDArray[np.float64] = c0 + np.sum(np.cos(angle) * amplitude, axis=-1)
        dphi: NDArray[np.float64] = -np.sum(np.sin(angle) * (amplitude * n * pi), axis=-1)
        return phi, dphi

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Cole-Hopf exact solution for the 1D viscous Burgers equation.

        The Cole-Hopf transformation ``u = -2*nu*(phi_x/phi)`` converts the
        nonlinear Burgers equation into the linear heat equation. For the
        benchmark data ``u(x, 0) = -sin(pi*x)`` on ``[0, 1]`` with
        ``u(0, t) = u(1, t) = 0`` (see :meth:`initial_condition` and
        :meth:`boundary_value`), the exact solution is::

            u(x, t) = -2*nu * sum_n  c_n * (-n*pi) * exp(-n^2*pi^2*nu*t) * sin(n*pi*x)
                             / ( c_0 + sum_n c_n * exp(-n^2*pi^2*nu*t) * cos(n*pi*x) )

            c_0 = ive(0, R),  c_n = 2*(-1)^n*ive(n, R),  R = 1/(2*pi*nu)

        with ``ive`` the exponentially-scaled modified Bessel function of the
        first kind. The coefficients are derived and cached in
        :func:`_cole_hopf_coefficients`; every one of them was previously
        hardcoded to 1, which is the Cole-Hopf image of a Dirac comb rather than
        of a sinusoid.

        The series is assembled in float64 (the denominator legitimately reaches
        ``1.5e-14`` at ``nu = 0.01``, which would underflow float32) and cast
        back to the input dtype on return.

        For ``nu < COLE_HOPF_MIN_RESOLVED_VISCOSITY`` accuracy is lost over the
        whole left-hand interval ``x < arccos(2*pi*nu*ln(1/eps) - 1)/pi``, not
        merely in a neighbourhood of ``x = 0``: that is ~20% of the domain at
        ``nu = 0.009``, ~50% at ``nu = 0.005`` and ~80% at ``nu = 0.001``, and
        inside it ``u`` is pinned to ~0 while the true solution is O(1). Only
        the far field can be trusted there; see
        :data:`COLE_HOPF_MIN_RESOLVED_VISCOSITY` for the measured table.

        Args:
            coords: Points (N, dim) with spatial coordinates.
            time: Time at which to evaluate (default 0).

        Returns:
            Exact solution values (N,) in the input's dtype/device, or None for
            a steady-state (non-time-dependent) configuration, which has no
            Cole-Hopf solution.

        """
        if not self.is_time_dependent:
            return None

        t = time if time is not None else 0.0
        nu = self.viscosity

        if isinstance(coords, Tensor):
            c0, n_np, coefficients_np = _cole_hopf_coefficients(nu)
            pi = float(np.pi)
            # torch.tensor (not as_tensor) copies, so the cached read-only
            # numpy arrays are never aliased into a mutable tensor.
            n = torch.tensor(n_np, dtype=torch.float64, device=coords.device)
            coefficients = torch.tensor(
                coefficients_np,
                dtype=torch.float64,
                device=coords.device,
            )
            # float64 assembly: gradients still flow through coords, since
            # .to(float64) and the final .to(coords.dtype) are differentiable.
            x = coords[:, 0].to(torch.float64)
            amplitude = coefficients * torch.exp(-((n * pi) ** 2) * nu * t)
            angle = x.unsqueeze(-1) * n.unsqueeze(0) * pi
            phi = c0 + (torch.cos(angle) * amplitude).sum(dim=-1)
            dphi = -(torch.sin(angle) * (amplitude * n * pi)).sum(dim=-1)
            u = -2.0 * nu * dphi / phi.clamp(min=COLE_HOPF_CLAMP_EPS)
            return u.to(coords.dtype)

        phi_np, dphi_np = self.cole_hopf_potential(coords, time=t)
        u_np = -2.0 * nu * dphi_np / np.maximum(phi_np, COLE_HOPF_CLAMP_EPS)
        return u_np.astype(np.float32)

    def convergence_rate(self, h_values: list[float], errors: list[float]) -> float:
        """Compute convergence rate from h-refinement study.

        Given errors at different mesh sizes h, fits log(error) = p*log(h) + C
        to estimate the convergence order p.

        Args:
            h_values: Mesh sizes (decreasing).
            errors: Corresponding L2 errors.

        Returns:
            Estimated convergence rate p.

        """
        log_h = np.log(np.array(h_values))
        log_e = np.log(np.array(errors))
        # Linear regression: log(e) = p * log(h) + C
        coeffs = np.polyfit(log_h, log_e, 1)
        return float(coeffs[0])


GAUSSIAN_PULSE_WIDTH_FRACTION: float = 0.1
"""Fraction of the mean domain extent used as the Gaussian-pulse standard
deviation (``sigma = GAUSSIAN_PULSE_WIDTH_FRACTION * np.mean(domain_size)``).
Shared verbatim by ``AdvectionDiffusionOperator.initial_condition``,
``AdvectionDiffusionOperator.exact_solution``, and
``HeatOperator.initial_condition``."""


class AdvectionDiffusionOperator(PDEOperator):
    """Advection-Diffusion equation: u_t + a·∇u = ν∇²u + f.

    This linear PDE models:
    - Heat/mass transport with convection
    - Pollutant dispersion
    - Tracer transport

    Supports:
    - Steady-state (a·∇u = ν∇²u + f)
    - Time-dependent
    - Variable advection velocity
    """

    name = "advection_diffusion"
    description = "Advection-Diffusion: u_t + a·∇u = ν∇²u + f"
    pde_type = PDEType.ADVECTION_DIFFUSION
    is_time_dependent = True
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        advection_velocity: list[float] | None = None,
        diffusion: float | None = None,
    ) -> None:
        """Initialize advection-diffusion operator.

        Args:
            config: PDE configuration.
            advection_velocity: Advection velocity vector.
            diffusion: Diffusion coefficient.

        """
        super().__init__(config)
        self.advection_velocity = np.array(
            advection_velocity if advection_velocity is not None else config.advection_coeff,
            dtype=np.float32,
        )
        self.diffusion = diffusion if diffusion is not None else config.diffusion_coeff
        self.is_time_dependent = config.is_time_dependent

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute advection-diffusion residual: R = a·∇u - ν∇²u - f."""
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u))
        source = self.source_term(coords)

        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        # Advection term: a · ∇u
        advection = torch.zeros_like(u.squeeze())
        velocity = torch.tensor(self.advection_velocity, dtype=coords.dtype, device=coords.device)
        for d in range(self.dim):
            du_dx = derivatives.get(f"u_x{d}", torch.zeros_like(u.squeeze()))
            advection = advection + velocity[d] * du_dx

        # Steady state: a·∇u = ν∇²u + f
        residual_values = advection - self.diffusion * laplacian - source

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute source term (default: zero)."""
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute boundary values."""
        if isinstance(coords, Tensor):
            return torch.full(
                (coords.shape[0],),
                self.config.boundary_value,
                dtype=coords.dtype,
                device=coords.device,
            )
        return np.full(coords.shape[0], self.config.boundary_value, dtype=np.float32)

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Compute initial condition (Gaussian pulse)."""
        center = (self.domain_min + self.domain_max) / 2
        sigma = GAUSSIAN_PULSE_WIDTH_FRACTION * np.mean(self.domain_size)

        if isinstance(coords, Tensor):
            center_t = torch.tensor(center, dtype=coords.dtype, device=coords.device)
            dist_sq = torch.sum((coords - center_t) ** 2, dim=-1)
            return torch.exp(-dist_sq / (2 * sigma**2))
        else:
            dist_sq = np.sum((coords - center) ** 2, axis=-1)
            return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Exact solution for constant advection (advected Gaussian)."""
        if time is None:
            return None

        # Advected center
        center = (self.domain_min + self.domain_max) / 2 + self.advection_velocity * time
        sigma = GAUSSIAN_PULSE_WIDTH_FRACTION * np.mean(self.domain_size) + np.sqrt(
            2 * self.diffusion * time
        )

        if isinstance(coords, Tensor):
            center_t = torch.tensor(center, dtype=coords.dtype, device=coords.device)
            dist_sq = torch.sum((coords - center_t) ** 2, dim=-1)
            return torch.exp(-dist_sq / (2 * sigma**2))
        else:
            dist_sq = np.sum((coords - center) ** 2, axis=-1)
            return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)


class HeatOperator(PDEOperator):
    """Heat equation operator: u_t = κ∇²u + f.

    The heat equation describes:
    - Heat conduction
    - Diffusion processes
    - Random walk/Brownian motion

    Supports:
    - Time-dependent evolution
    - Variable thermal diffusivity
    - Source terms (heat generation)
    """

    name = "heat"
    description = "Heat equation: u_t = κ∇²u + f"
    pde_type = PDEType.HEAT
    is_time_dependent = True
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        diffusivity: float | None = None,
    ) -> None:
        """Initialize heat operator.

        Args:
            config: PDE configuration.
            diffusivity: Thermal diffusivity κ.

        """
        super().__init__(config)
        self.diffusivity = diffusivity if diffusivity is not None else config.diffusion_coeff

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute heat equation residual (steady state): R = -κ∇²u - f."""
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u))
        source = self.source_term(coords)

        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        # Steady state: 0 = κ∇²u + f  =>  R = -κ∇²u - f
        residual_values = -self.diffusivity * laplacian - source

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute heat source term."""
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute boundary temperature."""
        if isinstance(coords, Tensor):
            return torch.full(
                (coords.shape[0],),
                self.config.boundary_value,
                dtype=coords.dtype,
                device=coords.device,
            )
        return np.full(coords.shape[0], self.config.boundary_value, dtype=np.float32)

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Compute initial temperature distribution."""
        # Hot spot in center
        center = (self.domain_min + self.domain_max) / 2
        sigma = GAUSSIAN_PULSE_WIDTH_FRACTION * np.mean(self.domain_size)

        if isinstance(coords, Tensor):
            center_t = torch.tensor(center, dtype=coords.dtype, device=coords.device)
            dist_sq = torch.sum((coords - center_t) ** 2, dim=-1)
            return torch.exp(-dist_sq / (2 * sigma**2))
        else:
            dist_sq = np.sum((coords - center) ** 2, axis=-1)
            return np.exp(-dist_sq / (2 * sigma**2)).astype(np.float32)


class NavierStokesOperator(PDEOperator):
    """Incompressible 2D Navier-Stokes operator.

    Governing equations:
        u_t + (u dot nabla)u = -nabla p + nu * laplacian u  (momentum)
        nabla dot u = 0                                       (continuity)

    Where u = (u_x, u_y) is velocity and p is pressure.

    Implements the Taylor-Green vortex benchmark with exact analytical solution,
    ideal for SBIR validation against PhysicsNeMo and classical solvers:
        u_x = -cos(x)sin(y)exp(-2*nu*t)
        u_y =  sin(x)cos(y)exp(-2*nu*t)
        p   = -(cos(2x) + cos(2y))exp(-4*nu*t) / 4
    """

    name = "navier_stokes"
    description = "Incompressible 2D Navier-Stokes: u_t + (u.nabla)u = -nabla p + nu*laplacian u"
    pde_type = PDEType.NAVIER_STOKES
    is_time_dependent = True
    is_linear = False
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        reynolds_number: float | None = None,
    ) -> None:
        """Initialize Navier-Stokes operator.

        Args:
            config: PDE configuration.
            reynolds_number: Reynolds number Re = UL/nu. If provided,
                viscosity is computed as nu = UL/Re with U=L=1.

        """
        super().__init__(config)
        if reynolds_number is not None:
            self.viscosity = 1.0 / reynolds_number
            self.reynolds_number = reynolds_number
        else:
            self.viscosity = config.diffusion_coeff
            self.reynolds_number = 1.0 / self.viscosity if self.viscosity > 0 else float("inf")

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
        time: float | None = None,
    ) -> PDEResidual:
        """Compute NS momentum residual for velocity field.

        Input u should have shape (N, 2) for (u_x, u_y).
        Computes: R = (u dot nabla)u_x - nu * laplacian(u_x).
        """
        if u.dim() == 1:
            u = u.unsqueeze(-1)

        coords = coords.requires_grad_(True)

        if u.shape[-1] >= 2:
            ux = u[:, 0:1]
            uy = u[:, 1:2]
        else:
            # 1D-u fallback: treat the single component as ``u_x`` with the
            # transverse component identically zero. ``uy`` is then a
            # constant tensor with no grad path through ``coords``, so
            # calling ``autograd.grad(uy, coords, ...)`` below would error
            # with "element 0 of tensors does not require grad". The
            # downstream code at the ``if grad_ux is not None and
            # grad_uy is not None`` check already handles ``grad_uy is None``
            # by routing into the zero-residual fallback, so we mirror that
            # contract here by short-circuiting to ``None`` rather than
            # making a doomed autograd call.
            ux = u
            uy = torch.zeros_like(u)

        derivatives: dict[str, Tensor] = {}

        # Symmetry guard: only call autograd on a velocity component when
        # it actually carries grad through ``coords``. The 1D-u fallback
        # above makes ``uy`` a constant zero placeholder with no grad
        # path; the same situation can also arise for ``ux`` when callers
        # pass a detached or constant tensor (e.g., a numerical-stencil
        # baseline being benchmarked against the autodiff residual). In
        # either case ``torch.autograd.grad(<no-grad tensor>, coords, ...)``
        # raises "element 0 of tensors does not require grad". Setting
        # the corresponding gradient to ``None`` is already a supported
        # state downstream (the ``if grad_ux is not None and grad_uy is
        # not None`` check below routes into the zero-residual fallback).
        if ux.requires_grad or ux.grad_fn is not None:
            grad_ux = torch.autograd.grad(
                ux,
                coords,
                grad_outputs=torch.ones_like(ux),
                create_graph=True,
                allow_unused=True,
            )[0]
        else:
            grad_ux = None
            logger.debug(
                "ns_residual_ux_no_grad_path",
                u_shape=tuple(u.shape),
                coords_shape=tuple(coords.shape),
            )

        if uy.requires_grad or uy.grad_fn is not None:
            grad_uy = torch.autograd.grad(
                uy,
                coords,
                grad_outputs=torch.ones_like(uy),
                create_graph=True,
                allow_unused=True,
            )[0]
        else:
            grad_uy = None
            logger.debug(
                "ns_residual_uy_no_grad_path",
                u_shape=tuple(u.shape),
                coords_shape=tuple(coords.shape),
                reason="1d_fallback" if u.shape[-1] < 2 else "detached_input",
            )

        if grad_ux is not None and grad_uy is not None:
            dux_dx = grad_ux[:, 0]
            dux_dy = grad_ux[:, 1] if self.dim > 1 else torch.zeros_like(dux_dx)
            duy_dx = grad_uy[:, 0]
            duy_dy = grad_uy[:, 1] if self.dim > 1 else torch.zeros_like(duy_dx)

            derivatives["ux_x"] = dux_dx
            derivatives["ux_y"] = dux_dy
            derivatives["uy_x"] = duy_dx
            derivatives["uy_y"] = duy_dy
            derivatives["continuity"] = dux_dx + duy_dy

            d2ux_dx2 = torch.autograd.grad(
                dux_dx.unsqueeze(-1),
                coords,
                grad_outputs=torch.ones(coords.shape[0], 1, device=coords.device),
                create_graph=True,
                allow_unused=True,
            )[0]
            d2ux_dy2 = torch.autograd.grad(
                dux_dy.unsqueeze(-1),
                coords,
                grad_outputs=torch.ones(coords.shape[0], 1, device=coords.device),
                create_graph=True,
                allow_unused=True,
            )[0]

            laplacian_ux = torch.zeros_like(dux_dx)
            if d2ux_dx2 is not None:
                laplacian_ux = laplacian_ux + d2ux_dx2[:, 0]
            if d2ux_dy2 is not None:
                laplacian_ux = laplacian_ux + d2ux_dy2[:, 1]

            advection_ux = ux.squeeze() * dux_dx + uy.squeeze() * dux_dy
            momentum_x = advection_ux - self.viscosity * laplacian_ux
            derivatives["momentum_x"] = momentum_x

            residual_values = momentum_x
        else:
            residual_values = torch.zeros(coords.shape[0], device=coords.device)

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """NS has no explicit source for Taylor-Green vortex."""
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Boundary values from exact solution."""
        return self.exact_solution(coords, time=time)

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Taylor-Green vortex exact solution.

        u_x(x,y,t) = -cos(x)sin(y)exp(-2*nu*t)
        u_y(x,y,t) =  sin(x)cos(y)exp(-2*nu*t)

        Args:
            coords: Points (N, 2) with x,y coordinates.
            time: Time value (default 0).

        Returns:
            Velocity field (N, 2) with [u_x, u_y] components.

        """
        t = time if time is not None else 0.0
        decay = np.exp(-2.0 * self.viscosity * t)

        if isinstance(coords, Tensor):
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else torch.zeros_like(x)
            ux = -torch.cos(x) * torch.sin(y) * decay
            uy = torch.sin(x) * torch.cos(y) * decay
            return torch.stack([ux, uy], dim=-1)
        else:
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else np.zeros_like(x)
            ux = -np.cos(x) * np.sin(y) * decay
            uy = np.sin(x) * np.cos(y) * decay
            return np.stack([ux, uy], axis=-1).astype(np.float32)

    def exact_pressure(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Taylor-Green exact pressure: p = -(cos(2x)+cos(2y))*exp(-4*nu*t)/4."""
        t = time if time is not None else 0.0
        decay = np.exp(-4.0 * self.viscosity * t)

        if isinstance(coords, Tensor):
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else torch.zeros_like(x)
            return -(torch.cos(2 * x) + torch.cos(2 * y)) * decay / 4.0
        else:
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else np.zeros_like(x)
            return (-(np.cos(2 * x) + np.cos(2 * y)) * decay / 4.0).astype(np.float32)

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Initial condition = exact solution at t=0."""
        return self.exact_solution(coords, time=0.0)


class LShapedPoissonOperator(PDEOperator):
    r"""Poisson equation on L-shaped domain.

    Solves -Delta u = f on the L-shaped domain [-1,1]^2 \\ [0,1]x[-1,0]
    with Dirichlet boundary conditions.

    Known singular solution near reentrant corner at the origin:
        u(r, theta) = r^(2/3) * sin(2*theta/3)

    where (r, theta) are polar coordinates centred at the origin.
    This is the standard benchmark for adaptive mesh refinement
    because the solution gradient is singular at the reentrant corner,
    requiring concentrated mesh refinement.

    The source term for the singular solution is f = 0 (harmonic).
    """

    name = "poisson_lshaped"
    description = "Poisson equation on L-shaped domain with corner singularity"
    pde_type = PDEType.POISSON
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        source_function: Callable[[NDArray | Tensor], NDArray | Tensor] | None = None,
    ) -> None:
        """Initialize L-shaped Poisson operator.

        Args:
            config: PDE configuration. The ``geometry`` field should have
                ``geometry_type=GeometryType.L_SHAPED``.
            source_function: Custom source term. Defaults to f=0 (for the
                singular benchmark solution).

        """
        super().__init__(config)
        self.diffusion = config.diffusion_coeff
        self._source_function = source_function

        # Build geometry from config or default to L-shaped
        if config.geometry.geometry_type == GeometryType.L_SHAPED:
            self.geometry: DomainGeometry = create_geometry(config.geometry)
        else:
            self.geometry = LShapedDomain(scale=config.geometry.scale)

        self._scale = (
            config.geometry.scale if config.geometry.geometry_type == GeometryType.L_SHAPED else 1.0
        )

        logger.info(
            "lshaped_poisson_operator_created",
            scale=self._scale,
            diffusion=self.diffusion,
        )

    # ------------------------------------------------------------------
    # Exact (singular) solution utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _polar_from_cartesian(
        x: Tensor,
        y: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Convert Cartesian (x, y) to polar (r, theta).

        The angle theta is measured from the positive x-axis in the range
        [0, 2*pi) so that the reentrant-corner singular solution is
        well-defined on the full L-shaped domain.
        """
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        # Map to [0, 2*pi)
        theta = torch.where(theta < 0, theta + 2 * np.pi, theta)
        return r, theta

    @staticmethod
    def _singular_solution(r: Tensor, theta: Tensor) -> Tensor:
        """Evaluate the benchmark singular solution.

        u(r, theta) = r^(2/3) * sin(2*theta/3)
        """
        return r.pow(2.0 / 3.0) * torch.sin(2.0 * theta / 3.0)

    @staticmethod
    def _singular_solution_np(
        x: NDArray[np.float32],
        y: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Numpy version of the singular solution."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0, theta + 2 * np.pi, theta)
        # Avoid 0^(2/3) producing nan
        result = np.where(
            r > 0,
            np.power(r, 2.0 / 3.0) * np.sin(2.0 * theta / 3.0),
            0.0,
        )
        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # PDEOperator interface
    # ------------------------------------------------------------------

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute exact singular solution u = r^(2/3) sin(2*theta/3).

        Args:
            coords: Point coordinates (N, 2).
            time: Unused (steady-state problem).

        Returns:
            Solution values (N,).

        """
        if isinstance(coords, Tensor):
            x, y = coords[:, 0], coords[:, 1]
            r, theta = self._polar_from_cartesian(x, y)
            return self._singular_solution(r, theta)
        else:
            return self._singular_solution_np(coords[:, 0], coords[:, 1])

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute source/forcing term.

        The default singular solution is harmonic (f = 0). A custom
        source function can be provided via the constructor.

        Args:
            coords: Point coordinates (N, 2).
            time: Unused.

        Returns:
            Source term values (N,).

        """
        if self._source_function is not None:
            return self._source_function(coords)

        # Default: f = 0 (the singular benchmark is harmonic)
        if isinstance(coords, Tensor):
            return torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
        return np.zeros(coords.shape[0], dtype=np.float32)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Compute Dirichlet boundary values from the exact solution.

        Args:
            coords: Boundary point coordinates (N_b, 2).
            time: Unused.

        Returns:
            Boundary values (N_b,).

        """
        return self.exact_solution(coords, time)

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute Poisson residual: R = -nu * Laplacian(u) - f.

        Uses automatic differentiation to compute the Laplacian.

        Args:
            u: Solution values at collocation points (N,) or (N, 1).
            coords: Collocation point coordinates (N, 2).
            compute_derivatives: Whether to return derivative tensors.

        Returns:
            PDEResidual with values and norms.

        """
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u.squeeze()))
        source = self.source_term(coords)

        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        # Residual: -nu * Laplacian(u) - f = 0
        residual_values = -self.diffusion * laplacian - source

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())

        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def is_boundary_point(
        self,
        coords: NDArray[np.float32] | Tensor,
        tolerance: float = DEFAULT_BOUNDARY_TOLERANCE,
    ) -> NDArray[np.bool_] | Tensor:
        """Determine which points are on the L-shaped boundary.

        Overrides the rectangular base implementation with geometry-aware
        boundary detection.

        Args:
            coords: Point coordinates (N, dim).
            tolerance: Distance tolerance for boundary detection.

        Returns:
            Boolean mask (N,) with True for boundary points.

        """
        if isinstance(coords, Tensor):
            return self.geometry.is_boundary(coords, tol=tolerance)
        else:
            coords_t = torch.from_numpy(coords)
            result = self.geometry.is_boundary(coords_t, tol=tolerance)
            return result.numpy()

    def generate_collocation_points(
        self,
        n_points: int,
        method: str = "random",
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate collocation points inside the L-shaped domain.

        Uses the geometry's interior sampling which handles rejection
        sampling automatically.

        Args:
            n_points: Number of interior points to generate.
            method: Sampling method (only 'random' supported for L-shaped).
            seed: Random seed for reproducibility.

        Returns:
            Collocation points (n_points, 2).

        """
        if seed is not None:
            torch.manual_seed(seed)
        points = self.geometry.sample_interior(n_points)
        return points.numpy().astype(np.float32)

    def generate_boundary_points(
        self,
        n_points_per_face: int,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Generate points on the L-shaped domain boundary.

        The L-shape has 6 boundary segments, so total points will be
        approximately 6 * n_points_per_face (distributed proportionally
        to segment length).

        Args:
            n_points_per_face: Approximate points per boundary segment.
            seed: Random seed.

        Returns:
            Boundary points (N_boundary, 2).

        """
        if seed is not None:
            torch.manual_seed(seed)
        # 6 segments total; distribute proportionally
        total = n_points_per_face * 6
        points = self.geometry.sample_boundary(total)
        return points.numpy().astype(np.float32)

    def compute_error(
        self,
        u_pred: Tensor,
        coords: Tensor,
    ) -> dict[str, float]:
        """Compute error metrics against the exact singular solution.

        Args:
            u_pred: Predicted solution values (N,).
            coords: Point coordinates (N, 2).

        Returns:
            Dictionary with 'l2_error', 'linf_error', and 'mse'.

        """
        u_exact = self.exact_solution(coords)
        assert isinstance(u_exact, Tensor)
        diff = u_pred - u_exact

        l2 = float(torch.sqrt(torch.mean(diff**2)).item())
        linf = float(torch.max(torch.abs(diff)).item())
        mse = float(torch.mean(diff**2).item())

        logger.debug(
            "error_computed",
            l2_error=l2,
            linf_error=linf,
            mse=mse,
        )

        return {"l2_error": l2, "linf_error": linf, "mse": mse}


# ---------------------------------------------------------------------------
# Out-of-distribution operators (held-out generalisation benchmarks)
#
# These two operators expose PDE residual structure the domain-trained FNet
# evaluator has never seen, providing the "held-out" generalisation test for
# the LLM-prior MCTS ablation: Helmholtz adds an oscillatory zeroth-order
# (reaction) term, and the biharmonic operator is fourth-order.
# ---------------------------------------------------------------------------

DEFAULT_HELMHOLTZ_WAVENUMBER: float = 1.0
"""Default Helmholtz wavenumber ``k`` when neither the constructor argument
nor a positive ``reaction_coeff`` supplies one. Surfaced as a named constant
so no magic number lives in the operator body."""


class HelmholtzOperator(PDEOperator):
    """Helmholtz equation operator: ∇²u + k²u = f.

    The Helmholtz equation models time-harmonic wave phenomena (acoustics,
    electromagnetics, scattering). Relative to Poisson it adds an
    oscillatory zeroth-order (reaction) term ``k²u`` that the FNet residual
    encoding trained on diffusion-dominated problems has not seen — the
    "benchmark-graveyard" out-of-distribution structure.

    Manufactured solution (homogeneous Dirichlet on ``[0, 1]^dim``)::

        u(x) = ∏_d sin(π x_d)
        ∇²u  = -dim · π² · u
        f    = (k² - dim · π²) · u

    The wavenumber ``k`` resolves in priority order: explicit ``wavenumber``
    argument → positive ``config.reaction_coeff`` (interpreted as ``k²``) →
    :data:`DEFAULT_HELMHOLTZ_WAVENUMBER`.
    """

    name = "helmholtz"
    description = "Helmholtz equation: ∇²u + k²u = f"
    pde_type = PDEType.HELMHOLTZ
    is_time_dependent = False
    is_linear = True
    order = 2

    def __init__(
        self,
        config: PDEConfig,
        wavenumber: float | None = None,
    ) -> None:
        """Initialize the Helmholtz operator.

        Args:
            config: PDE configuration.
            wavenumber: Helmholtz wavenumber ``k``. When ``None`` it falls
                back to ``sqrt(config.reaction_coeff)`` if ``reaction_coeff``
                is positive, otherwise :data:`DEFAULT_HELMHOLTZ_WAVENUMBER`.

        """
        super().__init__(config)
        if wavenumber is not None:
            resolved_k = float(wavenumber)
        elif config.reaction_coeff > 0.0:
            resolved_k = float(np.sqrt(config.reaction_coeff))
        else:
            resolved_k = DEFAULT_HELMHOLTZ_WAVENUMBER
        if resolved_k <= 0.0:
            raise ValueError(f"Helmholtz wavenumber must be positive, got {resolved_k}")
        self.wavenumber = resolved_k

    def _manufactured(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Evaluate the manufactured solution ``∏_d sin(π x_d)``."""
        if isinstance(coords, Tensor):
            product = torch.ones(coords.shape[0], dtype=coords.dtype, device=coords.device)
            for d in range(self.dim):
                product = product * torch.sin(np.pi * coords[:, d])
            return product
        product = np.ones(coords.shape[0], dtype=np.float32)
        for d in range(self.dim):
            product = product * np.sin(np.pi * coords[:, d])
        return product

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute Helmholtz residual: R = ∇²u + k²u - f."""
        derivatives = self.compute_derivatives(u, coords)
        laplacian = derivatives.get("laplacian", torch.zeros_like(u))

        source = self.source_term(coords)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        # Flatten every term to (N,) so a (N, 1) `u` (and the
        # `torch.zeros_like(u)` laplacian fallback) cannot trigger implicit
        # (N, N) broadcasting in the residual sum.
        u_flat = u.reshape(-1)
        residual_values = laplacian.reshape(-1) + (self.wavenumber**2) * u_flat - source.reshape(-1)

        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())
        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives=derivatives if compute_derivatives else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Source term ``f = (k² - dim · π²) · u`` for the manufactured solution."""
        coefficient = self.wavenumber**2 - self.dim * (np.pi**2)
        return coefficient * self._manufactured(coords)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Dirichlet boundary values (the manufactured solution vanishes there)."""
        if self.config.boundary_condition == BoundaryCondition.DIRICHLET:
            if isinstance(coords, Tensor):
                return torch.full(
                    (coords.shape[0],),
                    self.config.boundary_value,
                    dtype=coords.dtype,
                    device=coords.device,
                )
            return np.full(coords.shape[0], self.config.boundary_value, dtype=np.float32)
        return self._manufactured(coords)

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Manufactured exact solution ``∏_d sin(π x_d)``."""
        return self._manufactured(coords)


class BiharmonicOperator(PDEOperator):
    """Biharmonic equation operator: ∇⁴u = f.

    The biharmonic (fourth-order) operator models thin-plate bending and
    Stokes-flow stream functions. Its fourth-order residual structure is a
    qualitatively different "held-out" generalisation target from the
    second-order operators the FNet evaluator was trained on.

    Manufactured solution (homogeneous Dirichlet on ``[0, 1]^dim``)::

        u(x) = ∏_d sin(π x_d)
        ∇²u  = -dim · π² · u
        ∇⁴u  = (dim · π²)² · u
        f    = (dim · π²)² · u
    """

    name = "biharmonic"
    description = "Biharmonic equation: ∇⁴u = f"
    pde_type = PDEType.BIHARMONIC
    is_time_dependent = False
    is_linear = True
    order = 4

    def __init__(self, config: PDEConfig) -> None:
        """Initialize the biharmonic operator."""
        super().__init__(config)

    def _manufactured(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Evaluate the manufactured solution ``∏_d sin(π x_d)``."""
        if isinstance(coords, Tensor):
            product = torch.ones(coords.shape[0], dtype=coords.dtype, device=coords.device)
            for d in range(self.dim):
                product = product * torch.sin(np.pi * coords[:, d])
            return product
        product = np.ones(coords.shape[0], dtype=np.float32)
        for d in range(self.dim):
            product = product * np.sin(np.pi * coords[:, d])
        return product

    def _laplacian_autograd(self, field: Tensor, coords: Tensor) -> Tensor:
        """Laplacian of a scalar ``field`` w.r.t. ``coords`` via autograd.

        Returns a graph-connected tensor (``create_graph=True``) so it can be
        differentiated again to form the biharmonic. Returns zeros when
        ``field`` is disconnected from ``coords`` (mirrors
        :meth:`PDEOperator.compute_derivatives`).
        """
        n_points = coords.shape[0]
        zeros = torch.zeros(n_points, dtype=coords.dtype, device=coords.device)
        if not field.requires_grad and field.grad_fn is None:
            return zeros
        if field.dim() == 1:
            field = field.unsqueeze(-1)
        grad = torch.autograd.grad(
            field,
            coords,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return zeros
        laplacian = zeros
        for d in range(self.dim):
            grad_d = grad[:, d : d + 1]
            if grad_d.grad_fn is None and not grad_d.requires_grad:
                continue
            grad2 = torch.autograd.grad(
                grad_d,
                coords,
                grad_outputs=torch.ones_like(grad_d),
                create_graph=True,
                allow_unused=True,
            )[0]
            if grad2 is not None:
                laplacian = laplacian + grad2[:, d]
        return laplacian

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
    ) -> PDEResidual:
        """Compute biharmonic residual: R = ∇⁴u - f."""
        if not u.requires_grad and u.grad_fn is None:
            # Disconnected solution — derivatives are undefined; the
            # biharmonic term is zero (consistent with the base class).
            biharmonic = torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
            laplacian = biharmonic
        else:
            # Use the caller's coords directly when they already carry grad (the
            # connected case, so ``u`` stays attached); otherwise differentiate
            # against a private clone so we never flip the caller's leaf tensor
            # to ``requires_grad`` in place. A solution connected to parameters
            # but not to ``coords`` yields a zero biharmonic (∇⁴u w.r.t. coords
            # is undefined), consistent with the base class.
            if coords.requires_grad:
                work_coords = coords
            else:
                work_coords = coords.detach().clone().requires_grad_(True)
            laplacian = self._laplacian_autograd(u, work_coords)
            biharmonic = self._laplacian_autograd(laplacian, work_coords)

        source = self.source_term(coords)
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source).to(coords.device)

        residual_values = biharmonic - source
        l2_norm = float(torch.sqrt(torch.mean(residual_values**2)).item())
        max_norm = float(torch.max(torch.abs(residual_values)).item())
        return PDEResidual(
            values=residual_values,
            l2_norm=l2_norm,
            max_norm=max_norm,
            derivatives={"laplacian": laplacian, "biharmonic": biharmonic}
            if compute_derivatives
            else {},
        )

    def source_term(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Source term ``f = (dim · π²)² · u`` for the manufactured solution."""
        coefficient = (self.dim * (np.pi**2)) ** 2
        return coefficient * self._manufactured(coords)

    def boundary_value(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor:
        """Dirichlet boundary values (the manufactured solution vanishes there)."""
        if self.config.boundary_condition == BoundaryCondition.DIRICHLET:
            if isinstance(coords, Tensor):
                return torch.full(
                    (coords.shape[0],),
                    self.config.boundary_value,
                    dtype=coords.dtype,
                    device=coords.device,
                )
            return np.full(coords.shape[0], self.config.boundary_value, dtype=np.float32)
        return self._manufactured(coords)

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Manufactured exact solution ``∏_d sin(π x_d)``."""
        return self._manufactured(coords)
