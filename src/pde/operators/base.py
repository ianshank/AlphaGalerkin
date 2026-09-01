"""Shared base types for PDE operators.

Provides:
- ``PDEResidual``: container for PDE residual computation results.
- ``PDEOperator``: abstract base class defining the PDE operator interface
  (residual computation, source terms, boundary conditions, collocation-point
  generation, and automatic-differentiation derivative helpers).

Concrete operator implementations live in sibling modules of this package
(``poisson.py``, ``burgers.py``, ``advection_diffusion.py``, ``heat.py``,
``navier_stokes.py``, ``lshaped_poisson.py``, ``helmholtz.py``,
``biharmonic.py``) and import from here rather than duplicating this
machinery. This mirrors the pattern already used by
``src/pde/operators_picogk.py``, which imports ``PDEOperator``/``PDEResidual``
from this package (previously the single ``src/pde/operators.py`` module) for
its SDF-aware helical operators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.constants import DEFAULT_BOUNDARY_TOLERANCE
from src.pde.config import PDEConfig, PDEType

logger = structlog.get_logger(__name__)


GAUSSIAN_PULSE_WIDTH_FRACTION: float = 0.1
"""Fraction of the mean domain extent used as the Gaussian-pulse standard
deviation (``sigma = GAUSSIAN_PULSE_WIDTH_FRACTION * np.mean(domain_size)``).
Shared verbatim by ``AdvectionDiffusionOperator.initial_condition``,
``AdvectionDiffusionOperator.exact_solution``, and
``HeatOperator.initial_condition`` -- both of which live in separate modules
of this package, so the constant is declared here (rather than in either
operator's own file) to avoid re-declaring it twice."""


def _manufactured_sine_product(
    coords: NDArray[np.float32] | Tensor,
    dim: int,
) -> NDArray[np.float32] | Tensor:
    """Evaluate the manufactured solution ``prod_d sin(pi * x_d)``.

    Shared verbatim by :class:`~src.pde.operators.helmholtz.HelmholtzOperator`
    and :class:`~src.pde.operators.biharmonic.BiharmonicOperator` (previously
    duplicated method-for-method as each class's private ``_manufactured`` in
    the single ``src/pde/operators.py`` module). Declared here, rather than in
    either operator's file, so the two modules do not carry independent copies
    of the same helper.

    Args:
        coords: Point coordinates (N, dim).
        dim: Number of spatial dimensions to take the product over.

    Returns:
        The manufactured solution values (N,), matching ``coords``'s array
        type (Tensor in, Tensor out; ndarray in, ndarray out).

    """
    if isinstance(coords, Tensor):
        product = torch.ones(coords.shape[0], dtype=coords.dtype, device=coords.device)
        for d in range(dim):
            product = product * torch.sin(np.pi * coords[:, d])
        return product
    product = np.ones(coords.shape[0], dtype=np.float32)
    for d in range(dim):
        product = product * np.sin(np.pi * coords[:, d])
    return product


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
