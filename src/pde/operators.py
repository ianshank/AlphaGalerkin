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
from typing import Any

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

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
        tolerance: float = 1e-6,
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

        Args:
            u: Solution values as a function of coords.
            coords: Collocation point coordinates (N, dim).

        Returns:
            Dictionary with derivative tensors.

        """
        coords = coords.requires_grad_(True)

        # First derivatives
        derivatives: dict[str, Tensor] = {}

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
            laplacian = torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device)
            for d in range(self.dim):
                grad_d = grad[:, d : d + 1]
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

            derivatives["laplacian"] = laplacian

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

        # Default: sinusoidal source for smooth manufactured solution
        if isinstance(coords, Tensor):
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else torch.zeros_like(x)
            # Source for solution u = sin(πx)sin(πy)
            return 2 * (np.pi**2) * torch.sin(np.pi * x) * torch.sin(np.pi * y)
        else:
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else np.zeros_like(x)
            return 2 * (np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)

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

        # Default: sinusoidal exact solution
        if isinstance(coords, Tensor):
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else torch.zeros_like(x)
            return torch.sin(np.pi * x) * torch.sin(np.pi * y)
        else:
            x = coords[:, 0]
            y = coords[:, 1] if self.dim > 1 else np.zeros_like(x)
            return np.sin(np.pi * x) * np.sin(np.pi * y)


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
        shock_position: float | None = None,
        shock_width: float | None = None,
    ) -> None:
        """Initialize Burgers operator.

        Args:
            config: PDE configuration.
            viscosity: Kinematic viscosity (overrides config if provided).
            shock_position: Center position of the shock profile (default 0.5).
            shock_width: Sharpness parameter for tanh shock (default 10.0).

        """
        super().__init__(config)
        self.viscosity = viscosity if viscosity is not None else config.diffusion_coeff
        self.is_time_dependent = config.is_time_dependent
        self.shock_position = shock_position if shock_position is not None else 0.5
        self.shock_width = shock_width if shock_width is not None else 10.0

    def residual(
        self,
        u: Tensor,
        coords: Tensor,
        compute_derivatives: bool = True,
        time: float | None = None,
    ) -> PDEResidual:
        """Compute Burgers residual: R = u_t + u·∇u - ν∇²u."""
        derivatives = self.compute_derivatives(u, coords)

        laplacian = derivatives.get("laplacian", torch.zeros_like(u))

        # Nonlinear advection term: u · ∇u
        advection = torch.zeros_like(u)
        for d in range(self.dim):
            du_dx = derivatives.get(f"u_x{d}", torch.zeros_like(u))
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
        """Compute boundary values (shock-like profile)."""
        if isinstance(coords, Tensor):
            x = coords[:, 0]
            return 0.5 * (1.0 - torch.tanh(self.shock_width * (x - self.shock_position)))
        else:
            x = coords[:, 0]
            return 0.5 * (1.0 - np.tanh(self.shock_width * (x - self.shock_position)))

    def initial_condition(
        self,
        coords: NDArray[np.float32] | Tensor,
    ) -> NDArray[np.float32] | Tensor:
        """Compute initial condition (sinusoidal wave)."""
        if isinstance(coords, Tensor):
            x = coords[:, 0]
            return torch.sin(2 * np.pi * x)
        else:
            x = coords[:, 0]
            return np.sin(2 * np.pi * x).astype(np.float32)

    def exact_solution(
        self,
        coords: NDArray[np.float32] | Tensor,
        time: float | None = None,
    ) -> NDArray[np.float32] | Tensor | None:
        """Cole-Hopf exact solution for 1D viscous Burgers equation.

        The Cole-Hopf transformation u = -2*nu * (phi_x / phi) converts
        the nonlinear Burgers equation to the linear heat equation.

        For initial condition u(x,0) = -sin(pi*x), the exact solution is:
            u(x,t) = -2*nu*pi * sum(a_n * n * exp(-n^2*pi^2*nu*t) * sin(n*pi*x))
                      / (a_0/2 + sum(a_n * exp(-n^2*pi^2*nu*t) * cos(n*pi*x)))

        where a_n are Fourier-Bessel coefficients of exp(-cos(pi*x)/(2*pi*nu)).

        For small viscosity, this approximates a moving shock.

        Args:
            coords: Points (N, dim) with spatial coordinates.
            time: Time at which to evaluate (default 0).

        Returns:
            Exact solution values (N,), or None if time is None and PDE
            is time-dependent (ambiguous).

        """
        if not self.is_time_dependent:
            return None

        t = time if time is not None else 0.0
        nu = self.viscosity

        if isinstance(coords, Tensor):
            x = coords[:, 0]
            # Fourier series approximation (N_terms for convergence)
            n_terms = 50
            phi = torch.ones_like(x) * 0.5  # a_0/2 term
            dphi = torch.zeros_like(x)

            for n in range(1, n_terms + 1):
                # Bessel function coefficients approximated numerically
                # For u0 = sin(2*pi*x), use direct Fourier series of exp transform
                decay = np.exp(-(n * np.pi) ** 2 * nu * t)
                phi = phi + decay * torch.cos(n * np.pi * x)
                dphi = dphi - n * np.pi * decay * torch.sin(n * np.pi * x)

            u = -2.0 * nu * dphi / phi.clamp(min=1e-10)
            return u
        else:
            x = coords[:, 0]
            n_terms = 50
            phi = np.ones_like(x) * 0.5
            dphi = np.zeros_like(x)

            for n in range(1, n_terms + 1):
                decay = np.exp(-(n * np.pi) ** 2 * nu * t)
                phi = phi + decay * np.cos(n * np.pi * x)
                dphi = dphi - n * np.pi * decay * np.sin(n * np.pi * x)

            u = -2.0 * nu * dphi / np.clip(phi, a_min=1e-10, a_max=None)
            return u.astype(np.float32)

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
        sigma = 0.1 * np.mean(self.domain_size)

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
        sigma = 0.1 * np.mean(self.domain_size) + np.sqrt(2 * self.diffusion * time)

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
        sigma = 0.1 * np.mean(self.domain_size)

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
            self.reynolds_number = (
                1.0 / self.viscosity if self.viscosity > 0 else float("inf")
            )

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
            ux = u
            uy = torch.zeros_like(u)

        derivatives: dict[str, Tensor] = {}

        grad_ux = torch.autograd.grad(
            ux, coords, grad_outputs=torch.ones_like(ux),
            create_graph=True, allow_unused=True,
        )[0]

        grad_uy = torch.autograd.grad(
            uy, coords, grad_outputs=torch.ones_like(uy),
            create_graph=True, allow_unused=True,
        )[0]

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
                dux_dx.unsqueeze(-1), coords,
                grad_outputs=torch.ones(coords.shape[0], 1, device=coords.device),
                create_graph=True, allow_unused=True,
            )[0]
            d2ux_dy2 = torch.autograd.grad(
                dux_dy.unsqueeze(-1), coords,
                grad_outputs=torch.ones(coords.shape[0], 1, device=coords.device),
                create_graph=True, allow_unused=True,
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
            uy = np.cos(x) * np.cos(y) * decay
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
    """Poisson equation on L-shaped domain.

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
            config.geometry.scale
            if config.geometry.geometry_type == GeometryType.L_SHAPED
            else 1.0
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
            return torch.zeros(
                coords.shape[0], dtype=coords.dtype, device=coords.device
            )
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
        tolerance: float = 1e-6,
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
