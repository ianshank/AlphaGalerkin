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
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import BoundaryCondition, PDEConfig, PDEType


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
    """Poisson equation operator: -∇²u = f

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
    """Burgers equation operator: u_t + u·∇u = ν∇²u

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
    ) -> None:
        """Initialize Burgers operator.

        Args:
            config: PDE configuration.
            viscosity: Kinematic viscosity (overrides config if provided).

        """
        super().__init__(config)
        self.viscosity = viscosity if viscosity is not None else config.diffusion_coeff
        self.is_time_dependent = config.is_time_dependent

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
            # Tanh profile: transition from 1 to 0
            return 0.5 * (1.0 - torch.tanh(10.0 * (x - 0.5)))
        else:
            x = coords[:, 0]
            return 0.5 * (1.0 - np.tanh(10.0 * (x - 0.5)))

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


class AdvectionDiffusionOperator(PDEOperator):
    """Advection-Diffusion equation: u_t + a·∇u = ν∇²u + f

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
    """Heat equation operator: u_t = κ∇²u + f

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
