"""Advection-Diffusion equation operator: u_t + a.grad(u) = nu*grad^2(u) + f."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import PDEConfig, PDEType
from src.pde.operators.base import GAUSSIAN_PULSE_WIDTH_FRACTION, PDEOperator, PDEResidual


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
