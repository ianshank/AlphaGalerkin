"""Heat equation operator: u_t = kappa*grad^2(u) + f."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import PDEConfig, PDEType
from src.pde.operators.base import GAUSSIAN_PULSE_WIDTH_FRACTION, PDEOperator, PDEResidual


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
