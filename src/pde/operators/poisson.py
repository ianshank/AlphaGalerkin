"""Poisson equation operator: -grad^2 u = f."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.operators.base import PDEOperator, PDEResidual


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
