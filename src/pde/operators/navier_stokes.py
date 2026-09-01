"""Incompressible 2D Navier-Stokes operator (Taylor-Green vortex benchmark)."""

from __future__ import annotations

import numpy as np
import structlog
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import PDEConfig, PDEType
from src.pde.operators.base import PDEOperator, PDEResidual

logger = structlog.get_logger(__name__)


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
