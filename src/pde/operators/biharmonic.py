"""Biharmonic equation operator: grad^4(u) = f (out-of-distribution benchmark).

See ``helmholtz.py`` for the shared out-of-distribution rationale: this
operator's fourth-order residual structure is a qualitatively different
"held-out" generalisation target from the second-order operators the FNet
evaluator was trained on.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.operators.base import PDEOperator, PDEResidual, _manufactured_sine_product


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
        return _manufactured_sine_product(coords, self.dim)

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
