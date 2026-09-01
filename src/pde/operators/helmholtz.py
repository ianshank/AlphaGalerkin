"""Helmholtz equation operator: grad^2(u) + k^2*u = f (out-of-distribution benchmark).

These two operators (this one and ``biharmonic.py``) expose PDE residual
structure the domain-trained FNet evaluator has never seen, providing the
"held-out" generalisation test for the LLM-prior MCTS ablation: Helmholtz adds
an oscillatory zeroth-order (reaction) term, and the biharmonic operator is
fourth-order.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src.pde.config import BoundaryCondition, PDEConfig, PDEType
from src.pde.operators.base import PDEOperator, PDEResidual, _manufactured_sine_product

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
        return _manufactured_sine_product(coords, self.dim)

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
