"""Kolmogorov-forward generator ``L = A + D + J`` for the stochastic layer.

The generator bundles a drift model (linear ``f(x)=Ax+b`` exactly closed, or a
generic callable, documented approximate), a constant diffusion factor ``g``
(``Q = g gᵀ``), and an optional compound-Poisson jump term. Constructing a
generator whose config carries a jump term with positive rate **requires** a
jump-semigroup model — the jump component is never silently dropped (AC2).

Spec: specs/stochastic_galerkin_nke.spec.md (AC2, change-doc requirement 1b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from src.pde.stochastic.config import JumpConfig, StochasticGeneratorConfig
from src.pde.stochastic.errors import JumpModelMissingError, StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState


@runtime_checkable
class DriftModel(Protocol):
    """Pointwise drift ``f: (N, d) -> (N, d)`` evaluated on sample batches."""

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the drift at a batch of points."""
        ...


@runtime_checkable
class JumpSemigroup(Protocol):
    """Approximation of the jump semigroup ``e^{dt·J}`` on mixture states."""

    def apply(self, state: GaussianMixtureState, dt: float) -> GaussianMixtureState:
        """Advance the mixture state through the jump flow over ``dt``."""
        ...


@dataclass(frozen=True)
class LinearDrift:
    """Linear drift ``f(x) = A x + b`` — the exactly-closed moment path."""

    matrix: Tensor
    bias: Tensor

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or self.matrix.shape[0] != self.matrix.shape[1]:
            msg = f"drift matrix must be square; got {tuple(self.matrix.shape)}"
            raise StochasticConfigurationError(msg)
        if self.bias.shape != (self.matrix.shape[0],):
            msg = (
                f"drift bias shape {tuple(self.bias.shape)} does not match "
                f"matrix dim {self.matrix.shape[0]}"
            )
            raise StochasticConfigurationError(msg)

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate ``x Aᵀ + b`` on a batch of shape (N, d)."""
        return x @ self.matrix.T + self.bias


class KolmogorovGenerator:
    """Kolmogorov-forward generator ``L = A + D + J`` on Gaussian-mixture states."""

    def __init__(
        self,
        config: StochasticGeneratorConfig,
        *,
        drift: DriftModel | None = None,
        jump_semigroup: JumpSemigroup | None = None,
    ) -> None:
        """Build the generator; enforces the AC2 jump-model contract.

        Args:
            config: Validated generator configuration.
            drift: Drift model. Optional when ``config.drift_matrix`` is set
                (a ``LinearDrift`` is built from the config); required for a
                callable-drift generator.
            jump_semigroup: Jump-flow model. Required whenever the config
                carries a jump term with ``rate > 0``.

        Raises:
            JumpModelMissingError: Jump configured but no jump model supplied.
            StochasticConfigurationError: No drift available.

        """
        if config.has_jump and jump_semigroup is None:
            msg = (
                "generator has a jump term (rate "
                f"{config.jump.rate if config.jump else 0.0} > 0) but no jump-semigroup "
                "model was supplied; the jump component is never silently ignored. "
                "Pass jump_semigroup=MDNJumpSemigroup(...) (trained) or "
                "jump_semigroup=AnalyticCompoundPoissonMoments(...) (moment oracle), "
                "or set jump=None / rate=0 to drop the jump term explicitly."
            )
            raise JumpModelMissingError(msg)

        a_matrix = config.drift_matrix_tensor()
        if drift is None:
            if a_matrix is None:
                msg = (
                    "no drift available: config.drift_matrix is None and no callable "
                    "drift was supplied"
                )
                raise StochasticConfigurationError(msg)
            drift = LinearDrift(matrix=a_matrix, bias=config.drift_bias_tensor())

        self.config = config
        self.dim = config.dim
        self.drift: DriftModel = drift
        g = config.diffusion_tensor()
        self.diffusion: Tensor = g
        self.q_matrix: Tensor = g @ g.T
        self.jump_semigroup = jump_semigroup

    @property
    def has_jump(self) -> bool:
        """True iff the generator carries a jump term with positive rate."""
        return self.config.has_jump

    @property
    def jump(self) -> JumpConfig | None:
        """The jump-term configuration, if any."""
        return self.config.jump

    @property
    def is_linear(self) -> bool:
        """True iff the drift is the exactly-closed linear path."""
        return isinstance(self.drift, LinearDrift)

    def linear_drift(self) -> LinearDrift:
        """Return the drift as a ``LinearDrift`` (raises when callable-drift)."""
        if not isinstance(self.drift, LinearDrift):
            msg = "generator drift is a generic callable, not LinearDrift"
            raise StochasticConfigurationError(msg)
        return self.drift

    def drift_at(self, x: Tensor) -> Tensor:
        """Evaluate the drift on a batch of shape (N, d) in float64."""
        return self.drift(x.to(torch.float64))
