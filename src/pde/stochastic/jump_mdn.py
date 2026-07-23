"""Jump-semigroup models: the analytic moment oracle (and, in v1, the MDN).

The exact jump flow ``e^{dt·J}`` for a compound-Poisson term convolves the
density with a random-sum increment and leaves the Gaussian-mixture family;
models here approximate its action on mixture states.

``AnalyticCompoundPoissonMoments`` is the **exact first-two-moment oracle**:
for the isolated pure-jump generator, ``dm/dt = λ μ_ξ`` and
``dP/dt = λ E[ξξᵀ] = λ (Σ_ξ + μ_ξ μ_ξᵀ)`` hold exactly (independent
increments), so the finite-``dt`` update is exact, not merely O(dt). It is
the test/benchmark oracle for AC3/AC4 and the moment-matching target the
trained MDN is measured against.

Spec: specs/stochastic_galerkin_nke.spec.md (AC3, task 1.4).
"""

from __future__ import annotations

import torch

from src.pde.stochastic.config import JumpConfig
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState


class AnalyticCompoundPoissonMoments:
    """Exact first-two-moment jump flow for a compound-Poisson term."""

    def __init__(self, jump: JumpConfig) -> None:
        self.jump = jump
        self.rate = jump.rate
        mu = torch.tensor(jump.jump_mean, dtype=torch.float64)
        sigma = torch.tensor(jump.jump_cov, dtype=torch.float64)
        self._mu = mu
        self._second_moment = sigma + torch.outer(mu, mu)

    def apply(self, state: GaussianMixtureState, dt: float) -> GaussianMixtureState:
        """Advance the mixture through the exact jump moment flow over ``dt``."""
        shift = (self.rate * dt) * self._mu.to(state.dtype)
        production = (self.rate * dt) * self._second_moment.to(state.dtype)
        return GaussianMixtureState(
            weights=state.weights,
            means=state.means + shift,
            covariances=state.covariances + production,
        )
