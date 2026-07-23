"""Lagrangian Galerkin projection of the A+D generator onto Gaussian mixtures.

Projecting the Kolmogorov forward equation onto the sufficient statistics
``{1, x, x xᵀ}`` of the Gaussian family yields the moment-matching ODEs

    dm/dt = E[f(X)]
    dP/dt = E[(X − m) f(X)ᵀ] + E[f(X) (X − m)ᵀ] + Q,   Q = g gᵀ

which close **exactly** for linear drift ``f(x) = A x + b`` (``dm/dt = Am+b``,
``dP/dt = AP + PAᵀ + Q`` — the OU/Lyapunov system). Generic callable drift uses
cubature (spherical-radial sigma-point) expectations — documented approximate.
K>1 mixtures propagate per component with frozen weights (exact for linear
drift; weight dynamics are spec Out of Scope).

The split flows used by the Strang composition:

    A-flow (h):  m ← e^{Ah} m + (∫₀ʰ e^{As} ds) b,  P ← e^{Ah} P e^{Aᵀh}
    D-flow (h):  P ← P + h Q

These flows do not commute, so the Strang composition has genuine O(dt²) error
in the covariance (AC4 measures exactly that).

Spec: specs/stochastic_galerkin_nke.spec.md (AC1, task 1.3).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from src.pde.stochastic.config import DEFAULT_COV_JITTER, StrangSplittingConfig
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState, pack_moments
from src.pde.stochastic.generator import KolmogorovGenerator
from src.pde.time_stepping import RK4, TimeSteppingConfig, TimeSteppingMethod

MomentRhs = Callable[[GaussianMixtureState], tuple[Tensor, Tensor]]


def _cubature_points(mean: Tensor, cov: Tensor) -> tuple[Tensor, float]:
    """Spherical-radial cubature points for E[f(X)], X ~ N(mean, cov).

    Returns 2d points ``m ± √d · L eᵢ`` (L the jittered Cholesky factor), each
    with uniform weight ``1/(2d)``.
    """
    d = mean.shape[0]
    jitter = DEFAULT_COV_JITTER * torch.eye(d, dtype=mean.dtype, device=mean.device)
    chol = torch.linalg.cholesky(cov + jitter)
    offsets = math.sqrt(float(d)) * chol.T  # rows are the scaled directions
    points = torch.cat([mean + offsets, mean - offsets], dim=0)  # (2d, d)
    return points, 1.0 / (2.0 * d)


class GalerkinMomentProjection:
    """Projected A+D moment dynamics for a ``KolmogorovGenerator``."""

    def __init__(self, generator: KolmogorovGenerator, config: StrangSplittingConfig) -> None:
        self.generator = generator
        self.config = config
        if config.ad_integrator == "exact_expm" and not generator.is_linear:
            msg = (
                "ad_integrator='exact_expm' requires linear drift; use "
                "ad_integrator='rk4' for a callable drift model"
            )
            raise StochasticConfigurationError(msg)

    # ------------------------------------------------------------------
    # Split flows (Strang substeps)
    # ------------------------------------------------------------------

    def advection_flow(self, state: GaussianMixtureState, h: float) -> GaussianMixtureState:
        """Advance the mixture through the pure-advection flow over ``h``."""
        if self.config.ad_integrator == "exact_expm":
            return self._advection_flow_exact(state, h)
        return self._integrate_flow_rk4(state, h, self._advection_moment_derivatives)

    def diffusion_flow(self, state: GaussianMixtureState, h: float) -> GaussianMixtureState:
        """Advance through the pure-diffusion flow over ``h``: P ← P + h·Q (exact)."""
        q = self.generator.q_matrix.to(state.covariances)
        return GaussianMixtureState(
            weights=state.weights,
            means=state.means,
            covariances=state.covariances + h * q,
        )

    def advection_flow_matrices(
        self,
        h: float,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Exact advection-flow operators ``(e^{Ah}, ∫₀ʰ e^{As} ds · b)``.

        Exposed so the batched parallel-in-time trainer can apply the same
        exact flow to stacked (B, K, …) moment tensors without constructing
        per-slice states. Linear drift only. Device-agnostic: the operators
        are built on ``device``.
        """
        drift = self.generator.linear_drift()
        d = self.generator.dim
        a = drift.matrix.to(dtype=dtype, device=device)
        b = drift.bias.to(dtype=dtype, device=device)
        aug = torch.zeros(d + 1, d + 1, dtype=dtype, device=device)
        aug[:d, :d] = a
        aug[:d, d] = b
        phi_aug = torch.linalg.matrix_exp(aug * h)
        return phi_aug[:d, :d], phi_aug[:d, d]

    def _advection_flow_exact(self, state: GaussianMixtureState, h: float) -> GaussianMixtureState:
        expm_a, shift = self.advection_flow_matrices(h, state.dtype, state.means.device)
        means = state.means @ expm_a.T + shift
        covariances = expm_a @ state.covariances @ expm_a.T
        return GaussianMixtureState(
            weights=state.weights,
            means=means,
            covariances=0.5 * (covariances + covariances.transpose(-1, -2)),
        )

    # ------------------------------------------------------------------
    # Unsplit moment ODE (the AC1 reference path)
    # ------------------------------------------------------------------

    def moment_rhs(self, state: GaussianMixtureState) -> tuple[Tensor, Tensor]:
        """Unsplit A+D moment derivatives ``(dm (K,d), dP (K,d,d))``."""
        dm_a, dp_a = self._advection_moment_derivatives(state)
        q = self.generator.q_matrix.to(state.covariances)
        return dm_a, dp_a + q

    def _advection_moment_derivatives(self, state: GaussianMixtureState) -> tuple[Tensor, Tensor]:
        if self.generator.is_linear:
            drift = self.generator.linear_drift()
            a = drift.matrix.to(state.means)
            b = drift.bias.to(state.means)
            dm = state.means @ a.T + b
            ap = a @ state.covariances
            dp = ap + ap.transpose(-1, -2)
            return dm, dp
        # Cubature expectations for generic drift (documented approximate).
        dms: list[Tensor] = []
        dps: list[Tensor] = []
        for k in range(state.n_components):
            mean = state.means[k]
            cov = state.covariances[k]
            points, weight = _cubature_points(mean, cov)
            f_vals = self.generator.drift_at(points).to(state.dtype)
            centered = points - mean
            cross = weight * (centered.T @ f_vals)
            dms.append(weight * f_vals.sum(dim=0))
            dps.append(cross + cross.T)
        return torch.stack(dms), torch.stack(dps)

    def packed_rhs(
        self, n_components: int, dim: int, moment_fn: MomentRhs | None = None
    ) -> Callable[[Tensor, float], Tensor]:
        """Return ``rhs_fn(u, t)`` on packed states for ``src/pde/time_stepping.py``."""
        fn = moment_fn if moment_fn is not None else self.moment_rhs

        def rhs(u: Tensor, _t: float) -> Tensor:
            state = GaussianMixtureState.unpack(u, n_components, dim)
            dm, dp = fn(state)
            return pack_moments(torch.zeros_like(state.weights), dm, dp)

        return rhs

    def propagate(self, state: GaussianMixtureState, t_grid: Tensor) -> list[GaussianMixtureState]:
        """Integrate the unsplit A+D moment ODE along ``t_grid`` (RK4, fixed substeps).

        This is the AC1 reference path — it never uses the exact flows, so the
        comparison against the analytic OU solution exercises the projected
        moment ODE itself.
        """
        times = torch.as_tensor(t_grid, dtype=torch.float64).reshape(-1)
        if times.shape[0] < 2 or bool((times[1:] <= times[:-1]).any()):
            msg = "t_grid must be strictly increasing with at least two points"
            raise StochasticConfigurationError(msg)
        stepper = self._make_rk4_stepper()
        rhs = self.packed_rhs(state.n_components, state.dim)
        out = [state]
        current = state.pack()
        t = float(times[0])
        for target in times[1:].tolist():
            stepper.dt = (target - t) / self.config.rk4_substeps
            for _ in range(self.config.rk4_substeps):
                current, t = stepper.step(current, t, rhs)
            out.append(GaussianMixtureState.unpack(current, state.n_components, state.dim))
        return out

    def _integrate_flow_rk4(
        self,
        state: GaussianMixtureState,
        h: float,
        moment_fn: MomentRhs,
    ) -> GaussianMixtureState:
        """RK4-integrate a single split flow over ``h`` (shared stepper math)."""
        stepper = self._make_rk4_stepper()
        rhs = self.packed_rhs(state.n_components, state.dim, moment_fn)
        stepper.dt = h / self.config.rk4_substeps
        u = state.pack()
        t = 0.0
        for _ in range(self.config.rk4_substeps):
            u, t = stepper.step(u, t, rhs)
        return GaussianMixtureState.unpack(u, state.n_components, state.dim)

    def _make_rk4_stepper(self) -> RK4:
        """Build an RK4 stepper from ``src/pde/time_stepping.py`` (reuse, not reinvention)."""
        cfg = TimeSteppingConfig(
            name="stochastic_moment_rk4",
            method=TimeSteppingMethod.RK4,
            dt=self.config.dt,
            t_end=self.config.t_end,
        )
        return RK4(cfg)
