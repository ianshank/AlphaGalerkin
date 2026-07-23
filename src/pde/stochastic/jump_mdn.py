"""Jump-semigroup models: the analytic moment oracle and the trained MDN.

The exact jump flow ``e^{dt·J}`` for a compound-Poisson term convolves the
density with a random-sum increment and leaves the Gaussian-mixture family;
models here approximate its action on mixture states.

``AnalyticCompoundPoissonMoments`` is the **exact first-two-moment oracle**:
for the isolated pure-jump generator, ``dm/dt = λ μ_ξ`` and
``dP/dt = λ E[ξξᵀ] = λ (Σ_ξ + μ_ξ μ_ξᵀ)`` hold exactly (independent
increments), so the finite-``dt`` update is exact, not merely O(dt). It is
the test/benchmark oracle for AC3/AC4 and the moment-matching target the
trained MDN is measured against.

``MDNJumpSemigroup`` is the learned flow: packed mixture params + dt →
same-K packed params, **residual-parameterized with the delta scaled by dt**
so ``dt → 0`` is the identity: means shift additively, Cholesky diagonals
scale by ``exp(dt·Δ)`` (always positive, exact identity at dt=0), and a
``DEFAULT_MDN_MIN_SCALE²`` diagonal floor prevents covariance collapse. The
network runs float32; ``advance()`` converts at the float64 boundary.

Spec: specs/stochastic_galerkin_nke.spec.md (AC3, task 1.4).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.pde.stochastic.config import (
    DEFAULT_COV_JITTER,
    DEFAULT_MDN_MIN_SCALE,
    JumpConfig,
    MDNJumpConfig,
)
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState

_LOG_2PI = 1.8378770664093453
"""log(2π) as a float64 literal (avoids re-deriving per call)."""


class AnalyticCompoundPoissonMoments:
    """Exact first-two-moment jump flow for a compound-Poisson term."""

    def __init__(self, jump: JumpConfig) -> None:
        self.jump = jump
        self.rate = jump.rate
        mu = torch.tensor(jump.jump_mean, dtype=torch.float64)
        sigma = torch.tensor(jump.jump_cov, dtype=torch.float64)
        self._mu = mu
        self._second_moment = sigma + torch.outer(mu, mu)

    def advance(self, state: GaussianMixtureState, dt: float) -> GaussianMixtureState:
        """Advance the mixture through the exact jump moment flow over ``dt``."""
        shift = (self.rate * dt) * self._mu.to(state.dtype)
        production = (self.rate * dt) * self._second_moment.to(state.dtype)
        return GaussianMixtureState(
            weights=state.weights,
            means=state.means + shift,
            covariances=state.covariances + production,
        )


def _tril_size(dim: int) -> int:
    return dim * (dim + 1) // 2


def unpack_batch(packed: Tensor, k: int, d: int) -> tuple[Tensor, Tensor, Tensor]:
    """Unpack a (B, P) batch into weights (B,K), means (B,K,d), covs (B,K,d,d)."""
    b = packed.shape[0]
    expected = k + k * d + k * _tril_size(d)
    if packed.ndim != 2 or packed.shape[1] != expected:
        msg = f"packed batch must have shape (B, {expected}); got {tuple(packed.shape)}"
        raise StochasticConfigurationError(msg)
    weights = packed[:, :k]
    means = packed[:, k : k + k * d].reshape(b, k, d)
    tril_flat = packed[:, k + k * d :].reshape(b, k, _tril_size(d))
    rows, cols = torch.tril_indices(d, d, device=packed.device)
    cov = torch.zeros(b, k, d, d, dtype=packed.dtype, device=packed.device)
    cov[:, :, rows, cols] = tril_flat
    strict = torch.tril(cov, diagonal=-1)
    cov = cov + strict.transpose(-1, -2)
    return weights, means, cov


def pack_batch(weights: Tensor, means: Tensor, covariances: Tensor) -> Tensor:
    """Inverse of :func:`unpack_batch` (no state validation — batch form)."""
    b, k, d = means.shape
    rows, cols = torch.tril_indices(d, d, device=means.device)
    tril = covariances[:, :, rows, cols].reshape(b, k * _tril_size(d))
    return torch.cat([weights, means.reshape(b, k * d), tril], dim=1)


def batched_mixture_nll(packed: Tensor, samples: Tensor, k: int, d: int) -> Tensor:
    """Per-batch-element mixture NLL: (B, P) params × (B, S, d) samples → (B,).

    Computed with jittered Cholesky factors (matching the single-state
    ``GaussianMixtureState.log_prob`` math, batched over B).
    """
    weights, means, covs = unpack_batch(packed, k, d)
    jitter = DEFAULT_COV_JITTER * torch.eye(d, dtype=packed.dtype, device=packed.device)
    chol = torch.linalg.cholesky(covs + jitter)  # (B, K, d, d)
    diff = samples.unsqueeze(1) - means.unsqueeze(2)  # (B, K, S, d)
    solved = torch.linalg.solve_triangular(chol, diff.transpose(-1, -2), upper=False)
    mahalanobis = solved.pow(2).sum(dim=-2)  # (B, K, S)
    log_det = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)  # (B, K)
    comp_logp = -0.5 * (mahalanobis + d * _LOG_2PI) - log_det.unsqueeze(-1)
    log_w = torch.log(weights.clamp_min(torch.finfo(packed.dtype).tiny)).unsqueeze(-1)
    log_prob = torch.logsumexp(log_w + comp_logp, dim=1)  # (B, S)
    return -log_prob.mean(dim=1)


class MDNJumpSemigroup(nn.Module):
    """Learned jump semigroup: residual mixture-to-mixture map scaled by dt."""

    def __init__(self, config: MDNJumpConfig) -> None:
        super().__init__()
        self.config = config
        k, d = config.n_components, config.dim
        self._packed_size = k + k * d + k * _tril_size(d)
        self._delta_size = k + k * d + k * _tril_size(d)
        self.dt_embed = nn.Linear(1, config.dt_embed_dim)
        layers: list[nn.Module] = []
        in_dim = self._packed_size + config.dt_embed_dim
        for width in config.hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.GELU())
            in_dim = width
        layers.append(nn.Linear(in_dim, self._delta_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, packed: Tensor, dt: Tensor) -> Tensor:
        """Map packed mixtures (B, P) + dt (B, 1) → packed mixtures (B, P).

        Residual parameterization (identity at dt=0): weight logits and means
        shift by ``dt·Δ``; Cholesky off-diagonals shift by ``dt·Δ``; Cholesky
        diagonals scale by ``exp(dt·Δ)``; the output covariance gains a
        ``DEFAULT_MDN_MIN_SCALE²`` diagonal floor.
        """
        k, d = self.config.n_components, self.config.dim
        b = packed.shape[0]
        weights, means, covs = unpack_batch(packed, k, d)
        jitter = DEFAULT_COV_JITTER * torch.eye(d, dtype=packed.dtype, device=packed.device)
        chol = torch.linalg.cholesky(covs + jitter)  # (B, K, d, d)

        features = torch.cat([packed, self.dt_embed(dt)], dim=1)
        deltas = self.mlp(features) * dt  # residual scaled by dt
        d_logits = deltas[:, :k]
        d_means = deltas[:, k : k + k * d].reshape(b, k, d)
        d_tril = deltas[:, k + k * d :].reshape(b, k, _tril_size(d))

        tiny = torch.finfo(packed.dtype).tiny
        new_weights = torch.softmax(torch.log(weights.clamp_min(tiny)) + d_logits, dim=1)
        new_means = means + d_means

        rows, cols = torch.tril_indices(d, d, device=packed.device)
        diag_mask = rows == cols
        delta_l = torch.zeros_like(chol)
        delta_l[:, :, rows[~diag_mask], cols[~diag_mask]] = d_tril[:, :, ~diag_mask]
        diag_scale = torch.exp(d_tril[:, :, diag_mask])  # (B, K, d)
        new_chol = chol + delta_l
        new_chol = new_chol - torch.diag_embed(torch.diagonal(new_chol, dim1=-2, dim2=-1))
        new_chol = new_chol + torch.diag_embed(torch.diagonal(chol, dim1=-2, dim2=-1) * diag_scale)
        floor = (DEFAULT_MDN_MIN_SCALE**2) * torch.eye(d, dtype=packed.dtype, device=packed.device)
        new_covs = new_chol @ new_chol.transpose(-1, -2) + floor
        return pack_batch(new_weights, new_means, new_covs)

    def advance(self, state: GaussianMixtureState, dt: float) -> GaussianMixtureState:
        """``JumpSemigroup`` protocol: float64 state → float32 net → float64 state."""
        if state.n_components != self.config.n_components or state.dim != self.config.dim:
            msg = (
                f"MDN configured for K={self.config.n_components}, d={self.config.dim} "
                f"but state has K={state.n_components}, d={state.dim}"
            )
            raise StochasticConfigurationError(msg)
        packed32 = state.pack().to(torch.float32).unsqueeze(0)
        dt32 = torch.tensor([[dt]], dtype=torch.float32)
        out = self.forward(packed32, dt32).squeeze(0).to(state.dtype)
        result = GaussianMixtureState.unpack(out, state.n_components, state.dim)
        return result

    def nll(self, packed_out: Tensor, samples: Tensor) -> Tensor:
        """Mean mixture NLL of ``samples`` (B, S, d) under packed outputs (B, P)."""
        per_element = batched_mixture_nll(
            packed_out, samples, self.config.n_components, self.config.dim
        )
        return per_element.mean()
