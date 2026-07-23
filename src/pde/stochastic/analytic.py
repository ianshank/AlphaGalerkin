"""Closed-form OU and jump-diffusion-OU moment references (float64).

These are the *independent* analytic references the acceptance tests compare
against (AC1, AC3, AC4). They use van Loan block matrix exponentials — no
scipy dependency and no reuse of the package's own integrators, so the tests
are non-circular.

For ``dX = (A X + b) dt + g dW + dJ`` with compound-Poisson ``dJ``
(rate λ, jump sizes ξ ~ N(μ_ξ, Σ_ξ)):

- mean:        dm/dt = A m + b + λ μ_ξ
- covariance:  dP/dt = A P + P Aᵀ + Q + λ (Σ_ξ + μ_ξ μ_ξᵀ),  Q = g gᵀ

so the jump-OU forms reduce to the pure-OU forms with ``b_eff`` / ``Q_eff``.

Spec: specs/stochastic_galerkin_nke.spec.md (pinned toy problems).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _as_f64_matrix(a: Tensor, name: str) -> Tensor:
    a = torch.as_tensor(a, dtype=torch.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        msg = f"{name} must be a square matrix; got shape {tuple(a.shape)}"
        raise ValueError(msg)
    return a


def _as_f64_vector(v: Tensor, dim: int, name: str) -> Tensor:
    v = torch.as_tensor(v, dtype=torch.float64).reshape(-1)
    if v.shape[0] != dim:
        msg = f"{name} must have length {dim}; got {v.shape[0]}"
        raise ValueError(msg)
    return v


def ou_mean(a_matrix: Tensor, bias: Tensor, m0: Tensor, t: float) -> Tensor:
    """Mean of the OU process at time ``t``: solution of dm/dt = A m + b.

    Uses the augmented-system exponential ``expm([[A, b], [0, 0]] t)`` so no
    invertibility of A is assumed.

    Args:
        a_matrix: (d, d) drift matrix A.
        bias: (d,) drift bias b.
        m0: (d,) initial mean.
        t: Time (>= 0).

    Returns:
        (d,) mean m(t) in float64.

    """
    a = _as_f64_matrix(a_matrix, "a_matrix")
    d = a.shape[0]
    b = _as_f64_vector(bias, d, "bias")
    m = _as_f64_vector(m0, d, "m0")
    aug = torch.zeros(d + 1, d + 1, dtype=torch.float64)
    aug[:d, :d] = a
    aug[:d, d] = b
    phi = torch.linalg.matrix_exp(aug * t)
    state = torch.cat([m, torch.ones(1, dtype=torch.float64)])
    return (phi @ state)[:d]


def ou_covariance(a_matrix: Tensor, q_matrix: Tensor, p0: Tensor, t: float) -> Tensor:
    """Covariance of the OU process at time ``t``: solution of the Lyapunov ODE.

    dP/dt = A P + P Aᵀ + Q, via the van Loan (1978) block exponential
    ``expm([[-A, Q], [0, Aᵀ]] t) = [[·, M12], [0, M22]]`` with
    ``Ad = M22ᵀ = e^{At}`` and ``Qd = Ad @ M12 = ∫₀ᵗ e^{As} Q e^{Aᵀs} ds``,
    giving ``P(t) = Ad P0 Adᵀ + Qd``.

    Args:
        a_matrix: (d, d) drift matrix A.
        q_matrix: (d, d) diffusion production Q = g gᵀ (symmetric PSD).
        p0: (d, d) initial covariance.
        t: Time (>= 0).

    Returns:
        (d, d) covariance P(t) in float64 (symmetrized).

    """
    a = _as_f64_matrix(a_matrix, "a_matrix")
    q = _as_f64_matrix(q_matrix, "q_matrix")
    p = _as_f64_matrix(p0, "p0")
    d = a.shape[0]
    block = torch.zeros(2 * d, 2 * d, dtype=torch.float64)
    block[:d, :d] = -a
    block[:d, d:] = q
    block[d:, d:] = a.T
    exp_block = torch.linalg.matrix_exp(block * t)
    ad = exp_block[d:, d:].T
    qd = ad @ exp_block[:d, d:]
    p_t = ad @ p @ ad.T + qd
    return 0.5 * (p_t + p_t.T)


def _jump_effective_bias(bias: Tensor, rate: float, jump_mean: Tensor, dim: int) -> Tensor:
    b = _as_f64_vector(bias, dim, "bias")
    mu = _as_f64_vector(jump_mean, dim, "jump_mean")
    return b + rate * mu


def _jump_effective_q(
    q_matrix: Tensor, rate: float, jump_mean: Tensor, jump_cov: Tensor, dim: int
) -> Tensor:
    q = _as_f64_matrix(q_matrix, "q_matrix")
    mu = _as_f64_vector(jump_mean, dim, "jump_mean")
    sigma = _as_f64_matrix(jump_cov, "jump_cov")
    return q + rate * (sigma + torch.outer(mu, mu))


def jump_ou_mean(
    a_matrix: Tensor,
    bias: Tensor,
    rate: float,
    jump_mean: Tensor,
    m0: Tensor,
    t: float,
) -> Tensor:
    """Mean of the jump-diffusion OU process: OU mean with b_eff = b + λ μ_ξ."""
    a = _as_f64_matrix(a_matrix, "a_matrix")
    b_eff = _jump_effective_bias(bias, rate, jump_mean, a.shape[0])
    return ou_mean(a, b_eff, m0, t)


def jump_ou_covariance(
    a_matrix: Tensor,
    q_matrix: Tensor,
    rate: float,
    jump_mean: Tensor,
    jump_cov: Tensor,
    p0: Tensor,
    t: float,
) -> Tensor:
    """Covariance of the jump-diffusion OU: OU covariance with Q_eff = Q + λ E[ξξᵀ]."""
    a = _as_f64_matrix(a_matrix, "a_matrix")
    q_eff = _jump_effective_q(q_matrix, rate, jump_mean, jump_cov, a.shape[0])
    return ou_covariance(a, q_eff, p0, t)


def gaussian_density_on_grid(mean: Tensor, cov: Tensor, coords: Tensor) -> Tensor:
    """Density of N(mean, cov) at ``coords`` (N, d); returns (N,) in float64."""
    cov_m = _as_f64_matrix(cov, "cov")
    d = cov_m.shape[0]
    mu = _as_f64_vector(mean, d, "mean")
    pts = torch.as_tensor(coords, dtype=torch.float64)
    if pts.ndim != 2 or pts.shape[1] != d:
        msg = f"coords must have shape (N, {d}); got {tuple(pts.shape)}"
        raise ValueError(msg)
    chol = torch.linalg.cholesky(cov_m)
    diff = pts - mu
    solved = torch.linalg.solve_triangular(chol, diff.T, upper=False)
    mahalanobis = solved.pow(2).sum(dim=0)
    log_det = torch.log(torch.diagonal(chol)).sum()
    log_norm = -0.5 * d * torch.log(torch.tensor(2.0 * torch.pi, dtype=torch.float64))
    return torch.exp(log_norm - log_det - 0.5 * mahalanobis)
