"""Gaussian-mixture state and basis for the stochastic Galerkin layer.

The mixture ``p(x) = Σ_k w_k N(x; m_k, P_k)`` is the Galerkin trial space onto
which the Kolmogorov-forward generator is projected. ``GaussianMixtureState``
is the immutable moment state; ``pack()``/``unpack()`` give the flat-tensor
form consumed by ``src/pde/time_stepping.py`` steppers and the MDN.

Spec: specs/stochastic_galerkin_nke.spec.md (tasks 1.2, AC1/AC3/AC6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.pde.stochastic.config import (
    _WEIGHT_SUM_ATOL,
    DEFAULT_COV_JITTER,
    GaussianMixtureBasisConfig,
)
from src.pde.stochastic.errors import StochasticConfigurationError

_LOG_2PI = math.log(2.0 * math.pi)


def _tril_size(dim: int) -> int:
    """Number of lower-triangular entries of a dim×dim matrix."""
    return dim * (dim + 1) // 2


@dataclass(frozen=True)
class GaussianMixtureState:
    """Immutable Gaussian-mixture moment state (weights, means, covariances).

    Shapes: ``weights`` (K,), ``means`` (K, d), ``covariances`` (K, d, d) with
    each covariance symmetric positive-definite (jitter is added at Cholesky
    time, not stored).
    """

    weights: Tensor
    means: Tensor
    covariances: Tensor

    def __post_init__(self) -> None:
        if self.weights.ndim != 1 or self.means.ndim != 2 or self.covariances.ndim != 3:
            msg = (
                "expected weights (K,), means (K,d), covariances (K,d,d); got "
                f"{tuple(self.weights.shape)}, {tuple(self.means.shape)}, "
                f"{tuple(self.covariances.shape)}"
            )
            raise StochasticConfigurationError(msg)
        k, d = self.means.shape
        if self.weights.shape[0] != k or self.covariances.shape != (k, d, d):
            msg = (
                f"inconsistent component shapes: weights K={self.weights.shape[0]}, "
                f"means K={k}, covariances {tuple(self.covariances.shape)}"
            )
            raise StochasticConfigurationError(msg)
        weight_sum = float(self.weights.sum())
        if abs(weight_sum - 1.0) > _WEIGHT_SUM_ATOL:
            msg = f"mixture weights must sum to 1 (got {weight_sum:.8f})"
            raise StochasticConfigurationError(msg)
        if not torch.allclose(
            self.covariances, self.covariances.transpose(-1, -2), atol=1e-7, rtol=0.0
        ):
            msg = "covariances must be symmetric"
            raise StochasticConfigurationError(msg)

    @property
    def n_components(self) -> int:
        """Mixture size K."""
        return int(self.means.shape[0])

    @property
    def dim(self) -> int:
        """State dimension d."""
        return int(self.means.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the moment tensors."""
        return self.means.dtype

    def pack(self) -> Tensor:
        """Flatten to a 1-D tensor: [weights | means | tril(covariances)].

        The layout is ``K + K*d + K*d(d+1)/2`` entries, compatible with the
        ``rhs_fn(u, t)`` contract of ``src/pde/time_stepping.py`` steppers and
        with the MDN input head.
        """
        k, d = self.n_components, self.dim
        rows, cols = torch.tril_indices(d, d, device=self.means.device)
        tril = self.covariances[:, rows, cols].reshape(k * _tril_size(d))
        return torch.cat([self.weights, self.means.reshape(k * d), tril])

    @classmethod
    def unpack(cls, vec: Tensor, n_components: int, dim: int) -> GaussianMixtureState:
        """Inverse of :meth:`pack`; reconstructs symmetric covariances."""
        k, d = n_components, dim
        expected = k + k * d + k * _tril_size(d)
        if vec.ndim != 1 or vec.shape[0] != expected:
            msg = f"packed vector must have shape ({expected},); got {tuple(vec.shape)}"
            raise StochasticConfigurationError(msg)
        weights = vec[:k]
        means = vec[k : k + k * d].reshape(k, d)
        tril_flat = vec[k + k * d :].reshape(k, _tril_size(d))
        rows, cols = torch.tril_indices(d, d, device=vec.device)
        cov = torch.zeros(k, d, d, dtype=vec.dtype, device=vec.device)
        cov[:, rows, cols] = tril_flat
        strict = torch.tril(cov, diagonal=-1)
        cov = cov + strict.transpose(-1, -2)
        return cls(weights=weights, means=means, covariances=cov)

    def log_prob(self, x: Tensor) -> Tensor:
        """Mixture log-density at points ``x`` of shape (N, d); returns (N,).

        Implemented via Cholesky factors (with ``DEFAULT_COV_JITTER`` on the
        diagonal) rather than ``torch.distributions`` so that the reference
        test against ``MixtureSameFamily`` is non-circular.
        """
        if x.ndim != 2 or x.shape[1] != self.dim:
            msg = f"x must have shape (N, {self.dim}); got {tuple(x.shape)}"
            raise StochasticConfigurationError(msg)
        x = x.to(self.dtype)
        d = self.dim
        jitter = DEFAULT_COV_JITTER * torch.eye(d, dtype=self.dtype, device=x.device)
        chol = torch.linalg.cholesky(self.covariances + jitter)  # (K, d, d)
        diff = x.unsqueeze(0) - self.means.unsqueeze(1)  # (K, N, d)
        solved = torch.linalg.solve_triangular(
            chol, diff.transpose(-1, -2), upper=False
        )  # (K, d, N)
        mahalanobis = solved.pow(2).sum(dim=-2)  # (K, N)
        log_det = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)  # (K,)
        comp_logp = -0.5 * (mahalanobis + d * _LOG_2PI) - log_det.unsqueeze(-1)  # (K, N)
        log_w = torch.log(self.weights).unsqueeze(-1)  # (K, 1)
        return torch.logsumexp(log_w + comp_logp, dim=0)

    def density_on_grid(self, coords: Tensor) -> Tensor:
        """Mixture density at grid points ``coords`` of shape (N, d); returns (N,)."""
        return torch.exp(self.log_prob(coords))

    def to_dtype(self, dtype: torch.dtype) -> GaussianMixtureState:
        """Return a copy of this state with all tensors cast to ``dtype``."""
        return GaussianMixtureState(
            weights=self.weights.to(dtype),
            means=self.means.to(dtype),
            covariances=self.covariances.to(dtype),
        )


class GaussianMixtureBasis(nn.Module):
    """Gaussian-mixture basis factory (config-or-kwargs idiom).

    Mirrors the ``MultiScaleFourierFeatures`` construction pattern: pass a
    ``GaussianMixtureBasisConfig`` or the equivalent keyword arguments. Nothing
    is learnable in v1 (``weight_dynamics="frozen"``); the ``nn.Module`` base
    keeps the door open for learnable basis parameters without an API break.
    """

    def __init__(
        self,
        config: GaussianMixtureBasisConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = GaussianMixtureBasisConfig(**kwargs)
        elif kwargs:
            msg = "pass either a config object or keyword arguments, not both"
            raise StochasticConfigurationError(msg)
        self.config = config

    def initial_state(
        self,
        means: Tensor,
        covariances: Tensor,
        weights: Tensor | None = None,
    ) -> GaussianMixtureState:
        """Build a validated ``GaussianMixtureState`` in the configured dtype.

        Args:
            means: (K, d) component means.
            covariances: (K, d, d) symmetric SPD component covariances.
            weights: Optional (K,) mixture weights; defaults to uniform.

        Returns:
            The validated state, cast to ``config.torch_dtype``.

        """
        k, d = self.config.n_components, self.config.dim
        if tuple(means.shape) != (k, d):
            msg = f"means must have shape ({k}, {d}); got {tuple(means.shape)}"
            raise StochasticConfigurationError(msg)
        if tuple(covariances.shape) != (k, d, d):
            msg = f"covariances must have shape ({k}, {d}, {d}); got {tuple(covariances.shape)}"
            raise StochasticConfigurationError(msg)
        dtype = self.config.torch_dtype
        if weights is None:
            weights = torch.full((k,), 1.0 / k, dtype=dtype)
        return GaussianMixtureState(
            weights=weights.to(dtype),
            means=means.to(dtype),
            covariances=covariances.to(dtype),
        )
