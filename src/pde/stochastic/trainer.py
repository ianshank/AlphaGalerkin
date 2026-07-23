"""Parallel-in-time Strang-splitting trainer over precomputed particle clusters.

Every interval loss ``ℓ_i = NLL(S_dt(θ_i); particles(t_{i+1}))`` uses the
**precomputed** slice mixture ``θ_i`` — never the model's own rollout — so the
M−1 losses are mutually independent and evaluate in ONE batched forward pass
(no autoregressive dependency across timesteps; AC6). The total loss applies
trapezoid-style interval weights ``dt·[½, 1, …, 1, ½]`` (an implementation
choice, not asserted paper fidelity).

Only the MDN jump semigroup is trainable in v1 (drift/diffusion are known
config inputs — spec Out of Scope for learnable drift/diffusion), so a
no-jump generator has nothing to train and is rejected. The trainer can also
evaluate the same loss with the **exact oracle** substituted for the MDN,
giving an honest achievable floor: AC5 gates both the calibrated
``final/initial`` ratio and the closure of the (initial − oracle) gap.

Spec: specs/stochastic_galerkin_nke.spec.md (AC5, AC6, change-doc trainer req).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.pde.stochastic.config import StrangTrainerConfig
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.jump_mdn import (
    MDNJumpSemigroup,
    batched_mixture_nll,
    pack_batch,
    unpack_batch,
)
from src.pde.stochastic.particles import TimeSliceClusters
from src.pde.stochastic.projection import GalerkinMomentProjection

_UNIFORM_GRID_RTOL = 1e-9
"""Relative tolerance when validating that the coarse grid is uniform."""


@dataclass
class StrangTrainingResult:
    """Loss trajectory and diagnostics of a training run."""

    loss_history: list[float]
    initial_loss: float
    final_loss: float
    oracle_loss: float
    per_slice_losses: list[float]
    monotone_window: int

    @property
    def loss_ratio(self) -> float:
        """``final_loss / initial_loss`` (the AC5 absolute gate)."""
        return self.final_loss / self.initial_loss

    @property
    def gap_closure(self) -> float:
        """Fraction of the (initial − oracle) gap still open at the end.

        0 ⇒ the trainer reached the oracle-achievable loss; 1 ⇒ no progress.
        Guarded against a degenerate (≤0) initial gap.
        """
        gap = self.initial_loss - self.oracle_loss
        if gap <= 0.0:
            return 0.0
        return max(0.0, (self.final_loss - self.oracle_loss) / gap)

    def windowed_means(self) -> list[float]:
        """Means of consecutive non-overlapping ``monotone_window``-step windows."""
        w = self.monotone_window
        return [
            sum(self.loss_history[i : i + w]) / len(self.loss_history[i : i + w])
            for i in range(0, len(self.loss_history), w)
        ]

    def is_monotone_windowed(self, rel_tol: float) -> bool:
        """AC5: successive window means non-increasing within ``rel_tol``."""
        means = self.windowed_means()
        return all(
            later <= earlier * (1.0 + rel_tol)
            for earlier, later in zip(means[:-1], means[1:], strict=True)
        )


class StrangParallelTrainer:
    """Trains the MDN jump semigroup through the batched Strang composition."""

    def __init__(
        self,
        config: StrangTrainerConfig,
        projection: GalerkinMomentProjection,
        mdn: MDNJumpSemigroup,
        data: TimeSliceClusters,
    ) -> None:
        generator = projection.generator
        if not generator.has_jump:
            msg = (
                "nothing to train: the generator has no jump term and only the MDN "
                "jump semigroup is trainable in v1 (drift/diffusion are known inputs)"
            )
            raise StochasticConfigurationError(msg)
        if not generator.is_linear:
            msg = "the batched trainer requires linear drift (exact advection flows)"
            raise StochasticConfigurationError(msg)
        if data.n_slices != config.n_time_slices:
            msg = f"data has {data.n_slices} slices but config.n_time_slices={config.n_time_slices}"
            raise StochasticConfigurationError(msg)
        intervals = data.times[1:] - data.times[:-1]
        dt = float(intervals[0])
        if not torch.allclose(intervals, intervals[0], rtol=_UNIFORM_GRID_RTOL, atol=0.0):
            msg = "the coarse time grid must be uniform for the batched trainer"
            raise StochasticConfigurationError(msg)
        first = data.mixtures[0]
        if first.n_components != mdn.config.n_components or first.dim != mdn.config.dim:
            msg = (
                f"MDN configured for K={mdn.config.n_components}, d={mdn.config.dim} but "
                f"clusters have K={first.n_components}, d={first.dim}"
            )
            raise StochasticConfigurationError(msg)

        self.config = config
        self.projection = projection
        self.mdn = mdn
        self.data = data
        self.dt = dt
        self._k = first.n_components
        self._d = first.dim
        # Device-agnostic: everything trainable/batched lives on config.device
        # ('cpu' default; 'cuda'/'cuda:N' supported — resolve 'auto' upstream).
        self.device = torch.device(config.device)
        self.mdn.to(self.device)

        # Precomputed, grad-free training tensors (float32 — the net's dtype).
        self._inputs = torch.stack([m.pack() for m in data.mixtures[:-1]]).to(
            dtype=torch.float32, device=self.device
        )  # (B, P)
        self._targets = data.particles[1:].to(dtype=torch.float32, device=self.device)  # (B, N, d)
        self._dt_column = torch.full(
            (self._inputs.shape[0], 1), dt, dtype=torch.float32, device=self.device
        )
        # Exact half-step advection operators and diffusion production.
        expm_half, shift_half = projection.advection_flow_matrices(
            0.5 * dt, torch.float32, self.device
        )
        self._expm_half = expm_half
        self._shift_half = shift_half
        self._q_half = (0.5 * dt) * generator.q_matrix.to(dtype=torch.float32, device=self.device)
        jump = generator.jump
        assert jump is not None  # has_jump guarantees this
        mu = torch.tensor(jump.jump_mean, dtype=torch.float32, device=self.device)
        sigma = torch.tensor(jump.jump_cov, dtype=torch.float32, device=self.device)
        self._oracle_shift = (jump.rate * dt) * mu
        self._oracle_production = (jump.rate * dt) * (sigma + torch.outer(mu, mu))
        # Trapezoid-style interval weights dt·[½, 1, …, 1, ½].
        n_intervals = self._inputs.shape[0]
        weights = torch.full((n_intervals,), dt, dtype=torch.float32, device=self.device)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self._trapezoid_weights = weights

    # ------------------------------------------------------------------
    # Batched Strang loss (one forward pass over all intervals)
    # ------------------------------------------------------------------

    def _advection_half(self, means: Tensor, covs: Tensor) -> tuple[Tensor, Tensor]:
        new_means = means @ self._expm_half.T + self._shift_half
        new_covs = self._expm_half @ covs @ self._expm_half.T
        return new_means, 0.5 * (new_covs + new_covs.transpose(-1, -2))

    def _diffusion_half(self, covs: Tensor) -> Tensor:
        return covs + self._q_half

    def _oracle_jump(self, means: Tensor, covs: Tensor) -> tuple[Tensor, Tensor]:
        return means + self._oracle_shift, covs + self._oracle_production

    def compute_slice_losses(
        self, indices: Tensor | None = None, use_oracle: bool = False
    ) -> Tensor:
        """All interval losses ``ℓ_i`` in one batched forward pass; returns (B,).

        ``indices`` restricts to a subset of intervals (minibatch mode);
        ``use_oracle`` substitutes the exact compound-Poisson moment oracle
        for the MDN, giving the achievable-loss floor.
        """
        packed = self._inputs if indices is None else self._inputs[indices]
        targets = self._targets if indices is None else self._targets[indices]
        dt_col = self._dt_column if indices is None else self._dt_column[indices]

        weights, means, covs = unpack_batch(packed, self._k, self._d)
        means, covs = self._advection_half(means, covs)
        covs = self._diffusion_half(covs)
        mid = pack_batch(weights, means, covs)
        if use_oracle:
            w2, m2, c2 = unpack_batch(mid, self._k, self._d)
            m2, c2 = self._oracle_jump(m2, c2)
            out = pack_batch(w2, m2, c2)
        else:
            out = self.mdn.forward(mid, dt_col)
        w3, m3, c3 = unpack_batch(out, self._k, self._d)
        c3 = self._diffusion_half(c3)
        m3, c3 = self._advection_half(m3, c3)
        final = pack_batch(w3, m3, c3)
        return batched_mixture_nll(final, targets, self._k, self._d)

    def compute_total_loss(self, indices: Tensor | None = None, use_oracle: bool = False) -> Tensor:
        """Trapezoid-weighted sum of the interval losses."""
        losses = self.compute_slice_losses(indices, use_oracle=use_oracle)
        trapezoid = self._trapezoid_weights if indices is None else self._trapezoid_weights[indices]
        return (trapezoid * losses).sum()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> StrangTrainingResult:
        """Run the (full-batch by default) training loop; returns diagnostics."""
        optimizer = torch.optim.AdamW(self.mdn.parameters(), lr=self.config.learning_rate)
        sampler = torch.Generator().manual_seed(self.config.seed)
        n_intervals = self._inputs.shape[0]
        with torch.no_grad():
            oracle_loss = float(self.compute_total_loss(use_oracle=True))
        history: list[float] = []
        for _ in range(self.config.max_steps):
            if self.config.full_batch:
                indices = None
            else:
                take = min(self.config.minibatch_slices, n_intervals)
                indices = torch.randperm(n_intervals, generator=sampler)[:take]
            optimizer.zero_grad()
            loss = self.compute_total_loss(indices)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            history.append(float(loss.detach()))
        with torch.no_grad():
            per_slice = [float(v) for v in self.compute_slice_losses()]
        return StrangTrainingResult(
            loss_history=history,
            initial_loss=history[0],
            final_loss=history[-1],
            oracle_loss=oracle_loss,
            per_slice_losses=per_slice,
            monotone_window=self.config.monotone_window,
        )
