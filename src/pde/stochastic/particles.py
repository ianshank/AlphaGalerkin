"""Precomputed particle data for the parallel-in-time trainer.

Particles are simulated ONCE (no autograd) by Euler–Maruyama with compound-
Poisson jumps on a fine ``sim_dt`` grid, recorded at the coarse time slices,
and each slice is fitted with a K-component mixture via a small seeded torch
Lloyd's k-means (scikit-learn is deliberately not a dependency; k-means +
per-cluster empirical moments is a documented crude v1 fit — spec Out of
Scope for EM-quality fitting). Raw particles are retained as NLL targets.

Device note: simulation/clustering run on CPU by design (seeded
``torch.Generator`` determinism); the trainer moves the precomputed tensors
to its configured device, so the pipeline as a whole is GPU/CPU agnostic.

Spec: specs/stochastic_galerkin_nke.spec.md (trainer requirement, AC6).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.pde.stochastic.config import (
    DEFAULT_CLUSTER_COV_FLOOR,
    DEFAULT_KMEANS_MAX_ITERS,
    DEFAULT_KMEANS_TOL,
    JumpConfig,
)
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState
from src.pde.stochastic.generator import DriftModel


@dataclass(frozen=True)
class ParticleSimulationResult:
    """Particle positions recorded at the coarse time slices."""

    times: Tensor  # (M,)
    particles: Tensor  # (M, N, d)


@dataclass(frozen=True)
class TimeSliceClusters:
    """Per-slice K-mixture fits plus the raw particles (NLL targets)."""

    times: Tensor  # (M,)
    mixtures: list[GaussianMixtureState]
    particles: Tensor  # (M, N, d)

    @property
    def n_slices(self) -> int:
        """Number of coarse time slices M."""
        return int(self.times.shape[0])


def sample_gaussian(n: int, mean: Tensor, cov: Tensor, generator: torch.Generator) -> Tensor:
    """Draw ``n`` samples from N(mean, cov) in float64 with a seeded generator."""
    mean64 = torch.as_tensor(mean, dtype=torch.float64).reshape(-1)
    cov64 = torch.as_tensor(cov, dtype=torch.float64)
    chol = torch.linalg.cholesky(cov64)
    z = torch.randn(n, mean64.shape[0], dtype=torch.float64, generator=generator)
    return mean64 + z @ chol.T


def _compound_poisson_increment(
    n: int,
    h: float,
    jump: JumpConfig,
    generator: torch.Generator,
) -> Tensor:
    """Compound-Poisson increments over ``h`` for ``n`` particles: Σᵢ ξᵢ.

    Given a Poisson count k, the sum of k iid N(μ, Σ) draws is N(kμ, kΣ).
    """
    d = len(jump.jump_mean)
    rate_h = torch.full((n,), jump.rate * h, dtype=torch.float64)
    counts = torch.poisson(rate_h, generator=generator)
    mu = torch.tensor(jump.jump_mean, dtype=torch.float64)
    chol = torch.linalg.cholesky(torch.tensor(jump.jump_cov, dtype=torch.float64))
    z = torch.randn(n, d, dtype=torch.float64, generator=generator)
    return counts.unsqueeze(1) * mu + torch.sqrt(counts).unsqueeze(1) * (z @ chol.T)


def simulate_jump_diffusion(
    drift: DriftModel,
    diffusion: Tensor,
    jump: JumpConfig | None,
    x0: Tensor,
    t_grid: Tensor,
    sim_dt: float,
    seed: int,
) -> ParticleSimulationResult:
    """Euler–Maruyama with compound-Poisson jumps, recorded at ``t_grid``.

    Args:
        drift: Pointwise drift ``f: (N, d) -> (N, d)``.
        diffusion: Constant diffusion factor g of shape (d, m).
        jump: Optional compound-Poisson jump term.
        x0: Initial particles of shape (N, d).
        t_grid: Strictly increasing coarse times (M,), starting anywhere.
        sim_dt: Fine simulation step (each coarse interval is subdivided into
            ``round(interval / sim_dt)`` equal steps).
        seed: Seed for the dedicated torch.Generator (full determinism).

    Returns:
        Particle positions at every coarse grid time (initial slice included).

    """
    times = torch.as_tensor(t_grid, dtype=torch.float64).reshape(-1)
    if times.shape[0] < 2 or bool((times[1:] <= times[:-1]).any()):
        msg = "t_grid must be strictly increasing with at least two points"
        raise StochasticConfigurationError(msg)
    x = torch.as_tensor(x0, dtype=torch.float64).clone()
    if x.ndim != 2:
        msg = f"x0 must have shape (N, d); got {tuple(x.shape)}"
        raise StochasticConfigurationError(msg)
    n, d = x.shape
    g = torch.as_tensor(diffusion, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    slices = [x.clone()]
    with torch.no_grad():
        for t0, t1 in zip(times[:-1].tolist(), times[1:].tolist(), strict=True):
            interval = t1 - t0
            # round() keeps |n_sub·sim_dt − interval| ≤ sim_dt/2; the actual
            # substep h adapts so each interval is resolved exactly.
            n_sub = max(1, round(interval / sim_dt))
            h = interval / n_sub
            sqrt_h = h**0.5
            for _ in range(n_sub):
                dw = torch.randn(n, g.shape[1], dtype=torch.float64, generator=generator)
                x = x + drift(x) * h + sqrt_h * (dw @ g.T)
                if jump is not None and jump.rate > 0.0:
                    x = x + _compound_poisson_increment(n, h, jump, generator)
            slices.append(x.clone())
    return ParticleSimulationResult(times=times, particles=torch.stack(slices))


def _lloyd_kmeans(points: Tensor, k: int, generator: torch.Generator) -> Tensor:
    """Seeded torch Lloyd's iteration; returns hard assignments (N,).

    Deterministic under a fixed generator: initial centroids are k distinct
    random points; an emptied cluster is reseeded to a random point.
    """
    n = points.shape[0]
    perm = torch.randperm(n, generator=generator)
    centroids = points[perm[:k]].clone()
    assignments = torch.zeros(n, dtype=torch.long)
    scale = float(points.std()) + DEFAULT_KMEANS_TOL
    for _ in range(DEFAULT_KMEANS_MAX_ITERS):
        distances = torch.cdist(points, centroids)  # (N, k)
        assignments = distances.argmin(dim=1)
        new_centroids = centroids.clone()
        for j in range(k):
            mask = assignments == j
            if bool(mask.any()):
                new_centroids[j] = points[mask].mean(dim=0)
            else:
                reseed = int(torch.randint(n, (1,), generator=generator))
                new_centroids[j] = points[reseed]
        shift = float((new_centroids - centroids).norm())
        centroids = new_centroids
        if shift < DEFAULT_KMEANS_TOL * scale:
            break
    return assignments


def cluster_time_slices(
    sim: ParticleSimulationResult, n_components: int, seed: int
) -> TimeSliceClusters:
    """Fit a K-mixture to every time slice (weights/means/empirical covariances).

    Per-cluster covariances get a ``DEFAULT_CLUSTER_COV_FLOOR`` diagonal floor
    so degenerate clusters stay SPD.
    """
    m_slices, n, d = sim.particles.shape
    generator = torch.Generator().manual_seed(seed)
    floor = DEFAULT_CLUSTER_COV_FLOOR * torch.eye(d, dtype=torch.float64)
    mixtures: list[GaussianMixtureState] = []
    for i in range(m_slices):
        points = sim.particles[i]
        if n_components == 1:
            assignments = torch.zeros(n, dtype=torch.long)
        else:
            assignments = _lloyd_kmeans(points, n_components, generator)
        weights, means, covs = [], [], []
        for j in range(n_components):
            mask = assignments == j
            count = int(mask.sum())
            if count == 0:
                # Reseeded-but-still-empty cluster: keep a vanishing weight on
                # the slice mean so the mixture stays well-formed.
                weights.append(0.0)
                means.append(points.mean(dim=0))
                covs.append(floor.clone())
                continue
            cluster = points[mask]
            weights.append(count / n)
            means.append(cluster.mean(dim=0))
            if count == 1:
                covs.append(floor.clone())
            else:
                centered = cluster - cluster.mean(dim=0)
                covs.append((centered.T @ centered) / (count - 1) + floor)
        mixtures.append(
            GaussianMixtureState(
                weights=torch.tensor(weights, dtype=torch.float64),
                means=torch.stack(means),
                covariances=torch.stack(covs),
            )
        )
    return TimeSliceClusters(times=sim.times, mixtures=mixtures, particles=sim.particles)
