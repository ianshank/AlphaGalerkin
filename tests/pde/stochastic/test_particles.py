"""Tests for the precomputed particle pipeline (simulation + clustering).

Slice moments are checked against the analytic jump-OU forms within seeded
Monte-Carlo tolerance; clustering is deterministic under a fixed seed and
keeps covariances SPD via the diagonal floor.
"""

from __future__ import annotations

import pytest
import torch

from src.pde.stochastic.analytic import jump_ou_covariance, jump_ou_mean
from src.pde.stochastic.config import DEFAULT_CLUSTER_COV_FLOOR, JumpConfig
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.generator import LinearDrift
from src.pde.stochastic.particles import (
    cluster_time_slices,
    sample_gaussian,
    simulate_jump_diffusion,
)

F64 = torch.float64

PINNED_JUMP = JumpConfig(rate=2.0, jump_mean=[0.5], jump_cov=[[0.04]])
DRIFT = LinearDrift(matrix=torch.tensor([[-1.0]], dtype=F64), bias=torch.zeros(1, dtype=F64))
G = torch.tensor([[0.3]], dtype=F64)
T_GRID = torch.linspace(0.0, 1.0, 11, dtype=F64)
N_PARTICLES = 2000
SIM_DT = 0.005
SEED = 42

# Seeded Monte-Carlo tolerances (≈3σ for N=2000; deterministic under SEED).
_MC_MEAN_TOL = 0.06
_MC_VAR_TOL = 0.08


def _pinned_x0() -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return sample_gaussian(
        N_PARTICLES, torch.zeros(1, dtype=F64), torch.tensor([[0.1]], dtype=F64), gen
    )


def _pinned_sim():
    return simulate_jump_diffusion(
        drift=DRIFT,
        diffusion=G,
        jump=PINNED_JUMP,
        x0=_pinned_x0(),
        t_grid=T_GRID,
        sim_dt=SIM_DT,
        seed=SEED,
    )


class TestSampleGaussian:
    def test_moments(self):
        gen = torch.Generator().manual_seed(0)
        x = sample_gaussian(
            5000, torch.tensor([1.0, -1.0]), torch.tensor([[0.5, 0.1], [0.1, 0.3]]), gen
        )
        assert x.shape == (5000, 2)
        assert float((x.mean(dim=0) - torch.tensor([1.0, -1.0], dtype=F64)).abs().max()) < 0.05
        emp_cov = torch.cov(x.T)
        assert (
            float((emp_cov - torch.tensor([[0.5, 0.1], [0.1, 0.3]], dtype=F64)).abs().max()) < 0.05
        )


class TestSimulateJumpDiffusion:
    def test_slice_moments_match_analytic(self):
        sim = _pinned_sim()
        assert sim.particles.shape == (11, N_PARTICLES, 1)
        a = torch.tensor([[-1.0]], dtype=F64)
        b = torch.zeros(1, dtype=F64)
        q = torch.tensor([[0.09]], dtype=F64)
        mu = torch.tensor([0.5], dtype=F64)
        sigma = torch.tensor([[0.04]], dtype=F64)
        m0 = torch.zeros(1, dtype=F64)
        p0 = torch.tensor([[0.1]], dtype=F64)
        for i, t in enumerate(sim.times.tolist()):
            expected_m = float(jump_ou_mean(a, b, 2.0, mu, m0, t)[0])
            expected_p = float(jump_ou_covariance(a, q, 2.0, mu, sigma, p0, t)[0, 0])
            emp_m = float(sim.particles[i].mean())
            emp_p = float(sim.particles[i].var())
            assert abs(emp_m - expected_m) < _MC_MEAN_TOL, f"slice {i} mean"
            assert abs(emp_p - expected_p) < _MC_VAR_TOL, f"slice {i} var"

    def test_deterministic_under_seed(self):
        a = _pinned_sim()
        b = _pinned_sim()
        torch.testing.assert_close(a.particles, b.particles)

    def test_different_seed_differs(self):
        a = _pinned_sim()
        b = simulate_jump_diffusion(
            drift=DRIFT,
            diffusion=G,
            jump=PINNED_JUMP,
            x0=_pinned_x0(),
            t_grid=T_GRID,
            sim_dt=SIM_DT,
            seed=SEED + 1,
        )
        assert not torch.allclose(a.particles[-1], b.particles[-1])

    def test_no_jump_variant(self):
        sim = simulate_jump_diffusion(
            drift=DRIFT,
            diffusion=G,
            jump=None,
            x0=_pinned_x0(),
            t_grid=torch.tensor([0.0, 0.5], dtype=F64),
            sim_dt=SIM_DT,
            seed=SEED,
        )
        # OU with m0=0, b=0 keeps zero mean.
        assert abs(float(sim.particles[-1].mean())) < _MC_MEAN_TOL

    def test_grid_validation(self):
        with pytest.raises(StochasticConfigurationError, match="strictly increasing"):
            simulate_jump_diffusion(
                drift=DRIFT,
                diffusion=G,
                jump=None,
                x0=_pinned_x0(),
                t_grid=torch.tensor([0.0], dtype=F64),
                sim_dt=SIM_DT,
                seed=SEED,
            )

    def test_x0_shape_validation(self):
        with pytest.raises(StochasticConfigurationError, match="x0 must have shape"):
            simulate_jump_diffusion(
                drift=DRIFT,
                diffusion=G,
                jump=None,
                x0=torch.zeros(5, dtype=F64),
                t_grid=torch.tensor([0.0, 0.5], dtype=F64),
                sim_dt=SIM_DT,
                seed=SEED,
            )


class TestClusterTimeSlices:
    def test_k1_matches_empirical_moments(self):
        sim = _pinned_sim()
        clusters = cluster_time_slices(sim, n_components=1, seed=SEED)
        assert clusters.n_slices == 11
        for i in range(11):
            mixture = clusters.mixtures[i]
            assert float(mixture.weights.sum()) == pytest.approx(1.0)
            emp_mean = sim.particles[i].mean(dim=0)
            torch.testing.assert_close(mixture.means[0], emp_mean)
            emp_var = float(sim.particles[i].var())
            assert float(mixture.covariances[0, 0, 0]) == pytest.approx(
                emp_var + DEFAULT_CLUSTER_COV_FLOOR, rel=1e-6
            )

    def test_deterministic_under_seed(self):
        sim = _pinned_sim()
        a = cluster_time_slices(sim, n_components=2, seed=SEED)
        b = cluster_time_slices(sim, n_components=2, seed=SEED)
        for ma, mb in zip(a.mixtures, b.mixtures):
            torch.testing.assert_close(ma.means, mb.means)
            torch.testing.assert_close(ma.weights, mb.weights)

    def test_k2_separates_bimodal_data(self):
        gen = torch.Generator().manual_seed(3)
        left = sample_gaussian(500, torch.tensor([-3.0]), torch.tensor([[0.1]]), gen)
        right = sample_gaussian(500, torch.tensor([3.0]), torch.tensor([[0.1]]), gen)
        points = torch.cat([left, right]).unsqueeze(0)  # (1, 1000, 1)
        from src.pde.stochastic.particles import ParticleSimulationResult

        sim = ParticleSimulationResult(times=torch.tensor([0.0], dtype=F64), particles=points)
        clusters = cluster_time_slices(sim, n_components=2, seed=SEED)
        mixture = clusters.mixtures[0]
        centers = sorted(float(m) for m in mixture.means.reshape(-1))
        assert abs(centers[0] + 3.0) < 0.2
        assert abs(centers[1] - 3.0) < 0.2
        assert float(mixture.weights.min()) > 0.4

    def test_covariances_spd(self):
        sim = _pinned_sim()
        clusters = cluster_time_slices(sim, n_components=3, seed=SEED)
        for mixture in clusters.mixtures:
            eigvals = torch.linalg.eigvalsh(mixture.covariances)
            assert bool((eigvals > 0).all())
