"""AC5 (loss decrease within 500 steps) and AC6 (parallel independence) tests.

The pinned jump-OU pipeline (simulate → cluster → train, all seeded) gates the
calibrated windowed monotonicity / loss-ratio / gap-closure criteria; AC6
asserts batched slice losses equal a per-slice loop and are order-invariant.
Calibration (spec table): initial 0.6916 → final 0.65714, oracle floor
0.65727, max window increase 2.8e-6, trajectory errs 1.7e-2 / 7.4e-3.
"""

from __future__ import annotations

import pytest
import torch

from src.pde.stochastic.analytic import jump_ou_covariance, jump_ou_mean
from src.pde.stochastic.config import (
    DEFAULT_LOSS_GAP_CLOSURE,
    DEFAULT_LOSS_RATIO_GATE,
    DEFAULT_MONOTONE_REL_TOL,
    DEFAULT_TRAINED_MDN_MOMENT_TOL,
    JumpConfig,
    MDNJumpConfig,
    StochasticGeneratorConfig,
    StrangSplittingConfig,
    StrangTrainerConfig,
)
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState
from src.pde.stochastic.generator import KolmogorovGenerator
from src.pde.stochastic.jump_mdn import MDNJumpSemigroup
from src.pde.stochastic.particles import (
    ParticleSimulationResult,
    cluster_time_slices,
    sample_gaussian,
    simulate_jump_diffusion,
)
from src.pde.stochastic.projection import GalerkinMomentProjection
from src.pde.stochastic.strang import StrangSplitStep
from src.pde.stochastic.trainer import StrangParallelTrainer

F64 = torch.float64
PINNED_JUMP = JumpConfig(rate=2.0, jump_mean=[0.5], jump_cov=[[0.04]])
SEED = 42


def _build(max_steps: int = 500, **trainer_overrides):
    """The spec-pinned jump-OU training setup (fully seeded)."""
    torch.manual_seed(SEED)
    mdn = MDNJumpSemigroup(MDNJumpConfig(dim=1, n_components=1))
    cfg = StochasticGeneratorConfig(
        dim=1, drift_matrix=[[-1.0]], diffusion=[[0.3]], jump=PINNED_JUMP
    )
    gen = KolmogorovGenerator(cfg, jump_semigroup=mdn)
    proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=0.1, t_end=1.0))
    g0 = torch.Generator().manual_seed(SEED)
    x0 = sample_gaussian(2000, torch.zeros(1, dtype=F64), torch.tensor([[0.1]], dtype=F64), g0)
    t_grid = torch.linspace(0.0, 1.0, 11, dtype=F64)
    sim = simulate_jump_diffusion(gen.drift, gen.diffusion, PINNED_JUMP, x0, t_grid, 0.005, SEED)
    clusters = cluster_time_slices(sim, 1, SEED)
    trainer_cfg = StrangTrainerConfig(
        n_particles=2000,
        n_time_slices=11,
        sim_dt=0.005,
        max_steps=max_steps,
        learning_rate=1e-2,
        **trainer_overrides,
    )
    trainer = StrangParallelTrainer(trainer_cfg, proj, mdn, clusters)
    return trainer, proj, mdn


@pytest.fixture(scope="module")
def trained():
    trainer, proj, mdn = _build()
    result = trainer.train()
    return trainer, proj, mdn, result


class TestAC5TrainerConvergence:
    """AC5 on the pinned jump-OU problem (calibrated gates)."""

    def test_windowed_means_non_increasing(self, trained):
        _trainer, _proj, _mdn, result = trained
        assert result.is_monotone_windowed(DEFAULT_MONOTONE_REL_TOL)

    def test_loss_ratio_gate(self, trained):
        _trainer, _proj, _mdn, result = trained
        assert result.loss_ratio < DEFAULT_LOSS_RATIO_GATE

    def test_gap_closure_gate(self, trained):
        _trainer, _proj, _mdn, result = trained
        assert result.gap_closure < DEFAULT_LOSS_GAP_CLOSURE

    def test_within_500_steps(self, trained):
        _trainer, _proj, _mdn, result = trained
        assert len(result.loss_history) <= 500

    def test_trained_trajectory_matches_analytic(self, trained):
        """AC3 (trained, trajectory-level): full Strang with the trained MDN."""
        _trainer, proj, _mdn, _result = trained
        state = GaussianMixtureState(
            weights=torch.ones(1, dtype=F64),
            means=torch.zeros(1, 1, dtype=F64),
            covariances=torch.tensor([[[0.1]]], dtype=F64),
        )
        final, t_final = StrangSplitStep(proj).propagate(state)[-1]
        assert t_final == pytest.approx(1.0)
        expected_m = jump_ou_mean(
            torch.tensor([[-1.0]], dtype=F64),
            torch.zeros(1, dtype=F64),
            2.0,
            torch.tensor([0.5], dtype=F64),
            torch.zeros(1, dtype=F64),
            1.0,
        )
        expected_p = jump_ou_covariance(
            torch.tensor([[-1.0]], dtype=F64),
            torch.tensor([[0.09]], dtype=F64),
            2.0,
            torch.tensor([0.5], dtype=F64),
            torch.tensor([[0.04]], dtype=F64),
            torch.tensor([[0.1]], dtype=F64),
            1.0,
        )
        assert abs(float(final.means[0, 0] - expected_m[0])) < DEFAULT_TRAINED_MDN_MOMENT_TOL
        assert (
            abs(float(final.covariances[0, 0, 0] - expected_p[0, 0]))
            < DEFAULT_TRAINED_MDN_MOMENT_TOL
        )

    def test_oracle_floor_below_initial(self, trained):
        _trainer, _proj, _mdn, result = trained
        assert result.oracle_loss < result.initial_loss


class TestAC6ParallelIndependence:
    """AC6: all interval losses in one batched pass, no cross-slice coupling."""

    @pytest.fixture()
    def fresh_trainer(self):
        trainer, _proj, _mdn = _build(max_steps=1)
        return trainer

    def test_batched_equals_per_slice_loop(self, fresh_trainer):
        with torch.no_grad():
            batched = fresh_trainer.compute_slice_losses()
            looped = torch.cat(
                [
                    fresh_trainer.compute_slice_losses(indices=torch.tensor([i]))
                    for i in range(batched.shape[0])
                ]
            )
        torch.testing.assert_close(batched, looped, rtol=1e-5, atol=1e-7)

    def test_slice_order_permutation_invariance(self, fresh_trainer):
        with torch.no_grad():
            full = fresh_trainer.compute_slice_losses()
            perm = torch.randperm(full.shape[0], generator=torch.Generator().manual_seed(3))
            permuted = fresh_trainer.compute_slice_losses(indices=perm)
        torch.testing.assert_close(permuted, full[perm], rtol=1e-6, atol=1e-8)

    def test_trapezoid_weights(self, fresh_trainer):
        with torch.no_grad():
            losses = fresh_trainer.compute_slice_losses()
            total = fresh_trainer.compute_total_loss()
        dt = fresh_trainer.dt
        weights = torch.full((losses.shape[0],), dt)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        assert float(total) == pytest.approx(float((weights * losses).sum()), rel=1e-6)

    def test_batched_consistent_with_state_path(self, fresh_trainer):
        """The batched float32 pipeline tracks the float64 StrangSplitStep path."""
        trainer = fresh_trainer
        stepper = StrangSplitStep(trainer.projection)
        state_in = trainer.data.mixtures[0]
        out_state = stepper.step(state_in, trainer.dt)
        with torch.no_grad():
            batched_packed = trainer.compute_slice_losses(indices=torch.tensor([0]))
            # Reference NLL from the float64 state path on the same targets.
            targets64 = trainer.data.particles[1].to(F64)
            reference = -out_state.log_prob(targets64).mean()
        assert float(batched_packed[0]) == pytest.approx(float(reference), rel=1e-3)

    def test_oracle_loss_path(self, fresh_trainer):
        with torch.no_grad():
            oracle_losses = fresh_trainer.compute_slice_losses(use_oracle=True)
            mdn_losses = fresh_trainer.compute_slice_losses()
        assert oracle_losses.shape == mdn_losses.shape
        assert bool(torch.isfinite(oracle_losses).all())


class TestTrainerValidation:
    def test_no_jump_generator_rejected(self):
        cfg = StochasticGeneratorConfig(dim=1, drift_matrix=[[-1.0]], diffusion=[[0.3]])
        gen = KolmogorovGenerator(cfg)
        proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=0.1, t_end=1.0))
        torch.manual_seed(SEED)
        mdn = MDNJumpSemigroup(MDNJumpConfig(dim=1, n_components=1))
        clusters = cluster_time_slices(
            ParticleSimulationResult(
                times=torch.linspace(0.0, 1.0, 11, dtype=F64),
                particles=torch.randn(11, 32, 1, dtype=F64),
            ),
            1,
            SEED,
        )
        trainer_cfg = StrangTrainerConfig(n_particles=32, n_time_slices=11, sim_dt=0.005)
        with pytest.raises(StochasticConfigurationError, match="nothing to train"):
            StrangParallelTrainer(trainer_cfg, proj, mdn, clusters)

    def test_slice_count_mismatch_rejected(self):
        trainer, proj, mdn = _build(max_steps=1)
        bad_cfg = StrangTrainerConfig(n_particles=2000, n_time_slices=5, sim_dt=0.005, max_steps=1)
        with pytest.raises(StochasticConfigurationError, match="slices"):
            StrangParallelTrainer(bad_cfg, proj, mdn, trainer.data)

    def test_nonuniform_grid_rejected(self):
        trainer, proj, mdn = _build(max_steps=1)
        bad_times = torch.tensor([0.0, 0.1, 0.3], dtype=F64)
        clusters = cluster_time_slices(
            ParticleSimulationResult(times=bad_times, particles=trainer.data.particles[:3]),
            1,
            SEED,
        )
        cfg = StrangTrainerConfig(n_particles=2000, n_time_slices=3, sim_dt=0.005, max_steps=1)
        with pytest.raises(StochasticConfigurationError, match="uniform"):
            StrangParallelTrainer(cfg, proj, mdn, clusters)

    def test_mdn_shape_mismatch_rejected(self):
        trainer, proj, _mdn = _build(max_steps=1)
        torch.manual_seed(SEED)
        wrong_mdn = MDNJumpSemigroup(MDNJumpConfig(dim=1, n_components=2))
        cfg = StrangTrainerConfig(n_particles=2000, n_time_slices=11, sim_dt=0.005, max_steps=1)
        with pytest.raises(StochasticConfigurationError, match="clusters have"):
            StrangParallelTrainer(cfg, proj, wrong_mdn, trainer.data)

    def test_minibatch_mode_runs(self):
        trainer, _proj, _mdn = _build(max_steps=5, full_batch=False, minibatch_slices=3)
        result = trainer.train()
        assert len(result.loss_history) == 5
        assert all(torch.isfinite(torch.tensor(result.loss_history)))
