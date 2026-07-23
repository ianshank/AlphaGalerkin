"""AC3-oracle (jump-OU moments) and AC4 (second-order splitting) tests.

Strang composition with the exact compound-Poisson moment oracle is compared
against the independent jump-OU closed forms; the dt-halving sweep verifies
the O(dt²) covariance error signature with oracle/no-jump substeps only (a
trained MDN has an approximation floor that would falsify the order).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.pde.stochastic.analytic import jump_ou_covariance, jump_ou_mean, ou_covariance, ou_mean
from src.pde.stochastic.config import (
    DEFAULT_OU_MOMENT_TOL,
    DEFAULT_STRANG_SLOPE_MAX,
    DEFAULT_STRANG_SLOPE_MIN,
    JumpConfig,
    StochasticGeneratorConfig,
    StrangSplittingConfig,
)
from src.pde.stochastic.errors import JumpModelMissingError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureBasis
from src.pde.stochastic.generator import KolmogorovGenerator
from src.pde.stochastic.jump_mdn import AnalyticCompoundPoissonMoments
from src.pde.stochastic.projection import GalerkinMomentProjection
from src.pde.stochastic.strang import StrangSplitStep

F64 = torch.float64

# Spec-pinned jump-diffusion OU problem (AC3/AC5).
PINNED_JUMP = JumpConfig(rate=2.0, jump_mean=[0.5], jump_cov=[[0.04]])
PINNED_A = [[-1.0]]
PINNED_G = [[0.3]]
PINNED_M0 = [0.0]
PINNED_P0 = [[0.1]]
PINNED_T = 1.0
PINNED_DT = 0.1


def _jump_generator(dt: float = PINNED_DT, t_end: float = PINNED_T):
    cfg = StochasticGeneratorConfig(
        dim=1, drift_matrix=PINNED_A, diffusion=PINNED_G, jump=PINNED_JUMP
    )
    gen = KolmogorovGenerator(cfg, jump_semigroup=AnalyticCompoundPoissonMoments(PINNED_JUMP))
    proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=dt, t_end=t_end))
    return gen, proj


def _initial_state():
    basis = GaussianMixtureBasis(dim=1)
    return basis.initial_state(
        means=torch.tensor([PINNED_M0], dtype=F64),
        covariances=torch.tensor([PINNED_P0], dtype=F64),
    )


class TestOracle:
    def test_oracle_moment_update_is_exact_form(self):
        oracle = AnalyticCompoundPoissonMoments(PINNED_JUMP)
        state = _initial_state()
        dt = 0.3
        out = oracle.apply(state, dt)
        # m += λ·dt·μ ; P += λ·dt·(Σ + μμᵀ)
        assert abs(float(out.means[0, 0]) - 2.0 * dt * 0.5) < 1e-14
        expected_p = 0.1 + 2.0 * dt * (0.04 + 0.25)
        assert abs(float(out.covariances[0, 0, 0]) - expected_p) < 1e-14

    def test_oracle_preserves_weights(self):
        oracle = AnalyticCompoundPoissonMoments(PINNED_JUMP)
        state = _initial_state()
        torch.testing.assert_close(oracle.apply(state, 0.5).weights, state.weights)


class TestAC3OracleJumpOu:
    """AC3: Strang + exact jump oracle recovers the jump-OU closed forms."""

    def test_pinned_jump_ou_final_moments(self):
        _gen, proj = _jump_generator()
        final, t_final = StrangSplitStep(proj).propagate(_initial_state())[-1]
        assert abs(t_final - PINNED_T) < 1e-12
        a = torch.tensor(PINNED_A, dtype=F64)
        b = torch.zeros(1, dtype=F64)
        q = torch.tensor([[0.09]], dtype=F64)
        mu = torch.tensor([0.5], dtype=F64)
        sigma = torch.tensor([[0.04]], dtype=F64)
        expected_m = jump_ou_mean(a, b, 2.0, mu, torch.tensor(PINNED_M0, dtype=F64), PINNED_T)
        expected_p = jump_ou_covariance(
            a, q, 2.0, mu, sigma, torch.tensor(PINNED_P0, dtype=F64), PINNED_T
        )
        assert abs(float(final.means[0, 0] - expected_m[0])) < DEFAULT_OU_MOMENT_TOL
        assert abs(float(final.covariances[0, 0, 0] - expected_p[0, 0])) < DEFAULT_OU_MOMENT_TOL

    def test_no_jump_strang_matches_ou(self):
        """AC1 companion: the no-jump Strang path also recovers the OU forms."""
        cfg = StochasticGeneratorConfig(dim=1, drift_matrix=PINNED_A, diffusion=[[0.5]])
        gen = KolmogorovGenerator(cfg)
        proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=0.05, t_end=1.0))
        basis = GaussianMixtureBasis(dim=1)
        state = basis.initial_state(
            means=torch.tensor([[1.0]], dtype=F64),
            covariances=torch.tensor([[[0.5]]], dtype=F64),
        )
        final, _ = StrangSplitStep(proj).propagate(state)[-1]
        a = torch.tensor(PINNED_A, dtype=F64)
        expected_m = ou_mean(a, torch.zeros(1, dtype=F64), torch.tensor([1.0], dtype=F64), 1.0)
        expected_p = ou_covariance(a, gen.q_matrix, torch.tensor([[0.5]], dtype=F64), 1.0)
        assert abs(float(final.means[0, 0] - expected_m[0])) < DEFAULT_OU_MOMENT_TOL
        assert abs(float(final.covariances[0, 0, 0] - expected_p[0, 0])) < DEFAULT_OU_MOMENT_TOL


class TestAC4StrangOrder:
    """AC4: covariance error halves quadratically under dt halving."""

    def _covariance_error(self, dt: float) -> float:
        cfg = StochasticGeneratorConfig(dim=1, drift_matrix=[[-1.0]], diffusion=[[math.sqrt(0.5)]])
        gen = KolmogorovGenerator(cfg)
        proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=dt, t_end=0.8))
        basis = GaussianMixtureBasis(dim=1)
        state = basis.initial_state(
            means=torch.tensor([[1.0]], dtype=F64),
            covariances=torch.tensor([[[0.2]]], dtype=F64),
        )
        final, _ = StrangSplitStep(proj).propagate(state)[-1]
        expected = ou_covariance(
            torch.tensor([[-1.0]], dtype=F64),
            torch.tensor([[0.5]], dtype=F64),
            torch.tensor([[0.2]], dtype=F64),
            0.8,
        )
        return abs(float(final.covariances[0, 0, 0] - expected[0, 0]))

    def test_second_order_slope(self):
        dts = [0.2, 0.1, 0.05, 0.025]
        errors = [self._covariance_error(dt) for dt in dts]
        slopes = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        for slope in slopes:
            assert DEFAULT_STRANG_SLOPE_MIN <= slope <= DEFAULT_STRANG_SLOPE_MAX, f"slopes={slopes}"

    def test_second_order_with_oracle_jump(self):
        """The oracle jump substep preserves the O(dt²) order."""
        errors = []
        for dt in [0.2, 0.1, 0.05]:
            _gen, proj = _jump_generator(dt=dt)
            final, _ = StrangSplitStep(proj).propagate(_initial_state())[-1]
            expected_p = jump_ou_covariance(
                torch.tensor(PINNED_A, dtype=F64),
                torch.tensor([[0.09]], dtype=F64),
                2.0,
                torch.tensor([0.5], dtype=F64),
                torch.tensor([[0.04]], dtype=F64),
                torch.tensor(PINNED_P0, dtype=F64),
                PINNED_T,
            )
            errors.append(abs(float(final.covariances[0, 0, 0] - expected_p[0, 0])))
        slopes = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
        for slope in slopes:
            assert DEFAULT_STRANG_SLOPE_MIN <= slope <= DEFAULT_STRANG_SLOPE_MAX, f"slopes={slopes}"


class TestStrangComposition:
    def test_defense_in_depth_jump_check(self):
        gen, proj = _jump_generator()
        # Simulate misuse: strip the generator's jump model after construction.
        gen.jump_semigroup = None
        with pytest.raises(JumpModelMissingError, match="never silently ignored"):
            StrangSplitStep(proj)

    def test_explicit_jump_step_overrides_generator(self):
        _gen, proj = _jump_generator()

        class _Recording:
            def __init__(self):
                self.calls = 0

            def apply(self, state, dt):
                self.calls += 1
                return state

        recorder = _Recording()
        stepper = StrangSplitStep(proj, jump_step=recorder)
        stepper.step(_initial_state(), 0.1)
        assert recorder.calls == 1

    def test_no_jump_generator_skips_jump_flow(self):
        cfg = StochasticGeneratorConfig(dim=1, drift_matrix=PINNED_A, diffusion=PINNED_G)
        gen = KolmogorovGenerator(cfg)
        proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=0.1, t_end=1.0))
        stepper = StrangSplitStep(proj)
        assert stepper.jump_step is None

    def test_propagate_partial_final_step(self):
        cfg = StochasticGeneratorConfig(dim=1, drift_matrix=PINNED_A, diffusion=PINNED_G)
        gen = KolmogorovGenerator(cfg)
        proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=0.2, t_end=0.5))
        trajectory = StrangSplitStep(proj).propagate(
            GaussianMixtureBasis(dim=1).initial_state(
                means=torch.zeros(1, 1, dtype=F64),
                covariances=torch.tensor([[[0.1]]], dtype=F64),
            )
        )
        times = [t for _, t in trajectory]
        assert times == pytest.approx([0.0, 0.2, 0.4, 0.5])

    def test_single_step_local_error_is_cubic(self):
        """One Strang step vs the analytic jump-OU flow: local error O(dt³)."""
        dt = 1e-3
        _gen, proj = _jump_generator(dt=dt, t_end=1.0)
        state = _initial_state()
        strang = StrangSplitStep(proj).step(state, dt)
        expected_m = jump_ou_mean(
            torch.tensor(PINNED_A, dtype=F64),
            torch.zeros(1, dtype=F64),
            2.0,
            torch.tensor([0.5], dtype=F64),
            torch.tensor(PINNED_M0, dtype=F64),
            dt,
        )
        expected_p = jump_ou_covariance(
            torch.tensor(PINNED_A, dtype=F64),
            torch.tensor([[0.09]], dtype=F64),
            2.0,
            torch.tensor([0.5], dtype=F64),
            torch.tensor([[0.04]], dtype=F64),
            torch.tensor(PINNED_P0, dtype=F64),
            dt,
        )
        # Local error is O(dt³) = 1e-9 with O(1) constants for this problem.
        assert abs(float(strang.means[0, 0] - expected_m[0])) < 1e-8
        assert abs(float(strang.covariances[0, 0, 0] - expected_p[0, 0])) < 1e-8
