"""AC1 (OU moment recovery) and AC2 (jump config error) tests.

The projection's unsplit RK4 path is compared against the independent van
Loan closed forms on the spec-pinned OU problems; the jump-without-model
constructor path must raise ``JumpModelMissingError``.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pde.stochastic.analytic import ou_covariance, ou_mean
from src.pde.stochastic.config import (
    DEFAULT_OU_MOMENT_TOL,
    GaussianMixtureBasisConfig,
    JumpConfig,
    StochasticGeneratorConfig,
    StrangSplittingConfig,
)
from src.pde.stochastic.errors import JumpModelMissingError, StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureBasis, GaussianMixtureState
from src.pde.stochastic.generator import KolmogorovGenerator, LinearDrift
from src.pde.stochastic.projection import GalerkinMomentProjection

F64 = torch.float64

# Spec-pinned OU problems.
OU_1D = {
    "a": [[-1.0]],
    "b": [0.0],
    "g": [[0.5]],
    "m0": [1.0],
    "p0": [[0.5]],
}
OU_2D = {
    "a": [[-1.0, 0.3], [0.0, -0.8]],
    "b": [0.1, -0.2],
    "g": [[0.4, 0.0], [0.0, 0.3]],
    "m0": [1.0, -0.5],
    "p0": [[0.3, 0.0], [0.0, 0.2]],
}


def _generator(problem: dict, **kwargs) -> KolmogorovGenerator:
    dim = len(problem["m0"])
    cfg = StochasticGeneratorConfig(
        dim=dim,
        drift_matrix=problem["a"],
        drift_bias=problem["b"],
        diffusion=problem["g"],
        **kwargs,
    )
    return KolmogorovGenerator(cfg)


def _state(problem: dict) -> GaussianMixtureState:
    dim = len(problem["m0"])
    basis = GaussianMixtureBasis(GaussianMixtureBasisConfig(dim=dim))
    return basis.initial_state(
        means=torch.tensor([problem["m0"]], dtype=F64),
        covariances=torch.tensor([problem["p0"]], dtype=F64),
    )


def _splitting(dt: float = 0.05, t_end: float = 1.0, **kwargs) -> StrangSplittingConfig:
    return StrangSplittingConfig(dt=dt, t_end=t_end, **kwargs)


class TestAC2JumpConfigError:
    """AC2: jump term without a jump model is a hard configuration error."""

    def _config_with_jump(self, rate: float) -> StochasticGeneratorConfig:
        return StochasticGeneratorConfig(
            dim=1,
            drift_matrix=[[-1.0]],
            diffusion=[[0.3]],
            jump=JumpConfig(rate=rate, jump_mean=[0.5], jump_cov=[[0.04]]),
        )

    def test_jump_without_model_raises(self):
        with pytest.raises(JumpModelMissingError, match="jump-semigroup"):
            KolmogorovGenerator(self._config_with_jump(rate=2.0))

    def test_error_message_names_the_fix(self):
        with pytest.raises(JumpModelMissingError, match="MDNJumpSemigroup"):
            KolmogorovGenerator(self._config_with_jump(rate=2.0))
        with pytest.raises(JumpModelMissingError, match="AnalyticCompoundPoissonMoments"):
            KolmogorovGenerator(self._config_with_jump(rate=2.0))

    def test_zero_rate_does_not_raise(self):
        gen = KolmogorovGenerator(self._config_with_jump(rate=0.0))
        assert gen.has_jump is False

    def test_no_jump_config_does_not_raise(self):
        gen = _generator(OU_1D)
        assert gen.has_jump is False
        assert gen.jump is None

    def test_jump_with_model_accepted(self):
        class _IdentityJump:
            def advance(self, state: GaussianMixtureState, dt: float) -> GaussianMixtureState:
                return state

        gen = KolmogorovGenerator(self._config_with_jump(rate=2.0), jump_semigroup=_IdentityJump())
        assert gen.has_jump is True


class TestGeneratorConstruction:
    def test_linear_drift_built_from_config(self):
        gen = _generator(OU_2D)
        assert gen.is_linear
        drift = gen.linear_drift()
        x = torch.tensor([[1.0, 2.0]], dtype=F64)
        expected = x @ drift.matrix.T + drift.bias
        torch.testing.assert_close(gen.drift_at(x), expected)

    def test_missing_drift_raises(self):
        cfg = StochasticGeneratorConfig(dim=1, diffusion=[[0.3]])
        with pytest.raises(StochasticConfigurationError, match="no drift available"):
            KolmogorovGenerator(cfg)

    def test_callable_drift_accepted(self):
        cfg = StochasticGeneratorConfig(dim=1, diffusion=[[0.3]])
        gen = KolmogorovGenerator(cfg, drift=lambda x: -x)
        assert gen.is_linear is False
        with pytest.raises(StochasticConfigurationError, match="not LinearDrift"):
            gen.linear_drift()

    def test_q_matrix_is_ggt(self):
        gen = _generator(OU_2D)
        g = torch.tensor(OU_2D["g"], dtype=F64)
        torch.testing.assert_close(gen.q_matrix, g @ g.T)

    def test_linear_drift_shape_validation(self):
        with pytest.raises(StochasticConfigurationError, match="square"):
            LinearDrift(matrix=torch.zeros(2, 3, dtype=F64), bias=torch.zeros(2, dtype=F64))
        with pytest.raises(StochasticConfigurationError, match="bias shape"):
            LinearDrift(matrix=torch.zeros(2, 2, dtype=F64), bias=torch.zeros(3, dtype=F64))


class TestAC1OuRecovery:
    """AC1: the unsplit projected moment ODE recovers the OU closed forms."""

    @pytest.mark.parametrize("problem", [OU_1D, OU_2D], ids=["1d", "2d"])
    def test_unsplit_rk4_matches_analytic(self, problem):
        gen = _generator(problem)
        proj = GalerkinMomentProjection(gen, _splitting(rk4_substeps=8))
        state = _state(problem)
        t_grid = torch.linspace(0.0, 1.0, 5, dtype=F64)
        trajectory = proj.propagate(state, t_grid)
        a = torch.tensor(problem["a"], dtype=F64)
        b = torch.tensor(problem["b"], dtype=F64)
        q = gen.q_matrix
        m0 = torch.tensor(problem["m0"], dtype=F64)
        p0 = torch.tensor(problem["p0"], dtype=F64)
        for state_t, t in zip(trajectory, t_grid.tolist()):
            expected_m = ou_mean(a, b, m0, t)
            expected_p = ou_covariance(a, q, p0, t)
            assert float((state_t.means[0] - expected_m).abs().max()) < DEFAULT_OU_MOMENT_TOL
            assert float((state_t.covariances[0] - expected_p).abs().max()) < DEFAULT_OU_MOMENT_TOL

    def test_cubature_path_matches_analytic_for_linear_callable(self):
        """The sigma-point path is exact for linear drift — validates cubature."""
        a = torch.tensor(OU_2D["a"], dtype=F64)
        b = torch.tensor(OU_2D["b"], dtype=F64)
        cfg = StochasticGeneratorConfig(dim=2, diffusion=OU_2D["g"])
        gen = KolmogorovGenerator(cfg, drift=lambda x: x @ a.T + b)
        proj = GalerkinMomentProjection(gen, _splitting(rk4_substeps=8, ad_integrator="rk4"))
        state = _state(OU_2D)
        t_grid = torch.tensor([0.0, 0.5, 1.0], dtype=F64)
        trajectory = proj.propagate(state, t_grid)
        final = trajectory[-1]
        expected_m = ou_mean(a, b, torch.tensor(OU_2D["m0"], dtype=F64), 1.0)
        expected_p = ou_covariance(a, gen.q_matrix, torch.tensor(OU_2D["p0"], dtype=F64), 1.0)
        assert float((final.means[0] - expected_m).abs().max()) < DEFAULT_OU_MOMENT_TOL
        assert float((final.covariances[0] - expected_p).abs().max()) < DEFAULT_OU_MOMENT_TOL

    @settings(max_examples=20, deadline=None)
    @given(
        entries=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=4, max_size=4),
        scale=st.floats(min_value=0.1, max_value=1.0),
    )
    def test_hypothesis_stable_a_sweep(self, entries, scale):
        """Random stable A (bounded spectral radius by construction)."""
        m = torch.tensor([[entries[0], entries[1]], [entries[2], entries[3]]], dtype=F64)
        a = -(m @ m.T + 0.2 * torch.eye(2, dtype=F64)) * scale  # SND ⇒ stable
        cfg = StochasticGeneratorConfig(
            dim=2,
            drift_matrix=a.tolist(),
            diffusion=[[0.3, 0.0], [0.0, 0.2]],
        )
        gen = KolmogorovGenerator(cfg)
        proj = GalerkinMomentProjection(gen, _splitting(rk4_substeps=8))
        state = _state(
            {"m0": [0.5, -0.5], "p0": [[0.2, 0.0], [0.0, 0.3]], "a": None, "b": None, "g": None}
        )
        t_grid = torch.tensor([0.0, 1.0], dtype=F64)
        final = proj.propagate(state, t_grid)[-1]
        expected_m = ou_mean(
            a, torch.zeros(2, dtype=F64), torch.tensor([0.5, -0.5], dtype=F64), 1.0
        )
        expected_p = ou_covariance(
            a, gen.q_matrix, torch.tensor([[0.2, 0.0], [0.0, 0.3]], dtype=F64), 1.0
        )
        assert float((final.means[0] - expected_m).abs().max()) < DEFAULT_OU_MOMENT_TOL
        assert float((final.covariances[0] - expected_p).abs().max()) < DEFAULT_OU_MOMENT_TOL


class TestFlowIdentities:
    """Hand-computed 1D identities for the exact split flows."""

    def test_advection_flow_1d(self):
        theta, h = 1.0, 0.3
        gen = _generator(OU_1D)
        proj = GalerkinMomentProjection(gen, _splitting())
        state = _state(OU_1D)
        out = proj.advection_flow(state, h)
        assert abs(float(out.means[0, 0]) - 1.0 * math.exp(-theta * h)) < 1e-12
        assert abs(float(out.covariances[0, 0, 0]) - 0.5 * math.exp(-2 * theta * h)) < 1e-12

    def test_diffusion_flow_1d(self):
        h = 0.3
        gen = _generator(OU_1D)
        proj = GalerkinMomentProjection(gen, _splitting())
        state = _state(OU_1D)
        out = proj.diffusion_flow(state, h)
        torch.testing.assert_close(out.means, state.means)
        assert abs(float(out.covariances[0, 0, 0]) - (0.5 + h * 0.25)) < 1e-12

    def test_flows_do_not_commute(self):
        """A∘D vs D∘A differ in covariance — the AC4 signal is real."""
        h = 0.4
        gen = _generator(OU_1D)
        proj = GalerkinMomentProjection(gen, _splitting())
        state = _state(OU_1D)
        ad = proj.diffusion_flow(proj.advection_flow(state, h), h)
        da = proj.advection_flow(proj.diffusion_flow(state, h), h)
        assert abs(float(ad.covariances[0, 0, 0]) - float(da.covariances[0, 0, 0])) > 1e-3

    def test_weights_frozen_through_flows(self):
        gen = _generator(OU_2D)
        proj = GalerkinMomentProjection(gen, _splitting())
        basis = GaussianMixtureBasis(dim=2, n_components=2)
        state = basis.initial_state(
            means=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=F64),
            covariances=torch.eye(2, dtype=F64).expand(2, 2, 2).clone() * 0.3,
            weights=torch.tensor([0.7, 0.3], dtype=F64),
        )
        out = proj.diffusion_flow(proj.advection_flow(state, 0.2), 0.2)
        torch.testing.assert_close(out.weights, state.weights)


class TestProjectionValidation:
    def test_exact_expm_requires_linear_drift(self):
        cfg = StochasticGeneratorConfig(dim=1, diffusion=[[0.3]])
        gen = KolmogorovGenerator(cfg, drift=lambda x: -x)
        with pytest.raises(StochasticConfigurationError, match="requires linear drift"):
            GalerkinMomentProjection(gen, _splitting(ad_integrator="exact_expm"))

    def test_propagate_rejects_bad_grid(self):
        gen = _generator(OU_1D)
        proj = GalerkinMomentProjection(gen, _splitting())
        state = _state(OU_1D)
        with pytest.raises(StochasticConfigurationError, match="strictly increasing"):
            proj.propagate(state, torch.tensor([0.0], dtype=F64))
        with pytest.raises(StochasticConfigurationError, match="strictly increasing"):
            proj.propagate(state, torch.tensor([0.0, 0.5, 0.5], dtype=F64))

    def test_multi_component_propagates_independently(self):
        gen = _generator(OU_2D)
        proj = GalerkinMomentProjection(gen, _splitting(rk4_substeps=8))
        basis = GaussianMixtureBasis(dim=2, n_components=2)
        means = torch.tensor([[1.0, -0.5], [-1.0, 0.5]], dtype=F64)
        covs = torch.stack(
            [
                torch.diag(torch.tensor([0.3, 0.2], dtype=F64)),
                torch.diag(torch.tensor([0.1, 0.4], dtype=F64)),
            ]
        )
        state = basis.initial_state(means=means, covariances=covs)
        t_grid = torch.tensor([0.0, 1.0], dtype=F64)
        joint = proj.propagate(state, t_grid)[-1]
        for k in range(2):
            single_basis = GaussianMixtureBasis(dim=2, n_components=1)
            single = single_basis.initial_state(means=means[k : k + 1], covariances=covs[k : k + 1])
            single_out = proj.propagate(single, t_grid)[-1]
            torch.testing.assert_close(joint.means[k], single_out.means[0])
            torch.testing.assert_close(joint.covariances[k], single_out.covariances[0])

    def test_packed_rhs_zero_weight_derivative(self):
        gen = _generator(OU_1D)
        proj = GalerkinMomentProjection(gen, _splitting())
        state = _state(OU_1D)
        rhs = proj.packed_rhs(1, 1)
        derivative = rhs(state.pack(), 0.0)
        assert float(derivative[0]) == 0.0  # weight slot
