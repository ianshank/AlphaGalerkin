"""Unit tests for GaussianMixtureState / GaussianMixtureBasis.

Covers pack/unpack round-trips (incl. Hypothesis), the non-circular
``log_prob`` reference against ``torch.distributions``, density normalization,
state validation, and the basis config-or-kwargs idiom.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from src.pde.stochastic.config import GaussianMixtureBasisConfig
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureBasis, GaussianMixtureState


def _random_state(k: int, d: int, seed: int = 0) -> GaussianMixtureState:
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(k, generator=gen, dtype=torch.float64) + 0.1
    weights = weights / weights.sum()
    means = torch.randn(k, d, generator=gen, dtype=torch.float64)
    factors = torch.randn(k, d, d, generator=gen, dtype=torch.float64) * 0.3
    covariances = factors @ factors.transpose(-1, -2) + 0.5 * torch.eye(d, dtype=torch.float64)
    return GaussianMixtureState(weights=weights, means=means, covariances=covariances)


class TestStateValidation:
    def test_valid_state(self):
        state = _random_state(2, 3)
        assert state.n_components == 2
        assert state.dim == 3
        assert state.dtype is torch.float64

    def test_weights_must_sum_to_one(self):
        with pytest.raises(StochasticConfigurationError, match="sum to 1"):
            GaussianMixtureState(
                weights=torch.tensor([0.5, 0.2], dtype=torch.float64),
                means=torch.zeros(2, 1, dtype=torch.float64),
                covariances=torch.eye(1, dtype=torch.float64).expand(2, 1, 1).clone(),
            )

    def test_asymmetric_covariance_rejected(self):
        cov = torch.tensor([[[1.0, 0.5], [0.0, 1.0]]], dtype=torch.float64)
        with pytest.raises(StochasticConfigurationError, match="symmetric"):
            GaussianMixtureState(
                weights=torch.ones(1, dtype=torch.float64),
                means=torch.zeros(1, 2, dtype=torch.float64),
                covariances=cov,
            )

    def test_shape_mismatch_rejected(self):
        with pytest.raises(StochasticConfigurationError, match="inconsistent"):
            GaussianMixtureState(
                weights=torch.ones(2, dtype=torch.float64) / 2,
                means=torch.zeros(3, 1, dtype=torch.float64),
                covariances=torch.eye(1, dtype=torch.float64).expand(3, 1, 1).clone(),
            )

    def test_bad_ndim_rejected(self):
        with pytest.raises(StochasticConfigurationError, match="expected weights"):
            GaussianMixtureState(
                weights=torch.ones(1, 1, dtype=torch.float64),
                means=torch.zeros(1, 1, dtype=torch.float64),
                covariances=torch.eye(1, dtype=torch.float64).expand(1, 1, 1).clone(),
            )


class TestPackUnpack:
    def test_round_trip_2d(self):
        state = _random_state(3, 2)
        packed = state.pack()
        assert packed.shape == (3 + 3 * 2 + 3 * 3,)
        restored = GaussianMixtureState.unpack(packed, n_components=3, dim=2)
        torch.testing.assert_close(restored.weights, state.weights)
        torch.testing.assert_close(restored.means, state.means)
        torch.testing.assert_close(restored.covariances, state.covariances)

    @settings(max_examples=20, deadline=None)
    @given(k=st.integers(min_value=1, max_value=4), d=st.integers(min_value=1, max_value=4))
    def test_round_trip_hypothesis(self, k, d):
        state = _random_state(k, d, seed=k * 10 + d)
        restored = GaussianMixtureState.unpack(state.pack(), n_components=k, dim=d)
        torch.testing.assert_close(restored.covariances, state.covariances)
        torch.testing.assert_close(restored.means, state.means)

    def test_unpack_wrong_length_rejected(self):
        with pytest.raises(StochasticConfigurationError, match="packed vector"):
            GaussianMixtureState.unpack(torch.zeros(5, dtype=torch.float64), 2, 2)

    def test_unpacked_covariance_is_symmetric(self):
        state = _random_state(2, 3)
        restored = GaussianMixtureState.unpack(state.pack(), 2, 3)
        torch.testing.assert_close(restored.covariances, restored.covariances.transpose(-1, -2))


class TestLogProb:
    def test_matches_torch_distributions(self):
        state = _random_state(3, 2, seed=7)
        x = torch.randn(50, 2, dtype=torch.float64)
        mix = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(probs=state.weights),
            torch.distributions.MultivariateNormal(state.means, state.covariances),
        )
        torch.testing.assert_close(state.log_prob(x), mix.log_prob(x), rtol=1e-6, atol=1e-6)

    def test_density_integrates_to_one_1d(self):
        state = _random_state(2, 1, seed=3)
        grid = torch.linspace(-12.0, 12.0, 4001, dtype=torch.float64).unsqueeze(1)
        density = state.density_on_grid(grid)
        integral = torch.trapezoid(density, grid.squeeze(1))
        assert abs(float(integral) - 1.0) < 1e-4

    def test_bad_point_shape_rejected(self):
        state = _random_state(1, 2)
        with pytest.raises(StochasticConfigurationError, match="x must have shape"):
            state.log_prob(torch.zeros(5, 3, dtype=torch.float64))

    def test_to_dtype_round_trip(self):
        state = _random_state(1, 2)
        as32 = state.to_dtype(torch.float32)
        assert as32.dtype is torch.float32
        back = as32.to_dtype(torch.float64)
        torch.testing.assert_close(back.means, state.means, rtol=1e-6, atol=1e-6)


class TestBasis:
    def test_config_or_kwargs(self):
        via_cfg = GaussianMixtureBasis(GaussianMixtureBasisConfig(dim=2, n_components=2))
        via_kwargs = GaussianMixtureBasis(dim=2, n_components=2)
        assert via_cfg.config.dim == via_kwargs.config.dim

    def test_both_config_and_kwargs_rejected(self):
        with pytest.raises(StochasticConfigurationError, match="not both"):
            GaussianMixtureBasis(GaussianMixtureBasisConfig(dim=2), dim=2)

    def test_initial_state_uniform_weights_and_dtype(self):
        basis = GaussianMixtureBasis(dim=2, n_components=2)
        state = basis.initial_state(
            means=torch.zeros(2, 2, dtype=torch.float32),
            covariances=torch.eye(2, dtype=torch.float32).expand(2, 2, 2).clone(),
        )
        assert state.dtype is torch.float64
        torch.testing.assert_close(state.weights, torch.full((2,), 0.5, dtype=torch.float64))

    def test_initial_state_shape_mismatch_rejected(self):
        basis = GaussianMixtureBasis(dim=2, n_components=1)
        with pytest.raises(StochasticConfigurationError, match="means must have shape"):
            basis.initial_state(
                means=torch.zeros(2, 2, dtype=torch.float64),
                covariances=torch.eye(2, dtype=torch.float64).expand(2, 2, 2).clone(),
            )
        with pytest.raises(StochasticConfigurationError, match="covariances must have shape"):
            basis.initial_state(
                means=torch.zeros(1, 2, dtype=torch.float64),
                covariances=torch.eye(2, dtype=torch.float64).unsqueeze(0).expand(2, 2, 2).clone(),
            )
