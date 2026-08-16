"""Unit tests for the stochastic-layer Pydantic configs.

Guards field bounds, ``extra="forbid"``, cross-field shape validators, and
hash determinism (spec: Data Contract).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pde.stochastic.config import (
    DEFAULT_MONOTONE_WINDOW,
    GaussianMixtureBasisConfig,
    JumpConfig,
    MDNJumpConfig,
    StochasticGeneratorConfig,
    StrangSplittingConfig,
    StrangTrainerConfig,
)


def _jump(rate: float = 1.0, dim: int = 1) -> JumpConfig:
    return JumpConfig(
        rate=rate,
        jump_mean=[0.5] * dim,
        jump_cov=[[0.04 if i == j else 0.0 for j in range(dim)] for i in range(dim)],
    )


class TestGaussianMixtureBasisConfig:
    def test_defaults(self):
        cfg = GaussianMixtureBasisConfig(dim=2)
        assert cfg.n_components == 1
        assert cfg.dtype == "float64"
        assert cfg.weight_dynamics == "frozen"

    @pytest.mark.parametrize("dim", [0, 9])
    def test_dim_bounds(self, dim):
        with pytest.raises(ValidationError):
            GaussianMixtureBasisConfig(dim=dim)

    @pytest.mark.parametrize("k", [0, 33])
    def test_component_bounds(self, k):
        with pytest.raises(ValidationError):
            GaussianMixtureBasisConfig(dim=2, n_components=k)

    def test_weight_dynamics_locked_to_frozen(self):
        with pytest.raises(ValidationError):
            GaussianMixtureBasisConfig(dim=2, weight_dynamics="learned")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            GaussianMixtureBasisConfig(dim=2, bogus=1)

    def test_torch_dtype_property(self):
        import torch

        assert GaussianMixtureBasisConfig(dim=1).torch_dtype is torch.float64
        assert GaussianMixtureBasisConfig(dim=1, dtype="float32").torch_dtype is torch.float32

    def test_hash_deterministic(self):
        a = GaussianMixtureBasisConfig(dim=3, n_components=2)
        b = GaussianMixtureBasisConfig(dim=3, n_components=2)
        assert a.compute_hash() == b.compute_hash()
        c = GaussianMixtureBasisConfig(dim=3, n_components=3)
        assert a.compute_hash() != c.compute_hash()


class TestJumpConfig:
    def test_valid(self):
        cfg = _jump(rate=2.0, dim=2)
        assert cfg.rate == 2.0

    def test_negative_rate_rejected(self):
        with pytest.raises(ValidationError):
            _jump(rate=-0.1)

    def test_zero_rate_allowed(self):
        assert _jump(rate=0.0).rate == 0.0

    def test_cov_shape_mismatch(self):
        with pytest.raises(ValidationError):
            JumpConfig(rate=1.0, jump_mean=[0.5, 0.5], jump_cov=[[0.1]])

    def test_ragged_cov_rejected(self):
        with pytest.raises(ValidationError):
            JumpConfig(rate=1.0, jump_mean=[0.5, 0.5], jump_cov=[[0.1, 0.0], [0.0]])

    def test_asymmetric_cov_rejected(self):
        with pytest.raises(ValidationError):
            JumpConfig(rate=1.0, jump_mean=[0.0, 0.0], jump_cov=[[0.1, 0.5], [0.0, 0.1]])


class TestStochasticGeneratorConfig:
    def test_valid_linear(self):
        cfg = StochasticGeneratorConfig(
            dim=2,
            drift_matrix=[[-1.0, 0.3], [0.0, -0.8]],
            drift_bias=[0.1, -0.2],
            diffusion=[[0.4, 0.0], [0.0, 0.3]],
        )
        assert cfg.has_jump is False
        assert cfg.drift_matrix_tensor() is not None
        assert cfg.drift_bias_tensor().shape == (2,)
        assert cfg.diffusion_tensor().shape == (2, 2)

    def test_bias_defaults_to_zeros(self):
        cfg = StochasticGeneratorConfig(dim=1, drift_matrix=[[-1.0]], diffusion=[[0.5]])
        assert cfg.drift_bias_tensor().tolist() == [0.0]

    def test_drift_matrix_shape_rejected(self):
        with pytest.raises(ValidationError):
            StochasticGeneratorConfig(dim=2, drift_matrix=[[-1.0]], diffusion=[[0.4], [0.3]])

    def test_bias_length_rejected(self):
        with pytest.raises(ValidationError):
            StochasticGeneratorConfig(
                dim=2,
                drift_matrix=[[-1.0, 0.0], [0.0, -1.0]],
                drift_bias=[0.1],
                diffusion=[[0.4], [0.3]],
            )

    def test_diffusion_rows_must_match_dim(self):
        with pytest.raises(ValidationError):
            StochasticGeneratorConfig(
                dim=2, drift_matrix=[[-1.0, 0.0], [0.0, -1.0]], diffusion=[[0.4]]
            )

    def test_ragged_diffusion_rejected(self):
        with pytest.raises(ValidationError):
            StochasticGeneratorConfig(
                dim=2,
                drift_matrix=[[-1.0, 0.0], [0.0, -1.0]],
                diffusion=[[0.4, 0.0], [0.3]],
            )

    def test_jump_dim_mismatch_rejected(self):
        with pytest.raises(ValidationError):
            StochasticGeneratorConfig(
                dim=2,
                drift_matrix=[[-1.0, 0.0], [0.0, -1.0]],
                diffusion=[[0.4], [0.3]],
                jump=_jump(dim=1),
            )

    def test_has_jump_semantics(self):
        base = {
            "dim": 1,
            "drift_matrix": [[-1.0]],
            "diffusion": [[0.3]],
        }
        assert StochasticGeneratorConfig(**base).has_jump is False
        assert StochasticGeneratorConfig(**base, jump=_jump(rate=0.0)).has_jump is False
        assert StochasticGeneratorConfig(**base, jump=_jump(rate=2.0)).has_jump is True

    def test_callable_drift_allows_none_matrix(self):
        cfg = StochasticGeneratorConfig(dim=1, diffusion=[[0.3]])
        assert cfg.drift_matrix_tensor() is None


class TestMDNJumpConfig:
    def test_defaults(self):
        cfg = MDNJumpConfig(dim=1, n_components=1)
        assert cfg.hidden_dims == [64, 64]
        assert cfg.dt_embed_dim == 8

    def test_empty_hidden_dims_rejected(self):
        with pytest.raises(ValidationError):
            MDNJumpConfig(dim=1, n_components=1, hidden_dims=[])

    def test_nonpositive_hidden_dim_rejected(self):
        with pytest.raises(ValidationError):
            MDNJumpConfig(dim=1, n_components=1, hidden_dims=[64, 0])


class TestStrangSplittingConfig:
    def test_valid(self):
        cfg = StrangSplittingConfig(dt=0.1, t_end=1.0)
        assert cfg.ad_integrator == "exact_expm"

    def test_dt_exceeding_horizon_rejected(self):
        with pytest.raises(ValidationError):
            StrangSplittingConfig(dt=2.0, t_end=1.0)

    def test_unknown_integrator_rejected(self):
        with pytest.raises(ValidationError):
            StrangSplittingConfig(dt=0.1, t_end=1.0, ad_integrator="euler")


class TestStrangTrainerConfig:
    def test_defaults(self):
        cfg = StrangTrainerConfig(n_particles=100, n_time_slices=5, sim_dt=0.01)
        assert cfg.max_steps == 500
        assert cfg.full_batch is True
        assert cfg.monotone_window == DEFAULT_MONOTONE_WINDOW

    def test_particle_bounds(self):
        with pytest.raises(ValidationError):
            StrangTrainerConfig(n_particles=1, n_time_slices=5, sim_dt=0.01)

    def test_min_time_slices(self):
        with pytest.raises(ValidationError):
            StrangTrainerConfig(n_particles=100, n_time_slices=2, sim_dt=0.01)
