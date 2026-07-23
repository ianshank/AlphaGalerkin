"""AC3-trained tests: the MDN jump semigroup learns the compound-Poisson flow.

The MDN is trained on pure-jump transitions of the spec-pinned problem
(isolating the jump flow from A+D), then its applied moments are compared
against the exact oracle within the calibrated tolerance. Calibration run
(seeds below): NLL 0.803 → 0.727, mean err 6.7e-3, cov err 2.7e-2 — the
5e-2 gate carries ≈1.8× headroom.
"""

from __future__ import annotations

import pytest
import torch

from src.pde.stochastic.config import (
    DEFAULT_TRAINED_MDN_MOMENT_TOL,
    JumpConfig,
    MDNJumpConfig,
)
from src.pde.stochastic.errors import StochasticConfigurationError
from src.pde.stochastic.gaussian_mixture import GaussianMixtureState
from src.pde.stochastic.jump_mdn import (
    AnalyticCompoundPoissonMoments,
    MDNJumpSemigroup,
    batched_mixture_nll,
    pack_batch,
    unpack_batch,
)
from src.pde.stochastic.particles import _compound_poisson_increment, sample_gaussian

F64 = torch.float64
PINNED_JUMP = JumpConfig(rate=2.0, jump_mean=[0.5], jump_cov=[[0.04]])
PINNED_DT = 0.1


def _pinned_state() -> GaussianMixtureState:
    return GaussianMixtureState(
        weights=torch.ones(1, dtype=F64),
        means=torch.zeros(1, 1, dtype=F64),
        covariances=torch.tensor([[[0.1]]], dtype=F64),
    )


def _train_mdn(n_steps: int = 300) -> tuple[MDNJumpSemigroup, list[float]]:
    """Deterministic training run on pure-jump transitions (the AC3 setup)."""
    torch.manual_seed(42)
    batch, n_samples = 16, 512
    gen = torch.Generator().manual_seed(7)
    means = torch.rand(batch, 1, dtype=F64, generator=gen) * 2 - 1
    variances = torch.rand(batch, 1, dtype=F64, generator=gen) * 0.25 + 0.05
    packed_in = torch.cat([torch.ones(batch, 1), means.float(), variances.float()], dim=1)
    targets = []
    for i in range(batch):
        x = sample_gaussian(n_samples, means[i], variances[i].reshape(1, 1), gen)
        inc = _compound_poisson_increment(n_samples, PINNED_DT, PINNED_JUMP, gen)
        targets.append((x + inc).float())
    target_batch = torch.stack(targets)

    mdn = MDNJumpSemigroup(MDNJumpConfig(dim=1, n_components=1))
    optimizer = torch.optim.Adam(mdn.parameters(), lr=5e-3)
    dts = torch.full((batch, 1), PINNED_DT, dtype=torch.float32)
    losses: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        out = mdn.forward(packed_in, dts)
        loss = mdn.nll(out, target_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return mdn, losses


@pytest.fixture(scope="module")
def trained_mdn() -> tuple[MDNJumpSemigroup, list[float]]:
    return _train_mdn()


class TestResidualParameterization:
    def test_identity_at_dt_zero(self):
        torch.manual_seed(0)
        mdn = MDNJumpSemigroup(MDNJumpConfig(dim=2, n_components=2))
        state = GaussianMixtureState(
            weights=torch.tensor([0.6, 0.4], dtype=F64),
            means=torch.tensor([[0.5, -0.5], [1.0, 1.0]], dtype=F64),
            covariances=torch.stack(
                [
                    torch.tensor([[0.3, 0.05], [0.05, 0.2]], dtype=F64),
                    torch.tensor([[0.15, 0.0], [0.0, 0.25]], dtype=F64),
                ]
            ),
        )
        out = mdn.advance(state, 0.0)
        torch.testing.assert_close(out.weights, state.weights, rtol=0, atol=1e-6)
        torch.testing.assert_close(out.means, state.means, rtol=0, atol=1e-6)
        torch.testing.assert_close(out.covariances, state.covariances, rtol=0, atol=1e-6)

    def test_apply_dtype_round_trip(self):
        torch.manual_seed(0)
        mdn = MDNJumpSemigroup(MDNJumpConfig(dim=1, n_components=1))
        state = _pinned_state()
        out = mdn.advance(state, PINNED_DT)
        assert out.dtype is F64
        assert out.n_components == 1
        assert out.dim == 1
        # Output covariance stays SPD.
        assert float(out.covariances[0, 0, 0]) > 0

    def test_apply_shape_mismatch_raises(self):
        torch.manual_seed(0)
        mdn = MDNJumpSemigroup(MDNJumpConfig(dim=2, n_components=1))
        with pytest.raises(StochasticConfigurationError, match="MDN configured"):
            mdn.advance(_pinned_state(), PINNED_DT)


class TestBatchedHelpers:
    def test_pack_unpack_round_trip(self):
        gen = torch.Generator().manual_seed(1)
        b, k, d = 3, 2, 2
        weights = torch.softmax(torch.randn(b, k, generator=gen), dim=1)
        means = torch.randn(b, k, d, generator=gen)
        factors = torch.randn(b, k, d, d, generator=gen) * 0.3
        covs = factors @ factors.transpose(-1, -2) + 0.5 * torch.eye(d)
        packed = pack_batch(weights, means, covs)
        w2, m2, c2 = unpack_batch(packed, k, d)
        torch.testing.assert_close(w2, weights)
        torch.testing.assert_close(m2, means)
        torch.testing.assert_close(c2, covs)

    def test_unpack_bad_shape_raises(self):
        with pytest.raises(StochasticConfigurationError, match="packed batch"):
            unpack_batch(torch.zeros(2, 5), 2, 2)

    def test_batched_nll_matches_single_state_log_prob(self):
        state = GaussianMixtureState(
            weights=torch.tensor([0.7, 0.3], dtype=F64),
            means=torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=F64),
            covariances=torch.stack(
                [
                    torch.tensor([[0.4, 0.1], [0.1, 0.3]], dtype=F64),
                    torch.tensor([[0.2, 0.0], [0.0, 0.5]], dtype=F64),
                ]
            ),
        )
        gen = torch.Generator().manual_seed(2)
        samples = torch.randn(40, 2, dtype=F64, generator=gen)
        packed = state.pack().unsqueeze(0)
        batched = batched_mixture_nll(packed, samples.unsqueeze(0), 2, 2)
        reference = -state.log_prob(samples).mean()
        assert float(batched[0]) == pytest.approx(float(reference), rel=1e-9)


class TestAC3TrainedMdn:
    """AC3: trained MDN reproduces the oracle jump moments within tolerance."""

    def test_nll_strictly_decreased(self, trained_mdn):
        _mdn, losses = trained_mdn
        assert losses[-1] < losses[0]

    def test_trained_moments_match_oracle(self, trained_mdn):
        mdn, _losses = trained_mdn
        state = _pinned_state()
        out = mdn.advance(state, PINNED_DT)
        oracle = AnalyticCompoundPoissonMoments(PINNED_JUMP).advance(state, PINNED_DT)
        mean_err = abs(float(out.means[0, 0] - oracle.means[0, 0]))
        cov_err = abs(float(out.covariances[0, 0, 0] - oracle.covariances[0, 0, 0]))
        assert mean_err < DEFAULT_TRAINED_MDN_MOMENT_TOL
        assert cov_err < DEFAULT_TRAINED_MDN_MOMENT_TOL

    def test_weights_remain_normalized(self, trained_mdn):
        mdn, _losses = trained_mdn
        out = mdn.advance(_pinned_state(), PINNED_DT)
        assert float(out.weights.sum()) == pytest.approx(1.0, abs=1e-6)
