"""GPU smoke tests for the stochastic Galerkin layer (device-agnostic contract).

Auto-skipped on CPU CI via the root ``conftest.py`` ``gpu_required`` hook.
Asserts CUDA propagation matches the CPU result and that the trainer and
comparison harness run end-to-end on a CUDA device.
"""

from __future__ import annotations

import pytest
import torch

from src.pde.stochastic.config import (
    JumpConfig,
    StochasticGeneratorConfig,
    StrangSplittingConfig,
)
from src.pde.stochastic.gaussian_mixture import GaussianMixtureBasis
from src.pde.stochastic.generator import KolmogorovGenerator
from src.pde.stochastic.jump_mdn import AnalyticCompoundPoissonMoments
from src.pde.stochastic.projection import GalerkinMomentProjection
from src.pde.stochastic.strang import StrangSplitStep

F64 = torch.float64
PINNED_JUMP = JumpConfig(rate=2.0, jump_mean=[0.5], jump_cov=[[0.04]])


def _propagate(device: str):
    cfg = StochasticGeneratorConfig(
        dim=1, drift_matrix=[[-1.0]], diffusion=[[0.3]], jump=PINNED_JUMP
    )
    gen = KolmogorovGenerator(cfg, jump_semigroup=AnalyticCompoundPoissonMoments(PINNED_JUMP))
    proj = GalerkinMomentProjection(gen, StrangSplittingConfig(dt=0.1, t_end=1.0))
    basis = GaussianMixtureBasis(dim=1)
    state = basis.initial_state(
        means=torch.zeros(1, 1, dtype=F64, device=device),
        covariances=torch.tensor([[[0.1]]], dtype=F64, device=device),
    )
    final, _t = StrangSplitStep(proj).propagate(state)[-1]
    return final


@pytest.mark.gpu_required
class TestCudaSmoke:
    def test_cuda_propagation_matches_cpu(self):
        cpu = _propagate("cpu")
        cuda = _propagate("cuda")
        assert cuda.means.device.type == "cuda"
        torch.testing.assert_close(cuda.means.cpu(), cpu.means, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(cuda.covariances.cpu(), cpu.covariances, rtol=1e-10, atol=1e-12)

    def test_trainer_runs_on_cuda(self):
        from tests.pde.stochastic.test_trainer import _build

        trainer, _proj, _mdn = _build(max_steps=3, device="cuda")
        result = trainer.train()
        assert all(torch.isfinite(torch.tensor(result.loss_history)))

    def test_harness_runs_on_cuda(self):
        from src.research.stochastic_galerkin_compare import (
            StochasticCompareParams,
            run_stochastic_galerkin_comparison,
        )

        params = StochasticCompareParams(
            grid_n=12,
            n_train_samples=6,
            n_eval_samples=3,
            n_epochs=1,
            d_model=16,
            n_fourier_features=8,
            batch_size=4,
            device="cuda",
        )
        result = run_stochastic_galerkin_comparison(params)
        assert result.stochastic.density_mse < 1e-6
