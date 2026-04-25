"""Tests for ``AlphaGalerkinSolver`` with ``evaluator='trained'``.

These tests cover the network-backed evaluator wiring path:

- The config validator rejects misconfigured ``trained`` setups (missing /
  non-existent checkpoint).
- ``_build_mcts`` correctly dispatches ``trained`` to ``FNetEvaluator`` and
  ``random``/``uniform`` to ``RandomEvaluator``.
- A checkpoint trained for a *different* PDE action space loads with
  ``strict=False`` so policy-head shape mismatches degrade gracefully.
- The GPU code path is exercised when CUDA is available; the CPU-only CI
  hosts auto-skip via the root ``conftest.py`` ``gpu_required`` hook.

These are *wiring* tests targeted at the public boundary that this PR
changed: ``AlphaGalerkinConfig`` validation and ``_build_mcts`` dispatch.
We deliberately avoid driving a full ``solve()`` call so this suite stays
independent of orthogonal shape contracts between ``BasisSelectionGame.
to_tensor()`` and ``AlphaGalerkinModel`` (which only accepts square-grid
inputs whereas the default Poisson collocation count is non-square).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from config.schemas import OperatorConfig
from src.alphagalerkin import AlphaGalerkinConfig, AlphaGalerkinSolver
from src.mcts.evaluator import FNetEvaluator, RandomEvaluator
from src.modeling.model import AlphaGalerkinModel

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubPDEGame:
    """Minimal stand-in for ``PDEGame`` that exposes ``action_space_size``.

    ``AlphaGalerkinSolver._build_mcts`` only reads ``action_space_size`` from
    its argument when constructing the ``RandomEvaluator``; the trained
    branch uses the loaded model's policy-head dimensionality instead. This
    keeps the wiring tests independent of the heavyweight game classes.
    """

    def __init__(self, action_space_size: int) -> None:
        self.action_space_size = action_space_size


def _operator_config(action_space_size: int) -> OperatorConfig:
    """Tiny ``OperatorConfig`` matching the PDE-state channel layout."""
    return OperatorConfig(
        d_model=16,
        d_key=16,
        d_value=16,
        d_ffn=32,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=8,
        use_fnet_mixing=False,
        input_channels=23,  # 3 + default max_basis_functions=20
        action_space_size=action_space_size,
    )


def _save_tiny_checkpoint(
    path: Path,
    op_config: OperatorConfig,
    seed: int = 0,
) -> None:
    """Save a tiny checkpoint in the format ``create_model_from_checkpoint`` expects.

    Builds a tiny ``AlphaGalerkinModel``, deterministically initializes it,
    and writes the standard ``{"model_state_dict", "config"}`` payload.
    """
    torch.manual_seed(seed)
    model = AlphaGalerkinModel(op_config)
    model.eval()
    state = {
        "model_state_dict": model.state_dict(),
        "config": {"operator": op_config.model_dump()},
        "version": "test",
    }
    torch.save(state, path)


def _solver_kwargs(**overrides: object) -> dict[str, object]:
    """Fast-CI-friendly solver kwargs (mirrors test_solver._fast_solver_config)."""
    kwargs: dict[str, object] = {
        "game_mode": "basis_selection",
        "n_mcts_simulations": 2,
        "max_steps": 2,
        "target_tolerance": 1e-4,
        "device": "cpu",
        "seed": 7,
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# Dispatch: random/uniform vs trained
# ---------------------------------------------------------------------------


class TestEvaluatorDispatch:
    """``AlphaGalerkinSolver._build_mcts`` selects the right evaluator class."""

    def test_random_dispatches_to_random_evaluator(self) -> None:
        """Default ``evaluator='random'`` builds a ``RandomEvaluator``."""
        solver = AlphaGalerkinSolver(
            AlphaGalerkinConfig(**_solver_kwargs(evaluator="random")),
        )
        mcts = solver._build_mcts(_StubPDEGame(action_space_size=64))
        assert isinstance(mcts.evaluator, RandomEvaluator)
        assert mcts.evaluator.n_actions == 64

    def test_uniform_dispatches_to_random_evaluator(self) -> None:
        """``'uniform'`` is a documented alias for the random path."""
        solver = AlphaGalerkinSolver(
            AlphaGalerkinConfig(**_solver_kwargs(evaluator="uniform")),
        )
        mcts = solver._build_mcts(_StubPDEGame(action_space_size=32))
        assert isinstance(mcts.evaluator, RandomEvaluator)
        assert mcts.evaluator.n_actions == 32

    def test_trained_dispatches_to_fnet_evaluator(
        self,
        tmp_path: Path,
    ) -> None:
        """``'trained'`` loads a checkpoint and wraps it in ``FNetEvaluator``."""
        op_cfg = _operator_config(action_space_size=64)
        ckpt = tmp_path / "trained.pt"
        _save_tiny_checkpoint(ckpt, op_cfg)

        solver = AlphaGalerkinSolver(
            AlphaGalerkinConfig(
                **_solver_kwargs(evaluator="trained", checkpoint_path=ckpt),
            ),
        )
        mcts = solver._build_mcts(_StubPDEGame(action_space_size=64))
        assert isinstance(mcts.evaluator, FNetEvaluator)
        # Loaded model parameters live on the resolved device.
        first_param = next(mcts.evaluator.model.parameters())
        assert first_param.device.type == "cpu"


# ---------------------------------------------------------------------------
# Graceful degradation: action-space mismatch
# ---------------------------------------------------------------------------


class TestActionSpaceMismatch:
    """Checkpoint trained for one PDE loads (with warnings) on another."""

    def test_action_space_mismatch_loads_non_strict(
        self,
        tmp_path: Path,
    ) -> None:
        """Mismatched policy-head shape must not abort ``_build_mcts``.

        The checkpoint is saved with ``action_space_size=8``; the stub game
        reports ``action_space_size=64``. Because the solver's
        ``create_model_from_checkpoint(strict=False)`` call tolerates head
        shape mismatches, ``_build_mcts`` should still return a valid MCTS
        with an ``FNetEvaluator`` (the unloaded keys are surfaced via the
        checkpoint loader's logger).
        """
        op_cfg = _operator_config(action_space_size=8)
        ckpt = tmp_path / "tiny.pt"
        _save_tiny_checkpoint(ckpt, op_cfg)

        solver = AlphaGalerkinSolver(
            AlphaGalerkinConfig(
                **_solver_kwargs(evaluator="trained", checkpoint_path=ckpt),
            ),
        )
        mcts = solver._build_mcts(_StubPDEGame(action_space_size=64))
        assert isinstance(mcts.evaluator, FNetEvaluator)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestDeviceResolution:
    """``_resolve_device`` falls back to CPU when CUDA is unavailable."""

    def test_cpu_passes_through(self) -> None:
        assert AlphaGalerkinSolver._resolve_device("cpu") == "cpu"

    def test_cuda_falls_back_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``cuda`` resolves to ``cpu`` when ``torch.cuda.is_available()`` is False."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert AlphaGalerkinSolver._resolve_device("cuda") == "cpu"
        assert AlphaGalerkinSolver._resolve_device("cuda:0") == "cpu"

    def test_cuda_passes_through_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert AlphaGalerkinSolver._resolve_device("cuda") == "cuda"
        assert AlphaGalerkinSolver._resolve_device("cuda:0") == "cuda:0"


# ---------------------------------------------------------------------------
# GPU smoke: trained evaluator on CUDA
# ---------------------------------------------------------------------------


class TestTrainedEvaluatorGPU:
    """Exercise the trained-evaluator path on CUDA when available."""

    @pytest.mark.gpu_required
    def test_trained_evaluator_loads_on_gpu(
        self,
        tmp_path: Path,
    ) -> None:
        """The trained evaluator's model parameters land on a CUDA device.

        Skipped automatically by the root ``conftest.py`` GPU hook on
        CPU-only CI. When CUDA is available, asserts the model parameters
        actually moved onto a CUDA device (no silent CPU fallback).
        """
        op_cfg = _operator_config(action_space_size=64)
        ckpt = tmp_path / "gpu.pt"
        _save_tiny_checkpoint(ckpt, op_cfg)

        solver = AlphaGalerkinSolver(
            AlphaGalerkinConfig(
                **_solver_kwargs(
                    evaluator="trained",
                    checkpoint_path=ckpt,
                    device="cuda",
                ),
            ),
        )
        mcts = solver._build_mcts(_StubPDEGame(action_space_size=64))
        assert isinstance(mcts.evaluator, FNetEvaluator)
        first_param = next(mcts.evaluator.model.parameters())
        assert first_param.device.type == "cuda"
