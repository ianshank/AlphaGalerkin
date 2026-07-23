"""Coverage for FNetEvaluator using a minimal fake model.

FNetEvaluator only reads ``output.policy_logits`` / ``output.value`` and
dispatches on ``forward_fast``; a tiny stand-in exercises every branch without
constructing the full AlphaGalerkinModel.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.mcts.evaluator import FNetEvaluator


class _FakeOutput:
    def __init__(self, policy_logits: torch.Tensor, value: torch.Tensor) -> None:
        self.policy_logits = policy_logits
        self.value = value


class _FakeModelNoFast(torch.nn.Module):
    """Returns constant logits/value; no ``forward_fast`` method."""

    def __init__(self, n_actions: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self._lin = torch.nn.Linear(1, 1)  # gives .to()/.eval() something real
        self.slow_calls = 0

    def forward(self, x: torch.Tensor) -> _FakeOutput:
        self.slow_calls += 1
        b = x.shape[0]
        logits = torch.arange(self.n_actions, dtype=torch.float32).repeat(b, 1)
        value = torch.full((b, 1), 0.25)
        return _FakeOutput(logits, value)


class _FakeModel(_FakeModelNoFast):
    """Adds a ``forward_fast`` fast path over the base fake model."""

    def __init__(self, n_actions: int) -> None:
        super().__init__(n_actions)
        self.fast_calls = 0

    def forward_fast(self, x: torch.Tensor) -> _FakeOutput:
        self.fast_calls += 1
        return self.forward(x)


def _state() -> np.ndarray:
    return np.zeros((1, 2, 2), dtype=np.float32)


def test_evaluate_uses_fast_path_and_masks_illegal() -> None:
    model = _FakeModel(n_actions=4)
    ev = FNetEvaluator(model, use_fast_path=True)
    result = ev.evaluate(_state(), legal_actions=[0, 2])
    assert model.fast_calls == 1
    assert result.value == pytest.approx(0.25)
    # Illegal actions carry zero probability; legal ones sum to 1.
    assert result.policy[1] == 0.0
    assert result.policy[3] == 0.0
    assert result.policy[[0, 2]].sum() == pytest.approx(1.0, abs=1e-5)


def test_evaluate_regular_path_when_fast_disabled() -> None:
    model = _FakeModel(n_actions=3)
    ev = FNetEvaluator(model, use_fast_path=False)
    ev.evaluate(_state(), legal_actions=[0, 1, 2])
    assert model.fast_calls == 0
    assert model.slow_calls == 1


def test_evaluate_regular_path_when_no_fast_attr() -> None:
    model = _FakeModelNoFast(n_actions=3)
    assert not hasattr(model, "forward_fast")
    ev = FNetEvaluator(model, use_fast_path=True)
    ev.evaluate(_state(), legal_actions=[1])
    assert model.slow_calls == 1


def test_evaluate_batch_and_empty() -> None:
    model = _FakeModel(n_actions=3)
    ev = FNetEvaluator(model)
    assert ev.evaluate_batch([], []) == []
    results = ev.evaluate_batch([_state(), _state()], [[0], [1, 2]])
    assert len(results) == 2
    assert results[0].policy[0] == pytest.approx(1.0, abs=1e-5)
    assert results[1].policy[0] == 0.0


def test_temperature_zero_skips_scaling() -> None:
    model = _FakeModel(n_actions=3)
    ev = FNetEvaluator(model, temperature=0.0)
    result = ev.evaluate(_state(), legal_actions=[0, 1, 2])
    assert result.policy.sum() == pytest.approx(1.0, abs=1e-5)
