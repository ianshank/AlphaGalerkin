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


# ---------------------------------------------------------------------------
# _SOFTMAX_NORMALIZER_FLOOR binding
# ---------------------------------------------------------------------------


class TestSoftmaxNormalizerFloor:
    """Guards the extracted ``_SOFTMAX_NORMALIZER_FLOOR`` constant.

    The constant replaced an inline ``1e-8`` in ``_process_policy``'s softmax
    denominator, and its docstring claims parity with the same-named constant
    in ``src/integrations/lm_studio/evaluator.py``. Both claims are asserted
    here; the binding test would fail if the constant were defined but the
    call site left hardcoded.
    """

    def test_floor_is_the_softmax_denominator_addend(self) -> None:
        """Inflating the floor scales the policy down by sum/(sum + floor)."""
        import src.mcts.evaluator as evaluator_module

        model = _FakeModel(n_actions=4)
        ev = FNetEvaluator(model, use_fast_path=True)
        baseline = ev.evaluate(_state(), legal_actions=[0, 1, 2, 3]).policy
        assert baseline.sum() == pytest.approx(1.0, abs=1e-6)

        original = evaluator_module._SOFTMAX_NORMALIZER_FLOOR
        try:
            evaluator_module._SOFTMAX_NORMALIZER_FLOOR = 1.0
            inflated = ev.evaluate(_state(), legal_actions=[0, 1, 2, 3]).policy
        finally:
            evaluator_module._SOFTMAX_NORMALIZER_FLOOR = original

        assert inflated.sum() < baseline.sum()
        ratio = float(inflated.sum() / baseline.sum())
        # exp_logits.sum() is recoverable from the shrink ratio: s/(s+1).
        assert 0.0 < ratio < 1.0
        assert inflated == pytest.approx(baseline * ratio, abs=1e-6)

    def test_floor_is_positive_and_inert_at_shipped_value(self) -> None:
        """The shipped floor is small enough not to perturb a real softmax."""
        import src.mcts.evaluator as evaluator_module

        assert evaluator_module._SOFTMAX_NORMALIZER_FLOOR > 0.0
        assert evaluator_module._SOFTMAX_NORMALIZER_FLOOR < 1e-6

    def test_floor_matches_the_lm_studio_mirror(self) -> None:
        """Documented sync contract with the LM Studio evaluator mirror."""
        import src.integrations.lm_studio.evaluator as lm_evaluator
        import src.mcts.evaluator as evaluator_module

        assert evaluator_module._SOFTMAX_NORMALIZER_FLOOR == lm_evaluator._SOFTMAX_NORMALIZER_FLOOR


class TestProcessPolicyDegenerateMask:
    """``_process_policy`` must not emit NaN when no action is legal.

    Regression test: the softmax shift used a plain ``masked_logits.max()``.
    With an empty ``legal_actions`` every entry is ``-inf``, so the shift was
    ``-inf`` and ``(-inf) - (-inf)`` produced an all-NaN policy — which does not
    raise, and instead propagates into MCTS selection as silently corrupt
    priors. ``src/integrations/lm_studio/evaluator.py`` already guarded this
    while documenting itself as a mirror of this method, so the two differed in
    exactly the case that matters.
    """

    @staticmethod
    def _evaluator() -> FNetEvaluator:
        evaluator = FNetEvaluator.__new__(FNetEvaluator)
        evaluator.temperature = 1.0
        return evaluator

    def test_empty_legal_actions_yields_zeros_not_nan(self) -> None:
        """No legal actions degrades to an all-zero policy, never NaN."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        policy = self._evaluator()._process_policy(logits, [])

        assert not np.any(np.isnan(policy)), policy
        assert np.array_equal(policy, np.zeros(3, dtype=np.float32))

    def test_normal_masking_is_unchanged_by_the_shift_guard(self) -> None:
        """The finite-max shift leaves the ordinary path exactly as before."""
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        policy = self._evaluator()._process_policy(logits, [0, 2])

        assert policy[1] == 0.0
        assert policy.sum() == pytest.approx(1.0)
        assert policy[2] > policy[0]
