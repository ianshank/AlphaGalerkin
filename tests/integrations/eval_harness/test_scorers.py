"""Tests for the eval-harness scorer adapters (CPU; real harness types, no torch).

The harness types and the adapter's ``scorers`` module (which subclasses the
harness ``Scorer`` at import time) are resolved inside fixtures so this file
collects on a base install; see ``test_contract.py`` for why (R-13).
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.eval_harness_required


@pytest.fixture(scope="module")
def harness_types() -> ModuleType:
    from eval_harness.core import types

    return types


@pytest.fixture(scope="module")
def scorers() -> ModuleType:
    from src.integrations.eval_harness import scorers

    return scorers


@pytest.fixture(scope="module")
def ctx(harness_types: ModuleType) -> Any:
    return harness_types.RunContext(config=None)


def _output(
    harness_types: ModuleType, payload: dict[str, Any] | None, *, error: str | None = None
) -> Any:
    return harness_types.TargetOutput(output=payload, latency_ms=12.0, error=error)


def test_final_residual_scorer_passes_below_threshold(
    harness_types: ModuleType, scorers: ModuleType, ctx: Any
) -> None:
    scorer = scorers.FinalResidualScorer(target_residual=1e-2)
    item = harness_types.EvalItem(id="poisson/seed0", inputs={})
    result = scorer.score(
        item, _output(harness_types, {"final_residual": 5e-3, "rollouts_used": 32}), ctx
    )
    assert result.name == "final_residual"
    assert result.value == 5e-3
    assert result.passed is True
    assert result.metadata["rollouts_used"] == 32


def test_final_residual_scorer_fails_above_threshold(
    harness_types: ModuleType, scorers: ModuleType, ctx: Any
) -> None:
    scorer = scorers.FinalResidualScorer(target_residual=1e-3)
    item = harness_types.EvalItem(id="x", inputs={})
    result = scorer.score(item, _output(harness_types, {"final_residual": 0.5}), ctx)
    assert result.value == 0.5
    assert result.passed is False


def test_final_residual_scorer_handles_target_error(
    harness_types: ModuleType, scorers: ModuleType, ctx: Any
) -> None:
    scorer = scorers.FinalResidualScorer()
    item = harness_types.EvalItem(id="x", inputs={})
    result = scorer.score(item, _output(harness_types, None, error="boom"), ctx)
    assert result.value == scorers.FAILED_RESIDUAL_SENTINEL
    assert result.passed is False
    assert result.comment == "boom"


def test_policy_topk_scorer_hit(harness_types: ModuleType, scorers: ModuleType, ctx: Any) -> None:
    scorer = scorers.PolicyTopKScorer(k=3)
    item = harness_types.EvalItem(id="x", inputs={}, expected={"ranked_actions": [7, 3, 11, 1]})
    result = scorer.score(
        item, _output(harness_types, {"chosen_action": 11, "topk_actions": [11, 7]}), ctx
    )
    assert result.value == 1.0
    assert result.passed is True
    assert result.metadata["top1"] == 0.0  # 11 is in top-3 but not the oracle's top-1 (7)


def test_policy_topk_scorer_miss(harness_types: ModuleType, scorers: ModuleType, ctx: Any) -> None:
    scorer = scorers.PolicyTopKScorer(k=2)
    item = harness_types.EvalItem(id="x", inputs={}, expected={"ranked_actions": [7, 3, 11, 1]})
    result = scorer.score(item, _output(harness_types, {"chosen_action": 1}), ctx)
    assert result.value == 0.0
    assert result.passed is False


def test_policy_topk_scorer_top1(harness_types: ModuleType, scorers: ModuleType, ctx: Any) -> None:
    scorer = scorers.PolicyTopKScorer(k=3)
    item = harness_types.EvalItem(id="x", inputs={}, expected={"ranked_actions": [7, 3, 11]})
    result = scorer.score(item, _output(harness_types, {"chosen_action": 7}), ctx)
    assert result.metadata["top1"] == 1.0


def test_policy_topk_scorer_no_label_is_neither_pass_nor_fail(
    harness_types: ModuleType, scorers: ModuleType, ctx: Any
) -> None:
    scorer = scorers.PolicyTopKScorer()
    item = harness_types.EvalItem(id="x", inputs={}, expected=None)
    result = scorer.score(item, _output(harness_types, {"chosen_action": 1}), ctx)
    assert result.value == 0.0
    assert result.passed is None


def test_policy_topk_scorer_missing_choice_fails(
    harness_types: ModuleType, scorers: ModuleType, ctx: Any
) -> None:
    scorer = scorers.PolicyTopKScorer()
    item = harness_types.EvalItem(id="x", inputs={}, expected={"ranked_actions": [1, 2]})
    result = scorer.score(item, _output(harness_types, {}), ctx)
    assert result.passed is False


def test_final_residual_scorer_rejects_nonpositive_threshold(scorers: ModuleType) -> None:
    with pytest.raises(ValueError, match="target_residual must be > 0"):
        scorers.FinalResidualScorer(target_residual=0.0)


def test_policy_topk_scorer_rejects_nonpositive_k(scorers: ModuleType) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        scorers.PolicyTopKScorer(k=0)
