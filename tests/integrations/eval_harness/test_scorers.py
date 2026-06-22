"""Tests for the eval-harness scorer adapters (CPU; real harness types, no torch)."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("eval_harness")

from eval_harness.core.types import EvalItem, RunContext, TargetOutput  # noqa: E402

from src.integrations.eval_harness.scorers import (  # noqa: E402
    FAILED_RESIDUAL_SENTINEL,
    FinalResidualScorer,
    PolicyTopKScorer,
)

_CTX = RunContext(config=None)


def _output(payload: dict[str, Any] | None, *, error: str | None = None) -> TargetOutput:
    return TargetOutput(output=payload, latency_ms=12.0, error=error)


def test_final_residual_scorer_passes_below_threshold() -> None:
    scorer = FinalResidualScorer(target_residual=1e-2)
    item = EvalItem(id="poisson/seed0", inputs={})
    result = scorer.score(item, _output({"final_residual": 5e-3, "rollouts_used": 32}), _CTX)
    assert result.name == "final_residual"
    assert result.value == 5e-3
    assert result.passed is True
    assert result.metadata["rollouts_used"] == 32


def test_final_residual_scorer_fails_above_threshold() -> None:
    scorer = FinalResidualScorer(target_residual=1e-3)
    item = EvalItem(id="x", inputs={})
    result = scorer.score(item, _output({"final_residual": 0.5}), _CTX)
    assert result.value == 0.5
    assert result.passed is False


def test_final_residual_scorer_handles_target_error() -> None:
    scorer = FinalResidualScorer()
    item = EvalItem(id="x", inputs={})
    result = scorer.score(item, _output(None, error="boom"), _CTX)
    assert result.value == FAILED_RESIDUAL_SENTINEL
    assert result.passed is False
    assert result.comment == "boom"


def test_policy_topk_scorer_hit() -> None:
    scorer = PolicyTopKScorer(k=3)
    item = EvalItem(id="x", inputs={}, expected={"ranked_actions": [7, 3, 11, 1]})
    result = scorer.score(item, _output({"chosen_action": 11, "topk_actions": [11, 7]}), _CTX)
    assert result.value == 1.0
    assert result.passed is True
    assert result.metadata["top1"] == 0.0  # 11 is in top-3 but not the oracle's top-1 (7)


def test_policy_topk_scorer_miss() -> None:
    scorer = PolicyTopKScorer(k=2)
    item = EvalItem(id="x", inputs={}, expected={"ranked_actions": [7, 3, 11, 1]})
    result = scorer.score(item, _output({"chosen_action": 1}), _CTX)
    assert result.value == 0.0
    assert result.passed is False


def test_policy_topk_scorer_top1() -> None:
    scorer = PolicyTopKScorer(k=3)
    item = EvalItem(id="x", inputs={}, expected={"ranked_actions": [7, 3, 11]})
    result = scorer.score(item, _output({"chosen_action": 7}), _CTX)
    assert result.metadata["top1"] == 1.0


def test_policy_topk_scorer_no_label_is_neither_pass_nor_fail() -> None:
    scorer = PolicyTopKScorer()
    item = EvalItem(id="x", inputs={}, expected=None)
    result = scorer.score(item, _output({"chosen_action": 1}), _CTX)
    assert result.value == 0.0
    assert result.passed is None


def test_policy_topk_scorer_missing_choice_fails() -> None:
    scorer = PolicyTopKScorer()
    item = EvalItem(id="x", inputs={}, expected={"ranked_actions": [1, 2]})
    result = scorer.score(item, _output({}), _CTX)
    assert result.passed is False


def test_final_residual_scorer_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="target_residual must be > 0"):
        FinalResidualScorer(target_residual=0.0)


def test_policy_topk_scorer_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        PolicyTopKScorer(k=0)
