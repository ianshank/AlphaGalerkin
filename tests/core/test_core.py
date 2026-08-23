"""Tests for the generic core Registry and Protocols."""

from __future__ import annotations

import pytest

from src.core.protocols import EvaluatorProtocol
from src.core.registry import Registry
from src.mcts.evaluator import RandomEvaluator


def test_core_registry_basic_flow() -> None:
    """Test generic Registry registration, alias resolution, and retrieval."""
    registry: Registry[object] = Registry("TestModule")

    @registry.register("primary_key", deprecated_names=["alias_key"])
    class PrimaryImpl:
        pass

    assert registry.is_registered("primary_key")
    assert registry.is_registered("alias_key")
    assert registry.get("primary_key") is PrimaryImpl
    assert registry.get("alias_key") is PrimaryImpl
    assert registry.get_or_raise("primary_key") is PrimaryImpl
    assert registry.list_items() == ["primary_key"]
    assert len(registry) == 1
    assert "primary_key" in registry

    with pytest.raises(KeyError, match="Unknown TestModule: 'missing'"):
        registry.get_or_raise("missing")

    registry.clear()
    assert len(registry) == 0


def test_evaluator_protocol_runtime_check() -> None:
    """Verify built-in Evaluators satisfy EvaluatorProtocol."""
    rand_eval = RandomEvaluator(n_actions=10)
    assert isinstance(rand_eval, EvaluatorProtocol)
