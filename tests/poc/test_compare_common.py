"""Tests for compare-scenario shared helpers (B2).

Hash pins were captured against default configs *before* the CompareScenarioBase
extraction; they fail if a config field is added or a default changes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.poc.scenarios._compare_common import (
    CompareScenarioBase,
    comparison_metrics,
    empty_cuda_cache,
    lock_scenario_name,
)
from src.poc.scenarios._compare_lock import _name_locked
from src.poc.scenarios._compare_lock import lock_scenario_name as lock_from_lock_module
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME as LSHAPE_NAME,
)
from src.poc.scenarios.lshape_amr_compare_config import (
    LShapeAMRCompareConfig,
)
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME as ARENA_NAME,
)
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    MCTSClassicalAMRArenaConfig,
)
from src.poc.scenarios.stochastic_galerkin_compare_config import (
    StochasticGalerkinCompareConfig,
)
from src.poc.scenarios.transfer_baseline_compare_config import (
    SCENARIO_NAME as TRANSFER_NAME,
)
from src.poc.scenarios.transfer_baseline_compare_config import (
    TransferBaselineCompareConfig,
)

# Captured 2026-09-10 on the pre-B2 defaults. A changed hash means a config
# field or default drifted; that is an artifact-identity break, not a rename.
_PINNED_DEFAULT_HASHES = {
    "lshape_amr_compare": "dd907de55b71a04d",
    "transfer_baseline_compare": "1da7914fb9ff1e76",
    "stochastic_galerkin_compare": "fe4fc45bc1c91590",
}


class TestLockScenarioName:
    def test_accepts_expected(self) -> None:
        assert lock_scenario_name("foo", "foo") == "foo"

    def test_rejects_mismatch(self) -> None:
        with pytest.raises(ValueError, match="name must be 'foo'"):
            lock_scenario_name("foo", "bar")

    def test_yaml_dispatch_wording(self) -> None:
        with pytest.raises(ValueError, match="YAML dispatch key"):
            lock_scenario_name("foo", "bar", yaml_dispatch_note=True)

    def test_name_locked_alias_is_the_same_function(self) -> None:
        assert _name_locked is lock_from_lock_module

    def test_reexport_is_identity(self) -> None:
        assert lock_scenario_name is lock_from_lock_module

    def test_stochastic_config_keeps_name_locked_symbol(self) -> None:
        assert hasattr(StochasticGalerkinCompareConfig, "_name_locked")


class TestComparisonMetrics:
    def test_metrics_method(self) -> None:
        class _HasMethod:
            def metrics(self) -> dict[str, float]:
                return {"a": 1.0}

        assert dict(comparison_metrics(_HasMethod())) == {"a": 1.0}

    def test_metrics_mapping(self) -> None:
        obj = SimpleNamespace(metrics={"b": 2.0})
        assert dict(comparison_metrics(obj)) == {"b": 2.0}

    def test_missing_metrics_raises(self) -> None:
        with pytest.raises(TypeError, match="no metrics"):
            comparison_metrics(SimpleNamespace())

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            comparison_metrics(SimpleNamespace(metrics=3))


class TestEmptyCudaCache:
    def test_cpu_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        called = {"n": 0}

        def _empty() -> None:
            called["n"] += 1

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "empty_cache", _empty)
        empty_cuda_cache()
        assert called["n"] == 0

    def test_cuda_flushes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        called = {"n": 0}

        def _empty() -> None:
            called["n"] += 1

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", _empty)
        empty_cuda_cache()
        assert called["n"] == 1


class TestCompareScenarioBaseTeardown:
    def test_teardown_delegates_to_empty_cuda_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called = {"n": 0}

        def _empty() -> None:
            called["n"] += 1

        monkeypatch.setattr("src.poc.scenarios._compare_common.empty_cuda_cache", _empty)

        class _Stub(CompareScenarioBase):
            def execute(self) -> MagicMock:  # pragma: no cover - unused
                raise AssertionError("execute must not run")

            def _setup_log_fields(self) -> dict[str, object]:
                return {}

        _Stub(name="stub", description="teardown probe").teardown()
        assert called["n"] == 1


class TestDefaultConfigHashesAreStable:
    """compute_hash() must stay byte-stable across the B2 extraction."""

    def test_lshape(self) -> None:
        cfg = LShapeAMRCompareConfig(name=LSHAPE_NAME)
        assert cfg.compute_hash() == _PINNED_DEFAULT_HASHES["lshape_amr_compare"]

    def test_transfer(self) -> None:
        cfg = TransferBaselineCompareConfig(name=TRANSFER_NAME)
        assert cfg.compute_hash() == _PINNED_DEFAULT_HASHES["transfer_baseline_compare"]

    def test_stochastic(self) -> None:
        cfg = StochasticGalerkinCompareConfig()
        assert cfg.compute_hash() == _PINNED_DEFAULT_HASHES["stochastic_galerkin_compare"]

    def test_arena_hash_stable_with_frozen_substrate_timestamp(self) -> None:
        """Arena hash is stable once SubstrateConfig.created_at is frozen.

        Arena nests SubstrateConfig (BaseModuleConfig), whose created_at
        default_factory is wall-clock. Freeze it so the lock-function
        extraction cannot hide behind timestamp noise.
        """
        from datetime import datetime, timezone

        frozen = datetime(2026, 1, 1, tzinfo=timezone.utc)
        cfg = MCTSClassicalAMRArenaConfig(name=ARENA_NAME)
        cfg.substrate.created_at = frozen
        assert cfg.compute_hash() == "ec589acdca20a4f9"
