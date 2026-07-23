"""Tests for NoyronBasisConfig (Leap 71 v2.2)."""

from __future__ import annotations

import pytest

from src.poc.config import MetricThreshold
from src.poc.scenarios.noyron_basis_config import (
    SCENARIO_NAME,
    NoyronBasisConfig,
)


def _config(**overrides: object) -> NoyronBasisConfig:
    params: dict[str, object] = {"name": SCENARIO_NAME, "description": "test"}
    params.update(overrides)
    return NoyronBasisConfig(**params)  # type: ignore[arg-type]


class TestValidation:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.name == SCENARIO_NAME
        assert cfg.operator_name == "helical_heat"
        assert cfg.arms == ["random"]
        assert cfg.manufactured is True

    def test_name_locked(self) -> None:
        with pytest.raises(ValueError, match="name must be"):
            _config(name="something_else")

    def test_arms_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _config(arms=[])

    def test_arms_dedupe_preserves_order(self) -> None:
        cfg = _config(arms=["random", "llm", "random"])
        assert cfg.arms == ["random", "llm"]

    def test_unknown_arm_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown arm"):
            _config(arms=["random", "bogus"])

    def test_unknown_operator_rejected(self) -> None:
        # operator_name is a Literal — Pydantic rejects unknown values.
        with pytest.raises(ValueError, match="Input should be"):
            _config(operator_name="not_helical")

    @pytest.mark.parametrize("op", ["helical_heat", "helical_stokes", "helical_magnetostatics"])
    def test_all_helical_operators_accepted(self, op: str) -> None:
        assert _config(operator_name=op).operator_name == op

    def test_max_basis_cannot_exceed_candidates(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            _config(max_basis_functions=30, n_candidate_bases=10)

    def test_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            _config(n_seeds=0)
        with pytest.raises(ValueError):
            _config(manufactured_wavenumber=0)
        with pytest.raises(ValueError):
            _config(target_residual=0.0)


class TestDerived:
    def test_primary_arm(self) -> None:
        assert _config(arms=["llm", "random"]).primary_arm == "llm"

    def test_resolved_seeds_deterministic_and_decorrelated(self) -> None:
        cfg = _config(seed=42, n_seeds=3)
        seeds = cfg.resolved_seeds()
        assert len(seeds) == 3
        assert seeds[0] == 42
        assert len(set(seeds)) == 3  # decorrelated

    def test_max_rollouts_scales_with_budget(self) -> None:
        cfg = _config(n_simulations=8, max_basis_functions=6, rollout_headroom=2)
        assert cfg.max_rollouts_for_cell() == 8 * 6 * 2


class TestThresholds:
    def test_default_thresholds_shape(self) -> None:
        cfg = _config()
        thresholds = cfg.get_default_thresholds()
        by_name = {t.name: t for t in thresholds}
        assert set(by_name) == {"error_reduction_pct", "final_residual"}
        assert isinstance(by_name["error_reduction_pct"], MetricThreshold)

    def test_default_thresholds_operators_and_values(self) -> None:
        cfg = _config(min_error_reduction_pct=0.0, max_final_residual=1.0)
        by_name = {t.name: t for t in cfg.get_default_thresholds()}
        assert by_name["error_reduction_pct"].operator == ">="
        assert by_name["error_reduction_pct"].value == 0.0
        assert by_name["final_residual"].operator == "<="
        assert by_name["final_residual"].value == 1.0

    def test_thresholds_track_config(self) -> None:
        """AQA: spec ↔ config agreement — thresholds reflect the fields."""
        cfg = _config(min_error_reduction_pct=5.0, max_final_residual=2.0)
        by_name = {t.name: t.value for t in cfg.get_default_thresholds()}
        assert by_name["error_reduction_pct"] == 5.0
        assert by_name["final_residual"] == 2.0
