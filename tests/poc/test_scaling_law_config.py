"""Validation tests for ScalingLawConfig."""

from __future__ import annotations

import pytest

from src.poc.scenarios.scaling_law_config import (
    SCALING_SCENARIO_NAME,
    ScalingLawConfig,
)


def test_defaults_are_valid() -> None:
    cfg = ScalingLawConfig()
    assert cfg.name == SCALING_SCENARIO_NAME
    assert cfg.arms == ["random"]
    assert cfg.simulation_budgets == [8, 16, 32, 64]
    assert cfg.requires_gpu is True
    assert cfg.primary_arm == "random"


def test_name_is_locked() -> None:
    with pytest.raises(ValueError, match="name must be exactly"):
        ScalingLawConfig(name="not_scaling_law")


def test_arms_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="arms must be non-empty"):
        ScalingLawConfig(arms=[])


def test_arms_dedup_preserves_order() -> None:
    cfg = ScalingLawConfig(arms=["llm", "random", "llm"])
    assert cfg.arms == ["llm", "random"]
    assert cfg.primary_arm == "llm"


def test_budgets_need_two_distinct_values() -> None:
    with pytest.raises(ValueError, match=">= 2 distinct"):
        ScalingLawConfig(simulation_budgets=[16, 16])


def test_budgets_reject_non_positive() -> None:
    with pytest.raises(ValueError, match="must all be >= 1"):
        ScalingLawConfig(simulation_budgets=[0, 8])


def test_budgets_sorted_and_deduped() -> None:
    cfg = ScalingLawConfig(simulation_budgets=[64, 8, 8, 16])
    assert cfg.simulation_budgets == [8, 16, 64]


def test_seeds_non_empty_when_provided() -> None:
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        ScalingLawConfig(seeds=[])


def test_resolved_seeds_derived() -> None:
    cfg = ScalingLawConfig(seed=10, n_seeds=3)
    assert cfg.resolved_seeds() == [10, 10 + 1009, 10 + 2018]


def test_resolved_seeds_explicit_deduped() -> None:
    cfg = ScalingLawConfig(seeds=[5, 5, 7])
    assert cfg.resolved_seeds() == [5, 7]


def test_max_rollouts_for_budget_scales() -> None:
    cfg = ScalingLawConfig(max_basis_functions=10, rollout_headroom=3)
    assert cfg.max_rollouts_for_budget(8) == 8 * 10 * 3


def test_default_thresholds_match_fields() -> None:
    cfg = ScalingLawConfig(min_residual_decay=0.1, min_fit_r2=0.6)
    thresholds = {t.name: t for t in cfg.get_default_thresholds()}
    assert thresholds["residual_scaling_exponent"].operator == "<="
    assert thresholds["residual_scaling_exponent"].value == pytest.approx(-0.1)
    assert thresholds["residual_fit_r2"].operator == ">="
    assert thresholds["residual_fit_r2"].value == pytest.approx(0.6)


def test_explicit_valid_name_accepted() -> None:
    cfg = ScalingLawConfig(name=SCALING_SCENARIO_NAME)
    assert cfg.name == SCALING_SCENARIO_NAME
