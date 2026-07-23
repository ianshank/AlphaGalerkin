"""Pydantic validation tests for `LLMPriorAblationConfig`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.poc.config import MetricThreshold
from src.poc.scenarios.llm_prior_config import (
    SCENARIO_NAME,
    LLMPriorAblationConfig,
)


def test_defaults_are_valid() -> None:
    config = LLMPriorAblationConfig()
    assert config.name == SCENARIO_NAME
    assert config.id_pde == "poisson"
    assert config.ood_pde == "burgers"
    assert config.run_random_arm is True
    assert config.run_llm_arm is True
    assert config.run_trained_arm is True


def test_pde_enum_membership_enforced() -> None:
    with pytest.raises(ValidationError):
        LLMPriorAblationConfig(id_pde="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        LLMPriorAblationConfig(ood_pde="bogus")  # type: ignore[arg-type]


def test_default_thresholds_match_fields() -> None:
    config = LLMPriorAblationConfig(
        id_rollout_reduction_pct_min=33.0,
        ood_llm_residual_max=5e-3,
        ood_trained_residual_min=2e-1,
        llm_call_p95_latency_ms_max=4000.0,
    )
    thresholds = config.get_default_thresholds()
    by_name = {t.name: t for t in thresholds}
    assert isinstance(by_name["id_rollout_reduction_pct"], MetricThreshold)
    assert by_name["id_rollout_reduction_pct"].value == 33.0
    assert by_name["id_rollout_reduction_pct"].operator == ">="
    assert by_name["ood_llm_residual"].value == 5e-3
    assert by_name["ood_llm_residual"].operator == "<="
    assert by_name["ood_trained_residual"].value == 2e-1
    assert by_name["ood_trained_residual"].operator == ">"
    assert by_name["llm_call_p95_latency_ms"].value == 4000.0
    assert by_name["llm_call_p95_latency_ms"].operator == "<="


def test_seeds_derived_from_seed_when_none() -> None:
    config = LLMPriorAblationConfig(seed=10, n_seeds=4)
    seeds = config.resolved_seeds()
    assert len(seeds) == 4
    assert seeds[0] == 10
    assert all(isinstance(s, int) for s in seeds)
    # Prime stride should give strictly distinct seeds.
    assert len(set(seeds)) == 4


def test_explicit_seeds_passed_through() -> None:
    config = LLMPriorAblationConfig(seeds=[1, 2, 3])
    assert config.resolved_seeds() == [1, 2, 3]


def test_explicit_seeds_deduplicated_in_order() -> None:
    """Per docstring: duplicates are removed in first-seen order."""
    config = LLMPriorAblationConfig(seeds=[3, 1, 3, 2, 1])
    assert config.resolved_seeds() == [3, 1, 2]


def test_seeds_must_be_non_empty_when_provided() -> None:
    with pytest.raises(ValidationError):
        LLMPriorAblationConfig(seeds=[])


def test_name_locked_to_canonical() -> None:
    with pytest.raises(ValidationError):
        LLMPriorAblationConfig(name="something_else")


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMPriorAblationConfig(bogus_field=True)  # type: ignore[call-arg]


def test_at_least_one_arm_required() -> None:
    with pytest.raises(ValidationError):
        LLMPriorAblationConfig(
            run_random_arm=False,
            run_trained_arm=False,
            run_llm_arm=False,
        )


def test_load_config_dispatches_to_correct_class() -> None:
    # Local imports avoid an identity mismatch when sibling test files
    # have wiped ``src.poc.scenarios.*`` from ``sys.modules``: both imports
    # then resolve to the same freshly-loaded class.
    from src.poc.config import load_config_from_dict
    from src.poc.scenarios.llm_prior_config import (
        LLMPriorAblationConfig as _LLMPriorAblationConfig,
    )

    config = load_config_from_dict(
        {
            "name": SCENARIO_NAME,
            "description": "from dict",
        }
    )
    assert isinstance(config, _LLMPriorAblationConfig)
