"""Tests for scripts/run_mcts_classical_amr_arena.py helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from scripts.run_mcts_classical_amr_arena import (
    apply_overrides,
    build_config,
    build_parser,
    load_scenario_dict,
    main,
    stable_metrics,
)
from src.poc.baselines.registry import ScenarioBaselineRegistry
from src.poc.scenarios.mcts_classical_amr_arena_config import SCENARIO_NAME
from src.research.amr_arena_types import (
    DIVERGENCE_MAX_METRIC,
    DIVERGENCE_METRIC,
    GREEDY_OVER_DORFLER_METRIC,
    MCTS_OVER_GREEDY_METRIC,
)

pytest.importorskip("scipy", reason="scipy required when main() runs the scenario")


def _entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": SCENARIO_NAME,
        "device": "cpu",
        "require_adequacy_precondition": False,
        "operator_name": "poisson",
        "substrate": {
            "name": "tg",
            "kind": "tensor_grid",
            "initial_side": 4,
        },
        "max_dof": 60,
        "max_steps": 1,
        "max_refinements_classical": 1,
        "n_simulations": 1,
        "n_seeds": 1,
        "top_k_actions": 2,
        "max_action_space": 64,
    }
    entry.update(overrides)
    return entry


def _empty_args() -> argparse.Namespace:
    return argparse.Namespace(
        seed=None,
        n_seeds=None,
        n_simulations=None,
        max_dof=None,
        max_steps=None,
        output_dir=None,
        device=None,
        require_adequacy=None,
    )


def test_load_scenario_dict_from_list(tmp_path: Path) -> None:
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump({"scenarios": [_entry()]}), encoding="utf-8")
    found = load_scenario_dict(path)
    assert found["name"] == SCENARIO_NAME


def test_load_scenario_dict_bare_mapping(tmp_path: Path) -> None:
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump(_entry()), encoding="utf-8")
    found = load_scenario_dict(path)
    assert found["name"] == SCENARIO_NAME


def test_missing_scenario_raises(tmp_path: Path) -> None:
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump({"name": "transfer"}), encoding="utf-8")
    with pytest.raises(ValueError, match=SCENARIO_NAME):
        load_scenario_dict(path)


def test_apply_overrides_skips_none() -> None:
    merged = apply_overrides(_entry(), _empty_args())
    assert merged["max_dof"] == 60


def test_apply_overrides_writes_non_none() -> None:
    args = _empty_args()
    args.max_dof = 99
    args.require_adequacy = False
    merged = apply_overrides(_entry(), args)
    assert merged["max_dof"] == 99
    assert merged["require_adequacy_precondition"] is False


def test_build_parser_help() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "--proposal-grade" in help_text
    assert "--config" in help_text


def test_adequacy_flags_are_tristate_and_default_none() -> None:
    """Neither flag may override YAML when omitted.

    ``store_true``/``store_false`` sharing ``dest`` keep a tri-state default of
    ``None`` so ``apply_overrides`` leaves ``require_adequacy_precondition`` to
    the scenario YAML. ``default=None`` on both flags makes that order-independent.
    """
    parser = build_parser()
    assert parser.parse_args([]).require_adequacy is None
    assert parser.parse_args(["--require-adequacy"]).require_adequacy is True
    assert parser.parse_args(["--skip-adequacy"]).require_adequacy is False


def test_build_config_and_main_micro(tmp_path: Path) -> None:
    path = tmp_path / "arena.yaml"
    entry = _entry(output_dir=str(tmp_path), artifact_basename="cli_micro")
    path.write_text(yaml.safe_dump({"scenarios": [entry]}), encoding="utf-8")
    config = build_config(path, _empty_args())
    assert config.name == SCENARIO_NAME
    code = main(["--config", str(path), "--output-dir", str(tmp_path)])
    assert code in (0, 1)
    assert (tmp_path / "cli_micro.csv").exists() or list(tmp_path.glob("*.csv"))


#: An arena metrics dict as the scenario reports it: policy ratios, the two
#: divergence diagnostics, and the wall-clock-derived keys.
_ARENA_METRICS: dict[str, float] = {
    "l2_error_ratio_at_matched_dof": 0.9532,
    MCTS_OVER_GREEDY_METRIC: 1.0,
    GREEDY_OVER_DORFLER_METRIC: 0.9532,
    DIVERGENCE_METRIC: 0.0,
    DIVERGENCE_MAX_METRIC: 0.0,
    "matched_wall_time_seconds": 0.27,
    "error_per_dof_ratio_mcts_over_dorfler": 32.1,
}


def test_stable_metrics_keeps_policy_ratios_and_drops_diagnostics_and_wall_clock() -> None:
    assert stable_metrics(_ARENA_METRICS) == {
        "l2_error_ratio_at_matched_dof": 0.9532,
        MCTS_OVER_GREEDY_METRIC: 1.0,
        GREEDY_OVER_DORFLER_METRIC: 0.9532,
    }


def test_a_search_that_starts_diverging_does_not_fail_the_baseline_gate() -> None:
    """Divergence going 0 -> n is the search starting to matter, not a regression.

    Driven through the real registry: the registry records every metric it is
    given as lower-better unless told otherwise, so a divergence count that
    reached the baseline would turn the first configuration where MCTS departs
    from greedy into a gate failure. Killed mutation: dropping the diagnostic
    exclusion from ``stable_metrics`` turns this test red.
    """
    registry = ScenarioBaselineRegistry.from_observed(
        {SCENARIO_NAME: stable_metrics(_ARENA_METRICS)},
        higher_better_metrics=("mcts_win_fraction",),
    )
    later = {**_ARENA_METRICS, DIVERGENCE_METRIC: 3.0, DIVERGENCE_MAX_METRIC: 5.0}

    report = registry.compare({SCENARIO_NAME: later})

    assert report.is_clean, report.summary()
