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
)
from src.poc.scenarios.mcts_classical_amr_arena_config import SCENARIO_NAME

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


def test_build_config_and_main_micro(tmp_path: Path) -> None:
    path = tmp_path / "arena.yaml"
    entry = _entry(output_dir=str(tmp_path), artifact_basename="cli_micro")
    path.write_text(yaml.safe_dump({"scenarios": [entry]}), encoding="utf-8")
    config = build_config(path, _empty_args())
    assert config.name == SCENARIO_NAME
    code = main(["--config", str(path), "--output-dir", str(tmp_path)])
    assert code in (0, 1)
    assert (tmp_path / "cli_micro.csv").exists() or list(tmp_path.glob("*.csv"))
