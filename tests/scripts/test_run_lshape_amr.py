"""Tests for the run_lshape_amr CLI helpers and end-to-end main()."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from scripts.run_lshape_amr import (
    apply_overrides,
    build_config,
    build_parser,
    load_scenario_dict,
    main,
)
from src.poc.scenarios.lshape_amr_compare_config import (
    SCENARIO_NAME,
    LShapeAMRCompareConfig,
)

pytest.importorskip("scipy", reason="scipy required for the masked FD solve")


def _scenario_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": SCENARIO_NAME,
        "device": "cpu",
        "initial_side": 4,
        "max_dof": 90,
        "max_steps": 4,
        "n_candidate_elements": 4,
        "n_simulations": 2,
        "add_noise": False,
    }
    entry.update(overrides)
    return entry


def _write_yaml(path: Path, data: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _empty_args() -> argparse.Namespace:
    return argparse.Namespace(
        seed=None,
        max_dof=None,
        n_simulations=None,
        output_dir=None,
        device=None,
    )


class TestLoadScenarioDict:
    def test_finds_in_scenarios_list(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(
            tmp_path / "c.yaml", {"scenarios": [{"name": "other"}, _scenario_entry()]}
        )
        found = load_scenario_dict(cfg_path)
        assert found["name"] == SCENARIO_NAME

    def test_finds_bare_mapping(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path / "c.yaml", _scenario_entry())
        found = load_scenario_dict(cfg_path)
        assert found["name"] == SCENARIO_NAME

    def test_missing_in_list_raises(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path / "c.yaml", {"scenarios": [{"name": "other"}]})
        with pytest.raises(ValueError, match="No 'lshape_amr_compare' scenario"):
            load_scenario_dict(cfg_path)

    def test_missing_bare_raises(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path / "c.yaml", {"name": "other"})
        with pytest.raises(ValueError, match="No 'lshape_amr_compare' scenario"):
            load_scenario_dict(cfg_path)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="did not parse to a mapping"):
            load_scenario_dict(cfg_path)


class TestApplyOverrides:
    def test_only_applies_non_none(self) -> None:
        data = _scenario_entry(max_dof=90, n_simulations=2)
        args = _empty_args()
        args.max_dof = 200
        merged = apply_overrides(data, args)
        assert merged["max_dof"] == 200  # overridden
        assert merged["n_simulations"] == 2  # untouched (arg is None)
        # original dict not mutated
        assert data["max_dof"] == 90

    def test_all_none_is_noop_copy(self) -> None:
        data = _scenario_entry()
        merged = apply_overrides(data, _empty_args())
        assert merged == data
        assert merged is not data


class TestBuildConfig:
    def test_returns_valid_config(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path / "c.yaml", _scenario_entry())
        args = _empty_args()
        args.seed = 7
        config = build_config(cfg_path, args)
        # By class name, not isinstance (dual-import safe — see build_config).
        assert type(config).__name__ == LShapeAMRCompareConfig.__name__
        assert config.seed == 7


class TestParser:
    def test_parser_defaults(self) -> None:
        args = build_parser().parse_args([])
        assert args.seed is None
        assert args.max_dof is None
        assert args.log_level == "INFO"


class TestMain:
    def test_end_to_end_small_run(self, tmp_path: Path) -> None:
        cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"scenarios": [_scenario_entry()]})
        out_dir = tmp_path / "artifacts"
        exit_code = main(
            [
                "--config",
                str(cfg_path),
                "--output-dir",
                str(out_dir),
                "--max-dof",
                "80",
                "--n-simulations",
                "2",
            ]
        )
        assert exit_code in (0, 1)
        # The CSV artifact must have been written to the overridden directory.
        assert (out_dir / "lshape_mcts_vs_dorfler.csv").exists()
