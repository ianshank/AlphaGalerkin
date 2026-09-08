"""Tests for MCTSClassicalAMRArenaConfig (name-lock, scored flags, AQA)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.poc.config import MetricThreshold, load_config_from_dict
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    HEADLINE_EVALUATOR_NAME,
    SCENARIO_NAME,
    MCTSClassicalAMRArenaConfig,
)
from src.research.substrates.config import SubstrateConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides: object) -> MCTSClassicalAMRArenaConfig:
    params: dict[str, object] = {"name": SCENARIO_NAME}
    params.update(overrides)
    return MCTSClassicalAMRArenaConfig(**params)  # type: ignore[arg-type]


class TestValidation:
    def test_defaults(self) -> None:
        cfg = _config()
        assert cfg.name == SCENARIO_NAME
        assert cfg.substrate.kind == "skfem_tri"
        assert cfg.operator_name == "lshape_poisson"
        assert cfg.marking_fraction == 0.5
        assert cfg.add_noise is False
        assert cfg.temperature == 0.0
        assert cfg.search_mode == "single_agent"
        assert cfg.evaluator_name == HEADLINE_EVALUATOR_NAME
        assert cfg.require_adequacy_precondition is True

    def test_name_locked(self) -> None:
        with pytest.raises(ValidationError, match="name must be"):
            _config(name="wrong")

    def test_add_noise_rejected(self) -> None:
        with pytest.raises(ValidationError, match="add_noise"):
            _config(add_noise=True)

    def test_nonzero_temperature_rejected(self) -> None:
        with pytest.raises(ValidationError, match="temperature"):
            _config(temperature=1.0)

    def test_intermediate_rewards_rejected(self) -> None:
        with pytest.raises(ValidationError, match="use_intermediate_rewards"):
            _config(use_intermediate_rewards=True)

    def test_artifact_basename_rejects_extension(self) -> None:
        with pytest.raises(ValidationError, match="extension"):
            _config(artifact_basename="foo.csv")

    def test_adequacy_requires_skfem_lshape(self) -> None:
        with pytest.raises(ValidationError, match="adequacy precondition"):
            _config(
                substrate=SubstrateConfig(name="tg", kind="tensor_grid", initial_side=4),
                operator_name="poisson",
                require_adequacy_precondition=True,
            )

    def test_tensor_grid_allowed_without_adequacy(self) -> None:
        cfg = _config(
            substrate=SubstrateConfig(name="tg", kind="tensor_grid", initial_side=4),
            operator_name="poisson",
            require_adequacy_precondition=False,
        )
        assert cfg.substrate.kind == "tensor_grid"

    def test_resolved_seeds_use_stride(self) -> None:
        cfg = _config(seed=10, n_seeds=3, seed_stride=1009)
        assert cfg.resolved_seeds() == [10, 1019, 2028]


class TestThresholdsAQA:
    def test_single_matched_dof_gate(self) -> None:
        thresholds = _config().get_default_thresholds()
        assert len(thresholds) == 1
        gate = thresholds[0]
        assert isinstance(gate, MetricThreshold)
        assert gate.name == "l2_error_ratio_at_matched_dof"
        assert gate.operator == "<"
        assert gate.value == 1.0

    def test_gate_tracks_config_field(self) -> None:
        cfg = _config(max_l2_ratio_at_matched_dof=0.9)
        assert cfg.get_default_thresholds()[0].value == pytest.approx(0.9)

    def test_spec_table_names_the_same_gate(self) -> None:
        spec = (REPO_ROOT / "specs" / "mcts_classical_amr_arena.spec.md").read_text(
            encoding="utf-8"
        )
        assert "`l2_error_ratio_at_matched_dof`" in spec
        assert "| `<` |" in spec
        assert "ResidualPriorErrorValueEvaluator" in spec


class TestDispatch:
    def test_load_config_from_dict_by_name(self) -> None:
        cfg = load_config_from_dict({"name": SCENARIO_NAME, "device": "cpu"})
        assert type(cfg).__name__ == MCTSClassicalAMRArenaConfig.__name__

    def test_shipped_yamls_validate(self) -> None:
        for basename in (
            "mcts_classical_amr_arena.yaml",
            "mcts_classical_amr_arena_ci.yaml",
        ):
            path = REPO_ROOT / "config" / "scenarios" / basename
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            entry = raw["scenarios"][0]
            cfg = load_config_from_dict(entry)
            assert type(cfg).__name__ == MCTSClassicalAMRArenaConfig.__name__

    def test_ci_yaml_skips_adequacy_and_is_tiny(self) -> None:
        path = REPO_ROOT / "config" / "scenarios" / "mcts_classical_amr_arena_ci.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = raw["scenarios"][0]
        assert entry["require_adequacy_precondition"] is False
        assert entry["n_seeds"] == 1
        assert entry["max_steps"] <= 2
        assert entry["n_simulations"] <= 2
        assert entry["substrate"]["kind"] == "skfem_tri"
        assert entry["evaluator_name"] == HEADLINE_EVALUATOR_NAME
        assert entry["add_noise"] is False
        assert entry["search_mode"] == "single_agent"

    def test_headline_yaml_pins_adequacy_and_theta(self) -> None:
        path = REPO_ROOT / "config" / "scenarios" / "mcts_classical_amr_arena.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = raw["scenarios"][0]
        assert entry["require_adequacy_precondition"] is True
        assert entry["marking_fraction"] == 0.5
        assert entry["substrate"]["kind"] == "skfem_tri"
        assert entry["operator_name"] == "lshape_poisson"
        assert entry["max_action_space"] >= 16384
