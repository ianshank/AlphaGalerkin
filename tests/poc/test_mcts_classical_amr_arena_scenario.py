"""CPU micro-run of the arena scenario on tensor_grid (no FEM extra)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.poc.config import ScenarioStatus, load_config_from_dict
from src.poc.scenarios.mcts_classical_amr_arena import MCTSClassicalAMRArenaScenario
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME,
    MCTSClassicalAMRArenaConfig,
)
from src.research.substrates.config import SubstrateConfig

pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")


def _config(tmp_path: Path, **overrides: object) -> MCTSClassicalAMRArenaConfig:
    params: dict[str, object] = {
        "name": SCENARIO_NAME,
        "device": "cpu",
        "substrate": SubstrateConfig(
            name="arena_tg",
            kind="tensor_grid",
            initial_side=4,
            solve_cache_max_entries=64,
        ),
        "operator_name": "poisson",
        "require_adequacy_precondition": False,
        "max_dof": 80,
        "max_steps": 2,
        "max_refinements_classical": 2,
        "n_simulations": 2,
        "n_seeds": 1,
        "top_k_actions": 4,
        "max_action_space": 64,
        "output_dir": str(tmp_path),
        "artifact_basename": "arena_micro",
    }
    params.update(overrides)
    return MCTSClassicalAMRArenaConfig(**params)  # type: ignore[arg-type]


class TestDispatch:
    def test_load_config_from_dict_returns_config(self) -> None:
        cfg = load_config_from_dict({"name": SCENARIO_NAME, "device": "cpu"})
        assert type(cfg).__name__ == MCTSClassicalAMRArenaConfig.__name__


class TestMicroRun:
    def test_records_ratios_and_artifacts(self, tmp_path: Path) -> None:
        scenario = MCTSClassicalAMRArenaScenario(_config(tmp_path))
        result = scenario.run()
        assert result.status in {ScenarioStatus.PASSED, ScenarioStatus.FAILED}
        assert "l2_error_ratio_at_matched_dof" in result.metrics
        assert np.isfinite(result.metrics["l2_error_ratio_at_matched_dof"])
        assert "l2_error_ratio_at_matched_solves" in result.metrics
        assert "csv" in result.artifacts
        assert Path(result.artifacts["csv"]).exists()
        sidecar = Path(result.artifacts["csv"]).with_suffix(".run.json")
        assert sidecar.exists()
        assert result.metrics["n_seeds"] == pytest.approx(1.0)

    def test_git_is_snapshotted_before_artifact_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.research.run_manifest import GitProvenance, load_run_manifest

        order: list[str] = []

        def _probe() -> GitProvenance:
            order.append("git")
            return GitProvenance(sha="snap", branch="test", dirty=False)

        monkeypatch.setattr(
            "src.research.run_manifest.collect_git_provenance",
            _probe,
        )
        from src.research.mcts_classical_amr_arena import export_csv as original_export

        def _export(*args: object, **kwargs: object) -> object:
            order.append("csv")
            return original_export(*args, **kwargs)

        monkeypatch.setattr(
            "src.research.mcts_classical_amr_arena.export_csv",
            _export,
        )
        scenario = MCTSClassicalAMRArenaScenario(_config(tmp_path))
        result = scenario.run()
        assert order[:2] == ["git", "csv"]
        sidecar = Path(result.artifacts["csv"]).with_suffix(".run.json")
        loaded = load_run_manifest(sidecar)
        assert loaded.git.dirty is False
        assert loaded.git.sha == "snap"


GREEDY_RATIO_KEYS = (
    "l2_error_ratio_mcts_over_greedy_at_matched_dof",
    "l2_error_ratio_greedy_over_dorfler_at_matched_dof",
)
DIVERGENCE_KEYS = ("decisions_diverging_from_greedy", "decisions_diverging_from_greedy_max")


def _csv_methods(path: str) -> set[str]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["method"] for row in csv.DictReader(handle)}


class TestGreedyControl:
    def test_records_the_greedy_metrics_rows_and_arm(self, tmp_path: Path) -> None:
        from src.research.run_manifest import load_run_manifest

        result = MCTSClassicalAMRArenaScenario(_config(tmp_path)).run()
        assert set(GREEDY_RATIO_KEYS + DIVERGENCE_KEYS) <= set(result.metrics)
        assert _csv_methods(result.artifacts["csv"]) == {"uniform", "dorfler", "mcts", "greedy"}
        manifest = load_run_manifest(Path(result.artifacts["run_json"]))
        assert "greedy" in {arm.name for arm in manifest.arms}

    def test_off_writes_exactly_the_legacy_csv(self, tmp_path: Path) -> None:
        result = MCTSClassicalAMRArenaScenario(
            _config(tmp_path, include_greedy_control=False)
        ).run()
        assert _csv_methods(result.artifacts["csv"]) == {"uniform", "dorfler", "mcts"}
        assert set(GREEDY_RATIO_KEYS).isdisjoint(result.metrics)
        assert set(DIVERGENCE_KEYS) <= set(result.metrics)

    def test_setup_and_recorded_events_carry_the_greedy_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.poc.logging import ScenarioLogger

        events: dict[str, dict[str, object]] = {}
        original = ScenarioLogger.info

        def spy(self: ScenarioLogger, event: str, **kwargs: object) -> None:
            events[event] = kwargs
            original(self, event, **kwargs)

        monkeypatch.setattr(ScenarioLogger, "info", spy)
        result = MCTSClassicalAMRArenaScenario(_config(tmp_path)).run()
        assert events["setup_complete"]["include_greedy_control"] is True
        recorded = events["arena_recorded"]
        assert recorded["decisions_diverging_from_greedy"] == pytest.approx(
            result.metrics["decisions_diverging_from_greedy"]
        )
        assert recorded["l2_error_ratio_mcts_over_greedy"] == pytest.approx(
            result.metrics["l2_error_ratio_mcts_over_greedy_at_matched_dof"]
        )
        assert recorded["l2_error_ratio_greedy_over_dorfler"] == pytest.approx(
            result.metrics["l2_error_ratio_greedy_over_dorfler_at_matched_dof"]
        )
