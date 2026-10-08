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


#: The one CSV column that legitimately differs between two identical runs.
WALL_TIME_COLUMN = "wall_time_seconds"


def _csv_methods(path: str) -> set[str]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {row["method"] for row in csv.DictReader(handle)}


def _csv_rows_without_wall_time(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [
            {key: value for key, value in row.items() if key != WALL_TIME_COLUMN}
            for row in csv.DictReader(handle)
        ]


class TestGreedyControl:
    def test_records_the_greedy_metrics_rows_and_arm(self, tmp_path: Path) -> None:
        from src.research.run_manifest import load_run_manifest

        result = MCTSClassicalAMRArenaScenario(_config(tmp_path)).run()
        assert set(GREEDY_RATIO_KEYS + DIVERGENCE_KEYS) <= set(result.metrics)
        assert _csv_methods(result.artifacts["csv"]) == {"uniform", "dorfler", "mcts", "greedy"}
        manifest = load_run_manifest(Path(result.artifacts["run_json"]))
        assert "greedy" in {arm.name for arm in manifest.arms}

    def test_off_writes_the_legacy_rows_unperturbed_by_the_control(self, tmp_path: Path) -> None:
        """Off writes exactly the on-run's uniform/dorfler/mcts rows, minus greedy.

        That is the CSV half of "off reproduces the legacy artifact". The sidecar
        half does not hold and is not claimed: the config records
        ``include_greedy_control`` (so ``config_hash`` differs from a sidecar written
        before the field existed) and the divergence metrics are recorded either way.
        """
        on = MCTSClassicalAMRArenaScenario(_config(tmp_path / "on")).run()
        off = MCTSClassicalAMRArenaScenario(
            _config(tmp_path / "off", include_greedy_control=False)
        ).run()
        legacy_rows_of_on = [
            row
            for row in _csv_rows_without_wall_time(on.artifacts["csv"])
            if row["method"] != "greedy"
        ]
        assert _csv_methods(off.artifacts["csv"]) == {"uniform", "dorfler", "mcts"}
        assert _csv_rows_without_wall_time(off.artifacts["csv"]) == legacy_rows_of_on
        assert set(GREEDY_RATIO_KEYS).isdisjoint(off.metrics)
        assert set(DIVERGENCE_KEYS) <= set(off.metrics)

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
