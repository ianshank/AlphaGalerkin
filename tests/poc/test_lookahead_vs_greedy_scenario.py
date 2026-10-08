"""``lookahead_vs_greedy`` scenario: lifecycle, abort, provenance order (CPU, tensor_grid).

Defect classes:

* **S1 verdict/status drift** -- the scenario's PASSED/FAILED disagrees with
  the recorded ``verdict_go``.
* **S2 abort scored** -- an adequacy abort writes artifacts or a verdict.
* **S3 late snapshot** -- git is probed after an artifact exists (a clean tree
  then reads dirty), or a proposal-grade run computes on a dirty tree.

Per ``tests/poc/conftest.py``: every patch goes through the same module object
the scenario is instantiated from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import src.poc.scenarios.lookahead_vs_greedy as scenario_module
import src.research.lookahead_vs_greedy as harness_module
import src.research.lookahead_vs_greedy_artifacts as artifacts_module
from src.poc.config import ScenarioStatus
from src.poc.scenarios.lookahead_vs_greedy_config import LookaheadVsGreedyConfig
from src.research.lookahead_vs_greedy import (
    ADEQUACY_ABORTED_METRIC,
    AdequacyAbortedError,
)
from src.research.run_manifest import GitProvenance
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.sweep import RateSeparation

pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")


def _config(tmp_path: Path, **overrides: Any) -> LookaheadVsGreedyConfig:
    params: dict[str, Any] = {
        "substrate": SubstrateConfig(
            name="lvg_tg", kind="tensor_grid", initial_side=4, solve_cache_max_entries=256
        ),
        "operator_name": "poisson",
        "require_adequacy_precondition": False,
        "max_steps": 2,
        "classical_max_dof": 200,
        "max_refinements_classical": 20,
        "n_simulations": 4,
        "top_k_actions": 2,
        "max_action_space": 256,
        "robust_n_seeds": 2,
        "min_robust_wins": 1,
        "output_dir": str(tmp_path),
        "artifact_basename": "lvg_scenario",
    }
    params.update(overrides)
    return LookaheadVsGreedyConfig(**params)


def _separation() -> RateSeparation:
    return RateSeparation(
        adaptive_rate=-0.3,
        uniform_rate=-0.67,
        error_ratio_at_matched_dof=1.4,
        matched_dof=3000.0,
        n_adaptive_points=4,
        n_uniform_points=3,
    )


class TestRun:
    def test_status_is_the_verdict_and_the_artifacts_exist(self, tmp_path: Path) -> None:
        """S1: GO -> PASSED, NO-GO -> FAILED; both completed runs with artifacts."""
        result = scenario_module.LookaheadVsGreedyScenario(_config(tmp_path)).run()
        assert result.status in {ScenarioStatus.PASSED, ScenarioStatus.FAILED}
        assert result.passed is (result.metrics["verdict_go"] == 1.0)
        assert result.passed is (result.status is ScenarioStatus.PASSED)
        thresholds = _config(tmp_path).get_default_thresholds()
        assert set(result.threshold_results) == {t.name for t in thresholds}
        assert set(result.artifacts) == {"csv", "png", "run_json"}
        assert all(Path(path).is_file() for path in result.artifacts.values())

    def test_an_adequacy_abort_records_the_reading_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S2: no comparison ran, so no artifact and no verdict."""

        def abort(config: LookaheadVsGreedyConfig) -> None:
            raise AdequacyAbortedError(["adaptive_rate too shallow"], _separation())

        monkeypatch.setattr(scenario_module, "run_lookahead_vs_greedy", abort)
        result = scenario_module.LookaheadVsGreedyScenario(_config(tmp_path)).run()
        assert result.status is ScenarioStatus.SKIPPED
        assert not result.passed
        assert result.metrics[ADEQUACY_ABORTED_METRIC] == 1.0
        assert result.metrics["adequacy_adaptive_rate"] == -0.3
        assert "verdict_go" not in result.metrics
        assert result.artifacts == {}
        assert list(tmp_path.iterdir()) == []


class TestProvenanceOrder:
    def test_git_is_snapshotted_before_the_run_and_before_any_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """S3: snapshot -> compute -> write, in that order."""
        order: list[str] = []
        real_run = harness_module.run_lookahead_vs_greedy
        real_write = artifacts_module.write_artifacts

        def probe() -> GitProvenance:
            order.append("git")
            return GitProvenance(sha="snap", branch="t", dirty=False)

        def run(config: LookaheadVsGreedyConfig) -> Any:
            order.append("run")
            return real_run(config)

        def write(*args: Any, **kwargs: Any) -> Any:
            order.append("write")
            return real_write(*args, **kwargs)

        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", probe)
        monkeypatch.setattr(scenario_module, "run_lookahead_vs_greedy", run)
        monkeypatch.setattr(scenario_module, "write_artifacts", write)
        result = scenario_module.LookaheadVsGreedyScenario(
            _config(tmp_path), proposal_grade=True
        ).run()
        assert order == ["git", "run", "write"]
        assert result.status in {ScenarioStatus.PASSED, ScenarioStatus.FAILED}

    def test_a_proposal_grade_run_on_a_dirty_tree_computes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.research.run_manifest.collect_git_provenance",
            lambda: GitProvenance(sha="snap", branch="t", dirty=True),
        )
        monkeypatch.setattr(
            scenario_module,
            "run_lookahead_vs_greedy",
            lambda config: pytest.fail("computed on a dirty tree"),
        )
        result = scenario_module.LookaheadVsGreedyScenario(
            _config(tmp_path), proposal_grade=True
        ).run()
        assert result.status is ScenarioStatus.ERROR
        assert "dirty" in (result.error_message or "")


class TestLogging:
    def test_setup_and_recorded_events_carry_the_testbed_and_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.poc.logging import ScenarioLogger

        events: dict[str, dict[str, Any]] = {}
        original = ScenarioLogger.info

        def spy(self: ScenarioLogger, event: str, **kwargs: Any) -> None:
            events[event] = kwargs
            original(self, event, **kwargs)

        monkeypatch.setattr(ScenarioLogger, "info", spy)
        result = scenario_module.LookaheadVsGreedyScenario(_config(tmp_path)).run()
        assert events["setup_complete"]["testbed"] == "smoke_unit_square"
        assert events["setup_complete"]["proposal_grade"] is False
        recorded = events["lookahead_recorded"]
        assert recorded["verdict"] == ("GO" if result.passed else "NO-GO")
        assert recorded["primary_l2_ratio_vs_best_classical"] == pytest.approx(
            result.metrics["primary_l2_ratio_vs_best_classical"]
        )

    def test_the_recorded_event_refuses_a_foreign_comparison(self, tmp_path: Path) -> None:
        scenario = scenario_module.LookaheadVsGreedyScenario(_config(tmp_path))
        scenario.setup()
        with pytest.raises(TypeError, match="LookaheadVsGreedyResult"):
            scenario._log_metrics_recorded(object(), {})
