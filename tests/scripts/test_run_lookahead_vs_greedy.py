"""``scripts/run_lookahead_vs_greedy.py``: config loading and the exit-code contract.

Defect class, one sentence: **the exit code reports the verdict** -- a NO-GO, a
legitimate research outcome, exits non-zero and reads as a crash, or an abort /
error / non-proposal-grade run exits 0 and reads as a result.

Every exit code is asserted exactly, never as a set.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

import scripts.run_lookahead_vs_greedy as cli
from src.poc.config import ScenarioResult, ScenarioStatus
from src.poc.scenarios.lookahead_vs_greedy_config import SCENARIO_NAME
from src.research.lookahead_vs_greedy import ADEQUACY_ABORTED_METRIC, AdequacyAbortedError
from src.research.run_manifest import GitProvenance
from src.research.substrates.sweep import RateSeparation

pytest.importorskip("scipy", reason="scipy required when main() runs the scenario")

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAN = GitProvenance(sha="feed", branch="t", dirty=False)
DIRTY = GitProvenance(sha="feed", branch="t", dirty=True)


def _entry(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": SCENARIO_NAME,
        "description": "cli micro",
        "substrate": {
            "name": "tg",
            "kind": "tensor_grid",
            "initial_side": 4,
            "solve_cache_max_entries": 256,
        },
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
        "artifact_basename": "cli_micro",
    }
    entry.update(overrides)
    return entry


def _write_yaml(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "lvg.yaml"
    path.write_text(yaml.safe_dump({"scenarios": [_entry(tmp_path, **overrides)]}), "utf-8")
    return path


def _args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {"seed": None, "n_seeds": None, "output_dir": None}
    values.update(overrides)
    return argparse.Namespace(**values)


def _result(
    status: ScenarioStatus, metrics: dict[str, float] | None = None, **extra: Any
) -> ScenarioResult:
    now = datetime.now()
    return ScenarioResult(
        scenario_name=SCENARIO_NAME,
        config_hash="abc",
        status=status,
        passed=status is ScenarioStatus.PASSED,
        metrics=metrics or {},
        start_time=now,
        end_time=now,
        duration_seconds=0.0,
        **extra,
    )


class TestLoading:
    def test_list_and_bare_mappings(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path)
        assert cli.load_scenario_dict(path)["name"] == SCENARIO_NAME
        bare = tmp_path / "bare.yaml"
        bare.write_text(yaml.safe_dump(_entry(tmp_path)), "utf-8")
        assert cli.load_scenario_dict(bare)["max_steps"] == 2

    @pytest.mark.parametrize("document", [{"name": "transfer"}, ["not", "a", "mapping"]])
    def test_a_file_without_the_scenario_is_refused(self, tmp_path: Path, document: Any) -> None:
        path = tmp_path / "other.yaml"
        path.write_text(yaml.safe_dump(document), "utf-8")
        with pytest.raises(ValueError, match=SCENARIO_NAME + "|mapping"):
            cli.load_scenario_dict(path)

    def test_overrides_map_n_seeds_to_the_robustness_arm_and_skip_none(
        self, tmp_path: Path
    ) -> None:
        merged = cli.apply_overrides(_entry(tmp_path), _args(n_seeds=3, seed=7))
        assert (merged["robust_n_seeds"], merged["seed"]) == (3, 7)
        assert cli.apply_overrides(_entry(tmp_path), _args())["robust_n_seeds"] == 2

    @pytest.mark.parametrize(
        "basename", ["lookahead_vs_greedy_lshape", "lookahead_vs_greedy_zshape"]
    )
    def test_the_shipped_yamls_build(self, basename: str) -> None:
        config = cli.build_config(REPO_ROOT / "config" / "scenarios" / f"{basename}.yaml", _args())
        assert config.artifact_basename == basename
        assert config.require_adequacy_precondition is True

    def test_the_help_states_the_exit_codes(self) -> None:
        text = cli.build_parser().format_help()
        for flag in (
            "--config",
            "--output-dir",
            "--proposal-grade",
            "--log-level",
            "--seed",
            "--n-seeds",
        ):
            assert flag in text
        assert "3 adequacy abort" in text


class TestExitCodeFor:
    @pytest.mark.parametrize("status", [ScenarioStatus.PASSED, ScenarioStatus.FAILED])
    def test_go_and_no_go_both_exit_zero(self, status: ScenarioStatus) -> None:
        """The defect class: a NO-GO is a result, not a process failure."""
        assert cli.exit_code_for(_result(status), proposal_grade=False) == cli.EXIT_COMPLETED

    def test_an_abort_exits_three(self) -> None:
        aborted = _result(ScenarioStatus.SKIPPED, {ADEQUACY_ABORTED_METRIC: 1.0})
        assert cli.exit_code_for(aborted, proposal_grade=True) == cli.EXIT_ADEQUACY_ABORT

    def test_an_error_exits_one(self) -> None:
        errored = _result(ScenarioStatus.ERROR, error_message="boom")
        assert cli.exit_code_for(errored, proposal_grade=False) == cli.EXIT_ERROR

    def test_proposal_grade_without_a_csv_exits_two(self) -> None:
        assert (
            cli.exit_code_for(_result(ScenarioStatus.FAILED), proposal_grade=True)
            == cli.EXIT_NOT_PROPOSAL_GRADE
        )


class TestMain:
    def test_a_completed_micro_run_exits_zero_whatever_the_verdict(self, tmp_path: Path) -> None:
        code = cli.main(["--config", str(_write_yaml(tmp_path)), "--log-level", "WARNING"])
        assert code == cli.EXIT_COMPLETED
        assert {p.name for p in tmp_path.glob("cli_micro.*")} == {
            "cli_micro.csv",
            "cli_micro.png",
            "cli_micro.run.json",
        }

    def test_proposal_grade_on_a_clean_tree_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: CLEAN)
        code = cli.main(["--config", str(_write_yaml(tmp_path)), "--proposal-grade"])
        assert code == cli.EXIT_COMPLETED

    def test_proposal_grade_on_a_dirty_tree_exits_two_before_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: DIRTY)
        monkeypatch.setattr(
            cli, "LookaheadVsGreedyScenario", lambda *a, **k: pytest.fail("ran on a dirty tree")
        )
        code = cli.main(["--config", str(_write_yaml(tmp_path)), "--proposal-grade"])
        assert code == cli.EXIT_NOT_PROPOSAL_GRADE
        assert not list(tmp_path.glob("cli_micro.*"))

    def test_a_sidecar_that_fails_proposal_grade_after_the_run_exits_two(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Belt and braces: the written sidecar is re-checked from disk."""
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: CLEAN)
        monkeypatch.setattr(cli, "load_run_manifest", _dirty_manifest)
        code = cli.main(["--config", str(_write_yaml(tmp_path)), "--proposal-grade"])
        assert code == cli.EXIT_NOT_PROPOSAL_GRADE

    def test_an_adequacy_abort_exits_three(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.poc.scenarios.lookahead_vs_greedy as scenario_module

        def abort(config: Any) -> None:
            raise AdequacyAbortedError(
                ["synthetic"],
                RateSeparation(
                    adaptive_rate=-0.3,
                    uniform_rate=-0.67,
                    error_ratio_at_matched_dof=1.4,
                    matched_dof=3000.0,
                    n_adaptive_points=4,
                    n_uniform_points=3,
                ),
            )

        monkeypatch.setattr(scenario_module, "run_lookahead_vs_greedy", abort)
        monkeypatch.setattr(
            cli, "LookaheadVsGreedyScenario", scenario_module.LookaheadVsGreedyScenario
        )
        code = cli.main(["--config", str(_write_yaml(tmp_path))])
        assert code == cli.EXIT_ADEQUACY_ABORT

    @pytest.mark.parametrize(
        "argv",
        [
            ["--config", "/nonexistent/lookahead.yaml"],
            ["--config", "{yaml}", "--n-seeds", "1", "--seed", "3"],
        ],
    )
    def test_an_invalid_config_exits_one(self, tmp_path: Path, argv: list[str]) -> None:
        """Including C3 made unreachable: 2 robust wins needed of 1 seed."""
        path = _write_yaml(tmp_path, min_robust_wins=2)
        argv = [part.replace("{yaml}", str(path)) for part in argv]
        assert cli.main(argv) == cli.EXIT_ERROR

    def test_a_run_that_errors_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.poc.scenarios.lookahead_vs_greedy as scenario_module

        def explode(config: Any) -> None:
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(scenario_module, "run_lookahead_vs_greedy", explode)
        monkeypatch.setattr(
            cli, "LookaheadVsGreedyScenario", scenario_module.LookaheadVsGreedyScenario
        )
        assert cli.main(["--config", str(_write_yaml(tmp_path))]) == cli.EXIT_ERROR


def _dirty_manifest(path: Any) -> Any:
    from src.research.run_manifest import load_run_manifest

    manifest = load_run_manifest(path)
    return manifest.model_copy(update={"git": DIRTY})
