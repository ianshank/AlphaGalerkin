"""Gate 1 harness: arms, decision rule, adequacy abort, artifacts (CPU, tensor_grid).

The pre-registration is ``specs/lookahead_vs_greedy.spec.md``. Defect classes,
one sentence each:

* **H1 noise in the game** -- root noise reaches the arena game or the primary
  arm, so the "deterministic" run is not, or the robustness arm runs without it.
* **H2 depth/divergence drift** -- the recorded tree depth is not read from the
  search root before the action, or the divergence count disagrees with the
  per-step record.
* **H3 comparison before adequacy** -- an arm runs before the adequacy
  precondition, or a failed gate is scored instead of aborted.
* **H4 unfair arms** -- two game-driven arms share a solve cache, or a
  classical arm that stops short silently shrinks the matched DOF.
* **H5 cycle** -- the harness gains a runtime ``src.poc`` import, which would
  turn the scenario's module-level import into an import cycle.
* **H6 dishonest provenance** -- the sidecar's hash or git state is not the
  pre-run snapshot, or its config cannot reproduce its hash.

Mutation-kill record (Gate 1 surface, 2026-10-08): see the end of this module.
"""

from __future__ import annotations

import ast
import builtins
import csv
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import src.research.lookahead_vs_greedy as harness
import src.research.lookahead_vs_greedy_artifacts as artifacts
from src.mcts.node import MCTSNode
from src.mcts.search import MCTS
from src.pde.games.substrate_refinement import SubstrateEpisodeState
from src.poc.config import load_config_from_dict
from src.poc.scenarios.lookahead_vs_greedy_config import LookaheadVsGreedyConfig
from src.research.amr_arena_types import CSV_COLUMNS, DETERMINISTIC_ARM_SEED
from src.research.greedy_control import GreedyDivergence, greedy_action
from src.research.lookahead_vs_greedy import (
    PRIMARY_ARM,
    ROBUST_ARM,
    AdequacyAbortedError,
    LookaheadVsGreedyResult,
    SearchLog,
    adequacy_metrics,
    check_adequacy,
    run_lookahead_vs_greedy,
    run_search_arm,
    search_decision_rule,
)
from src.research.lookahead_vs_greedy_artifacts import (
    HARNESS_NAME,
    SidecarHashMismatchError,
    build_run_manifest,
    export_csv,
    export_plot,
    write_artifacts,
)
from src.research.lookahead_vs_greedy_metrics import (
    GREEDY_LABEL,
    PRIMARY_LABEL,
    ROBUST_LABEL,
    UNIFORM_LABEL,
    ClassicalBudgetError,
    break_even,
    dorfler_label,
)
from src.research.lookahead_vs_greedy_verdict import evaluate_verdict
from src.research.run_manifest import (
    GitProvenance,
    ProposalGradeError,
    RunRecorder,
    load_run_manifest,
)
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.sweep import RateSeparation

pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_FILES = (
    REPO_ROOT / "src" / "research" / "lookahead_vs_greedy.py",
    REPO_ROOT / "src" / "research" / "lookahead_vs_greedy_metrics.py",
    REPO_ROOT / "src" / "research" / "lookahead_vs_greedy_verdict.py",
    REPO_ROOT / "src" / "research" / "lookahead_vs_greedy_artifacts.py",
)
CLEAN = GitProvenance(sha="feed", branch="test", dirty=False)
DIRTY = GitProvenance(sha="feed", branch="test", dirty=True)


def micro_config(output_dir: Path, **overrides: Any) -> LookaheadVsGreedyConfig:
    """A tensor-grid mechanism config: small budgets, adequacy off (no FEM extra)."""
    params: dict[str, Any] = {
        "substrate": SubstrateConfig(
            name="lvg_tg", kind="tensor_grid", initial_side=4, solve_cache_max_entries=256
        ),
        "operator_name": "poisson",
        "require_adequacy_precondition": False,
        "max_steps": 3,
        "classical_max_dof": 200,
        "max_refinements_classical": 20,
        "n_simulations": 5,
        "top_k_actions": 2,
        "max_action_space": 256,
        "robust_n_seeds": 2,
        "min_robust_wins": 1,
        "output_dir": str(output_dir),
        "artifact_basename": "lvg_micro",
    }
    params.update(overrides)
    return LookaheadVsGreedyConfig(**params)


@pytest.fixture(scope="module")
def micro(tmp_path_factory: pytest.TempPathFactory) -> tuple[LookaheadVsGreedyConfig, Any]:
    config = micro_config(tmp_path_factory.mktemp("lvg"))
    return config, run_lookahead_vs_greedy(config)


def _rate_separation(**overrides: float) -> RateSeparation:
    values: dict[str, Any] = {
        "adaptive_rate": -1.3,
        "uniform_rate": -0.67,
        "error_ratio_at_matched_dof": 0.1,
        "matched_dof": 3000.0,
        "n_adaptive_points": 4,
        "n_uniform_points": 3,
    }
    values.update(overrides)
    return RateSeparation(**values)


class TestImportIsolation:
    def test_the_harness_loads_no_poc_module(self) -> None:
        """H5: the scenario imports the harness at module level; this keeps that acyclic."""
        code = (
            "import sys\n"
            "import src.research.lookahead_vs_greedy\n"
            "import src.research.lookahead_vs_greedy_artifacts\n"
            "print(sorted(m for m in sys.modules if m.startswith('src.poc')))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=120,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.strip().splitlines()[-1] == "[]"

    def test_the_harness_names_only_the_headline_evaluator(self) -> None:
        """The forbidden arena evaluators never appear; the search mode is the adapter's."""
        names: set[str] = set()
        for path in HARNESS_FILES:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert names.isdisjoint({"EncodedValueEvaluator", "RandomEvaluator"})
        source = HARNESS_FILES[0].read_text(encoding="utf-8")
        assert "ResidualPriorErrorValueEvaluator(" in source
        assert "search_mode=adapter.search_mode" in source
        assert "use_intermediate_rewards=False" in source


class _StubSearch:
    """Stands in for ``MCTS`` in the decision rule: fixed choice, a crafted root."""

    def __init__(self, action: int, root: MCTSNode | None) -> None:
        self.action = action
        self.root = root
        self.calls: list[dict[str, Any]] = []
        self.advanced: list[int] = []

    def get_action(self, game: Any, **kwargs: Any) -> int:
        self.calls.append(kwargs)
        return self.action

    def advance(self, action: int) -> None:
        """Tree reuse, as the engine does it: the root moves to the chosen child.

        So a depth read *after* ``advance`` differs from the one before it --
        here it drops to 0, the chosen action not being a child of the crafted root.
        """
        self.advanced.append(action)
        self.root = None if self.root is None else self.root.children.get(action)


def _depth_two_root() -> MCTSNode:
    root = MCTSNode(visit_count=3)
    child = MCTSNode(parent=root, action=0, visit_count=2)
    grandchild = MCTSNode(parent=child, action=1, visit_count=1)
    root.children[0] = child
    child.children[1] = grandchild
    grandchild.expand({5: 1.0})  # an unvisited frontier: not a ply
    return root


def _state(indicators: list[float], step: int = 3, dof: int = 40) -> SubstrateEpisodeState:
    return SubstrateEpisodeState(
        values=np.zeros(1, dtype=np.float32),
        indicators=np.array(indicators, dtype=np.float32),
        error_estimate=0.1,
        dof=dof,
        step=step,
        budget_remaining=5.0,
        history=[],
    )


class TestDecisionRule:
    @pytest.mark.parametrize("add_noise", [False, True])
    def test_records_depth_greedy_and_divergence_then_advances(self, add_noise: bool) -> None:
        """H1/H2: flags reach get_action; depth comes from the root before the action."""
        stub = _StubSearch(action=2, root=_depth_two_root())
        log = SearchLog(divergence=GreedyDivergence(seed=7))
        choose = search_decision_rule(
            stub,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            add_noise=add_noise,
            temperature=0.0,
            log=log,
        )
        action = choose(_state([0.1, 0.9, 0.3]), [0, 1, 2])
        assert action == 2
        assert stub.calls == [{"temperature": 0.0, "add_noise": add_noise}]
        assert stub.advanced == [2]
        (step,) = log.steps
        assert (step.step, step.action, step.greedy_action) == (3, 2, 1)
        assert (step.tree_depth, step.n_legal, step.dof) == (2, 3, 40)
        assert step.diverged and log.divergence.count == 1

    def test_agreeing_with_greedy_is_not_counted_and_a_missing_root_is_depth_zero(self) -> None:
        stub = _StubSearch(action=1, root=None)
        log = SearchLog(divergence=GreedyDivergence(seed=7))
        choose = search_decision_rule(
            stub,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            add_noise=False,
            temperature=0.0,
            log=log,
        )
        choose(_state([0.1, 0.9, 0.3]), [0, 1, 2])
        assert log.divergence.count == 0
        assert log.steps[0].tree_depth == 0
        assert not log.steps[0].diverged


class _RecordingMCTS(MCTS):
    """The real engine, recording every ``get_action`` call's flags."""

    seen: list[tuple[float, bool]] = []

    def get_action(self, game: Any, temperature: float = 1.0, add_noise: bool = True) -> int:
        type(self).seen.append((temperature, add_noise))
        action: int = super().get_action(game, temperature=temperature, add_noise=add_noise)
        return action


class _ContrarianMCTS(MCTS):
    """The real search, then always a legal action other than greedy's.

    Makes the divergence count non-zero by construction, so a harness that
    drops or zeroes the count it measured cannot pass as "never diverged".
    """

    def get_action(self, game: Any, temperature: float = 1.0, add_noise: bool = True) -> int:
        super().get_action(game, temperature=temperature, add_noise=add_noise)
        legal = game.get_legal_actions()
        greedy = greedy_action(game.state.indicators, legal)
        return int(next(action for action in legal if action != greedy))


class TestSearchArms:
    def test_the_game_is_built_with_noise_off_and_only_the_robust_rule_turns_it_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H1: noise is a decision-rule argument, never a property of the game."""
        config = micro_config(tmp_path)
        arena = config.arena_config()
        assert (arena.add_noise, arena.temperature) == (False, 0.0)
        monkeypatch.setattr(harness, "MCTS", _RecordingMCTS)
        _RecordingMCTS.seen = []
        run_search_arm(config, PRIMARY_ARM, config.seed)
        assert _RecordingMCTS.seen and set(_RecordingMCTS.seen) == {(0.0, False)}
        _RecordingMCTS.seen = []
        run_search_arm(config, ROBUST_ARM, config.seed)
        assert _RecordingMCTS.seen and set(_RecordingMCTS.seen) == {(0.0, True)}

    def test_the_primary_arm_is_deterministic_and_its_record_is_consistent(
        self, tmp_path: Path
    ) -> None:
        """H2: one run per committed action; divergences equal the per-step record."""
        config = micro_config(tmp_path)
        first, cache = run_search_arm(config, PRIMARY_ARM, config.seed)
        second, _ = run_search_arm(config, PRIMARY_ARM, config.seed + 1)
        curve = [(p.n_dof, p.l2_error) for p in first.trajectory.points]
        assert curve == [(p.n_dof, p.l2_error) for p in second.trajectory.points]
        assert [s.action for s in first.steps] == [s.action for s in second.steps]
        assert len(first.steps) == len(first.trajectory.points) - 1 == config.max_steps
        assert first.trajectory.decisions_diverging_from_greedy == sum(
            step.diverged for step in first.steps
        )
        assert all(1 <= step.tree_depth <= config.n_simulations for step in first.steps)
        assert cache.misses > len(first.steps)

    def test_a_departing_search_is_counted_into_the_trajectory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H2: every departure from greedy reaches the trajectory's count and the step record."""
        config = micro_config(tmp_path)
        monkeypatch.setattr(harness, "MCTS", _ContrarianMCTS)
        run, _ = run_search_arm(config, PRIMARY_ARM, config.seed)
        assert len(run.steps) == config.max_steps
        assert all(step.diverged for step in run.steps)
        assert run.trajectory.decisions_diverging_from_greedy == config.max_steps


class TestAdequacy:
    def test_off_measures_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            harness, "measure_adequacy", lambda *a, **k: pytest.fail("measured while off")
        )
        assert check_adequacy(micro_config(tmp_path)) is None

    @pytest.mark.parametrize(
        ("operator_name", "window"),
        [("lshape_poisson", (200.0, 4000.0)), ("zshape_poisson", (200.0, 5000.0))],
    )
    def test_on_uses_the_operator_gate_and_aborts_on_a_violation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        operator_name: str,
        window: tuple[float, float],
    ) -> None:
        """H3: the testbed's own gate; a failing separation aborts with the reading."""
        config = micro_config(
            tmp_path,
            substrate=SubstrateConfig(name="lvg_skfem", kind="skfem_tri"),
            operator_name=operator_name,
            require_adequacy_precondition=True,
        )
        seen: dict[str, Any] = {}

        def measure(substrate: object, *, theta: float, gate: Any) -> RateSeparation:
            seen.update(theta=theta, window=gate.rate_fit_dof_range)
            return _rate_separation(adaptive_rate=-0.2)

        monkeypatch.setattr(harness, "_fresh_substrate", lambda cfg: object())
        monkeypatch.setattr(harness, "measure_adequacy", measure)
        with pytest.raises(AdequacyAbortedError, match="adequacy precondition") as caught:
            check_adequacy(config)
        assert seen == {"theta": config.adequacy_theta, "window": window}
        assert caught.value.separation.adaptive_rate == -0.2
        assert caught.value.violations

        monkeypatch.setattr(harness, "measure_adequacy", lambda *a, **k: _rate_separation())
        assert check_adequacy(config) == _rate_separation()
        assert adequacy_metrics(_rate_separation())["adequacy_error_ratio"] == 0.1

    def test_an_abort_happens_before_any_arm_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def abort(config: LookaheadVsGreedyConfig) -> None:
            raise AdequacyAbortedError(["synthetic"], _rate_separation())

        monkeypatch.setattr(harness, "check_adequacy", abort)
        monkeypatch.setattr(
            harness, "run_sweep_arms", lambda config: pytest.fail("an arm ran before adequacy")
        )
        with pytest.raises(AdequacyAbortedError):
            run_lookahead_vs_greedy(micro_config(tmp_path))

    def test_a_passing_gate_is_recorded_beside_the_verdict_and_never_gated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The adequacy readings ride along in the metrics; no threshold reads them."""
        monkeypatch.setattr(harness, "check_adequacy", lambda config: _rate_separation())
        config = micro_config(tmp_path)
        result = run_lookahead_vs_greedy(config)
        assert result.adequacy == _rate_separation()
        recorded = adequacy_metrics(_rate_separation())
        assert {name: result.metrics()[name] for name in recorded} == recorded
        assert not {t.name for t in config.get_default_thresholds()} & set(recorded)


class TestFairness:
    def test_a_classical_arm_that_stops_short_is_an_error(self, tmp_path: Path) -> None:
        """H4: one classical level cannot span the game-driven DOF."""
        with pytest.raises(ClassicalBudgetError, match="dorfler_theta0.1"):
            run_lookahead_vs_greedy(micro_config(tmp_path, max_refinements_classical=1))

    def test_a_shared_solve_cache_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """H4: two MCTS runs through one cache would hand each other free solves."""
        real = harness.run_search_arm
        shared: list[Any] = []

        def sharing(config: LookaheadVsGreedyConfig, arm: Any, seed: int) -> Any:
            run, cache = real(config, arm, seed)
            shared.append(cache)
            return run, shared[0]

        monkeypatch.setattr(harness, "run_search_arm", sharing)
        with pytest.raises(RuntimeError, match="must not share"):
            run_lookahead_vs_greedy(micro_config(tmp_path))


class TestRunLookaheadVsGreedy:
    def test_every_arm_and_seed_runs_and_reads_against_one_matched_dof(
        self, micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult]
    ) -> None:
        config, result = micro
        assert list(result.classical()) == [
            GREEDY_LABEL,
            *(dorfler_label(theta) for theta in config.dorfler_thetas),
            UNIFORM_LABEL,
        ]
        assert result.primary.seed == config.seed
        assert [run.seed for run in result.robust] == config.robust_seeds()
        assert (result.primary.add_noise, {run.add_noise for run in result.robust}) == (
            False,
            {True},
        )
        for run in result.runs():
            arms = [run.trajectory, *result.classical().values()]
            assert run.reading.matched_dof == min(float(t.dofs().max()) for t in arms)
        assert set(result.wall_seconds) == {
            GREEDY_LABEL,
            f"{PRIMARY_LABEL}_seed{config.seed}",
            *(f"{ROBUST_LABEL}_seed{seed}" for seed in config.robust_seeds()),
        }
        assert result.adequacy is None

    def test_the_recorded_verdict_is_the_thresholds_on_the_recorded_metrics(
        self, micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult]
    ) -> None:
        config, result = micro
        metrics = result.metrics()
        assert all(np.isfinite(value) for value in metrics.values())
        again = evaluate_verdict(result.testbed, metrics, config.get_default_thresholds())
        assert again == result.verdict
        assert metrics["verdict_go"] == (1.0 if result.verdict.go else 0.0)
        assert {t.name for t in config.get_default_thresholds()} <= set(metrics)

    def test_k_star_is_against_greedy_at_the_run_s_matched_dof(
        self, micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult]
    ) -> None:
        config, result = micro
        primary = result.primary
        expected = break_even(
            primary.trajectory,
            result.greedy,
            matched_dof=primary.reading.matched_dof,
            alphas=config.break_even_alphas,
        )
        assert primary.break_even.dof_saving == expected.dof_saving
        assert primary.break_even.reuses == expected.reuses
        best = result.classical()[primary.reading.best_classical]
        assert primary.versus_best.matched_dof == min(
            float(best.dofs().max()), float(primary.trajectory.dofs().max())
        )


class TestArtifacts:
    def test_the_csv_is_the_arena_schema_with_every_arm(
        self, micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult], tmp_path: Path
    ) -> None:
        config, result = micro
        path = export_csv(result, tmp_path / "lvg.csv")
        text = path.read_text(encoding="utf-8")
        assert "\r\n" not in text
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or ()) == CSV_COLUMNS
            rows = list(reader)
        methods = {row["method"] for row in rows}
        assert methods == {
            UNIFORM_LABEL,
            *(dorfler_label(theta) for theta in config.dorfler_thetas),
            GREEDY_LABEL,
            PRIMARY_LABEL,
            ROBUST_LABEL,
        }
        deterministic = {
            r["seed"] for r in rows if r["method"] not in {PRIMARY_LABEL, ROBUST_LABEL}
        }
        assert deterministic == {str(DETERMINISTIC_ARM_SEED)}
        robust_seeds = {int(r["seed"]) for r in rows if r["method"] == ROBUST_LABEL}
        assert robust_seeds == set(config.robust_seeds())
        assert len(rows) == sum(len(t.points) for t in result.classical().values()) + sum(
            len(run.trajectory.points) for run in result.runs()
        )

    def test_the_plot_is_drawn_or_skipped_without_matplotlib(
        self,
        micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _, result = micro
        png = export_plot(result, tmp_path / "lvg.png")
        assert png is not None and png.stat().st_size > 0
        real_import = builtins.__import__

        def no_matplotlib(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("matplotlib"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_matplotlib)
        assert export_plot(result, tmp_path / "skipped.png") is None

    def test_the_sidecar_carries_the_pre_run_snapshot_and_every_reading(
        self,
        micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """H6: the snapshot taken before any write, a hash the config reproduces."""
        config, result = micro
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: CLEAN)
        recorder = RunRecorder.start(config_hash=config.compute_hash(), proposal_grade=True)
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: DIRTY)
        written = write_artifacts(result, config, recorder)
        assert set(written) == {"csv", "png", "run_json"}
        manifest = load_run_manifest(written["run_json"])
        assert manifest.git == CLEAN
        assert manifest.harness == HARNESS_NAME
        assert manifest.config_hash == config.compute_hash()
        # The committed-sidecar guard's expression, verbatim (dispatch by name).
        assert load_config_from_dict(manifest.config).compute_hash() == manifest.config_hash
        assert manifest.metrics == result.metrics()
        assert manifest.seeds == [run.seed for run in result.runs()]
        arms = {arm.name: arm for arm in manifest.arms}
        assert set(arms) >= {GREEDY_LABEL, UNIFORM_LABEL, f"{PRIMARY_LABEL}_seed{config.seed}"}
        primary = arms[f"{PRIMARY_LABEL}_seed{config.seed}"]
        assert primary.parameters["tree_depth_per_step"] == [
            s.tree_depth for s in result.primary.steps
        ]
        assert primary.parameters["add_noise"] is False
        assert arms[f"{ROBUST_LABEL}_seed{config.robust_seeds()[0]}"].parameters["add_noise"]
        assert result.verdict.label in manifest.notes
        assert "pre-registered" in manifest.notes

    def test_proposal_grade_refuses_a_dirty_tree_before_anything_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: DIRTY)
        with pytest.raises(ProposalGradeError, match="dirty"):
            RunRecorder.start(config_hash="abc", proposal_grade=True)

    def test_a_finite_k_star_is_recorded_per_run_and_an_infinite_one_is_omitted(
        self,
        micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """JSON has no infinity: a run's K*_alpha is a counter only when it is finite."""
        config, result = micro
        finite_at_one = dataclasses.replace(result.primary.break_even, reuses={1.0: 7.5, 1.5: None})
        patched = dataclasses.replace(
            result, primary=dataclasses.replace(result.primary, break_even=finite_at_one)
        )
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: CLEAN)
        recorder = RunRecorder.start(config_hash=config.compute_hash())
        manifest = build_run_manifest(patched, config, recorder, {})
        primary = next(arm for arm in manifest.arms if arm.name.startswith(PRIMARY_LABEL))
        assert primary.counters["break_even_reuses_alpha1"] == 7.5
        assert "break_even_reuses_alpha1.5" not in primary.counters

    def test_a_run_without_matplotlib_still_writes_the_csv_and_the_sidecar(
        self,
        micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        config, result = micro
        monkeypatch.setattr(artifacts, "export_plot", lambda result, path: None)
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: CLEAN)
        here = config.model_copy(update={"output_dir": str(tmp_path)})
        written = write_artifacts(result, here, RunRecorder.start(config_hash=here.compute_hash()))
        assert set(written) == {"csv", "run_json"}
        assert set(load_run_manifest(written["run_json"]).artifacts) == {"csv"}

    def test_a_config_changed_after_the_snapshot_is_refused_before_anything_is_written(
        self,
        micro: tuple[LookaheadVsGreedyConfig, LookaheadVsGreedyResult],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """H6: the recorded config must reproduce the recorded (pre-run) hash."""
        config, result = micro
        monkeypatch.setattr("src.research.run_manifest.collect_git_provenance", lambda: CLEAN)
        recorder = RunRecorder.start(config_hash=config.compute_hash())
        drifted = config.model_copy(update={"output_dir": str(tmp_path)})
        assert drifted.compute_hash() != recorder.config_hash
        with pytest.raises(SidecarHashMismatchError, match="would not reproduce its hash"):
            build_run_manifest(result, drifted, recorder, {})
        with pytest.raises(SidecarHashMismatchError):
            write_artifacts(result, drifted, recorder)
        assert list(tmp_path.iterdir()) == []


# Mutation-kill record (Gate 1 surface, 2026-10-08) is appended by the
# harden-a-guard pass; see the report for the planted defects and the named
# killers.
