"""Harness unit tests: adequacy abort, interpolation, forbidden evaluators."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.research.mcts_classical_amr_arena import (
    AdequacyPreconditionError,
    ArenaPoint,
    ArenaTrajectory,
    abort_if_inadequate,
    compare_mcts_vs_dorfler,
    export_csv,
    run_comparison,
    write_arena_manifest,
)
from src.research.substrates.config import AdequacyGateConfig
from src.research.substrates.sweep import RateSeparation

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "src" / "research" / "mcts_classical_amr_arena.py"


def _traj(method: str, dofs: list[int], errors: list[float], solves: list[int]) -> ArenaTrajectory:
    traj = ArenaTrajectory(method=method)  # type: ignore[arg-type]
    for level, (dof, err, nsol) in enumerate(zip(dofs, errors, solves, strict=True)):
        traj.points.append(
            ArenaPoint(
                level=level,
                n_dof=dof,
                l2_error=err,
                wall_time_seconds=float(level + 1),
                n_cache_misses=nsol,
                n_cache_hits=0,
                n_apply_actions=level,
            )
        )
    return traj


def _separation(**overrides: float) -> RateSeparation:
    values = {
        "adaptive_rate": -1.31,
        "uniform_rate": -0.67,
        "error_ratio_at_matched_dof": 0.1,
        "matched_dof": 2000.0,
        "n_adaptive_points": 4,
        "n_uniform_points": 4,
    }
    values.update(overrides)
    return RateSeparation(**values)  # type: ignore[arg-type]


class TestAdequacyAbort:
    def test_nonempty_violations_raise(self) -> None:
        with pytest.raises(AdequacyPreconditionError, match="adequacy precondition"):
            abort_if_inadequate(
                _separation(adaptive_rate=-0.2, error_ratio_at_matched_dof=2.0),
                AdequacyGateConfig(name="t"),
            )

    def test_passing_separation_is_silent(self) -> None:
        abort_if_inadequate(_separation(), AdequacyGateConfig(name="t"))


class TestMatchedMetrics:
    def test_matched_dof_ratio_mcts_better(self) -> None:
        dorfler = _traj("dorfler", [100, 200], [1.0, 0.4], [1, 2])
        mcts = _traj("mcts", [100, 200], [1.0, 0.2], [1, 8])
        l2, _solves, _wall, matched, *_rest = compare_mcts_vs_dorfler(dorfler, mcts)
        assert matched == pytest.approx(200.0)
        assert l2 == pytest.approx(0.5)

    def test_matched_solves_uses_cache_misses(self) -> None:
        dorfler = _traj("dorfler", [100, 200], [1.0, 0.5], [1, 2])
        mcts = _traj("mcts", [100, 150], [1.0, 0.8], [1, 20])
        _l2, solve_ratio, _wall, _mdof, matched_solves, _mt = compare_mcts_vs_dorfler(dorfler, mcts)
        assert matched_solves == pytest.approx(2.0)
        assert np.isfinite(solve_ratio)


class TestForbiddenEvaluators:
    def test_harness_does_not_name_encoded_or_random(self) -> None:
        tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        banned = {"EncodedValueEvaluator", "RandomEvaluator"}
        assert names.isdisjoint(banned)
        assert attrs.isdisjoint(banned)
        source = HARNESS.read_text(encoding="utf-8")
        assert "ResidualPriorErrorValueEvaluator" in source
        assert "search_mode=adapter.search_mode" in source


class TestCsvAndCaches:
    def test_export_csv_has_all_three_methods(self, tmp_path: Path) -> None:
        from src.research.mcts_classical_amr_arena import MultiSeedArena, SeedComparison

        dorfler = _traj("dorfler", [10, 20], [1.0, 0.5], [1, 2])
        uniform = _traj("uniform", [10, 40], [1.0, 0.4], [1, 2])
        mcts = _traj("mcts", [10, 18], [1.0, 0.6], [1, 5])
        mcts.cache_id = 99
        seed = SeedComparison(
            seed=7,
            mcts=mcts,
            l2_error_ratio_at_matched_dof=1.2,
            l2_error_ratio_at_matched_solves=1.5,
            error_per_dof_ratio_at_matched_wall_clock=2.0,
            matched_dof=18.0,
            matched_solves=2.0,
            matched_wall_time_seconds=1.0,
        )
        arena = MultiSeedArena(
            dorfler=dorfler,
            uniform=uniform,
            per_seed=[seed],
            seeds=[7],
            marking_fraction=0.5,
            dof_convention="fem_basis_dofs",
        )
        path = export_csv(arena, tmp_path / "arena.csv")
        text = path.read_text(encoding="utf-8")
        assert "method,seed,level,n_dof" in text
        assert "\r\n" not in text
        assert "dorfler" in text and "uniform" in text and "mcts" in text

    def test_classical_cache_id_is_none(self) -> None:
        dorfler = _traj("dorfler", [10], [1.0], [1])
        assert dorfler.cache_id is None


class TestManifestAndActionSpace:
    def test_write_arena_manifest_records_describe(self, tmp_path: Path) -> None:
        from src.poc.scenarios.mcts_classical_amr_arena_config import (
            MCTSClassicalAMRArenaConfig,
        )
        from src.research.mcts_classical_amr_arena import MultiSeedArena, SeedComparison
        from src.research.run_manifest import load_run_manifest
        from src.research.substrates.config import SubstrateConfig

        dorfler = _traj("dorfler", [10, 20], [1.0, 0.5], [1, 2])
        uniform = _traj("uniform", [10, 40], [1.0, 0.4], [1, 2])
        mcts = _traj("mcts", [10, 18], [1.0, 0.6], [1, 5])
        seed = SeedComparison(
            seed=7,
            mcts=mcts,
            l2_error_ratio_at_matched_dof=1.2,
            l2_error_ratio_at_matched_solves=1.5,
            error_per_dof_ratio_at_matched_wall_clock=2.0,
            matched_dof=18.0,
            matched_solves=2.0,
            matched_wall_time_seconds=1.0,
        )
        arena = MultiSeedArena(
            dorfler=dorfler,
            uniform=uniform,
            per_seed=[seed],
            seeds=[7],
            marking_fraction=0.5,
            dof_convention="fem_basis_dofs",
            substrate_describe={"kind": "skfem_tri", "dof_convention": "fem_basis_dofs"},
        )
        csv_path = tmp_path / "arena.csv"
        export_csv(arena, csv_path)
        config = MCTSClassicalAMRArenaConfig(
            name="mcts_classical_amr_arena",
            substrate=SubstrateConfig(name="tg", kind="tensor_grid", initial_side=4),
            operator_name="poisson",
            require_adequacy_precondition=False,
        )
        sidecar = write_arena_manifest(arena, config, csv_path, None, proposal_grade=False)
        loaded = load_run_manifest(sidecar)
        assert loaded.arms[0].parameters["dof_convention"] == "fem_basis_dofs"
        assert loaded.arms[0].parameters["substrate_describe"]["kind"] == "skfem_tri"
        assert loaded.config_hash != "unknown"

    def test_run_comparison_tensor_grid_micro(self, tmp_path: Path) -> None:
        pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")
        from src.poc.scenarios.mcts_classical_amr_arena_config import (
            MCTSClassicalAMRArenaConfig,
        )
        from src.research.substrates.config import SubstrateConfig

        config = MCTSClassicalAMRArenaConfig(
            name="mcts_classical_amr_arena",
            substrate=SubstrateConfig(
                name="arena_tg",
                kind="tensor_grid",
                initial_side=4,
                solve_cache_max_entries=64,
            ),
            operator_name="poisson",
            require_adequacy_precondition=False,
            max_dof=80,
            max_steps=2,
            max_refinements_classical=2,
            n_simulations=2,
            n_seeds=1,
            top_k_actions=4,
            max_action_space=64,
            output_dir=str(tmp_path),
        )
        arena = run_comparison(config)
        assert arena.dorfler.cache_id is None
        assert arena.per_seed[0].mcts.cache_id is not None
        assert arena.dorfler.cache_id != arena.per_seed[0].mcts.cache_id
        assert np.isfinite(arena.metrics()["l2_error_ratio_at_matched_dof"])

    def test_truncated_action_space_raises(self) -> None:
        pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")
        from src.poc.scenarios.mcts_classical_amr_arena_config import (
            MCTSClassicalAMRArenaConfig,
        )
        from src.research.mcts_classical_amr_arena import run_mcts_arm
        from src.research.substrates.config import SubstrateConfig

        config = MCTSClassicalAMRArenaConfig(
            name="mcts_classical_amr_arena",
            substrate=SubstrateConfig(
                name="tiny_space",
                kind="tensor_grid",
                initial_side=4,
                solve_cache_max_entries=16,
            ),
            operator_name="poisson",
            require_adequacy_precondition=False,
            max_action_space=1,
            max_steps=1,
            n_simulations=1,
            n_seeds=1,
            top_k_actions=0,
        )
        with pytest.raises(ValueError, match="truncates"):
            run_mcts_arm(config, seed=0)

    def test_mcts_stops_when_initial_dof_meets_budget(self) -> None:
        pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")
        from src.poc.scenarios.mcts_classical_amr_arena_config import (
            MCTSClassicalAMRArenaConfig,
        )
        from src.research.mcts_classical_amr_arena import run_mcts_arm
        from src.research.substrates.config import SubstrateConfig

        config = MCTSClassicalAMRArenaConfig(
            name="mcts_classical_amr_arena",
            substrate=SubstrateConfig(
                name="dof_cap",
                kind="tensor_grid",
                initial_side=4,
                solve_cache_max_entries=16,
            ),
            operator_name="poisson",
            require_adequacy_precondition=False,
            max_dof=10,
            max_steps=4,
            n_simulations=1,
            n_seeds=1,
            top_k_actions=2,
            max_action_space=64,
        )
        traj = run_mcts_arm(config, seed=0)
        assert len(traj.points) == 1

    def test_export_plot_and_proposal_grade(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.poc.scenarios.mcts_classical_amr_arena_config import (
            MCTSClassicalAMRArenaConfig,
        )
        from src.research.mcts_classical_amr_arena import (
            MultiSeedArena,
            SeedComparison,
            export_plot,
        )
        from src.research.run_manifest import GitProvenance, ProposalGradeError
        from src.research.substrates.config import SubstrateConfig

        dorfler = _traj("dorfler", [10, 20], [1.0, 0.5], [1, 2])
        uniform = _traj("uniform", [10, 40], [1.0, 0.4], [1, 2])
        mcts = _traj("mcts", [10, 18], [1.0, 0.6], [1, 5])
        seed = SeedComparison(
            seed=7,
            mcts=mcts,
            l2_error_ratio_at_matched_dof=1.2,
            l2_error_ratio_at_matched_solves=1.5,
            error_per_dof_ratio_at_matched_wall_clock=2.0,
            matched_dof=18.0,
            matched_solves=2.0,
            matched_wall_time_seconds=1.0,
        )
        arena = MultiSeedArena(
            dorfler=dorfler,
            uniform=uniform,
            per_seed=[seed],
            seeds=[7],
            marking_fraction=0.5,
            dof_convention="fem_basis_dofs",
        )
        png = export_plot(arena, tmp_path / "arena.png")
        assert png is not None
        assert png.exists()

        csv_path = tmp_path / "arena.csv"
        export_csv(arena, csv_path)
        config = MCTSClassicalAMRArenaConfig(
            name="mcts_classical_amr_arena",
            substrate=SubstrateConfig(name="tg", kind="tensor_grid", initial_side=4),
            operator_name="poisson",
            require_adequacy_precondition=False,
        )
        monkeypatch.setattr(
            "src.research.mcts_classical_amr_arena.collect_git_provenance",
            lambda: GitProvenance(sha="abc", dirty=True),
        )
        with pytest.raises(ProposalGradeError, match="dirty"):
            write_arena_manifest(arena, config, csv_path, png, proposal_grade=True)
        monkeypatch.setattr(
            "src.research.mcts_classical_amr_arena.collect_git_provenance",
            lambda: GitProvenance(sha="abc", dirty=False),
        )
        sidecar = write_arena_manifest(arena, config, csv_path, png, proposal_grade=True)
        assert sidecar.exists()

    def test_write_arena_manifest_uses_injected_git(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.poc.scenarios.mcts_classical_amr_arena_config import (
            MCTSClassicalAMRArenaConfig,
        )
        from src.research.mcts_classical_amr_arena import (
            MultiSeedArena,
            SeedComparison,
        )
        from src.research.run_manifest import GitProvenance, load_run_manifest
        from src.research.substrates.config import SubstrateConfig

        dorfler = _traj("dorfler", [10, 20], [1.0, 0.5], [1, 2])
        uniform = _traj("uniform", [10, 40], [1.0, 0.4], [1, 2])
        mcts = _traj("mcts", [10, 18], [1.0, 0.6], [1, 5])
        seed = SeedComparison(
            seed=7,
            mcts=mcts,
            l2_error_ratio_at_matched_dof=1.2,
            l2_error_ratio_at_matched_solves=1.5,
            error_per_dof_ratio_at_matched_wall_clock=2.0,
            matched_dof=18.0,
            matched_solves=2.0,
            matched_wall_time_seconds=1.0,
        )
        arena = MultiSeedArena(
            dorfler=dorfler,
            uniform=uniform,
            per_seed=[seed],
            seeds=[7],
            marking_fraction=0.5,
            dof_convention="fem_basis_dofs",
        )
        csv_path = tmp_path / "arena.csv"
        export_csv(arena, csv_path)
        config = MCTSClassicalAMRArenaConfig(
            name="mcts_classical_amr_arena",
            substrate=SubstrateConfig(name="tg", kind="tensor_grid", initial_side=4),
            operator_name="poisson",
            require_adequacy_precondition=False,
        )

        def _must_not_collect() -> GitProvenance:
            raise AssertionError("live git probe must not run when git is injected")

        monkeypatch.setattr(
            "src.research.mcts_classical_amr_arena.collect_git_provenance",
            _must_not_collect,
        )
        sidecar = write_arena_manifest(
            arena,
            config,
            csv_path,
            None,
            proposal_grade=True,
            git=GitProvenance(sha="deadbeef", branch="clean", dirty=False),
        )
        loaded = load_run_manifest(sidecar)
        assert loaded.git.dirty is False
        assert loaded.git.sha == "deadbeef"

    def test_classical_arm_empty_sweep_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.research.mcts_classical_amr_arena import run_classical_arm

        monkeypatch.setattr(
            "src.research.mcts_classical_amr_arena.run_refinement_sweep",
            lambda *args, **kwargs: [],
        )

        class _Host:
            def describe(self) -> dict[str, str]:
                return {"kind": "tensor_grid"}

        traj = run_classical_arm(
            _Host(),  # type: ignore[arg-type]
            policy="adaptive",
            theta=0.5,
            max_levels=1,
            max_dof=10,
            error_tolerance=1e-8,
        )
        assert traj.points == []
        assert traj.method == "dorfler"
