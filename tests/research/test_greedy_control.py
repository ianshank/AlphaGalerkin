"""Single-element greedy control and ``decisions_diverging_from_greedy``.

The arena's committed MCTS trajectory could not be told apart from greedy
single-element marking, because nothing measured the difference. These tests
guard the control arm and the counter that now do. Defect classes, one
sentence each:

* **G1 drift** -- the greedy choice and the game's top-k legal ranking score or
  tie-break differently, so "greedy" and the set the search sees diverge.
* **G2 shared search** -- the greedy control reaches the search engine, so the
  control moves with the candidate's defects and MCTS/greedy reads 1 for the
  wrong reason.
* **G3 blind counter** -- the divergence counter cannot count: it never
  increments, or it compares against ``legal[0]``, which is greedy only while
  the legal set is indicator-ranked.
* **G4 shared cache** -- two game-driven arms solve through one
  ``FingerprintSolveCache``, so one arm's solves become the other's free hits.

Every oracle below is written independently of ``greedy_action``: a test that
checked greedy against itself would pass for any definition of greedy.

Mutation-kill record (2026-09-25). 13 planted defects in this file's surface,
each confirmed applied by an anchor assertion and run against this file alone;
each was killed by the NAMED test (others that also went red are omitted). No
killer is ``fem_required`` or ``gpu_required``, so every kill holds on CPU CI.

1. G1 -- ``greedy_action`` ranks ``legal_actions`` as given, not ``sorted`` ->
   ``TestGreedyAction::test_ties_resolve_to_the_lowest_index``.
2. G1 -- ``get_valid_actions`` ranks ``reversed(refinable)`` (ties to the
   highest index) ->
   ``TestAgreesWithTheGameRanking::test_greedy_is_the_head_of_the_top_k_set``.
   Also red: ``test_mcts_with_one_simulation_never_diverges`` -- the counter
   itself notices the ranking drifting from greedy.
3. G1 -- ``indicator_score`` scores an out-of-range index ``-inf``, not ``0.0``
   -> ``TestGreedyAction::test_an_out_of_range_index_scores_zero``.
4. G2 -- ``run_greedy_arm`` delegates to the MCTS arm with one simulation (the
   peer review's own shortcut) ->
   ``TestRunGreedyArm::test_never_constructs_the_search_engine``. The import
   contract stayed GREEN on this one (a lazy import of the harness is not
   ``src.mcts``), which is why this semantic test exists.
5. G3 -- ``GreedyDivergence.record`` never increments ->
   ``TestDivergenceCounter::test_a_search_that_avoids_greedy_is_counted_every_step``.
6. G3 -- ``GreedyDivergence.record`` compares against ``legal[0]`` ->
   ``TestDivergenceCounter::test_the_reference_is_greedy_not_the_first_legal_action``.
7. G4 -- ``_assert_private_caches``'s identity check made inert ->
   ``TestPrivateCaches::test_a_cache_shared_by_greedy_and_mcts_is_refused``.
8. ``run_comparison`` ignores ``include_greedy_control`` ->
   ``TestArtifacts::test_without_greedy_the_csv_is_exactly_the_legacy_method_set``.
9. A legacy metric key renamed (``mcts_win_fraction``) ->
   ``TestComparisonMetrics::test_every_legacy_key_is_still_published``.
10. An unmeasured divergence count coerced to 0 ->
    ``TestComparisonMetrics::test_an_unmeasured_count_is_absent_not_zero``.
11. The CSV ``l2_error`` format drifts (``.12g`` -> ``.10g``) ->
    ``TestArtifacts::test_export_reproduces_the_committed_artifact_byte_for_byte``.
12. Greedy plotted in the MCTS style ->
    ``TestArtifacts::test_the_plot_draws_greedy_in_a_distinct_style``.
13. The episode driver ignores ``max_dof`` ->
    ``TestRunGreedyArm::test_stops_at_max_dof_like_the_mcts_arm``.

The import-contract plant (``from src.mcts.search import MCTS`` in
``greedy_control.py``) is recorded in ``tests/regression/test_import_contracts.py``.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import src.research.greedy_control as greedy_module
from src.pde.games.substrate_refinement import (
    SubstrateEpisodeState,
    SubstrateRefinementGame,
    indicator_score,
    rank_by_indicator,
)
from src.pde.games.substrate_refinement_config import SubstrateRefinementConfig
from src.poc.scenarios.mcts_classical_amr_arena_config import (
    SCENARIO_NAME,
    MCTSClassicalAMRArenaConfig,
)
from src.refinement.state import RefinementState
from src.research.amr_arena_episode import (
    ArenaEpisode,
    assert_action_space_covers,
    build_arena_episode,
    run_arena_episode,
)
from src.research.greedy_control import (
    GREEDY_METHOD,
    GreedyDivergence,
    greedy_action,
    run_greedy_arm,
    run_greedy_arm_with_cache,
)
from src.research.mcts_classical_amr_arena import (
    ArenaPoint,
    ArenaTrajectory,
    MultiSeedArena,
    SeedComparison,
    _assert_private_caches,
    export_csv,
    export_plot,
    run_comparison,
    run_mcts_arm,
    write_arena_manifest,
)
from src.research.run_manifest import GitProvenance, load_run_manifest
from src.research.substrates.config import SubstrateConfig
from src.research.substrates.solve_cache import FingerprintSolveCache

pytest.importorskip("scipy", reason="scipy required for the tensor-grid solve")

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CSV = REPO_ROOT / "results" / "mcts_classical_amr_arena.csv"
LEGACY_METHODS = {"uniform", "dorfler", "mcts"}
#: Deterministic metrics (no wall-clock) that must not move when greedy is added.
DETERMINISTIC_LEGACY_KEYS = (
    "l2_error_ratio_at_matched_dof",
    "l2_error_ratio_at_matched_solves",
    "matched_dof",
    "matched_solves",
    "mcts_win_fraction",
    "l2_ratio_seed_min",
    "l2_ratio_seed_max",
    "l2_ratio_seed_std",
    "n_seeds",
    "marking_fraction",
)
#: Every key ``MultiSeedArena.metrics()`` published before the greedy control.
LEGACY_KEYS = (
    *DETERMINISTIC_LEGACY_KEYS,
    "error_per_dof_ratio_mcts_over_dorfler",
    "matched_wall_time_seconds",
)
GREEDY_RATIO_KEYS = (
    "l2_error_ratio_mcts_over_greedy_at_matched_dof",
    "l2_error_ratio_greedy_over_dorfler_at_matched_dof",
)
DIVERGENCE_KEYS = ("decisions_diverging_from_greedy", "decisions_diverging_from_greedy_max")
#: Values exactly representable in float32, few enough that ties are common.
TIE_PRONE_INDICATORS = st.sampled_from([0.0, 0.25, 0.5, 1.0, 2.0])


def _reference_greedy(indicators: Sequence[float], legal: Sequence[int]) -> int:
    """Independent oracle: largest score, lowest index on ties, out of range scores 0."""

    def score(action: int) -> float:
        return float(indicators[action]) if 0 <= action < len(indicators) else 0.0

    best = max(score(action) for action in legal)
    return min(action for action in legal if score(action) == best)


def _config(**overrides: object) -> MCTSClassicalAMRArenaConfig:
    """A CPU tensor-grid arena: 16 elements at the start, one grid line per step."""
    params: dict[str, object] = {
        "name": SCENARIO_NAME,
        "substrate": SubstrateConfig(
            name="greedy_tg",
            kind="tensor_grid",
            initial_side=4,
            solve_cache_max_entries=64,
        ),
        "operator_name": "poisson",
        "require_adequacy_precondition": False,
        "max_dof": 10_000,
        "max_steps": 3,
        "max_refinements_classical": 2,
        "n_simulations": 1,
        "n_seeds": 1,
        "top_k_actions": 4,
        "max_action_space": 256,
    }
    params.update(overrides)
    return MCTSClassicalAMRArenaConfig(**params)  # type: ignore[arg-type]


def _curve(traj: ArenaTrajectory) -> list[tuple[int, float]]:
    return [(point.n_dof, point.l2_error) for point in traj.points]


def _traj(method: str, dofs: list[int], errors: list[float]) -> ArenaTrajectory:
    traj = ArenaTrajectory(method=method)  # type: ignore[arg-type]
    for level, (dof, err) in enumerate(zip(dofs, errors, strict=True)):
        traj.points.append(
            ArenaPoint(
                level=level,
                n_dof=dof,
                l2_error=err,
                wall_time_seconds=float(level + 1),
                n_cache_misses=level + 1,
                n_cache_hits=0,
                n_apply_actions=level,
            )
        )
    return traj


def _seed(seed: int, mcts: ArenaTrajectory, **overrides: object) -> SeedComparison:
    fields: dict[str, Any] = {
        "seed": seed,
        "mcts": mcts,
        "l2_error_ratio_at_matched_dof": 0.9,
        "l2_error_ratio_at_matched_solves": 2.0,
        "error_per_dof_ratio_at_matched_wall_clock": 3.0,
        "matched_dof": 20.0,
        "matched_solves": 2.0,
        "matched_wall_time_seconds": 1.0,
    }
    fields.update(overrides)
    return SeedComparison(**fields)


def _hand_built_arena(**overrides: object) -> MultiSeedArena:
    fields: dict[str, Any] = {
        "dorfler": _traj("dorfler", [10, 20], [1.0, 0.5]),
        "uniform": _traj("uniform", [10, 40], [1.0, 0.4]),
        "per_seed": [_seed(7, _traj("mcts", [10, 18], [1.0, 0.6]))],
        "seeds": [7],
        "marking_fraction": 0.5,
        "dof_convention": "fem_basis_dofs",
    }
    fields.update(overrides)
    return MultiSeedArena(**fields)


def _stub_search(
    pick: Callable[[list[int], int], int],
    seen: list[tuple[int, int, int]],
) -> type:
    """A stand-in for ``src.mcts.search.MCTS`` whose decision is ``pick(legal, greedy)``.

    ``greedy`` comes from the independent oracle, never from ``greedy_action``.
    ``seen`` records ``(legal[0], greedy, action)`` per decision.
    """

    class _StubSearch:
        def __init__(self, **_kwargs: object) -> None:
            self.advanced: list[int] = []

        def get_action(self, game: Any, temperature: float = 0.0, add_noise: bool = False) -> int:
            del temperature, add_noise
            legal = list(game.get_legal_actions())
            indicators = np.asarray(game.state.indicators, dtype=np.float64).tolist()
            greedy = _reference_greedy(indicators, legal)
            action = pick(legal, greedy)
            seen.append((legal[0], greedy, action))
            return action

        def advance(self, action: int) -> None:
            self.advanced.append(action)

    return _StubSearch


class _RecordingLogger:
    """Double for a cached structlog logger (``capture_logs`` records nothing here)."""

    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def bind(self, **_kwargs: object) -> _RecordingLogger:
        return self

    def debug(self, event: str, **kwargs: object) -> None:
        self.records.append((event, kwargs))

    def info(self, event: str, **kwargs: object) -> None:
        self.records.append((event, kwargs))

    def warning(self, event: str, **kwargs: object) -> None:
        self.records.append((event, kwargs))


class _FlatSubstrate:
    """Every unit refinable; the mesh handle *is* the unit count."""

    def refinable_mask(self, mesh: int) -> np.ndarray:
        return np.ones(mesh, dtype=bool)

    def n_units(self, mesh: int) -> int:
        return mesh


class TestGreedyAction:
    def test_returns_the_largest_indicator(self) -> None:
        assert greedy_action([0.1, 0.9, 0.3], [0, 1, 2]) == 1

    def test_ties_resolve_to_the_lowest_index(self) -> None:
        """The legal set arrives out of index order; the tie still goes to index 1."""
        assert greedy_action([0.5, 0.9, 0.9, 0.9], [3, 2, 1]) == 1

    def test_only_legal_actions_are_candidates(self) -> None:
        assert greedy_action([5.0, 0.1, 0.2], [1, 2]) == 2

    def test_the_first_legal_action_is_not_assumed_greedy(self) -> None:
        """Unranked (index-ordered) legal set, as the game returns without a top-k cap."""
        assert greedy_action([0.1, 0.2, 0.7, 0.3], [0, 1, 2, 3]) == 2

    def test_an_out_of_range_index_scores_zero(self) -> None:
        """The game's rule: a unit with no indicator ranks as unmarked (0.0), not -inf."""
        assert greedy_action([-1.0, -2.0], [0, 7]) == 7
        assert greedy_action([0.2, 0.1], [0, 5]) == 0

    def test_empty_legal_set_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            greedy_action([0.1, 0.2], [])

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_legal_indicator_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            greedy_action([bad, 0.5], [0, 1])

    def test_a_non_finite_indicator_outside_the_legal_set_is_ignored(self) -> None:
        assert greedy_action([math.nan, 0.5, 0.2], [1, 2]) == 1

    @settings(max_examples=150)
    @given(
        indicators=st.lists(TIE_PRONE_INDICATORS, min_size=1, max_size=12),
        data=st.data(),
    )
    def test_returns_a_legal_action_with_the_maximal_score(
        self, indicators: list[float], data: st.DataObject
    ) -> None:
        """Legal (out-of-range indices included), maximal, lowest index among ties."""
        universe = list(range(len(indicators) + 3))
        legal = data.draw(st.lists(st.sampled_from(universe), min_size=1, unique=True))
        chosen = greedy_action(indicators, legal)
        flat = np.asarray(indicators, dtype=np.float64)
        best = max(indicator_score(flat, action) for action in legal)
        assert chosen in legal
        assert indicator_score(flat, chosen) == best
        assert chosen == _reference_greedy(indicators, legal)


class TestRankByIndicator:
    def test_pairs_are_score_then_action_largest_first_and_stable(self) -> None:
        # float32 in (as the game stores indicators), exact in float64 out.
        ranked = rank_by_indicator(np.array([0.5, 0.75, 0.5], dtype=np.float32), [0, 1, 2, 9])
        assert ranked == [(0.75, 1), (0.5, 0), (0.5, 2), (0.0, 9)]


class TestAgreesWithTheGameRanking:
    """G1: greedy and the top-k legal set read one ranking, so they cannot drift."""

    @settings(max_examples=100)
    @given(
        indicators=st.lists(TIE_PRONE_INDICATORS, min_size=2, max_size=10),
        data=st.data(),
    )
    def test_greedy_is_the_head_of_the_top_k_set(
        self, indicators: list[float], data: st.DataObject
    ) -> None:
        n = len(indicators)
        top_k = data.draw(st.integers(min_value=1, max_value=n - 1))
        game = SubstrateRefinementGame(
            SubstrateRefinementConfig(
                name="agree",
                max_steps=5,
                top_k_actions=top_k,
                max_action_space=64,
            ),
            substrate=_FlatSubstrate(),  # type: ignore[arg-type]
            solve_cache=FingerprintSolveCache(4),
        )
        state = SubstrateEpisodeState(
            values=np.zeros(n, dtype=np.float32),
            indicators=np.asarray(indicators, dtype=np.float32),
            error_estimate=1.0,
            dof=n,
            budget_remaining=10.0,
            mesh=n,
        )
        legal = game.get_valid_actions(state)
        oracle_top_k = sorted(range(n), key=lambda idx: (-indicators[idx], idx))[:top_k]
        assert legal == oracle_top_k
        assert greedy_action(state.indicators, legal) == legal[0]
        assert greedy_action(state.indicators, list(range(n))) == legal[0]


class TestRunGreedyArm:
    def test_one_point_per_step_non_decreasing_dof_and_its_own_cache(self) -> None:
        config = _config(max_steps=3)
        traj, cache = run_greedy_arm_with_cache(config)
        _, other_cache = run_greedy_arm_with_cache(config)
        assert traj.method == GREEDY_METHOD
        assert [point.level for point in traj.points] == [0, 1, 2, 3]
        assert [point.n_apply_actions for point in traj.points] == [0, 1, 2, 3]
        dofs = traj.dofs()
        assert np.all(np.diff(dofs) >= 0)
        assert np.all(np.isfinite(traj.errors())) and np.all(traj.errors() > 0)
        assert traj.cache_id == id(cache)
        assert other_cache is not cache
        # One unique mesh per committed state: greedy never replays a path.
        assert cache.misses == len(traj.points)
        assert traj.points[-1].n_cache_misses == cache.misses
        assert traj.decisions_diverging_from_greedy is None

    def test_follows_the_head_of_the_ranked_legal_set(self) -> None:
        """Replay by the game's own ranking (``legal[0]`` under top-k) gives the same curve."""
        config = _config(max_steps=3, top_k_actions=4)
        seen: list[int] = []

        def head_of_ranking(_state: SubstrateEpisodeState, legal: list[int]) -> int:
            seen.append(legal[0])
            return legal[0]

        replay = run_arena_episode(
            build_arena_episode(config, name="replay"),
            method=GREEDY_METHOD,
            config=config,
            choose=head_of_ranking,
        )
        assert len(seen) == config.max_steps
        assert _curve(run_greedy_arm(config)) == _curve(replay)

    def test_stops_when_the_game_is_terminal_like_the_mcts_arm(self) -> None:
        """A tolerance above the initial error makes the first state terminal."""
        config = _config(max_steps=3, error_tolerance=0.9)
        greedy = run_greedy_arm(config)
        mcts = run_mcts_arm(config, seed=0)
        assert greedy.points[0].l2_error <= config.error_tolerance
        assert len(greedy.points) == len(mcts.points) == 1

    def test_stops_at_max_dof_like_the_mcts_arm(self) -> None:
        """Initial DOF is 25 on this grid; one step reaches 36, past a budget of 30."""
        config = _config(max_steps=5, max_dof=30)
        greedy = run_greedy_arm(config)
        mcts = run_mcts_arm(config, seed=0)
        assert len(greedy.points) == len(mcts.points) == 2
        assert greedy.points[-1].n_dof >= config.max_dof

    def test_never_constructs_the_search_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """G2, semantically: with the search engine rigged to explode, greedy still runs.

        The import contract catches a *direct* ``src.mcts`` import; this catches
        the indirect route -- greedy delegating to the MCTS arm with one
        simulation -- which that contract cannot see.
        """

        class _Exploding:
            def __init__(self, **_kwargs: object) -> None:
                raise AssertionError("the greedy control constructed the search engine")

        monkeypatch.setattr("src.mcts.search.MCTS", _Exploding)
        config = _config(max_steps=2)
        assert len(run_greedy_arm(config).points) == 3
        with pytest.raises(AssertionError, match="constructed the search engine"):
            run_mcts_arm(config, seed=0)  # control: the rig is live


class TestDivergenceCounter:
    def test_mcts_with_one_simulation_never_diverges(self) -> None:
        """Holds by construction while the legal set is indicator-ranked (0 < top_k < n).

        At an unvisited root every PUCT score is 0 (the exploration term scales
        with sqrt(parent visits)), so the first child -- ``legal[0]``, the
        greedy choice -- wins. At a reused root with one visit, the largest
        prior wins, a monotone softmax of the same indicators.
        """
        config = _config(max_steps=3, n_simulations=1, top_k_actions=4)
        mcts = run_mcts_arm(config, seed=0)
        assert mcts.decisions_diverging_from_greedy == 0
        assert _curve(mcts) == _curve(run_greedy_arm(config))

    def test_divergence_is_not_sufficient_evidence_of_look_ahead(self) -> None:
        """The real engine diverges at one simulation once the legal set is unranked.

        With ``top_k_actions=0`` the legal set is in index order and every PUCT
        score at an unvisited root is 0, so the first child -- the lowest index,
        not the largest indicator -- wins. That divergence is a tie-break, not
        look-ahead. If this starts failing, the engine's root selection changed
        and ``GreedyDivergence``'s docstring is stale.
        """
        config = _config(max_steps=3, n_simulations=1, top_k_actions=0)
        mcts = run_mcts_arm(config, seed=0)
        assert mcts.decisions_diverging_from_greedy is not None
        assert mcts.decisions_diverging_from_greedy >= 1

    def test_a_search_that_avoids_greedy_is_counted_every_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[int, int, int]] = []

        def avoid_greedy(legal: list[int], greedy: int) -> int:
            return next(action for action in legal if action != greedy)

        monkeypatch.setattr("src.mcts.search.MCTS", _stub_search(avoid_greedy, seen))
        mcts = run_mcts_arm(_config(max_steps=3, top_k_actions=4), seed=0)
        assert len(seen) == 3 == len(mcts.points) - 1
        assert all(action != greedy for _first, greedy, action in seen)
        assert mcts.decisions_diverging_from_greedy == 3

    def test_the_reference_is_greedy_not_the_first_legal_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """G3: with no top-k cap the legal set is index-ordered, so legal[0] is not greedy."""
        seen: list[tuple[int, int, int]] = []

        def follow_greedy(_legal: list[int], greedy: int) -> int:
            return greedy

        monkeypatch.setattr("src.mcts.search.MCTS", _stub_search(follow_greedy, seen))
        mcts = run_mcts_arm(_config(max_steps=3, top_k_actions=0), seed=0)
        assert any(first != greedy for first, greedy, _action in seen), (
            "vacuous: legal[0] was greedy at every step, so this cannot tell them apart"
        )
        assert mcts.decisions_diverging_from_greedy == 0

    def test_each_divergence_is_logged_with_both_indicators(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(greedy_module, "logger", recorder)
        state = SubstrateEpisodeState(
            values=np.zeros(3, dtype=np.float32),
            indicators=np.array([0.2, 0.9, 0.4], dtype=np.float32),
            dof=12,
            step=4,
        )
        counter = GreedyDivergence(seed=11)
        counter.record(state, [0, 1, 2], 1)
        assert counter.count == 0
        assert recorder.records == []
        counter.record(state, [0, 1, 2], 2)
        assert counter.count == 1
        (event, fields), *_ = recorder.records
        assert event == "arena_decision_diverged_from_greedy"
        assert fields["seed"] == 11
        assert fields["step"] == 4
        assert fields["action"] == 2
        assert fields["greedy_action"] == 1
        assert fields["action_indicator"] == pytest.approx(0.4)
        assert fields["greedy_indicator"] == pytest.approx(0.9)
        assert fields["dof"] == 12


class TestPrivateCaches:
    def test_a_cache_shared_by_greedy_and_mcts_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shared = FingerprintSolveCache(64)
        monkeypatch.setattr(
            "src.research.amr_arena_episode.FingerprintSolveCache",
            lambda _max_entries: shared,
        )
        with pytest.raises(RuntimeError, match="greedy and mcts seed .* must not share"):
            run_comparison(_config(max_steps=1))

    def test_a_cache_shared_across_mcts_seeds_is_refused(self) -> None:
        shared = FingerprintSolveCache(8)
        caches = [("mcts seed 1", shared), ("mcts seed 2", shared)]
        with pytest.raises(RuntimeError, match="mcts seed 1 and mcts seed 2"):
            _assert_private_caches(_traj("dorfler", [10], [1.0]), caches)

    def test_a_cache_shared_with_dorfler_is_refused(self) -> None:
        cache = FingerprintSolveCache(8)
        dorfler = _traj("dorfler", [10], [1.0])
        dorfler.cache_id = id(cache)
        with pytest.raises(RuntimeError, match="greedy and Dörfler"):
            _assert_private_caches(dorfler, [("greedy", cache)])

    def test_distinct_live_caches_pass(self) -> None:
        live = [("greedy", FingerprintSolveCache(8)), ("mcts seed 1", FingerprintSolveCache(8))]
        _assert_private_caches(_traj("dorfler", [10], [1.0]), live)


@pytest.fixture(scope="module")
def arenas() -> dict[bool, MultiSeedArena]:
    """One two-seed tensor-grid arena with the greedy control, and one without."""
    return {
        include: run_comparison(_config(include_greedy_control=include, n_seeds=2))
        for include in (True, False)
    }


class TestComparisonMetrics:
    def test_every_legacy_key_is_still_published(self, arenas: dict[bool, MultiSeedArena]) -> None:
        for arena in arenas.values():
            assert set(LEGACY_KEYS) <= set(arena.metrics())

    def test_adding_greedy_moves_no_deterministic_legacy_metric(
        self, arenas: dict[bool, MultiSeedArena]
    ) -> None:
        with_greedy, without = arenas[True].metrics(), arenas[False].metrics()
        for key in DETERMINISTIC_LEGACY_KEYS:
            assert with_greedy[key] == without[key], key

    def test_new_keys_are_present_with_the_greedy_control(
        self, arenas: dict[bool, MultiSeedArena]
    ) -> None:
        metrics = arenas[True].metrics()
        assert set(DIVERGENCE_KEYS + GREEDY_RATIO_KEYS) <= set(metrics)
        assert metrics["decisions_diverging_from_greedy"] == 0.0
        assert metrics["l2_error_ratio_mcts_over_greedy_at_matched_dof"] == pytest.approx(1.0)
        # One simulation is greedy here, so greedy/Dörfler is MCTS/Dörfler.
        assert metrics["l2_error_ratio_greedy_over_dorfler_at_matched_dof"] == pytest.approx(
            metrics["l2_error_ratio_at_matched_dof"]
        )

    def test_without_greedy_the_divergence_keys_remain_and_the_ratios_do_not(
        self, arenas: dict[bool, MultiSeedArena]
    ) -> None:
        metrics = arenas[False].metrics()
        assert set(DIVERGENCE_KEYS) <= set(metrics)
        assert set(GREEDY_RATIO_KEYS).isdisjoint(metrics)
        assert arenas[False].greedy is None

    def test_greedy_and_mcts_caches_differ(self, arenas: dict[bool, MultiSeedArena]) -> None:
        arena = arenas[True]
        assert arena.greedy is not None
        ids = [arena.greedy.cache_id, *(item.mcts.cache_id for item in arena.per_seed)]
        assert None not in ids
        assert len(set(ids)) == len(ids)

    def test_median_and_max_divergence_over_seeds(self) -> None:
        per_seed = []
        for seed, count, ratio in ((1, 0, 1.0), (2, 2, 0.8), (3, 5, 1.2)):
            mcts = _traj("mcts", [10, 18], [1.0, 0.6])
            mcts.decisions_diverging_from_greedy = count
            per_seed.append(_seed(seed, mcts, l2_error_ratio_mcts_over_greedy_at_matched_dof=ratio))
        arena = _hand_built_arena(
            per_seed=per_seed,
            seeds=[1, 2, 3],
            greedy=_traj("greedy", [10, 18], [1.0, 0.6]),
            l2_error_ratio_greedy_over_dorfler_at_matched_dof=0.95,
        )
        metrics = arena.metrics()
        assert metrics["decisions_diverging_from_greedy"] == 2.0
        assert metrics["decisions_diverging_from_greedy_max"] == 5.0
        assert metrics["l2_error_ratio_mcts_over_greedy_at_matched_dof"] == pytest.approx(1.0)
        assert metrics["l2_error_ratio_greedy_over_dorfler_at_matched_dof"] == pytest.approx(0.95)

    def test_an_unmeasured_count_is_absent_not_zero(self) -> None:
        metrics = _hand_built_arena().metrics()
        assert set(DIVERGENCE_KEYS).isdisjoint(metrics)
        assert set(GREEDY_RATIO_KEYS).isdisjoint(metrics)

    def test_a_half_built_greedy_arena_is_refused(self) -> None:
        arena = _hand_built_arena(greedy=_traj("greedy", [10, 18], [1.0, 0.6]))
        with pytest.raises(ValueError, match="greedy control ran but"):
            arena.metrics()


class TestArtifacts:
    @pytest.fixture
    def arena(self, arenas: dict[bool, MultiSeedArena]) -> MultiSeedArena:
        return arenas[True]

    def _rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_greedy_rows_are_written_once_with_seed_minus_one(
        self, arena: MultiSeedArena, tmp_path: Path
    ) -> None:
        rows = self._rows(export_csv(arena, tmp_path / "arena.csv"))
        assert {row["method"] for row in rows} == LEGACY_METHODS | {GREEDY_METHOD}
        greedy_rows = [row for row in rows if row["method"] == GREEDY_METHOD]
        assert arena.greedy is not None
        assert len(greedy_rows) == len(arena.greedy.points)
        assert {row["seed"] for row in greedy_rows} == {"-1"}

    def test_without_greedy_the_csv_is_exactly_the_legacy_method_set(self, tmp_path: Path) -> None:
        arena = run_comparison(_config(include_greedy_control=False))
        rows = self._rows(export_csv(arena, tmp_path / "legacy.csv"))
        assert {row["method"] for row in rows} == LEGACY_METHODS
        order = [row["method"] for row in rows]
        assert order == sorted(order, key=["uniform", "dorfler", "mcts"].index)

    def test_export_reproduces_the_committed_artifact_byte_for_byte(self, tmp_path: Path) -> None:
        """The row-writing refactor keeps the committed format, row order and bytes."""
        by_arm: dict[tuple[str, int], ArenaTrajectory] = {}
        for row in self._rows(COMMITTED_CSV):
            key = (row["method"], int(row["seed"]))
            traj = by_arm.setdefault(key, ArenaTrajectory(method=row["method"]))  # type: ignore[arg-type]
            traj.points.append(
                ArenaPoint(
                    level=int(row["level"]),
                    n_dof=int(row["n_dof"]),
                    l2_error=float(row["l2_error"]),
                    wall_time_seconds=float(row["wall_time_seconds"]),
                    n_cache_misses=int(row["n_cache_misses"]),
                    n_cache_hits=int(row["n_cache_hits"]),
                    n_apply_actions=int(row["n_apply_actions"]),
                )
            )
        per_seed = [
            _seed(seed, traj) for (method, seed), traj in by_arm.items() if method == "mcts"
        ]
        arena = _hand_built_arena(
            dorfler=by_arm[("dorfler", -1)],
            uniform=by_arm[("uniform", -1)],
            per_seed=per_seed,
            seeds=[item.seed for item in per_seed],
            # The committed artifact carries the greedy control since its 2026-10-08
            # re-record; an artifact without one round-trips with ``greedy=None``.
            greedy=by_arm.get((GREEDY_METHOD, -1)),
        )
        exported = export_csv(arena, tmp_path / "roundtrip.csv")
        assert exported.read_bytes() == COMMITTED_CSV.read_bytes()

    def test_the_plot_draws_greedy_in_a_distinct_style(
        self, arena: MultiSeedArena, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import matplotlib.pyplot as plt

        figures: list[Any] = []
        close = plt.close

        def keep_and_close(fig: Any = None) -> None:
            figures.append(fig)
            close(fig)

        monkeypatch.setattr(plt, "close", keep_and_close)
        assert export_plot(arena, tmp_path / "arena.png") is not None
        lines = {line.get_label(): line for line in figures[0].axes[0].get_lines()}
        assert {"uniform", "dorfler", "mcts", GREEDY_METHOD} <= set(lines)
        greedy, mcts = lines[GREEDY_METHOD], lines["mcts"]
        assert (greedy.get_linestyle(), greedy.get_marker()) != (
            mcts.get_linestyle(),
            mcts.get_marker(),
        )

    def test_the_manifest_records_the_greedy_arm_and_says_so(
        self, arena: MultiSeedArena, tmp_path: Path
    ) -> None:
        config = _config(n_seeds=2)
        csv_path = export_csv(arena, tmp_path / "arena.csv")
        sidecar = write_arena_manifest(
            arena, config, csv_path, None, git=GitProvenance(sha="abc", dirty=False)
        )
        manifest = load_run_manifest(sidecar)
        arms = {arm.name: arm for arm in manifest.arms}
        assert set(arms) == LEGACY_METHODS | {GREEDY_METHOD}
        greedy = arms[GREEDY_METHOD]
        assert greedy.parameters["selection_rule"] == "single_element_max_residual_indicator"
        assert greedy.parameters["top_k_actions"] == config.top_k_actions
        assert greedy.parameters["cache"] == "private_FingerprintSolveCache"
        assert arena.greedy is not None
        assert greedy.counters["n_levels"] == float(len(arena.greedy.points))
        assert greedy.counters["final_dof"] == float(arena.greedy.points[-1].n_dof)
        assert "Greedy control" in manifest.notes
        assert "decisions_diverging_from_greedy" in manifest.notes
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        assert raw["metrics"]["decisions_diverging_from_greedy"] == 0.0

    def test_without_greedy_the_manifest_has_no_greedy_arm(self, tmp_path: Path) -> None:
        config = _config(include_greedy_control=False)
        arena = run_comparison(config)
        csv_path = export_csv(arena, tmp_path / "legacy.csv")
        manifest = load_run_manifest(
            write_arena_manifest(
                arena, config, csv_path, None, git=GitProvenance(sha="abc", dirty=False)
            )
        )
        assert {arm.name for arm in manifest.arms} == LEGACY_METHODS
        assert "did not run" in manifest.notes


class TestEpisodeGuards:
    def test_the_action_space_check_needs_a_substrate_mesh(self) -> None:
        """Without a mesh there is nothing to count; the game is not even consulted."""
        plain = RefinementState(values=np.zeros(1, np.float32), indicators=np.ones(1, np.float32))
        no_mesh = SubstrateEpisodeState(
            values=np.zeros(1, np.float32), indicators=np.ones(1, np.float32), mesh=None
        )
        for state in (plain, no_mesh):
            assert assert_action_space_covers(None, state) is None  # type: ignore[arg-type]

    def test_a_non_substrate_state_is_refused_before_the_decision(self) -> None:
        """Decision rules are typed on SubstrateEpisodeState; anything else is a wiring bug."""
        decisions: list[int] = []

        class _PlainStateAdapter:
            state = RefinementState(
                values=np.zeros(2, dtype=np.float32),
                indicators=np.ones(2, dtype=np.float32),
            )

            def is_terminal(self) -> bool:
                return False

            def get_legal_actions(self) -> list[int]:
                return [0, 1]

        def choose(_state: SubstrateEpisodeState, legal: list[int]) -> int:
            decisions.append(legal[0])
            return legal[0]

        episode = ArenaEpisode(
            game=None,  # type: ignore[arg-type]
            adapter=_PlainStateAdapter(),  # type: ignore[arg-type]
            cache=FingerprintSolveCache(4),
        )
        with pytest.raises(TypeError, match="SubstrateEpisodeState, got RefinementState"):
            run_arena_episode(episode, method="mcts", config=_config(), choose=choose)
        assert decisions == []
