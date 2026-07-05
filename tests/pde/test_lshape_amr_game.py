"""Tests for the L-shaped Poisson AMR MCTS refinement game.

Covers ``LShapeAMRGame`` (constructor validation, action space, refinement /
re-solve, reward sign, termination, cloning, encoding) and the bundled
``EncodedValueEvaluator``. Fast unit paths inject a trivial deterministic
``solve_fn``; one micro-run uses the real masked solver via ``make_solve_fn``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import PDEConfig, PDEGameConfig, PDEType
from src.pde.game import GamePhase, PDEState
from src.pde.games.lshape_amr import (
    DEFAULT_VALUE_SCALE,
    EncodedValueEvaluator,
    GridSolveResult,
    LShapeAMRGame,
)
from src.pde.geometry import GeometryConfig, GeometryType
from src.pde.operators import LShapedPoissonOperator

# --------------------------------------------------------------------------- #
# Fixtures / builders                                                          #
# --------------------------------------------------------------------------- #


def _operator() -> LShapedPoissonOperator:
    cfg = PDEConfig(
        name="l",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-1.0, -1.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
        geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=1.0),
    )
    return LShapedPoissonOperator(cfg)


def _game_config(**overrides: object) -> PDEGameConfig:
    pde_cfg = PDEConfig(
        name="l",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[-1.0, -1.0],
        domain_max=[1.0, 1.0],
        advection_coeff=[0.0, 0.0],
        geometry=GeometryConfig(geometry_type=GeometryType.L_SHAPED, scale=1.0),
    )
    params: dict[str, object] = {
        "name": "g",
        "pde_config": pde_cfg,
        "game_mode": "mesh_refinement",
        "max_dof": 300,
        "max_steps": 12,
        "error_tolerance": 1e-6,
    }
    params.update(overrides)
    return PDEGameConfig(**params)  # type: ignore[arg-type]


def _fake_solve_fn():  # type: ignore[no-untyped-def]
    """Deterministic solve: DOF = node count, error = 1/DOF, ones indicators."""

    def solve(xs: np.ndarray, ys: np.ndarray) -> GridSolveResult:
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        grid = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float64)
        n_nodes = grid.shape[0]
        nx, ny = len(xs) - 1, len(ys) - 1
        indicators = np.ones((nx, ny), dtype=np.float64)
        return GridSolveResult(
            solution=np.zeros(n_nodes, dtype=np.float64),
            grid=grid,
            l2_error=1.0 / n_nodes,
            n_dof=n_nodes,
            indicators=indicators,
        )

    return solve


def _make_game(**overrides: object) -> LShapeAMRGame:
    kwargs: dict[str, object] = {
        "solve_fn": _fake_solve_fn(),
        "initial_side": 4,
        "n_candidate_elements": 6,
    }
    kwargs.update(overrides)
    return LShapeAMRGame(_operator(), _game_config(), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Constructor validation                                                       #
# --------------------------------------------------------------------------- #


class TestConstructor:
    def test_initial_side_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="initial_side must be >= 1"):
            _make_game(initial_side=0)

    def test_n_candidate_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_candidate_elements must be >= 1"):
            _make_game(n_candidate_elements=0)

    def test_action_space_equals_candidates(self) -> None:
        game = _make_game(n_candidate_elements=5)
        assert game.action_space_size == 5

    def test_state_channels_is_one(self) -> None:
        assert _make_game().state_channels == 1

    def test_default_value_scale(self) -> None:
        game = _make_game()
        assert game._value_scale == DEFAULT_VALUE_SCALE


# --------------------------------------------------------------------------- #
# Lifecycle: solve / actions / refinement                                      #
# --------------------------------------------------------------------------- #


class TestLifecycle:
    def test_initial_state_shape(self) -> None:
        game = _make_game()
        state = game.get_initial_state()
        assert state.step == 0
        assert state.phase == GamePhase.INITIAL
        assert state.dof > 0
        assert np.isfinite(state.error_estimate)

    def test_valid_actions_bounded_by_candidates(self) -> None:
        game = _make_game(n_candidate_elements=3)
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        assert len(actions) <= 3
        assert actions == list(range(len(actions)))

    def test_action_mask_matches_valid_actions(self) -> None:
        game = _make_game(n_candidate_elements=6)
        state = game.get_initial_state()
        mask = game.get_action_mask(state)
        assert mask.shape == (6,)
        assert mask.sum() == len(game.get_valid_actions(state))

    def test_apply_action_grows_dof_and_resolves(self) -> None:
        game = _make_game()
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        assert new_state.dof > state.dof
        assert new_state.step == state.step + 1
        assert new_state.history == [0]
        assert new_state.phase == GamePhase.REFINING

    def test_illegal_action_raises(self) -> None:
        game = _make_game(n_candidate_elements=6)
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="Illegal action"):
            game.apply_action(state, 999)

    def test_reward_sign_is_positive_on_efficiency_gain(self) -> None:
        game = _make_game()
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        reward = game.get_reward(new_state, state)
        # error-per-DOF drops (1/n falls, DOF rises) => positive reward.
        assert reward > 0.0

    def test_bisect_edge_out_of_range_noop(self) -> None:
        axis = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        # edge == len-1 is out of range -> returned unchanged.
        out = LShapeAMRGame._bisect_edge(axis, 2, 1e-12)
        np.testing.assert_array_equal(out, axis)

    def test_bisect_edge_duplicate_guard(self) -> None:
        axis = np.array([0.0, 1.0], dtype=np.float64)
        first = LShapeAMRGame._bisect_edge(axis, 0, 1e-12)
        assert first.size == 3  # midpoint inserted
        # A huge merge tolerance suppresses the insertion.
        again = LShapeAMRGame._bisect_edge(axis, 0, 10.0)
        np.testing.assert_array_equal(again, axis)


# --------------------------------------------------------------------------- #
# Termination                                                                  #
# --------------------------------------------------------------------------- #


class TestTermination:
    def _state(self, **overrides: object) -> PDEState:
        base: dict[str, object] = {
            "coords": np.zeros((4, 2), dtype=np.float32),
            "solution": np.zeros(4, dtype=np.float32),
            "residuals": np.zeros(4, dtype=np.float32),
            "error_estimate": 1.0,
            "dof": 10,
            "step": 1,
        }
        base.update(overrides)
        return PDEState(**base)  # type: ignore[arg-type]

    def test_terminal_on_dof_budget(self) -> None:
        game = _make_game()
        game.get_initial_state()
        assert game.is_terminal(self._state(dof=300))

    def test_terminal_on_step_cap(self) -> None:
        game = _make_game()
        game.get_initial_state()
        assert game.is_terminal(self._state(step=12))

    def test_terminal_on_tolerance(self) -> None:
        game = _make_game()
        game.get_initial_state()
        assert game.is_terminal(self._state(error_estimate=1e-9))

    def test_terminal_on_no_valid_actions(self) -> None:
        game = _make_game()
        game.get_initial_state()
        # Wipe the cached indicators so no element is refinable.
        game._last_indicators = np.zeros((1, 1), dtype=np.float64)
        assert game.is_terminal(self._state(dof=10, step=1, error_estimate=1.0))

    def test_non_terminal_mid_episode(self) -> None:
        game = _make_game()
        game.get_initial_state()
        assert not game.is_terminal(self._state(dof=10, step=1, error_estimate=1.0))


# --------------------------------------------------------------------------- #
# Encoding + clone                                                             #
# --------------------------------------------------------------------------- #


class TestEncodingAndClone:
    def test_to_tensor_in_range(self) -> None:
        game = _make_game()
        state = game.get_initial_state()
        refined = game.apply_action(state, 0)
        tensor = game.to_tensor(refined)
        assert tensor.shape == (1, 1, 1)
        value = float(tensor.reshape(-1)[0])
        assert -1.0 <= value <= 1.0

    def test_to_tensor_without_initial_epd(self) -> None:
        """`to_tensor` falls back gracefully when no initial epd was recorded."""
        game = _make_game()
        state = game.get_initial_state()
        game._initial_epd = None  # force the fallback branch
        tensor = game.to_tensor(state)
        assert -1.0 <= float(tensor.reshape(-1)[0]) <= 1.0

    def test_clone_is_independent(self) -> None:
        game = _make_game()
        game.get_initial_state()
        original_xs = game._xs.copy()
        clone = game.clone()

        # Mutating the clone's grid must not affect the original.
        state = clone.get_initial_state()
        clone.apply_action(state, 0)
        assert clone._xs.size >= original_xs.size
        np.testing.assert_array_equal(game._xs, original_xs)

    def test_get_result_and_exact_error(self) -> None:
        game = _make_game()
        state = game.get_initial_state()
        result = game.get_result(state, [state.error_estimate])
        assert result.final_dof == state.dof
        errs = game.compute_exact_error(state)
        assert set(errs) == {"l2", "h1", "linf", "residual"}


# --------------------------------------------------------------------------- #
# EncodedValueEvaluator                                                        #
# --------------------------------------------------------------------------- #


class TestEncodedValueEvaluator:
    def test_n_actions_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_actions must be >= 1"):
            EncodedValueEvaluator(n_actions=0)

    def test_uniform_policy_over_legal(self) -> None:
        ev = EncodedValueEvaluator(n_actions=4)
        state = np.array([[[0.5]]], dtype=np.float32)
        res = ev.evaluate(state, [0, 2])
        assert res.policy[0] == pytest.approx(0.5)
        assert res.policy[2] == pytest.approx(0.5)
        assert res.policy[1] == 0.0
        assert res.policy[3] == 0.0

    def test_value_reads_state_zero(self) -> None:
        ev = EncodedValueEvaluator(n_actions=2)
        res = ev.evaluate(np.array([0.3, 9.9], dtype=np.float32), [0, 1])
        assert res.value == pytest.approx(0.3)

    def test_value_clamped_high_and_low(self) -> None:
        ev = EncodedValueEvaluator(n_actions=2)
        assert ev.evaluate(np.array([5.0], dtype=np.float32), [0]).value == 1.0
        assert ev.evaluate(np.array([-5.0], dtype=np.float32), [0]).value == -1.0

    def test_empty_state_value_zero(self) -> None:
        ev = EncodedValueEvaluator(n_actions=2)
        res = ev.evaluate(np.array([], dtype=np.float32), [0])
        assert res.value == 0.0

    def test_no_legal_actions_zero_policy(self) -> None:
        ev = EncodedValueEvaluator(n_actions=3)
        res = ev.evaluate(np.array([0.1], dtype=np.float32), [])
        assert np.all(res.policy == 0.0)

    def test_evaluate_batch(self) -> None:
        ev = EncodedValueEvaluator(n_actions=2)
        out = ev.evaluate_batch(
            [np.array([0.2], dtype=np.float32), np.array([0.8], dtype=np.float32)],
            [[0], [1]],
        )
        assert len(out) == 2
        assert out[0].value == pytest.approx(0.2)
        assert out[1].policy[1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Real micro-run against the masked solver                                     #
# --------------------------------------------------------------------------- #


class TestRealSolveMicroRun:
    def test_real_solve_grows_dof(self) -> None:
        from src.research.lshape_amr_compare import (
            lshape_inside_predicate,
            make_solve_fn,
        )

        pytest.importorskip("scipy", reason="scipy required for the real solver")
        op = _operator()
        solve_fn = make_solve_fn(op, lshape_inside_predicate(1.0))
        game = LShapeAMRGame(
            op,
            _game_config(),
            solve_fn=solve_fn,
            initial_side=4,
            n_candidate_elements=4,
        )
        state = game.get_initial_state()
        assert np.isfinite(state.error_estimate)
        assert state.dof > 0

        actions = game.get_valid_actions(state)
        assert actions, "the coarse L-shape must offer a refinable element"
        new_state = game.apply_action(state, actions[0])
        assert new_state.dof >= state.dof
