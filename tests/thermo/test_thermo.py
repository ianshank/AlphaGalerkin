"""Tests for the λ-window scheduling ablation (mechanics, not the headline).

The headline MCTS-vs-greedy result is a **negative** one (see
``specs/lambda_scheduling.spec.md``); these tests gate the *mechanics* that make
that result trustworthy: deterministic apply_action, monotone-under-allocate /
non-monotone-under-split, surrogate correctness, config validation, and that the
three schedulers run at a matched budget.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.thermo.config import (
    HardnessProfileConfig,
    LambdaSchedulingConfig,
    SchedulingParams,
)
from src.thermo.game import LambdaSchedulingGame, total_stderr
from src.thermo.outer_loop import (
    run_bias_sweep,
    run_comparison,
    run_greedy,
    run_uniform,
)
from src.thermo.surrogate import (
    AnalyticSurrogate,
    MismatchedSurrogate,
    OperatorSurrogate,
    RecordedSurrogate,
    VarianceSurrogate,
)


def _truth() -> AnalyticSurrogate:
    return AnalyticSurrogate(HardnessProfileConfig(name="h"))


def _game(**overrides: object) -> LambdaSchedulingGame:
    params = SchedulingParams(**overrides)  # type: ignore[arg-type]
    return LambdaSchedulingGame(params, _truth())


# --------------------------------------------------------------------------- #
# Surrogates                                                                  #
# --------------------------------------------------------------------------- #


class TestSurrogates:
    def test_analytic_positive_and_protocol(self) -> None:
        s = _truth()
        assert isinstance(s, VarianceSurrogate)
        assert s.variance_coeff(0.0, 0.2) > 0.0

    def test_analytic_zero_width(self) -> None:
        assert _truth().variance_coeff(0.5, 0.5) == 0.0

    def test_mismatched_scales_by_bias(self) -> None:
        truth = _truth()
        base = truth.variance_coeff(0.4, 0.6)
        mis = MismatchedSurrogate(truth, bias=0.25)
        assert mis.variance_coeff(0.4, 0.6) == pytest.approx(base * 1.25)

    def test_mismatched_noise_deterministic(self) -> None:
        mis = MismatchedSurrogate(_truth(), bias=0.0, noise_amplitude=0.1)
        a = mis.variance_coeff(0.3, 0.5)
        b = mis.variance_coeff(0.3, 0.5)
        assert a == b  # no RNG

    def test_recorded_nearest_and_empty(self) -> None:
        table = [(0.0, 0.5, 2.0), (0.5, 1.0, 5.0)]
        rec = RecordedSurrogate(table)
        assert rec.variance_coeff(0.1, 0.2) == 2.0
        assert rec.variance_coeff(0.8, 0.9) == 5.0
        with pytest.raises(ValueError):
            RecordedSurrogate([])

    def test_operator_requires_fit(self) -> None:
        with pytest.raises(NotImplementedError):
            OperatorSurrogate()
        op = OperatorSurrogate(predict_fn=lambda lo, hi: 3.0)
        assert op.variance_coeff(0.0, 1.0) == 3.0


# --------------------------------------------------------------------------- #
# Determinism (the F2 invariant)                                             #
# --------------------------------------------------------------------------- #


class TestDeterminism:
    @settings(max_examples=25, deadline=None)
    @given(seq=st.lists(st.integers(min_value=0, max_value=11), min_size=1, max_size=8))
    def test_apply_action_is_deterministic(self, seq: list[int]) -> None:
        game = _game()

        def replay() -> np.ndarray:
            state = game.get_initial_state()
            for a in seq:
                if a in game.get_valid_actions(state):
                    state = game.apply_action(state, a)
            return state.values

        assert np.array_equal(replay(), replay())


# --------------------------------------------------------------------------- #
# Monotone-under-allocate / non-monotone-under-split                          #
# --------------------------------------------------------------------------- #


class TestMonotonicity:
    def test_allocate_never_increases_error(self) -> None:
        game = _game()
        state = game.get_initial_state()
        for _ in range(10):
            valid = [a for a in game.get_valid_actions(state) if a < game.params.max_windows]
            if not valid:
                break
            nxt = game.apply_action(state, valid[0])
            assert nxt.error_estimate <= state.error_estimate + 1e-9
            state = nxt

    def test_split_can_increase_error(self) -> None:
        """A split with credit 0.5 conserves samples and raises total variance."""
        game = _game(sample_split_credit=0.5)
        state = game.get_initial_state()
        maxw = game.params.max_windows
        split_actions = [a for a in game.get_valid_actions(state) if a >= maxw]
        assert split_actions, "expected split actions to be available"
        after = game.apply_action(state, split_actions[0])
        assert after.error_estimate > state.error_estimate

    def test_split_reachable_non_monotone(self) -> None:
        """Non-monotonicity must be *reachable*, not just theoretical."""
        game = _game()
        state = game.get_initial_state()
        maxw = game.params.max_windows
        increased = False
        for a in game.get_valid_actions(state):
            if a >= maxw:
                if game.apply_action(state, a).error_estimate > state.error_estimate:
                    increased = True
                    break
        assert increased


# --------------------------------------------------------------------------- #
# Game interface                                                              #
# --------------------------------------------------------------------------- #


class TestGame:
    def test_action_space_size(self) -> None:
        game = _game(max_windows=16)
        assert game.action_space_size == 32

    def test_initial_windows_partition_unit_interval(self) -> None:
        game = _game(n_initial_windows=4)
        w = game.get_initial_state().values.reshape(-1, 3)
        assert w.shape[0] == 4
        assert w[0, 0] == pytest.approx(0.0)
        assert w[-1, 1] == pytest.approx(1.0)

    def test_terminal_on_budget_exhaustion(self) -> None:
        game = _game(n_initial_windows=4, batch_samples=100, sample_budget=400)
        state = game.get_initial_state()
        # Initial already uses 400 samples → no allocation possible.
        assert game.is_terminal(state)

    def test_get_reward_and_winner(self) -> None:
        game = _game()
        state = game.get_initial_state()
        valid = [a for a in game.get_valid_actions(state) if a < game.params.max_windows]
        nxt = game.apply_action(state, valid[0])
        # Allocation reduces error → reward positive minus small cost.
        assert game.get_reward(nxt, state) > -1.0
        assert game.get_winner(state) in (-1, 1)

    def test_total_stderr_ignores_empty_windows(self) -> None:
        w = np.array([[0.0, 0.5, 0.0], [0.5, 1.0, 100.0]], dtype=np.float64)
        # Zero-sample window contributes nothing (no div-by-zero).
        assert total_stderr(w, _truth()) > 0.0


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_kcal_tolerance_rejected_when_too_small(self) -> None:
        with pytest.raises(ValueError):
            LambdaSchedulingConfig(name="c", error_tolerance=1e-4)

    def test_primary_bias_must_be_in_sweep(self) -> None:
        with pytest.raises(ValueError):
            LambdaSchedulingConfig(name="c", surrogate_bias_sweep=[0.0, 0.1], primary_bias=0.25)

    def test_max_windows_consistency(self) -> None:
        with pytest.raises(ValueError):
            LambdaSchedulingConfig(name="c", n_initial_windows=10, max_windows=4)

    def test_to_params_and_thresholds(self) -> None:
        cfg = LambdaSchedulingConfig(name="c", primary_bias=0.25)
        params = cfg.to_params()
        assert isinstance(params, SchedulingParams)
        names = {t.name for t in cfg.get_default_thresholds()}
        assert "dG_stderr_ratio_at_bias_0p25_median" in names
        assert "dG_stderr_ratio_mcts_over_greedy_median" in names


# --------------------------------------------------------------------------- #
# Schedulers + harness                                                        #
# --------------------------------------------------------------------------- #


class TestSchedulers:
    def test_greedy_and_uniform_respect_budget(self) -> None:
        game = _game(n_initial_windows=4, batch_samples=100, sample_budget=1200)
        for runner in (run_greedy, run_uniform):
            terminal = runner(game)
            used = terminal.values.reshape(-1, 3)[:, 2].sum()
            assert used <= game.params.sample_budget + 1e-6
            assert game.is_terminal(terminal)

    def test_greedy_beats_uniform_on_peaked_profile(self) -> None:
        """Sanity: at matched *full* budget, variance-weighted greedy beats uniform.

        A low tolerance makes the sample budget (not the convergence tolerance)
        the binding constraint, so both spend the same budget and the comparison
        reflects allocation quality, not who crossed the tolerance first.
        """
        game = _game(
            n_initial_windows=6,
            batch_samples=100,
            sample_budget=2000,
            error_tolerance=0.001,
        )
        g = run_greedy(game)
        u = run_uniform(game)
        assert g.error_estimate <= u.error_estimate + 1e-9

    def test_run_comparison_returns_ratios(self) -> None:
        truth = _truth()
        params = SchedulingParams(
            n_initial_windows=4, batch_samples=100, sample_budget=1200, max_steps=8
        )
        result = run_comparison(truth, truth, params, seed=0, bias=0.0, n_simulations=4, c_puct=1.4)
        assert result.ratio_mcts_over_greedy > 0.0
        assert result.ratio_mcts_over_uniform > 0.0

    def test_bias_sweep_aggregates(self) -> None:
        truth = _truth()
        params = SchedulingParams(
            n_initial_windows=4, batch_samples=100, sample_budget=1000, max_steps=6
        )

        def make_planner(bias: float) -> VarianceSurrogate:
            return truth if bias == 0.0 else MismatchedSurrogate(truth, bias=bias)

        cells = run_bias_sweep(
            truth,
            make_planner,
            params,
            biases=[0.0, 0.25],
            base_seed=1,
            n_seeds=2,
            n_simulations=4,
            c_puct=1.4,
        )
        assert len(cells) == 2
        for cell in cells:
            m = cell.metrics()
            assert set(m) >= {"median_ratio", "win_fraction", "ratio_min", "ratio_max"}
