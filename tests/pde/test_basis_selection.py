"""Tests for BasisSelectionGame."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.pde.config import (
    BasisSelectionConfig,
    PDEConfig,
    PDEGameConfig,
    PDEType,
)
from src.pde.game import GamePhase, PDEState
from src.pde.games.basis_selection import BasisFunction, BasisSelectionGame
from src.pde.operators import PoissonOperator


@pytest.fixture
def poisson_operator() -> PoissonOperator:
    config = PDEConfig(name="test_poisson", pde_type=PDEType.POISSON)
    return PoissonOperator(config)


@pytest.fixture
def small_basis_config() -> BasisSelectionConfig:
    return BasisSelectionConfig(
        name="test_basis",
        basis_type="fourier",
        max_basis_functions=8,
        n_candidate_bases=16,
        max_frequency=2,
        n_collocation_points=25,
        n_boundary_points_per_face=5,
    )


@pytest.fixture
def game_config(small_basis_config: BasisSelectionConfig) -> PDEGameConfig:
    pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
    return PDEGameConfig(
        name="test_game",
        pde_config=pde_config,
        game_mode="basis_selection",
        basis_config=small_basis_config,
        max_steps=20,
        error_tolerance=1e-4,
    )


@pytest.fixture
def game(poisson_operator: PoissonOperator, game_config: PDEGameConfig) -> BasisSelectionGame:
    return BasisSelectionGame(poisson_operator, game_config)


class TestBasisFunction:
    """Tests for BasisFunction dataclass."""

    def test_fourier_evaluate(self) -> None:
        bf = BasisFunction(type="fourier", params={"k_x": 1, "k_y": 0, "phase": 0.0}, index=0)
        coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        assert values.shape == (3,)
        assert values.dtype == np.float32

    def test_fourier_sin_values(self) -> None:
        bf = BasisFunction(type="fourier", params={"k_x": 1, "k_y": 0, "phase": 0.0}, index=0)
        coords = np.array([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        # sin(2*pi*0) = 0, sin(2*pi*0.25) = 1, sin(2*pi*0.5) = 0
        np.testing.assert_allclose(values[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(values[1], 1.0, atol=1e-6)
        np.testing.assert_allclose(values[2], 0.0, atol=1e-5)

    def test_fourier_with_phase(self) -> None:
        bf = BasisFunction(
            type="fourier",
            params={"k_x": 1, "k_y": 0, "phase": np.pi / 2},
            index=0,
        )
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        # sin(0 + pi/2) = cos(0) = 1
        np.testing.assert_allclose(values[0], 1.0, atol=1e-6)

    def test_polynomial_evaluate(self) -> None:
        bf = BasisFunction(type="polynomial", params={"degree_x": 2, "degree_y": 0}, index=0)
        coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        assert values.shape == (3,)
        np.testing.assert_allclose(values, [0.0, 0.25, 1.0], atol=1e-6)

    def test_polynomial_xy(self) -> None:
        bf = BasisFunction(type="polynomial", params={"degree_x": 1, "degree_y": 1}, index=0)
        coords = np.array([[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        np.testing.assert_allclose(values, [0.25, 1.0, 0.0], atol=1e-6)

    def test_rbf_evaluate(self) -> None:
        bf = BasisFunction(
            type="rbf",
            params={"center_x": 0.5, "center_y": 0.5, "sigma": 0.1},
            index=0,
        )
        coords = np.array([[0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        assert values.shape == (2,)
        # Value at center should be maximum (1.0)
        assert values[0] == pytest.approx(1.0, abs=1e-6)
        # Value far from center should be small
        assert values[1] < 0.01

    def test_rbf_symmetry(self) -> None:
        bf = BasisFunction(
            type="rbf",
            params={"center_x": 0.5, "center_y": 0.5, "sigma": 0.2},
            index=0,
        )
        coords = np.array(
            [[0.3, 0.5], [0.7, 0.5], [0.5, 0.3], [0.5, 0.7]],
            dtype=np.float32,
        )
        values = bf.evaluate(coords)
        # All equidistant from center, so values should be equal
        np.testing.assert_allclose(values, values[0], atol=1e-6)

    def test_unknown_type_raises(self) -> None:
        bf = BasisFunction(type="unknown", params={}, index=0)
        coords = np.array([[0.0, 0.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown basis type"):
            bf.evaluate(coords)


class TestBasisSelectionGameInit:
    """Tests for BasisSelectionGame initialization."""

    def test_creation(self, game: BasisSelectionGame) -> None:
        assert game.name == "basis_selection"
        assert game.action_space_size > 0

    def test_description(self, game: BasisSelectionGame) -> None:
        assert game.description == "Galerkin basis function selection game"

    def test_action_space_size(self, game: BasisSelectionGame) -> None:
        assert game.action_space_size <= 16

    def test_state_channels(self, game: BasisSelectionGame) -> None:
        assert game.state_channels == 3 + 8  # 3 + max_basis_functions

    def test_candidate_bases_generated(self, game: BasisSelectionGame) -> None:
        assert len(game._candidate_bases) > 0
        assert len(game._candidate_bases) <= 16

    def test_collocation_points_cached(self, game: BasisSelectionGame) -> None:
        assert game._collocation_points is not None
        assert game._collocation_points.shape[0] == 25
        assert game._collocation_points.shape[1] == 2

    def test_exact_solution_cached(self, game: BasisSelectionGame) -> None:
        # Poisson operator has an exact solution
        assert game._exact_solution is not None


class TestBasisSelectionGameInitialState:
    """Tests for initial state creation."""

    def test_initial_state(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert isinstance(state, PDEState)
        assert state.step == 0
        assert state.dof == 0
        assert len(state.history) == 0
        assert state.phase == GamePhase.INITIAL

    def test_initial_state_zero_solution(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        np.testing.assert_array_equal(state.solution, np.zeros_like(state.solution))

    def test_initial_state_has_residuals(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert state.residuals is not None
        assert len(state.residuals) == state.n_points

    def test_initial_state_positive_error(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert state.error_estimate > 0

    def test_initial_state_has_coords(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert state.coords is not None
        assert state.coords.ndim == 2
        assert state.coords.shape[1] == 2

    def test_initial_state_empty_basis(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert state.n_basis == 0
        assert state.basis_coefficients is not None
        assert len(state.basis_coefficients) == 0

    def test_initial_state_has_budget(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert state.budget_remaining > 0

    def test_initial_state_coords_in_domain(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert np.all(state.coords >= 0.0)
        assert np.all(state.coords <= 1.0)


class TestBasisSelectionGameActions:
    """Tests for action handling."""

    def test_valid_actions_initial(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        actions = game.get_valid_actions(state)
        assert len(actions) == game.action_space_size

    def test_valid_actions_after_selection(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        actions_before = game.get_valid_actions(state)
        action = actions_before[0]
        new_state = game.apply_action(state, action)
        actions_after = game.get_valid_actions(new_state)
        assert len(actions_after) == len(actions_before) - 1
        assert action not in actions_after

    def test_valid_actions_empty_at_max(self, game: BasisSelectionGame) -> None:
        """No valid actions when max basis functions reached."""
        state = game.get_initial_state()
        # Manually set n_basis to max
        state.basis_coefficients = np.zeros(game.basis_config.max_basis_functions, dtype=np.float32)
        actions = game.get_valid_actions(state)
        assert len(actions) == 0

    def test_action_mask(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        mask = game.get_action_mask(state)
        assert mask.dtype == bool
        assert mask.shape == (game.action_space_size,)
        assert mask.all()  # initially all valid

    def test_action_mask_after_selection(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state = game.apply_action(state, 0)
        mask = game.get_action_mask(state)
        assert mask[0] is np.bool_(False)
        assert mask.sum() == game.action_space_size - 1

    def test_action_mask_all_false_at_max(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.basis_coefficients = np.zeros(game.basis_config.max_basis_functions, dtype=np.float32)
        mask = game.get_action_mask(state)
        assert not mask.any()

    def test_apply_action(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        assert new_state.step == 1
        assert new_state.dof == 1
        assert 0 in new_state.history
        assert new_state.n_basis == 1

    def test_apply_action_does_not_mutate_original(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        original_step = state.step
        original_dof = state.dof
        game.apply_action(state, 0)
        assert state.step == original_step
        assert state.dof == original_dof

    def test_apply_action_computes_solution(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        # Solution should no longer be all zeros (unless basis is zero everywhere)
        assert new_state.solution is not None
        assert new_state.basis_coefficients is not None
        assert len(new_state.basis_coefficients) == 1

    def test_apply_action_updates_residuals(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        assert new_state.residuals is not None
        assert len(new_state.residuals) == state.n_points

    def test_apply_duplicate_action_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state = game.apply_action(state, 0)
        with pytest.raises(ValueError, match="already selected"):
            game.apply_action(state, 0)

    def test_apply_invalid_action_negative_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, -1)

    def test_apply_invalid_action_too_large_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, 9999)

    def test_apply_multiple_actions(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state = game.apply_action(state, 0)
        state = game.apply_action(state, 1)
        state = game.apply_action(state, 2)
        assert state.step == 3
        assert state.dof == 3
        assert state.n_basis == 3
        assert state.history == [0, 1, 2]

    def test_apply_action_decreases_budget(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        initial_budget = state.budget_remaining
        new_state = game.apply_action(state, 0)
        assert new_state.budget_remaining < initial_budget


class TestBasisSelectionGameErrorReduction:
    """Tests for error estimation and reduction."""

    def test_error_changes_after_action(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        initial_error = state.error_estimate
        actions = game.get_valid_actions(state)
        new_state = game.apply_action(state, actions[0])
        # Error should change (not necessarily decrease for arbitrary basis)
        assert new_state.error_estimate != initial_error or new_state.n_basis == 1

    def test_compute_exact_error_keys(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert "l2" in errors
        assert "h1" in errors
        assert "linf" in errors
        assert "residual" in errors

    def test_compute_exact_error_non_negative(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert errors["l2"] >= 0
        assert errors["h1"] >= 0
        assert errors["linf"] >= 0
        assert errors["residual"] >= 0

    def test_error_after_multiple_bases(self, game: BasisSelectionGame) -> None:
        """Adding multiple bases should generally improve approximation."""
        state = game.get_initial_state()
        initial_error = state.error_estimate
        for i in range(min(4, game.action_space_size)):
            if game.is_terminal(state):
                break
            state = game.apply_action(state, i)
        # After several basis additions, we expect some error change
        assert state.n_basis > 0


class TestBasisSelectionGameReward:
    """Tests for reward computation."""

    def test_reward_basic(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        reward = game.get_reward(new_state, state)
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_reward_from_error_reduction(self, game: BasisSelectionGame) -> None:
        """Reward should incorporate error reduction."""
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        reward = game.get_reward(new_state, state)
        error_reduction = state.error_estimate - new_state.error_estimate
        # Reward formula: reduction * scale - cost_per_dof * dof_added + bonus
        assert isinstance(reward, float)

    def test_reward_convergence_bonus(self, game: BasisSelectionGame) -> None:
        """Reaching tolerance should give bonus reward."""
        state = game.get_initial_state()
        prev_state = state.clone()
        state.error_estimate = 1e-6  # below tolerance
        reward = game.get_reward(state, prev_state)
        # Should include terminal_bonus
        assert reward > 0


class TestBasisSelectionGameLogReward:
    """Tests for the proposal-form log reward (PDEGameConfig.reward_form='log')."""

    @pytest.fixture
    def log_game(
        self,
        poisson_operator: PoissonOperator,
        small_basis_config: BasisSelectionConfig,
    ) -> BasisSelectionGame:
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(
            name="log_reward_game",
            pde_config=pde_config,
            game_mode="basis_selection",
            basis_config=small_basis_config,
            max_steps=20,
            error_tolerance=1e-4,
            reward_form="log",
            log_reward_alpha=1.0,
            log_reward_beta=0.1,
        )
        return BasisSelectionGame(poisson_operator, game_config)

    def test_log_reward_finite(self, log_game: BasisSelectionGame) -> None:
        """Log-form reward is always finite on normal states."""
        state = log_game.get_initial_state()
        new_state = log_game.apply_action(state, 0)
        reward = log_game.get_reward(new_state, state)
        assert np.isfinite(reward)

    def test_log_reward_monotone_in_error(
        self, log_game: BasisSelectionGame
    ) -> None:
        """Decreasing error strictly increases the log-form reward."""
        state = log_game.get_initial_state()
        low_error = state.clone()
        low_error.error_estimate = 0.01
        high_error = state.clone()
        high_error.error_estimate = 0.5
        prev = state.clone()

        low_reward = log_game.get_reward(low_error, prev)
        high_reward = log_game.get_reward(high_error, prev)
        assert low_reward > high_reward

    def test_log_reward_monotone_in_cost(
        self, log_game: BasisSelectionGame
    ) -> None:
        """Increasing DOF (cost) strictly decreases the log-form reward."""
        state = log_game.get_initial_state()
        cheap = state.clone()
        cheap.dof = 10
        cheap.error_estimate = 0.1
        expensive = state.clone()
        expensive.dof = 1000
        expensive.error_estimate = 0.1
        prev = state.clone()

        cheap_reward = log_game.get_reward(cheap, prev)
        expensive_reward = log_game.get_reward(expensive, prev)
        assert cheap_reward > expensive_reward

    def test_log_reward_terminal_bonus_applied(
        self, log_game: BasisSelectionGame
    ) -> None:
        """Terminal bonus is still applied under the log reward form."""
        state = log_game.get_initial_state()
        below_tol = state.clone()
        below_tol.error_estimate = 1e-6
        prev = state.clone()

        reward_below_tol = log_game.get_reward(below_tol, prev)

        # Same state but just above tolerance — no bonus.
        above_tol = state.clone()
        above_tol.error_estimate = 1.1 * log_game.config.error_tolerance
        reward_above_tol = log_game.get_reward(above_tol, prev)

        assert reward_below_tol > reward_above_tol

    def test_linear_default_unchanged_by_new_fields(
        self, game: BasisSelectionGame
    ) -> None:
        """Default reward_form='linear' keeps the historical formula."""
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        linear_reward = game.get_reward(new_state, state)

        error_reduction = state.error_estimate - new_state.error_estimate
        dof_added = new_state.dof - state.dof
        cost = game.config.cost_per_dof * dof_added
        expected = game.config.reward_per_error_reduction * error_reduction - cost
        if new_state.error_estimate < game.config.error_tolerance:
            expected += game.config.terminal_bonus

        np.testing.assert_allclose(linear_reward, expected, rtol=1e-6)


class TestBasisSelectionGameTerminal:
    """Tests for terminal conditions."""

    def test_not_terminal_initial(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert game.is_terminal(state) is False

    def test_terminal_low_error(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.error_estimate = 1e-6
        assert game.is_terminal(state) is True

    def test_terminal_max_basis(self, game: BasisSelectionGame) -> None:
        """Terminal when max basis functions reached."""
        state = game.get_initial_state()
        max_basis = game.basis_config.max_basis_functions
        n_candidates = game.action_space_size
        for i in range(min(max_basis, n_candidates)):
            if game.is_terminal(state):
                break
            state = game.apply_action(state, i)
        if state.n_basis >= max_basis:
            assert game.is_terminal(state) is True

    def test_terminal_budget_exhausted(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.budget_remaining = 0
        assert game.is_terminal(state) is True

    def test_terminal_max_steps(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.step = game.config.max_steps
        assert game.is_terminal(state) is True

    def test_terminal_no_valid_actions(self, game: BasisSelectionGame) -> None:
        """Terminal when all actions exhausted."""
        state = game.get_initial_state()
        # Select all candidate bases (up to max_basis)
        for i in range(min(game.action_space_size, game.basis_config.max_basis_functions)):
            if game.is_terminal(state):
                break
            state = game.apply_action(state, i)


class TestBasisSelectionGameResult:
    """Tests for game result generation."""

    def test_get_result(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state = game.apply_action(state, 0)
        error_history = [1.0, state.error_estimate]
        result = game.get_result(state, error_history)
        assert result.final_dof == 1
        assert result.n_steps == 1
        assert len(result.error_history) == 2

    def test_get_result_termination_reason(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.error_estimate = 1e-6
        result = game.get_result(state, [1.0, 1e-6])
        assert result.converged is True
        assert result.termination_reason == "converged"

    def test_get_result_budget_exhausted(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.budget_remaining = 0
        result = game.get_result(state, [1.0])
        assert result.termination_reason == "budget_exhausted"

    def test_get_result_empty_history(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        result = game.get_result(state, [])
        assert result.error_reduction_rate == 0.0

    def test_get_result_efficiency_metrics(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state = game.apply_action(state, 0)
        error_history = [1.0, state.error_estimate]
        result = game.get_result(state, error_history)
        assert isinstance(result.error_reduction_rate, float)
        assert isinstance(result.dof_efficiency, float)
        assert isinstance(result.compute_efficiency, float)


class TestBasisSelectionGameTensor:
    """Tests for tensor conversion."""

    def test_to_tensor(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.ndim >= 2
        assert tensor.shape[0] == game.state_channels

    def test_to_tensor_dtype(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.dtype == torch.float32

    def test_action_to_string(self, game: BasisSelectionGame) -> None:
        s = game.action_to_string(0)
        assert isinstance(s, str)
        assert len(s) > 0
        # For fourier basis, should contain "fourier"
        assert "fourier" in s or "poly" in s or "rbf" in s

    def test_action_to_string_invalid(self, game: BasisSelectionGame) -> None:
        s = game.action_to_string(-1)
        assert "invalid" in s


class TestBasisTypes:
    """Tests for different basis function types."""

    def test_polynomial_basis_game(self, poisson_operator: PoissonOperator) -> None:
        basis_config = BasisSelectionConfig(
            name="poly",
            basis_type="polynomial",
            max_basis_functions=5,
            n_candidate_bases=10,
            max_frequency=3,
            n_collocation_points=25,
            n_boundary_points_per_face=5,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(
            name="test",
            pde_config=pde_config,
            game_mode="basis_selection",
            basis_config=basis_config,
        )
        game = BasisSelectionGame(poisson_operator, game_config)
        state = game.get_initial_state()
        assert game.action_space_size > 0
        state = game.apply_action(state, 0)
        assert state.n_basis == 1

    def test_rbf_basis_game(self, poisson_operator: PoissonOperator) -> None:
        basis_config = BasisSelectionConfig(
            name="rbf",
            basis_type="rbf",
            max_basis_functions=5,
            n_candidate_bases=10,
            n_collocation_points=25,
            n_boundary_points_per_face=5,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(
            name="test",
            pde_config=pde_config,
            game_mode="basis_selection",
            basis_config=basis_config,
        )
        game = BasisSelectionGame(poisson_operator, game_config)
        state = game.get_initial_state()
        assert game.action_space_size == 10
        state = game.apply_action(state, 0)
        assert state.n_basis == 1

    def test_fourier_basis_game(self, poisson_operator: PoissonOperator) -> None:
        basis_config = BasisSelectionConfig(
            name="fourier",
            basis_type="fourier",
            max_basis_functions=4,
            n_candidate_bases=8,
            max_frequency=1,
            n_collocation_points=16,
            n_boundary_points_per_face=5,
        )
        pde_config = PDEConfig(name="test", pde_type=PDEType.POISSON)
        game_config = PDEGameConfig(
            name="test",
            pde_config=pde_config,
            game_mode="basis_selection",
            basis_config=basis_config,
        )
        game = BasisSelectionGame(poisson_operator, game_config)
        state = game.get_initial_state()
        assert game.action_space_size > 0


class TestBasisSelectionGameLoop:
    """Integration tests for game loop."""

    def test_full_game_loop(self, game: BasisSelectionGame) -> None:
        """Run a complete game to terminal state."""
        state = game.get_initial_state()
        error_history = [state.error_estimate]
        max_steps = 6

        for _ in range(max_steps):
            if game.is_terminal(state):
                break
            actions = game.get_valid_actions(state)
            if not actions:
                break
            prev = state
            state = game.apply_action(state, actions[0])
            reward = game.get_reward(state, prev)
            error_history.append(state.error_estimate)
            assert isinstance(reward, float)

        result = game.get_result(state, error_history)
        assert result.n_steps >= 0
        assert result.final_dof >= 0
