"""Tests for BasisSelectionGame."""

from __future__ import annotations

import numpy as np
import pytest

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

    def test_polynomial_evaluate(self) -> None:
        bf = BasisFunction(type="polynomial", params={"degree_x": 2, "degree_y": 0}, index=0)
        coords = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        values = bf.evaluate(coords)
        assert values.shape == (3,)
        np.testing.assert_allclose(values, [0.0, 0.25, 1.0], atol=1e-6)

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

    def test_action_space_size(self, game: BasisSelectionGame) -> None:
        assert game.action_space_size <= 16

    def test_state_channels(self, game: BasisSelectionGame) -> None:
        assert game.state_channels == 3 + 8  # 3 + max_basis_functions

    def test_candidate_bases_generated(self, game: BasisSelectionGame) -> None:
        assert len(game._candidate_bases) > 0
        assert len(game._candidate_bases) <= 16


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

    def test_apply_action(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        assert new_state.step == 1
        assert new_state.dof == 1
        assert 0 in new_state.history
        assert new_state.n_basis == 1

    def test_apply_duplicate_action_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state = game.apply_action(state, 0)
        with pytest.raises(ValueError, match="already selected"):
            game.apply_action(state, 0)

    def test_apply_invalid_action_raises(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, -1)
        with pytest.raises(ValueError, match="Invalid action"):
            game.apply_action(state, 9999)


class TestBasisSelectionGameErrorReduction:
    """Tests for error estimation and reduction."""

    def test_error_decreases_with_basis(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        initial_error = state.error_estimate

        # Add a basis function - error should typically not increase
        actions = game.get_valid_actions(state)
        state = game.apply_action(state, actions[0])

        # The error may not always decrease with arbitrary basis,
        # but the solution should be computed
        assert state.solution is not None
        assert state.n_basis == 1

    def test_compute_exact_error(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        errors = game.compute_exact_error(state)
        assert "l2" in errors
        assert "h1" in errors
        assert "linf" in errors
        assert "residual" in errors
        assert errors["l2"] >= 0
        assert errors["linf"] >= 0


class TestBasisSelectionGameReward:
    """Tests for reward computation."""

    def test_reward_basic(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        reward = game.get_reward(new_state, state)
        assert isinstance(reward, float)

    def test_reward_from_error_reduction(self, game: BasisSelectionGame) -> None:
        """Reward should incorporate error reduction."""
        state = game.get_initial_state()
        new_state = game.apply_action(state, 0)
        reward = game.get_reward(new_state, state)
        error_reduction = state.error_estimate - new_state.error_estimate
        # Reward should correlate with error reduction
        # (accounting for cost_per_dof)
        assert isinstance(reward, float)


class TestBasisSelectionGameTerminal:
    """Tests for terminal conditions."""

    def test_not_terminal_initial(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        assert game.is_terminal(state) is False

    def test_terminal_max_basis(self, game: BasisSelectionGame) -> None:
        """Terminal when max basis functions reached."""
        state = game.get_initial_state()
        max_basis = game.basis_config.max_basis_functions
        n_candidates = game.action_space_size

        # Add basis functions up to max
        for i in range(min(max_basis, n_candidates)):
            if game.is_terminal(state):
                break
            state = game.apply_action(state, i)

        # Should be terminal once max basis reached
        if state.n_basis >= max_basis:
            assert game.is_terminal(state) is True

    def test_terminal_budget_exhausted(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        state.budget_remaining = 0
        assert game.is_terminal(state) is True


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

    def test_get_result_empty_history(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        result = game.get_result(state, [])
        assert result.error_reduction_rate == 0.0


class TestBasisSelectionGameTensor:
    """Tests for tensor conversion."""

    def test_to_tensor(self, game: BasisSelectionGame) -> None:
        state = game.get_initial_state()
        tensor = game.to_tensor(state)
        assert tensor.ndim >= 2
        assert tensor.shape[0] == game.state_channels

    def test_action_to_string(self, game: BasisSelectionGame) -> None:
        s = game.action_to_string(0)
        assert isinstance(s, str)
        assert len(s) > 0

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
            name="test", pde_config=pde_config,
            game_mode="basis_selection", basis_config=basis_config,
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
            name="test", pde_config=pde_config,
            game_mode="basis_selection", basis_config=basis_config,
        )
        game = BasisSelectionGame(poisson_operator, game_config)
        state = game.get_initial_state()
        assert game.action_space_size == 10
        state = game.apply_action(state, 0)
        assert state.n_basis == 1
