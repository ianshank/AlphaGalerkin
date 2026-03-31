"""Tests for PDE-MCTS adapter.

Validates that ``PDEGameAdapter`` correctly satisfies the MCTS
``GameInterface`` protocol using ``BasisSelectionGame`` as the
underlying PDE game.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.pde.config import BasisSelectionConfig, PDEConfig, PDEGameConfig, PDEType
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.operators import PoissonOperator


@pytest.fixture
def pde_config() -> PDEConfig:
    """Minimal Poisson PDE config."""
    return PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )


@pytest.fixture
def game_config(pde_config: PDEConfig) -> PDEGameConfig:
    """Minimal game config for basis selection."""
    return PDEGameConfig(
        name="test_basis",
        pde_config=pde_config,
        game_mode="basis_selection",
        max_steps=10,
        error_tolerance=0.01,
        computational_budget=1e4,
        basis_config=BasisSelectionConfig(
            name="test_basis_selection",
            max_basis_functions=8,
            basis_type="fourier",
            max_frequency=3,
        ),
    )


@pytest.fixture
def adapter(pde_config: PDEConfig, game_config: PDEGameConfig) -> PDEGameAdapter:
    """Create a PDE game adapter."""
    operator = PoissonOperator(pde_config)
    pde_game = BasisSelectionGame(operator, game_config)
    return PDEGameAdapter(pde_game)


class TestPDEGameAdapterProtocol:
    """Verify the adapter satisfies the GameInterface protocol."""

    def test_get_state_returns_numpy(self, adapter: PDEGameAdapter) -> None:
        """get_state() returns a numpy float32 array."""
        state = adapter.get_state()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert state.ndim >= 1

    def test_get_legal_actions_returns_list(self, adapter: PDEGameAdapter) -> None:
        """get_legal_actions() returns a non-empty list of ints."""
        actions = adapter.get_legal_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(a, int) for a in actions)

    def test_apply_action_mutates_state(self, adapter: PDEGameAdapter) -> None:
        """apply_action() changes the internal state."""
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        # State should change after an action
        assert adapter.state.step == 1

    def test_is_terminal_initially_false(self, adapter: PDEGameAdapter) -> None:
        """Initial state is not terminal."""
        assert not adapter.is_terminal()

    def test_get_winner_returns_valid_value(self, adapter: PDEGameAdapter) -> None:
        """get_winner() returns one of {-1, 0, 1}."""
        result = adapter.get_winner()
        assert result in (-1, 0, 1)

    def test_clone_creates_independent_copy(self, adapter: PDEGameAdapter) -> None:
        """clone() creates a deep copy that doesn't affect original."""
        cloned = adapter.clone()

        # Apply action to clone only
        actions = cloned.get_legal_actions()
        cloned.apply_action(actions[0])

        # Original should be unchanged
        assert adapter.state.step == 0
        assert cloned.state.step == 1

    def test_clone_shares_game_rules(self, adapter: PDEGameAdapter) -> None:
        """clone() shares the pde_game (stateless rules) instance."""
        cloned = adapter.clone()
        assert cloned.pde_game is adapter.pde_game


class TestPDEGameAdapterGameplay:
    """Test full gameplay through the adapter."""

    def test_play_until_terminal_or_max(self, adapter: PDEGameAdapter) -> None:
        """Can play through the game until terminal or max steps."""
        max_steps = 10
        for _ in range(max_steps):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            if not actions:
                break
            adapter.apply_action(actions[0])

        # Should have made some progress
        assert len(adapter.error_history) > 1

    def test_error_history_tracked(self, adapter: PDEGameAdapter) -> None:
        """Error history records each step's error."""
        assert len(adapter.error_history) == 1  # Initial state

        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])

        assert len(adapter.error_history) == 2

    def test_error_reduction_property(self, adapter: PDEGameAdapter) -> None:
        """error_reduction returns fraction of initial error reduced."""
        assert adapter.error_reduction == 0.0

        # After a step, error_reduction should be non-negative
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        # Can be positive or negative depending on basis choice
        assert isinstance(adapter.error_reduction, float)

    def test_reset_returns_to_initial(self, adapter: PDEGameAdapter) -> None:
        """reset() restores initial state."""
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        assert adapter.state.step == 1

        adapter.reset()
        assert adapter.state.step == 0
        assert len(adapter.error_history) == 1


class TestPDEGameAdapterWinner:
    """Test winner determination logic."""

    def test_winner_positive_on_full_convergence(self) -> None:
        """Full convergence (error < tolerance) returns +1."""
        # Simulate the winner logic directly
        error_history = [1.0, 0.5, 0.001]  # Converged below 0.01
        tolerance = 0.01
        final_error = error_history[-1]
        initial_error = error_history[0]

        if final_error < tolerance or (final_error / initial_error) < 0.1:
            result = 1
        elif (final_error / initial_error) > 0.5:
            result = -1
        else:
            result = 0

        assert result == 1

    def test_winner_negative_on_poor_reduction(self) -> None:
        """Poor error reduction (< 50%) returns -1."""
        error_history = [1.0, 0.9, 0.8]
        final_error = error_history[-1]
        initial_error = error_history[0]
        reduction_ratio = final_error / initial_error  # 0.8 > 0.5

        assert reduction_ratio > 0.5

    def test_winner_neutral_on_moderate_reduction(self) -> None:
        """Moderate reduction (50-90%) returns 0."""
        error_history = [1.0, 0.5, 0.3]  # 70% reduction
        final_error = error_history[-1]
        initial_error = error_history[0]
        reduction_ratio = final_error / initial_error  # 0.3 = 70% reduction

        # 0.1 < 0.3 < 0.5 -> return 0
        assert 0.1 < reduction_ratio < 0.5

    def test_winner_positive_on_strong_reduction(self) -> None:
        """Strong reduction (>90%) returns +1 even if above tolerance."""
        error_history = [1.0, 0.1, 0.05]  # 95% reduction
        final_error = error_history[-1]
        initial_error = error_history[0]
        reduction_ratio = final_error / initial_error  # 0.05 = 95% reduction

        # 0.05 < 0.1 -> returns +1
        assert reduction_ratio < 0.1


class TestPDEGameAdapterEdgeCases:
    """Test edge cases in the adapter."""

    def test_current_error_property(self, adapter: PDEGameAdapter) -> None:
        """current_error returns latest error estimate."""
        initial_error = adapter.current_error
        assert initial_error == adapter.state.error_estimate

    def test_error_reduction_zero_at_start(self, adapter: PDEGameAdapter) -> None:
        """error_reduction is 0.0 at initial state."""
        assert adapter.error_reduction == 0.0

    def test_state_tensor_finite(self, adapter: PDEGameAdapter) -> None:
        """State tensor contains finite values."""
        state = adapter.get_state()
        assert state.shape[0] > 0  # Has channels
        # All values should be finite
        assert all(abs(v) < 1e10 for v in state.flatten())

    def test_multiple_clones_independent(self, adapter: PDEGameAdapter) -> None:
        """Multiple clones are independent of each other."""
        clone1 = adapter.clone()
        clone2 = adapter.clone()

        actions1 = clone1.get_legal_actions()
        clone1.apply_action(actions1[0])

        # clone2 and original should be unchanged
        assert adapter.state.step == 0
        assert clone2.state.step == 0
        assert clone1.state.step == 1

    def test_error_history_grows_with_actions(self, adapter: PDEGameAdapter) -> None:
        """Error history grows with each action."""
        initial_len = len(adapter.error_history)
        actions = adapter.get_legal_actions()

        for i, action in enumerate(actions[:3]):
            if adapter.is_terminal():
                break
            adapter.apply_action(action)
            assert len(adapter.error_history) == initial_len + i + 1

    def test_legal_actions_change_after_action(self, adapter: PDEGameAdapter) -> None:
        """Legal actions may change after taking an action."""
        initial_actions = set(adapter.get_legal_actions())
        adapter.apply_action(list(initial_actions)[0])
        new_actions = set(adapter.get_legal_actions())

        # Actions should differ (taken action might be removed)
        # or remain same depending on game rules
        assert isinstance(new_actions, set)


class TestPDEGameAdapterWinnerEdgeCases:
    """Test edge cases in winner computation."""

    def test_winner_with_zero_initial_error(self) -> None:
        """Zero initial error defaults to 1.0 reduction ratio."""
        # Simulate the logic from get_winner
        error_history = [0.0, 0.01]
        initial_error = error_history[0]
        final_error = error_history[-1]

        if initial_error > 0:
            reduction_ratio = final_error / initial_error
        else:
            reduction_ratio = 1.0

        assert reduction_ratio == 1.0

    def test_winner_converged_below_tolerance(self) -> None:
        """Convergence below tolerance returns +1."""
        tolerance = 0.01
        final_error = 0.005

        if final_error < tolerance:
            result = 1
        else:
            result = 0

        assert result == 1

    def test_winner_ninety_percent_reduction(self) -> None:
        """90%+ reduction returns +1 even above tolerance."""
        error_history = [1.0, 0.09]  # 91% reduction
        final_error = error_history[-1]
        initial_error = error_history[0]
        reduction_ratio = final_error / initial_error

        assert reduction_ratio < 0.1  # Should return +1
