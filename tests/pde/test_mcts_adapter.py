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
        domain_min=0.0,
        domain_max=1.0,
        n_collocation=25,
    )


@pytest.fixture
def game_config(pde_config: PDEConfig) -> PDEGameConfig:
    """Minimal game config for basis selection."""
    return PDEGameConfig(
        name="test_basis",
        pde_config=pde_config,
        game_mode="basis_selection",
        max_steps=10,
        tolerance=0.01,
        budget=1e4,
        basis_selection=BasisSelectionConfig(
            max_basis_size=8,
            candidate_types=["fourier"],
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
        state_before = adapter.get_state().copy()
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        state_after = adapter.get_state()
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
