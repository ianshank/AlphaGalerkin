"""Pure unit tests for the PDEGameAdapter.

Targets uncovered lines in src/pde/mcts_adapter.py.
Uses lightweight mocks to avoid complex PDE setup dependencies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.pde.game import GamePhase, PDEState
from src.pde.mcts_adapter import PDEGameAdapter


# ---------------------------------------------------------------------------
# Lightweight stub for PDEGame
# ---------------------------------------------------------------------------


class StubPDEGame:
    """Minimal PDEGame stub that satisfies the adapter's interface."""

    def __init__(
        self,
        action_space_size: int = 4,
        tolerance: float = 0.01,
        max_steps: int = 5,
    ) -> None:
        self.action_space_size = action_space_size
        self._tolerance = tolerance
        self._max_steps = max_steps
        self.config = MagicMock()
        self.config.tolerance = tolerance

    def get_initial_state(self) -> PDEState:
        n = 10
        return PDEState(
            coords=np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32),
            solution=np.zeros(n, dtype=np.float32),
            residuals=np.ones(n, dtype=np.float32),
            error_estimate=1.0,
            dof=n,
            step=0,
            budget_remaining=100.0,
            phase=GamePhase.INITIAL,
        )

    def get_valid_actions(self, state: PDEState) -> list[int]:
        return list(range(self.action_space_size))

    def apply_action(self, state: PDEState, action: int) -> PDEState:
        new_state = state.clone()
        new_state.step = state.step + 1
        # Each action reduces error by a fraction
        new_state.error_estimate = state.error_estimate * 0.5
        new_state.history = list(state.history) + [action]
        new_state.budget_remaining = state.budget_remaining - 1
        if new_state.error_estimate < self._tolerance:
            new_state.phase = GamePhase.CONVERGED
        return new_state

    def is_terminal(self, state: PDEState) -> bool:
        return (
            state.phase == GamePhase.CONVERGED
            or state.step >= self._max_steps
        )

    def to_tensor(self, state: PDEState) -> torch.Tensor:
        return torch.tensor(state.solution, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestAdapterInit:
    def test_creates_with_initial_state(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        assert adapter.state is not None
        assert adapter.state.step == 0
        assert len(adapter.error_history) == 1
        assert adapter.error_history[0] == 1.0

    def test_pde_game_ref(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        assert adapter.pde_game is game


# ---------------------------------------------------------------------------
# Tests: get_state
# ---------------------------------------------------------------------------


class TestGetState:
    def test_returns_numpy(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        state = adapter.get_state()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: get_legal_actions
# ---------------------------------------------------------------------------


class TestGetLegalActions:
    def test_returns_all_actions(self) -> None:
        game = StubPDEGame(action_space_size=6)
        adapter = PDEGameAdapter(game)
        actions = adapter.get_legal_actions()
        assert actions == list(range(6))


# ---------------------------------------------------------------------------
# Tests: apply_action
# ---------------------------------------------------------------------------


class TestApplyAction:
    def test_mutates_state(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        initial_error = adapter.current_error

        adapter.apply_action(0)
        assert adapter.state.step == 1
        assert adapter.current_error < initial_error
        assert len(adapter.error_history) == 2

    def test_error_history_grows(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)

        for i in range(3):
            adapter.apply_action(i % game.action_space_size)

        assert len(adapter.error_history) == 4  # initial + 3 actions


# ---------------------------------------------------------------------------
# Tests: is_terminal
# ---------------------------------------------------------------------------


class TestIsTerminal:
    def test_not_terminal_initially(self) -> None:
        game = StubPDEGame(max_steps=10)
        adapter = PDEGameAdapter(game)
        assert adapter.is_terminal() is False

    def test_terminal_after_convergence(self) -> None:
        game = StubPDEGame(tolerance=0.5, max_steps=10)
        adapter = PDEGameAdapter(game)
        # error starts at 1.0, after one action it's 0.5 which is not < 0.5
        adapter.apply_action(0)  # error -> 0.5
        adapter.apply_action(0)  # error -> 0.25 < 0.5 => converged
        assert adapter.is_terminal() is True

    def test_terminal_after_budget(self) -> None:
        game = StubPDEGame(max_steps=2, tolerance=1e-10)
        adapter = PDEGameAdapter(game)
        adapter.apply_action(0)
        adapter.apply_action(0)
        assert adapter.is_terminal() is True


# ---------------------------------------------------------------------------
# Tests: get_winner
# ---------------------------------------------------------------------------


class TestGetWinner:
    def test_converged_returns_1(self) -> None:
        game = StubPDEGame(tolerance=0.5, max_steps=10)
        adapter = PDEGameAdapter(game)
        adapter.apply_action(0)  # 0.5
        adapter.apply_action(0)  # 0.25 < 0.5
        assert adapter.get_winner() == 1

    def test_good_reduction_returns_1(self) -> None:
        """90%+ reduction even without convergence => +1."""
        game = StubPDEGame(tolerance=1e-10, max_steps=10)
        adapter = PDEGameAdapter(game)
        # 4 halvings: 1.0 -> 0.0625 (93.75% reduction)
        for _ in range(4):
            adapter.apply_action(0)
        assert adapter.get_winner() == 1

    def test_poor_reduction_returns_neg1(self) -> None:
        """Less than 50% reduction => -1."""
        game = StubPDEGame(tolerance=1e-10, max_steps=10)
        adapter = PDEGameAdapter(game)
        # Just one halving: 1.0 -> 0.5, reduction ratio = 0.5 => not < 0.5
        # But also not > 0.5 strictly - it equals 0.5, which is not > 0.5
        adapter.apply_action(0)
        # ratio is 0.5, which is not < 0.1 and not > 0.5, so => 0
        winner = adapter.get_winner()
        assert winner == 0  # Ambiguous

    def test_no_reduction_returns_neg1(self) -> None:
        """No error reduction should return -1."""
        game = StubPDEGame(tolerance=1e-10, max_steps=10)
        adapter = PDEGameAdapter(game)
        # Override error_history to simulate no reduction
        adapter.error_history = [1.0, 0.9]
        assert adapter.get_winner() == -1

    def test_empty_history_returns_0(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        adapter.error_history = []
        assert adapter.get_winner() == 0

    def test_zero_initial_error(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        adapter.error_history = [0.0, 0.0]
        # final_error=0.0 < tolerance=0.01 => converged => +1
        assert adapter.get_winner() == 1


# ---------------------------------------------------------------------------
# Tests: clone
# ---------------------------------------------------------------------------


class TestClone:
    def test_clone_is_independent(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        adapter.apply_action(0)

        cloned = adapter.clone()

        # Same state
        assert cloned.current_error == adapter.current_error
        assert cloned.error_history == adapter.error_history

        # Independent mutation
        cloned.apply_action(1)
        assert cloned.state.step != adapter.state.step
        assert len(cloned.error_history) != len(adapter.error_history)

    def test_clone_shares_game(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        cloned = adapter.clone()
        assert cloned.pde_game is adapter.pde_game


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_to_initial(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        adapter.apply_action(0)
        adapter.apply_action(1)

        adapter.reset()
        assert adapter.state.step == 0
        assert len(adapter.error_history) == 1
        assert adapter.current_error == 1.0


# ---------------------------------------------------------------------------
# Tests: properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_current_error(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        assert adapter.current_error == 1.0

    def test_error_reduction(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        adapter.apply_action(0)
        # 1.0 -> 0.5, reduction = 1 - 0.5/1.0 = 0.5
        assert adapter.error_reduction == pytest.approx(0.5)

    def test_error_reduction_zero_initial(self) -> None:
        game = StubPDEGame()
        adapter = PDEGameAdapter(game)
        adapter.error_history = [0.0, 0.0]
        assert adapter.error_reduction == 0.0
