"""Integration test: MCTS game for fire spread mesh refinement."""

from __future__ import annotations

import pytest

jaxtyping = pytest.importorskip("jaxtyping")

from src.firefighting.mcts.register import create_fire_spread_adapter  # noqa: E402
from src.pde.mcts_adapter import PDEGameAdapter  # noqa: E402


class TestFireSpreadMCTSIntegration:
    """Verify MCTS game protocol compliance for fire spread."""

    @pytest.fixture
    def adapter(self) -> PDEGameAdapter:
        return create_fire_spread_adapter(
            n_regions=4,
            max_budget=150,
            max_steps=10,
        )

    def test_adapter_creation(self, adapter: PDEGameAdapter) -> None:
        assert adapter is not None
        assert adapter.state is not None

    def test_game_protocol_methods(self, adapter: PDEGameAdapter) -> None:
        state = adapter.get_state()
        assert state is not None

        actions = adapter.get_legal_actions()
        assert len(actions) > 0

        adapter.apply_action(actions[0])
        new_state = adapter.get_state()
        assert new_state is not None

    def test_terminal_condition(self, adapter: PDEGameAdapter) -> None:
        for _ in range(20):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            adapter.apply_action(actions[0])
        assert adapter.is_terminal()

    def test_clone_independence(self, adapter: PDEGameAdapter) -> None:
        clone = adapter.clone()
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        assert clone.state.step == 0

    def test_winner_range(self, adapter: PDEGameAdapter) -> None:
        for _ in range(20):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            adapter.apply_action(actions[0])
        winner = adapter.get_winner()
        assert -1 <= winner <= 1

    def test_error_reduces_with_refinement(self, adapter: PDEGameAdapter) -> None:
        """Error should decrease as we refine."""
        initial_error = adapter.state.error_estimate
        for _ in range(5):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            # Pick first non-hold action
            for a in actions:
                if a != adapter.pde_game.action_space_size - 1:
                    adapter.apply_action(a)
                    break
        assert adapter.state.error_estimate <= initial_error
