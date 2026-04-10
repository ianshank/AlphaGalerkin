"""Integration test: MCTS game for compressible flow mesh refinement.

Tests the full chain: CompressibleFlowGame → PDEGameAdapter → MCTS-compatible.
"""

from __future__ import annotations

import pytest

jaxtyping = pytest.importorskip("jaxtyping")

from src.pde.mcts_adapter import PDEGameAdapter  # noqa: E402
from src.reentry.mcts.register import create_compressible_flow_adapter  # noqa: E402


class TestCompressibleFlowMCTSIntegration:
    """Verify MCTS game protocol compliance for compressible flow."""

    @pytest.fixture
    def adapter(self) -> PDEGameAdapter:
        return create_compressible_flow_adapter(
            n_regions=4,
            max_budget=200,
            max_steps=10,
        )

    def test_adapter_creation(self, adapter: PDEGameAdapter) -> None:
        assert adapter is not None
        assert adapter.state is not None

    def test_game_protocol_methods(self, adapter: PDEGameAdapter) -> None:
        """Adapter must expose get_state, get_legal_actions, apply_action, etc."""
        state = adapter.get_state()
        assert state is not None

        actions = adapter.get_legal_actions()
        assert len(actions) > 0

        # Apply an action
        adapter.apply_action(actions[0])
        new_state = adapter.get_state()
        assert new_state is not None

    def test_terminal_condition(self, adapter: PDEGameAdapter) -> None:
        """Game should eventually terminate."""
        for _ in range(20):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            adapter.apply_action(actions[0])
        # After many actions, should be terminal (budget exhausted)
        assert adapter.is_terminal()

    def test_clone_independence(self, adapter: PDEGameAdapter) -> None:
        """Cloned adapter should be independent of original."""
        clone = adapter.clone()
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])

        # Clone state should be unchanged
        assert clone.state.step == 0

    def test_winner_range(self, adapter: PDEGameAdapter) -> None:
        """Winner should be in [-1, 1]."""
        # Run to terminal
        for _ in range(20):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            adapter.apply_action(actions[0])
        winner = adapter.get_winner()
        assert -1 <= winner <= 1
