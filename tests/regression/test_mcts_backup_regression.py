"""Regression tests pinning MCTS backup sign semantics across search modes."""

from __future__ import annotations

import pytest

from src.mcts.node import MCTSNode
from src.mcts.search import SearchMode, invert_on_backup


@pytest.mark.regression
def test_mcts_backup_single_agent_no_sign_flip() -> None:
    """Regression test: SINGLE_AGENT mode must NOT invert value on backup at any depth."""
    root = MCTSNode()
    child = MCTSNode(parent=root, action=0, prior=1.0)
    root.children[0] = child
    grandchild = MCTSNode(parent=child, action=1, prior=1.0)
    child.children[1] = grandchild

    # Backup from depth 2 (grandchild) with invert=False (SINGLE_AGENT mode)
    assert not invert_on_backup(SearchMode.SINGLE_AGENT)
    grandchild.backup(0.8, invert=False)

    assert grandchild.visit_count == 1
    assert grandchild.total_value == 0.8
    assert grandchild.q_value == 0.8

    assert child.visit_count == 1
    assert child.total_value == 0.8
    assert child.q_value == 0.8

    assert root.visit_count == 1
    assert root.total_value == 0.8
    assert root.q_value == 0.8


@pytest.mark.regression
def test_mcts_backup_zero_sum_alternating_sign_flip() -> None:
    """Regression test: ZERO_SUM mode must invert values at alternating levels."""
    root = MCTSNode()
    child = MCTSNode(parent=root, action=0, prior=1.0)
    root.children[0] = child
    grandchild = MCTSNode(parent=child, action=1, prior=1.0)
    child.children[1] = grandchild

    # Backup from depth 2 with invert=True (ZERO_SUM mode)
    assert invert_on_backup(SearchMode.ZERO_SUM)
    grandchild.backup(0.8, invert=True)

    # Grandchild: +0.8 (evaluated state)
    assert grandchild.visit_count == 1
    assert grandchild.total_value == 0.8

    # Child (opponent's turn): -0.8
    assert child.visit_count == 1
    assert child.total_value == -0.8

    # Root (player's turn): +0.8
    assert root.visit_count == 1
    assert root.total_value == 0.8


@pytest.mark.regression
def test_mcts_backup_root_node_single_step() -> None:
    """Regression test: Backing up directly into root node handles both modes correctly."""
    root_single = MCTSNode()
    root_single.backup(0.5, invert=False)
    assert root_single.q_value == 0.5

    root_zero_sum = MCTSNode()
    root_zero_sum.backup(0.5, invert=True)
    assert root_zero_sum.q_value == 0.5
