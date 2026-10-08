"""Realized search-tree depth: ``subtree_depth`` and the read-only ``MCTS.root``.

Gate 1 (``specs/lookahead_vs_greedy.spec.md``) records the realized depth of the
tree each committed action was chosen on, rather than inferring it from the
``n_simulations / top_k_actions`` ratio. Defect classes, one sentence each:

* **D1 phantom ply** -- children that ``expand`` created but no simulation
  visited are counted, so a one-step look-ahead reads as depth 2.
* **D2 truncated walk** -- the walk stops at the first branch or the first
  child, so a deep line behind a shallow sibling is missed.
* **D3 side effects** -- measuring the tree changes it (visit counts, children),
  which would change the next search.
* **D4 behaviour change** -- exposing the root alters search results.

Mutation-kill record: see ``tests/research/test_lookahead_vs_greedy.py``'s
module docstring (one record for the Gate 1 surface).
"""

from __future__ import annotations

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.mcts.evaluator import RandomEvaluator
from src.mcts.node import MCTSNode, subtree_depth
from src.mcts.search import MCTS, SearchMode

#: Depth of the chain built for the iterative-walk test: far beyond the default
#: recursion limit, so a recursive implementation would raise RecursionError.
DEEP_CHAIN = 5000


def _chain(depth: int, *, visited: bool = True) -> tuple[MCTSNode, MCTSNode]:
    """A root with a single line of ``depth`` children; returns (root, leaf)."""
    root = MCTSNode()
    node = root
    for ply in range(depth):
        child = MCTSNode(parent=node, action=ply, prior=1.0, visit_count=1 if visited else 0)
        node.children[ply] = child
        node = child
    return root, node


class _LineGame:
    """Two legal actions per ply until ``horizon`` plies; deterministic, single-agent."""

    def __init__(self, horizon: int = 6) -> None:
        self.horizon = horizon
        self.moves: list[int] = []

    def get_state(self) -> numpy.ndarray:
        return numpy.array([float(len(self.moves))], dtype=numpy.float32)

    def get_legal_actions(self) -> list[int]:
        return [] if self.is_terminal() else [0, 1]

    def apply_action(self, action: int) -> None:
        self.moves.append(action)

    def is_terminal(self) -> bool:
        return len(self.moves) >= self.horizon

    def get_winner(self) -> int:
        return 0

    def clone(self) -> _LineGame:
        copy = _LineGame(self.horizon)
        copy.moves = list(self.moves)
        return copy


def _search(n_simulations: int, *, horizon: int = 6) -> MCTS:
    numpy.random.seed(0)
    mcts = MCTS(
        evaluator=RandomEvaluator(n_actions=2),
        n_simulations=n_simulations,
        search_mode=SearchMode.SINGLE_AGENT,
    )
    mcts.search(_LineGame(horizon), add_noise=False)
    return mcts


class TestSubtreeDepth:
    def test_a_lone_node_has_depth_zero(self) -> None:
        assert subtree_depth(MCTSNode()) == 0

    def test_unvisited_children_are_not_a_ply(self) -> None:
        """D1: expansion alone creates priors, not evaluated states."""
        root = MCTSNode()
        root.expand({0: 0.5, 1: 0.5})
        assert subtree_depth(root) == 0

    def test_a_visited_line_counts_every_edge(self) -> None:
        root, _ = _chain(4)
        assert subtree_depth(root) == 4

    def test_the_unvisited_frontier_below_a_visited_leaf_is_not_counted(self) -> None:
        """D1: the evaluated leaf's expansion adds no ply."""
        root, leaf = _chain(3)
        leaf.expand({7: 0.5, 8: 0.5})
        assert subtree_depth(root) == 3

    def test_a_deep_line_behind_a_shallow_sibling_is_found(self) -> None:
        """D2: the deepest visited descendant anywhere, not along the first branch."""
        root = MCTSNode()
        shallow = MCTSNode(parent=root, action=0, visit_count=3)
        root.children[0] = shallow
        deep_root, _ = _chain(5)
        deep_child = next(iter(deep_root.children.values()))
        deep_child.parent = root
        deep_child.action = 1
        root.children[1] = deep_child
        assert subtree_depth(root) == 5

    def test_a_deep_tree_does_not_recurse(self) -> None:
        root, _ = _chain(DEEP_CHAIN)
        assert subtree_depth(root) == DEEP_CHAIN

    def test_measuring_does_not_change_the_tree(self) -> None:
        """D3: pure read."""
        root, leaf = _chain(3)
        leaf.expand({9: 1.0})
        before = repr([(n.visit_count, len(n.children)) for n in _walk(root)])
        subtree_depth(root)
        assert repr([(n.visit_count, len(n.children)) for n in _walk(root)]) == before

    @settings(max_examples=60, deadline=None)
    @given(st.data())
    def test_matches_a_recursive_oracle_on_random_trees(self, data: st.DataObject) -> None:
        """An independent recursive definition agrees on arbitrary visit patterns."""
        root = MCTSNode(visit_count=1)
        frontier = [root]
        for _ in range(data.draw(st.integers(min_value=0, max_value=25))):
            parent = data.draw(st.sampled_from(frontier))
            action = len(parent.children)
            visited = data.draw(st.booleans()) and parent.visit_count > 0
            child = MCTSNode(parent=parent, action=action, visit_count=1 if visited else 0)
            parent.children[action] = child
            frontier.append(child)
        assert subtree_depth(root) == _oracle(root)


def _walk(node: MCTSNode) -> list[MCTSNode]:
    out = [node]
    for child in node.children.values():
        out.extend(_walk(child))
    return out


def _oracle(node: MCTSNode) -> int:
    visited = [child for child in node.children.values() if child.visit_count > 0]
    return max((1 + _oracle(child) for child in visited), default=0)


class TestRootProperty:
    def test_none_before_the_first_search(self) -> None:
        mcts = MCTS(evaluator=RandomEvaluator(n_actions=2), n_simulations=1)
        assert mcts.root is None
        assert mcts.get_root_value() == 0.0

    def test_is_the_search_root_and_follows_advance(self) -> None:
        mcts = _search(8)
        root = mcts.root
        assert root is not None
        assert root.is_root
        assert root.visit_count == mcts.n_simulations
        action = mcts.get_pv()[0]
        child = root.children[action]
        mcts.advance(action)
        assert mcts.root is child
        assert mcts.get_root_value() == child.q_value

    def test_is_read_only(self) -> None:
        mcts = _search(2)
        with pytest.raises(AttributeError):
            mcts.root = MCTSNode()  # type: ignore[misc]

    def test_reset_clears_it(self) -> None:
        mcts = _search(2)
        mcts.reset()
        assert mcts.root is None


class TestRealizedDepthOfARealSearch:
    def test_one_simulation_looks_exactly_one_ply_ahead(self) -> None:
        """D1 on the engine itself: one simulation evaluates one child state."""
        mcts = _search(1)
        assert mcts.root is not None
        assert subtree_depth(mcts.root) == 1

    def test_more_simulations_grow_the_tree_and_never_past_the_simulation_count(self) -> None:
        depths = []
        for n_simulations in (1, 2, 4, 16):
            mcts = _search(n_simulations)
            assert mcts.root is not None
            depth = subtree_depth(mcts.root)
            assert 1 <= depth <= n_simulations
            depths.append(depth)
        assert depths == sorted(depths)
        assert depths[-1] >= 2

    def test_the_horizon_caps_the_depth(self) -> None:
        mcts = _search(64, horizon=2)
        assert mcts.root is not None
        assert subtree_depth(mcts.root) <= 2

    def test_reading_the_root_does_not_change_the_search(self) -> None:
        """D4: identical visit distributions with and without reading the depth."""
        observed = _search(16)
        assert observed.root is not None
        subtree_depth(observed.root)
        control = _search(16)
        assert control.root is not None
        assert observed.root.get_visit_distribution() == control.root.get_visit_distribution()
