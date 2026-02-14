"""Property-based tests for MCTS invariants."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.alphagalerkin.env.state import DiscretizationState
from src.alphagalerkin.mcts.node import MCTSNode

_SUPPRESS = [HealthCheck.function_scoped_fixture]


class TestMCTSInvariants:
    """Hypothesis-driven invariant checks for MCTSNode."""

    @given(
        values=st.lists(
            st.floats(min_value=-1.0, max_value=1.0),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=30,
        suppress_health_check=_SUPPRESS,
    )
    def test_visit_count_equals_backup_count(
        self,
        initial_state: DiscretizationState,
        values: list[float],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        for v in values:
            node.backup(v)
        assert node.visit_count == len(values)

    @given(
        values=st.lists(
            st.floats(min_value=-1.0, max_value=1.0),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=30,
        suppress_health_check=_SUPPRESS,
    )
    def test_mean_value_is_average(
        self,
        initial_state: DiscretizationState,
        values: list[float],
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        for v in values:
            node.backup(v)
        expected = sum(values) / len(values)
        assert abs(node.mean_value - expected) < 1e-8

    @given(
        cpuct=st.floats(
            min_value=0.1,
            max_value=10.0,
        ),
        parent_visits=st.integers(
            min_value=1,
            max_value=10000,
        ),
    )
    @settings(
        max_examples=30,
        suppress_health_check=_SUPPRESS,
    )
    def test_ucb_score_finite_after_visit(
        self,
        initial_state: DiscretizationState,
        cpuct: float,
        parent_visits: int,
    ) -> None:
        node = MCTSNode(state=initial_state, prior=0.5)
        node.backup(0.5)
        score = node.ucb_score(cpuct, parent_visits)
        assert score != float("inf")
        assert score != float("-inf")
        assert score == score  # not NaN

    @given(
        prior=st.floats(min_value=0.0, max_value=1.0),
        value=st.floats(min_value=-1.0, max_value=1.0),
    )
    @settings(
        max_examples=30,
        suppress_health_check=_SUPPRESS,
    )
    def test_ucb_components_nonnegative_exploitation(
        self,
        initial_state: DiscretizationState,
        prior: float,
        value: float,
    ) -> None:
        """After one backup the mean_value should equal the backed-up value."""
        node = MCTSNode(state=initial_state, prior=prior)
        node.backup(value)
        assert abs(node.mean_value - value) < 1e-10

    @given(
        n_backups=st.integers(min_value=1, max_value=100),
    )
    @settings(
        max_examples=20,
        suppress_health_check=_SUPPRESS,
    )
    def test_total_value_equals_sum(
        self,
        initial_state: DiscretizationState,
        n_backups: int,
    ) -> None:
        """total_value should be exactly n_backups * value for constant value."""
        node = MCTSNode(state=initial_state, prior=0.5)
        for _ in range(n_backups):
            node.backup(0.3)
        expected = 0.3 * n_backups
        assert abs(node.total_value - expected) < 1e-6
