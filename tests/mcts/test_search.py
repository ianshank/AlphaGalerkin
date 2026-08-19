"""Tests for MCTS search implementation.

Tests cover:
- MCTS: Core MCTS with PUCT selection
- BatchMCTS: Batched leaf evaluation
- GameInterface: Protocol compliance
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

import src.mcts.search as search_module
from src.mcts.evaluator import EvaluationResult, RandomEvaluator
from src.mcts.search import MCTS, BatchMCTS

# --- Mock Game Implementation ---


class MockGame:
    """Mock game for testing MCTS."""

    def __init__(
        self,
        board_size: int = 3,
        terminal_after: int = 10,
    ):
        """Initialize mock game.

        Args:
            board_size: Size of the board.
            terminal_after: Number of moves before game ends.

        """
        self.board_size = board_size
        self.terminal_after = terminal_after
        self.move_count = 0
        self._state = numpy.zeros((1, board_size, board_size), dtype=numpy.float32)

    def get_state(self) -> numpy.ndarray:
        """Get current game state."""
        return self._state.copy()

    def get_legal_actions(self) -> list[int]:
        """Get list of legal actions."""
        if self.move_count >= self.terminal_after:
            return []
        # Return all positions as legal
        n_positions = self.board_size**2
        return list(range(n_positions))

    def apply_action(self, action: int) -> None:
        """Apply action to game state."""
        self.move_count += 1
        row = action // self.board_size
        col = action % self.board_size
        self._state[0, row, col] = 1.0

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self.move_count >= self.terminal_after

    def get_winner(self) -> int:
        """Get winner."""
        return 1 if self.move_count % 2 == 0 else -1

    def clone(self) -> MockGame:
        """Create deep copy."""
        new_game = MockGame(self.board_size, self.terminal_after)
        new_game.move_count = self.move_count
        new_game._state = self._state.copy()
        return new_game


# --- Fixtures ---


@pytest.fixture
def mock_game() -> MockGame:
    """Create a mock game."""
    return MockGame(board_size=3, terminal_after=5)


@pytest.fixture
def random_evaluator() -> RandomEvaluator:
    """Create a random evaluator."""
    return RandomEvaluator(n_actions=10)  # 3x3 + pass


@pytest.fixture
def mcts(random_evaluator: RandomEvaluator) -> MCTS:
    """Create MCTS with random evaluator."""
    return MCTS(
        evaluator=random_evaluator,
        c_puct=1.5,
        n_simulations=10,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )


@pytest.fixture
def batch_mcts(random_evaluator: RandomEvaluator) -> BatchMCTS:
    """Create BatchMCTS with random evaluator."""
    return BatchMCTS(
        evaluator=random_evaluator,
        batch_size=4,
        c_puct=1.5,
        n_simulations=16,
    )


# --- MCTS Initialization Tests ---


class TestMCTSInit:
    """Tests for MCTS initialization."""

    def test_init_stores_parameters(self, random_evaluator: RandomEvaluator):
        """Test initialization stores parameters."""
        mcts = MCTS(
            evaluator=random_evaluator,
            c_puct=2.0,
            n_simulations=100,
            dirichlet_alpha=0.5,
            dirichlet_epsilon=0.3,
            virtual_loss=2.0,
        )

        assert mcts.evaluator is random_evaluator
        assert mcts.c_puct == 2.0
        assert mcts.n_simulations == 100
        assert mcts.dirichlet_alpha == 0.5
        assert mcts.dirichlet_epsilon == 0.3
        assert mcts.virtual_loss == 2.0

    def test_init_default_root_none(self, random_evaluator: RandomEvaluator):
        """Test root is None initially."""
        mcts = MCTS(evaluator=random_evaluator)
        assert mcts._root is None


# --- MCTS Search Tests ---


class TestMCTSSearch:
    """Tests for MCTS search."""

    def test_search_returns_distribution(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test search returns action distribution."""
        distribution = mcts.search(mock_game)

        assert isinstance(distribution, dict)
        assert len(distribution) > 0
        # Probabilities should sum to 1
        assert abs(sum(distribution.values()) - 1.0) < 0.01

    def test_search_creates_root(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test search creates root node."""
        assert mcts._root is None
        mcts.search(mock_game)
        assert mcts._root is not None

    def test_search_expands_root(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test search expands root node."""
        mcts.search(mock_game)
        assert not mcts._root.is_leaf

    def test_search_performs_simulations(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test search performs correct number of simulations."""
        mcts.search(mock_game)

        # Root visit count should be >= n_simulations
        assert mcts._root.visit_count >= mcts.n_simulations

    def test_search_with_noise(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test search adds Dirichlet noise when requested."""
        # Run multiple searches with noise
        distributions = []
        for _ in range(5):
            mcts.reset()
            dist = mcts.search(mock_game, add_noise=True)
            distributions.append(tuple(sorted(dist.items())))

        # Distributions might vary due to noise (though not guaranteed)
        # Just verify search completes
        assert all(len(d) > 0 for d in distributions)

    def test_search_without_noise(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test search without Dirichlet noise."""
        distribution = mcts.search(mock_game, add_noise=False)
        assert len(distribution) > 0


# --- MCTS Get Action Tests ---


class TestMCTSGetAction:
    """Tests for MCTS action selection."""

    def test_get_action_returns_int(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test get_action returns integer action."""
        action = mcts.get_action(mock_game)
        assert isinstance(action, int)

    def test_get_action_returns_legal(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test get_action returns legal action."""
        legal_actions = mock_game.get_legal_actions()
        action = mcts.get_action(mock_game)
        assert action in legal_actions

    def test_get_action_temperature_zero(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test deterministic selection with temperature=0."""
        # Run multiple times - should be consistent
        mcts.search(mock_game, add_noise=False)
        action1 = mcts.get_action(mock_game, temperature=0, add_noise=False)
        action2 = mcts.get_action(mock_game, temperature=0, add_noise=False)

        # Since search already completed, actions should be same
        assert action1 == action2

    def test_get_action_high_temperature(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test stochastic selection with high temperature."""
        # Just verify it completes and returns valid action
        action = mcts.get_action(mock_game, temperature=2.0)
        assert action in mock_game.get_legal_actions()


# --- MCTS Tree Management Tests ---


class TestMCTSTreeManagement:
    """Tests for MCTS tree reuse and management."""

    def test_advance_reuses_subtree(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test advance reuses subtree for chosen action."""
        mcts.search(mock_game)
        original_root = mcts._root
        best_action = original_root.get_best_action()

        expected_new_root = original_root.children[best_action]
        mcts.advance(best_action)

        # New root should be the old child
        assert mcts._root is expected_new_root

    def test_advance_nonexistent_action(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test advance with non-existent action clears root."""
        mcts.search(mock_game)
        mcts.advance(999)  # Non-existent action

        assert mcts._root is None

    def test_reset_clears_tree(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test reset clears the search tree."""
        mcts.search(mock_game)
        assert mcts._root is not None

        mcts.reset()
        assert mcts._root is None

    def test_get_pv_empty_initially(self, mcts: MCTS):
        """Test get_pv returns empty list initially."""
        pv = mcts.get_pv()
        assert pv == []

    def test_get_pv_after_search(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test get_pv returns principal variation after search."""
        mcts.search(mock_game)
        pv = mcts.get_pv()

        # Should have at least one action
        assert len(pv) >= 1

    def test_get_root_value_zero_initially(self, mcts: MCTS):
        """Test get_root_value returns 0 initially."""
        value = mcts.get_root_value()
        assert value == 0.0

    def test_get_root_value_after_search(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test get_root_value returns Q-value after search."""
        mcts.search(mock_game)
        value = mcts.get_root_value()

        # Value should be in valid range
        assert -1.0 <= value <= 1.0


# --- BatchMCTS Tests ---


class TestBatchMCTS:
    """Tests for BatchMCTS."""

    def test_init_stores_batch_size(self, random_evaluator: RandomEvaluator):
        """Test BatchMCTS stores batch_size parameter."""
        batch_mcts = BatchMCTS(
            evaluator=random_evaluator,
            batch_size=8,
        )
        assert batch_mcts.batch_size == 8

    def test_search_returns_distribution(
        self,
        batch_mcts: BatchMCTS,
        mock_game: MockGame,
    ):
        """Test batch search returns distribution."""
        distribution = batch_mcts.search(mock_game)

        assert isinstance(distribution, dict)
        assert len(distribution) > 0

    def test_search_performs_simulations(
        self,
        batch_mcts: BatchMCTS,
        mock_game: MockGame,
    ):
        """Test batch search performs simulations."""
        batch_mcts.search(mock_game)
        assert batch_mcts._root.visit_count > 0

    def test_inherits_mcts_methods(self, batch_mcts: BatchMCTS):
        """Test BatchMCTS inherits MCTS methods."""
        assert hasattr(batch_mcts, "get_action")
        assert hasattr(batch_mcts, "advance")
        assert hasattr(batch_mcts, "reset")
        assert hasattr(batch_mcts, "get_pv")


# --- Terminal State Tests ---


class TestTerminalStates:
    """Tests for MCTS behavior at terminal states."""

    def test_search_at_terminal(
        self,
        mcts: MCTS,
    ):
        """Test search behavior at terminal state."""
        # Create game that's already terminal
        terminal_game = MockGame(terminal_after=0)

        # Search should still work but return empty or handle gracefully
        distribution = mcts.search(terminal_game)

        # At terminal state, might return empty distribution
        assert isinstance(distribution, dict)

    def test_simulation_stops_at_terminal(
        self,
        mcts: MCTS,
    ):
        """Test simulations stop at terminal states."""
        # Create game that terminates quickly
        quick_game = MockGame(terminal_after=2)

        mcts.search(quick_game)

        # Search should complete without error
        assert mcts._root is not None


# --- Integration Tests ---


class TestMCTSIntegration:
    """Integration tests for MCTS."""

    def test_full_game_simulation(
        self,
        mcts: MCTS,
    ):
        """Test MCTS through a full game."""
        game = MockGame(terminal_after=5)

        move_count = 0
        while not game.is_terminal() and move_count < 10:
            action = mcts.get_action(game, temperature=1.0)
            game.apply_action(action)
            mcts.advance(action)
            move_count += 1

        assert move_count > 0

    def test_tree_reuse_across_moves(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """Test tree is reused across moves."""
        # First move
        mcts.search(mock_game, add_noise=False)
        first_visits = sum(child.visit_count for child in mcts._root.children.values())

        # Get best action and advance
        action = mcts.get_action(mock_game, temperature=0, add_noise=False)
        mock_game.apply_action(action)
        mcts.advance(action)

        # Second search should reuse subtree
        if mcts._root is not None and not mock_game.is_terminal():
            initial_visits = mcts._root.visit_count
            assert initial_visits > 0  # Reused from previous search

    def test_different_board_sizes(
        self,
        random_evaluator: RandomEvaluator,
    ):
        """Test MCTS works with different game configurations."""
        # Small board
        small_game = MockGame(board_size=2, terminal_after=3)
        small_evaluator = RandomEvaluator(n_actions=5)  # 2x2 + pass
        small_mcts = MCTS(evaluator=small_evaluator, n_simulations=5)

        distribution = small_mcts.search(small_game)
        assert len(distribution) > 0

        # Large board
        large_game = MockGame(board_size=5, terminal_after=10)
        large_evaluator = RandomEvaluator(n_actions=26)  # 5x5 + pass
        large_mcts = MCTS(evaluator=large_evaluator, n_simulations=5)

        distribution = large_mcts.search(large_game)
        assert len(distribution) > 0


# --- Single Legal Action Tests ---


class _SingleActionGame:
    """Game with exactly one legal action at every non-terminal step.

    Exercises ``select_child``/``get_visit_distribution`` with a trivial
    (size-1) policy: no other action ever competes for the PUCT score, so
    this is the degenerate case where a division-by-zero or a malformed
    single-element softmax would first surface.
    """

    def __init__(self, moves_remaining: int = 3) -> None:
        self.moves_remaining = moves_remaining

    def get_state(self) -> numpy.ndarray:
        return numpy.zeros(1, dtype=numpy.float32)

    def get_legal_actions(self) -> list[int]:
        return [] if self.moves_remaining <= 0 else [0]

    def apply_action(self, action: int) -> None:
        self.moves_remaining -= 1

    def is_terminal(self) -> bool:
        return self.moves_remaining <= 0

    def get_winner(self) -> int:
        return 1

    def clone(self) -> _SingleActionGame:
        return _SingleActionGame(self.moves_remaining)


class TestSingleLegalAction:
    """Edge case: every node in the tree has exactly one legal action."""

    def test_search_assigns_full_mass_to_the_sole_action(self) -> None:
        evaluator = RandomEvaluator(n_actions=1)
        mcts = MCTS(evaluator=evaluator, n_simulations=10, c_puct=1.5)

        distribution = mcts.search(_SingleActionGame(), add_noise=False)

        assert distribution == {0: pytest.approx(1.0)}

    def test_get_action_returns_the_sole_action_at_every_temperature(self) -> None:
        evaluator = RandomEvaluator(n_actions=1)
        mcts = MCTS(evaluator=evaluator, n_simulations=10, c_puct=1.5)

        for temperature in (0.0, 1.0, 5.0):
            mcts.reset()
            action = mcts.get_action(_SingleActionGame(), temperature=temperature, add_noise=False)
            assert action == 0


# --- Terminal-At-Root Observability Tests ---


class TestTerminalAtRoot:
    """A root that is already terminal before any simulation runs.

    ``search()`` must return sanely (empty distribution, no crash);
    ``get_action()`` must fail with a clear, actionable error rather than a
    low-level one (previously an unguarded ``numpy.random.choice`` on an
    empty array: ``ValueError: 'a' cannot be empty unless no samples are
    taken`` at temperature != 0, or a bare "no children" from
    ``get_best_action()`` at temperature == 0 -- neither names the actual
    cause).
    """

    def test_search_returns_empty_distribution(self) -> None:
        evaluator = RandomEvaluator(n_actions=10)
        mcts = MCTS(evaluator=evaluator, n_simulations=5, c_puct=1.5)
        terminal_game = MockGame(terminal_after=0)

        distribution = mcts.search(terminal_game)

        assert distribution == {}
        assert mcts._root is not None
        assert mcts._root.children == {}

    def test_get_action_raises_clear_error_at_default_temperature(self) -> None:
        evaluator = RandomEvaluator(n_actions=10)
        mcts = MCTS(evaluator=evaluator, n_simulations=5, c_puct=1.5)
        terminal_game = MockGame(terminal_after=0)

        with pytest.raises(ValueError, match="already-terminal"):
            mcts.get_action(terminal_game)

    def test_get_action_raises_clear_error_at_temperature_zero(self) -> None:
        """The deterministic (argmax) branch hits the same guard."""
        evaluator = RandomEvaluator(n_actions=10)
        mcts = MCTS(evaluator=evaluator, n_simulations=5, c_puct=1.5)
        terminal_game = MockGame(terminal_after=0)

        with pytest.raises(ValueError, match="already-terminal"):
            mcts.get_action(terminal_game, temperature=0.0)


# --- c_puct / Temperature Boundary Value Tests ---


class TestExtremeSearchParameters:
    """c_puct and temperature at their documented boundary values."""

    def test_search_with_zero_c_puct_is_pure_exploitation(
        self,
        random_evaluator: RandomEvaluator,
        mock_game: MockGame,
    ):
        """c_puct=0.0 must not crash; the exploration term always vanishes."""
        mcts = MCTS(evaluator=random_evaluator, c_puct=0.0, n_simulations=20)

        distribution = mcts.search(mock_game, add_noise=False)

        assert len(distribution) > 0
        assert all(numpy.isfinite(p) for p in distribution.values())
        assert abs(sum(distribution.values()) - 1.0) < 1e-6

    def test_search_with_very_large_c_puct_stays_finite(
        self,
        random_evaluator: RandomEvaluator,
        mock_game: MockGame,
    ):
        mcts = MCTS(evaluator=random_evaluator, c_puct=1e6, n_simulations=20)

        distribution = mcts.search(mock_game, add_noise=False)

        assert len(distribution) > 0
        assert all(numpy.isfinite(p) for p in distribution.values())
        assert abs(sum(distribution.values()) - 1.0) < 1e-6

    def test_get_action_with_very_large_temperature_stays_finite(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        """1/temperature -> 0 as temperature grows; must not produce NaN."""
        action = mcts.get_action(mock_game, temperature=1e6, add_noise=False)
        assert action in mock_game.get_legal_actions()

    def test_get_visit_distribution_with_very_large_temperature(
        self,
        mcts: MCTS,
        mock_game: MockGame,
    ):
        mcts.search(mock_game, add_noise=False)
        distribution = mcts._root.get_visit_distribution(temperature=1e6)

        assert len(distribution) > 0
        assert all(numpy.isfinite(p) for p in distribution.values())
        assert abs(sum(distribution.values()) - 1.0) < 1e-6


# --- Observability: non-finite evaluator output (never silent) ---


class _NonFiniteValueEvaluator:
    """Uniform, finite policy paired with a NaN value.

    A stand-in for a value head that diverged during training.
    """

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def evaluate(self, state: numpy.ndarray, legal_actions: list[int]) -> EvaluationResult:
        policy = numpy.zeros(self.n_actions, dtype=numpy.float32)
        if legal_actions:
            for a in legal_actions:
                policy[a] = 1.0 / len(legal_actions)
        return EvaluationResult(policy=policy, value=float("nan"))

    def evaluate_batch(
        self,
        states: list[numpy.ndarray],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=False)]


class _NonFinitePolicyEvaluator:
    """Finite value paired with an Inf-poisoned policy entry."""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def evaluate(self, state: numpy.ndarray, legal_actions: list[int]) -> EvaluationResult:
        policy = numpy.zeros(self.n_actions, dtype=numpy.float32)
        if legal_actions:
            policy[legal_actions[0]] = float("inf")
        return EvaluationResult(policy=policy, value=0.0)

    def evaluate_batch(
        self,
        states: list[numpy.ndarray],
        legal_actions_batch: list[list[int]],
    ) -> list[EvaluationResult]:
        return [self.evaluate(s, la) for s, la in zip(states, legal_actions_batch, strict=False)]


class _RecordingLogger:
    """Minimal structlog-shaped stand-in that records ``warning``/``debug`` calls.

    Replaces the module-level ``logger`` directly instead of using
    ``structlog.testing.capture_logs``: ``tests/pde/test_mesh_refinement.py``
    documents that ``capture_logs`` can silently record nothing once any
    earlier test in the session has called
    ``src.poc.logging.configure_logging`` (it caches a stdlib-backed bound
    logger that stops consulting the patched processor chain) -- a failure
    mode that depends on collection order across the whole session.
    Replacing the proxy object outright is robust regardless of what ran
    before this test.
    """

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []
        self.debugs: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.warnings.append((event, kwargs))

    def debug(self, event: str, **kwargs: object) -> None:
        self.debugs.append((event, kwargs))


class TestNonFiniteEvaluationObservability:
    """A NaN/Inf evaluator output must leave a log trail, not fail silently.

    ``MCTS._check_finite_evaluation`` is the sole guard against a value
    network (or a broken test double) silently poisoning every ancestor's
    ``total_value`` with NaN -- these tests exercise both call sites
    (``_expand_node`` for ``MCTS``, the batch leaf-expansion loop for
    ``BatchMCTS``) and confirm the ordinary path stays silent (no false
    positives).
    """

    def test_expand_logs_non_finite_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_game: MockGame,
    ) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(search_module, "logger", recorder)

        mcts = MCTS(evaluator=_NonFiniteValueEvaluator(9), n_simulations=1, c_puct=1.0)
        mcts.search(mock_game, add_noise=False)

        events = [event for event, _ in recorder.warnings]
        assert "mcts_evaluator_non_finite_value" in events

    def test_expand_logs_non_finite_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_game: MockGame,
    ) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(search_module, "logger", recorder)

        mcts = MCTS(evaluator=_NonFinitePolicyEvaluator(9), n_simulations=1, c_puct=1.0)
        mcts.search(mock_game, add_noise=False)

        events = [event for event, _ in recorder.warnings]
        assert "mcts_evaluator_non_finite_policy" in events

    def test_normal_evaluation_does_not_warn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        random_evaluator: RandomEvaluator,
        mock_game: MockGame,
    ) -> None:
        """No false positives: the ordinary RandomEvaluator path stays silent."""
        recorder = _RecordingLogger()
        monkeypatch.setattr(search_module, "logger", recorder)

        mcts = MCTS(evaluator=random_evaluator, n_simulations=10, c_puct=1.5)
        mcts.search(mock_game, add_noise=False)

        assert recorder.warnings == []

    def test_batch_expand_logs_non_finite_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_game: MockGame,
    ) -> None:
        """The batched leaf-expansion path (not just root expansion) is checked.

        Asserts on ``context="batch_expand"`` specifically, since root
        expansion alone (shared with ``MCTS``) would already emit a
        ``context="expand"`` warning regardless of this call site.
        """
        recorder = _RecordingLogger()
        monkeypatch.setattr(search_module, "logger", recorder)

        batch_mcts = BatchMCTS(
            evaluator=_NonFiniteValueEvaluator(9),
            batch_size=4,
            n_simulations=8,
            c_puct=1.0,
        )
        batch_mcts.search(mock_game, add_noise=False)

        batch_contexts = [
            kwargs.get("context")
            for event, kwargs in recorder.warnings
            if event == "mcts_evaluator_non_finite_value"
        ]
        assert "batch_expand" in batch_contexts


# --- Observability: game-contract violation (non-terminal, no legal actions) ---


class _StuckAfterFirstMoveGame:
    """Reports zero legal actions from a state that claims to be non-terminal.

    A deliberate game-contract inconsistency (``is_terminal()`` always
    returns False): exercises the one degenerate case
    ``BatchMCTS._simulate_batch`` does not treat as terminal-like, unlike the
    single-simulation ``_expand_node`` path.
    """

    def __init__(self, moved: bool = False) -> None:
        self.moved = moved

    def get_state(self) -> numpy.ndarray:
        return numpy.zeros(1, dtype=numpy.float32)

    def get_legal_actions(self) -> list[int]:
        return [] if self.moved else [0, 1]

    def apply_action(self, action: int) -> None:
        self.moved = True

    def is_terminal(self) -> bool:
        return False

    def get_winner(self) -> int:
        return 0

    def clone(self) -> _StuckAfterFirstMoveGame:
        return _StuckAfterFirstMoveGame(self.moved)


class TestBatchLeafMissingLegalActionsObservability:
    def test_batch_expand_logs_when_leaf_has_no_legal_actions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorder = _RecordingLogger()
        monkeypatch.setattr(search_module, "logger", recorder)

        evaluator = RandomEvaluator(n_actions=2)
        batch_mcts = BatchMCTS(
            evaluator=evaluator,
            batch_size=4,
            n_simulations=4,
            c_puct=1.0,
        )
        batch_mcts.search(_StuckAfterFirstMoveGame(), add_noise=False)

        events = [event for event, _ in recorder.warnings]
        assert "mcts_batch_leaf_no_legal_actions" in events
