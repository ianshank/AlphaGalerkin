"""Integration tests for GumbelMCTS search algorithm.

Tests cover:
- GumbelMCTS.search(): core search with real model + mock game
- GumbelMCTS._sequential_halving(): simulation allocation correctness
- GumbelMCTS._simulate(): terminal and non-terminal node evaluation
- GumbelMCTS._evaluate(): neural network inference
- GumbelMCTS.get_improved_policy(): temperature-adjusted policy
- create_gumbel_mcts(): factory function
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config.schemas import OperatorConfig
from src.games.state import ActionMask, GameState
from src.mcts.gumbel import (
    GumbelMCTS,
    GumbelMCTSConfig,
    GumbelNode,
    GumbelSearchResult,
    create_gumbel_mcts,
)
from src.modeling.model import AlphaGalerkinModel

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

ACTION_SPACE = 10  # small action space for fast tests
BOARD_SIZE = 3
INPUT_CHANNELS = 17


@pytest.fixture
def op_config() -> OperatorConfig:
    """Minimal operator config for tests."""
    return OperatorConfig(
        d_model=16,
        d_key=8,
        d_value=8,
        d_ffn=32,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=8,
        use_fnet_mixing=False,
    )


@pytest.fixture
def small_model(op_config: OperatorConfig) -> AlphaGalerkinModel:
    """Small model for fast inference."""
    model = AlphaGalerkinModel(op_config)
    model.eval()
    return model


def _make_state(player: int = 1, move_number: int = 0) -> GameState:
    """Create a simple game state."""
    return GameState(
        board=np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        current_player=player,
        move_number=move_number,
    )


class _MockGame:
    """Minimal mock game for testing Gumbel MCTS.

    - 10 actions total, all legal at start
    - is_terminal() returns True after 3 moves
    - apply_action() increments move_number and cycles player
    - to_tensor() returns a (17, 3, 3) tensor
    """

    def get_legal_actions(self, state: GameState) -> list[int]:
        if state.move_number >= 3:
            return []
        return list(range(ACTION_SPACE))

    def get_action_mask(self, state: GameState) -> ActionMask:
        if state.move_number >= 3:
            mask = np.zeros(ACTION_SPACE, dtype=bool)
        else:
            mask = np.ones(ACTION_SPACE, dtype=bool)
        return ActionMask(mask=mask, action_space_size=ACTION_SPACE)

    def apply_action(self, state: GameState, action: int) -> GameState:
        new_board = state.board.copy()
        new_board[0, 0, action % BOARD_SIZE] += 1.0
        return GameState(
            board=new_board,
            current_player=-state.current_player,
            move_number=state.move_number + 1,
            move_history=[*state.move_history, action],
        )

    def is_terminal(self, state: GameState) -> bool:
        return state.move_number >= 3

    def get_winner(self, state: GameState) -> int | None:
        if not self.is_terminal(state):
            return None
        return 1  # player 1 always wins in tests

    def to_tensor(self, state: GameState) -> torch.Tensor:
        return torch.from_numpy(state.board).float()


@pytest.fixture
def mock_game() -> _MockGame:
    return _MockGame()


@pytest.fixture
def gumbel_config() -> GumbelMCTSConfig:
    return GumbelMCTSConfig(
        n_simulations=8,
        max_num_considered_actions=4,
        gumbel_scale=1.0,
        c_visit=10.0,
        c_scale=1.0,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
    )


@pytest.fixture
def gumbel_mcts(
    gumbel_config: GumbelMCTSConfig,
    mock_game: _MockGame,
    small_model: AlphaGalerkinModel,
) -> GumbelMCTS:
    return GumbelMCTS(
        config=gumbel_config,
        game=mock_game,
        model=small_model,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# TestGumbelMCTSInit
# ---------------------------------------------------------------------------


class TestGumbelMCTSInit:
    """Tests for GumbelMCTS initialization."""

    def test_init_stores_config(
        self,
        gumbel_config: GumbelMCTSConfig,
        mock_game: _MockGame,
        small_model: AlphaGalerkinModel,
    ) -> None:
        mcts = GumbelMCTS(config=gumbel_config, game=mock_game, model=small_model, device="cpu")
        assert mcts.config is gumbel_config
        assert mcts.game is mock_game
        assert mcts.model is small_model

    def test_init_device_str_converted(
        self,
        gumbel_config: GumbelMCTSConfig,
        mock_game: _MockGame,
        small_model: AlphaGalerkinModel,
    ) -> None:
        mcts = GumbelMCTS(config=gumbel_config, game=mock_game, model=small_model, device="cpu")
        assert mcts.device == torch.device("cpu")

    def test_init_device_tensor(
        self,
        gumbel_config: GumbelMCTSConfig,
        mock_game: _MockGame,
        small_model: AlphaGalerkinModel,
    ) -> None:
        device = torch.device("cpu")
        mcts = GumbelMCTS(config=gumbel_config, game=mock_game, model=small_model, device=device)
        assert mcts.device == device


# ---------------------------------------------------------------------------
# TestGumbelEvaluate
# ---------------------------------------------------------------------------


class TestGumbelEvaluate:
    """Tests for GumbelMCTS._evaluate()."""

    def test_evaluate_returns_policy_and_value(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        policy, value = gumbel_mcts._evaluate(state)

        assert isinstance(policy, np.ndarray)
        assert isinstance(value, float)

    def test_evaluate_policy_sums_to_one(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        policy, _ = gumbel_mcts._evaluate(state)
        # policy is softmax output, should sum to ~1
        assert abs(policy.sum() - 1.0) < 1e-3

    def test_evaluate_value_is_finite(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        _, value = gumbel_mcts._evaluate(state)
        assert np.isfinite(value)

    def test_evaluate_sets_model_to_eval(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        gumbel_mcts.model.train()
        gumbel_mcts._evaluate(state)
        assert not gumbel_mcts.model.training


# ---------------------------------------------------------------------------
# TestGumbelSimulate
# ---------------------------------------------------------------------------


class TestGumbelSimulate:
    """Tests for GumbelMCTS._simulate()."""

    def test_simulate_terminal_node_returns_stored_value(self, gumbel_mcts: GumbelMCTS) -> None:
        node = GumbelNode()
        node._terminal_value = 1.0
        value = gumbel_mcts._simulate(node)
        assert value == 1.0

    def test_simulate_none_state_returns_zero(self, gumbel_mcts: GumbelMCTS) -> None:
        node = GumbelNode()  # state is None by default
        value = gumbel_mcts._simulate(node)
        assert value == 0.0

    def test_simulate_terminal_game_state_sets_value(self, gumbel_mcts: GumbelMCTS) -> None:
        # State with move_number >= 3 is terminal in _MockGame
        state = _make_state(player=1, move_number=3)
        node = GumbelNode(state=state)
        value = gumbel_mcts._simulate(node)
        assert value in (-1.0, 0.0, 1.0)
        assert node._terminal_value is not None

    def test_simulate_non_terminal_calls_evaluate(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state(player=1, move_number=0)
        node = GumbelNode(state=state)
        value = gumbel_mcts._simulate(node)
        assert np.isfinite(value)

    def test_simulate_winner_current_player_returns_positive(self, gumbel_mcts: GumbelMCTS) -> None:
        # In _MockGame get_winner always returns 1, and we set player=1
        state = _make_state(player=1, move_number=3)
        node = GumbelNode(state=state)
        value = gumbel_mcts._simulate(node)
        assert value == 1.0

    def test_simulate_winner_opponent_returns_negative(self, gumbel_mcts: GumbelMCTS) -> None:
        # In _MockGame get_winner always returns 1, but current player is -1
        state = _make_state(player=-1, move_number=3)
        node = GumbelNode(state=state)
        value = gumbel_mcts._simulate(node)
        assert value == -1.0


# ---------------------------------------------------------------------------
# TestSequentialHalving
# ---------------------------------------------------------------------------


class TestSequentialHalving:
    """Tests for GumbelMCTS._sequential_halving()."""

    def test_returns_action_in_input_list(self, gumbel_mcts: GumbelMCTS) -> None:
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        actions = [0, 1, 2, 3]
        for a in actions:
            root.children[a] = GumbelNode(
                state=gumbel_mcts.game.apply_action(root_state, a),
                prior=0.25,
                gumbel=float(np.random.gumbel()),
            )

        best_action, visit_counts = gumbel_mcts._sequential_halving(
            root, actions, total_simulations=8
        )
        assert best_action in actions

    def test_visit_counts_keys_match_actions(self, gumbel_mcts: GumbelMCTS) -> None:
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        actions = [0, 1, 2, 3]
        for a in actions:
            root.children[a] = GumbelNode(
                state=gumbel_mcts.game.apply_action(root_state, a),
                prior=0.25,
                gumbel=0.0,
            )

        _, visit_counts = gumbel_mcts._sequential_halving(root, actions, total_simulations=8)
        assert set(visit_counts.keys()) == set(actions)

    def test_total_visits_within_budget(self, gumbel_mcts: GumbelMCTS) -> None:
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        actions = [0, 1, 2, 3]
        budget = 8
        for a in actions:
            root.children[a] = GumbelNode(
                state=gumbel_mcts.game.apply_action(root_state, a),
                prior=0.25,
                gumbel=0.0,
            )

        _, visit_counts = gumbel_mcts._sequential_halving(root, actions, total_simulations=budget)
        total_visits = sum(visit_counts.values())
        assert total_visits <= budget

    def test_single_action_returns_it(self, gumbel_mcts: GumbelMCTS) -> None:
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        actions = [5]
        root.children[5] = GumbelNode(
            state=gumbel_mcts.game.apply_action(root_state, 5),
            prior=1.0,
            gumbel=0.0,
        )
        best_action, _ = gumbel_mcts._sequential_halving(root, actions, total_simulations=4)
        assert best_action == 5

    def test_zero_budget_returns_action(self, gumbel_mcts: GumbelMCTS) -> None:
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        actions = [0, 1]
        for a in actions:
            root.children[a] = GumbelNode(
                state=gumbel_mcts.game.apply_action(root_state, a),
                prior=0.5,
                gumbel=float(a),
            )
        best_action, _ = gumbel_mcts._sequential_halving(root, actions, total_simulations=0)
        assert best_action in actions


# ---------------------------------------------------------------------------
# TestSequentialHalvingConfigKnobs
#
# Proves GumbelMCTSConfig.use_mixed_value and .discount actually change
# search behaviour -- the exact dead-config gap flagged by
# docs/CODE_HYGIENE_AUDIT.md P2 ("Config fields that are declared but never
# read"): before this fix, toggling either changed nothing.
# ---------------------------------------------------------------------------


class TestSequentialHalvingConfigKnobs:
    """Deterministic proofs that both knobs change ``_sequential_halving``.

    All nodes below pre-set ``state`` (to something non-``None``, so the
    ``if child.state is None`` expand branch is skipped) and
    ``_terminal_value`` (so ``_simulate`` returns a fixed value immediately
    without touching the game or the model). This makes every assertion
    exact and independent of ``np.random``/model weights.
    """

    @staticmethod
    def _budget_starved_root(root_state: GameState, visited_value: float) -> GumbelNode:
        """Root with 4 actions and a budget of 3 forces action 3 unvisited.

        ``_sequential_halving``'s per-action budget accounting allocates
        ``sims_per_action = max(1, remaining_budget // (2 * n_actions))``
        simulations per round, then walks ``remaining_actions`` in order,
        breaking out of the inner loop the moment the budget hits zero. With
        4 actions and a budget of 3, actions 0/1/2 each consume one unit of
        budget before it is exhausted, and action 3 -- last in iteration
        order -- never gets simulated at all (``visit_count`` stays 0).
        """
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        for a in range(4):
            node = GumbelNode(prior=0.25, gumbel=0.0)
            node.state = root_state
            node._terminal_value = visited_value
            root.children[a] = node
        return root

    def test_action_3_is_starved_of_budget_by_construction(self, gumbel_mcts: GumbelMCTS) -> None:
        """Pins the fixture's own precondition.

        If a future change to the halving loop's budget accounting altered
        this, the two knob tests below would silently stop testing what
        their docstrings claim.
        """
        root = self._budget_starved_root(_make_state(), visited_value=-5.0)
        _, visit_counts = gumbel_mcts._sequential_halving(
            root, [0, 1, 2, 3], total_simulations=3, raw_value=0.0
        )
        assert visit_counts == {0: 1, 1: 1, 2: 1, 3: 0}

    def test_use_mixed_value_toggle_changes_best_action(self, gumbel_mcts: GumbelMCTS) -> None:
        """Toggling ``use_mixed_value`` flips which starved action wins.

        Action 3 never gets visited (see ``_budget_starved_root``).
        Disabled, its completed Q defaults to a flat ``0.0`` -- strictly
        better than actions 0-2's real (bad) observed value of -5.0, so it
        wins purely by being unexplored. Enabled, its completed Q is
        ``v_mix``, grounded in a very negative raw root value, so it loses
        instead -- proving the flag is read and changes the outcome.
        """
        actions = [0, 1, 2, 3]
        raw_value = -100.0
        visited_value = -5.0

        gumbel_mcts.config.use_mixed_value = False
        root_disabled = self._budget_starved_root(_make_state(), visited_value)
        best_disabled, _ = gumbel_mcts._sequential_halving(
            root_disabled, actions, total_simulations=3, raw_value=raw_value
        )
        assert best_disabled == 3

        gumbel_mcts.config.use_mixed_value = True
        root_enabled = self._budget_starved_root(_make_state(), visited_value)
        best_enabled, _ = gumbel_mcts._sequential_halving(
            root_enabled, actions, total_simulations=3, raw_value=raw_value
        )
        assert best_enabled != 3
        assert best_enabled == 0

    def test_discount_scales_the_backed_up_value(self, gumbel_mcts: GumbelMCTS) -> None:
        """A discount < 1.0 shrinks the value backed up into value_sum."""
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        for a in (0, 1):
            node = GumbelNode(prior=0.5, gumbel=0.0)
            node.state = root_state
            node._terminal_value = 8.0
            root.children[a] = node

        gumbel_mcts.config.discount = 0.25
        gumbel_mcts._sequential_halving(root, [0, 1], total_simulations=2, raw_value=0.0)

        for a in (0, 1):
            assert root.children[a].visit_count == 1
            assert root.children[a].value_sum == pytest.approx(0.25 * 8.0)
            assert root.children[a].value == pytest.approx(2.0)

    def test_discount_default_is_a_backward_compatible_noop(self, gumbel_mcts: GumbelMCTS) -> None:
        """Discount defaults to 1.0: byte-identical to the pre-fix, undiscounted backup."""
        root_state = _make_state()
        root = GumbelNode(state=root_state)
        root._is_expanded = True
        for a in (0, 1):
            node = GumbelNode(prior=0.5, gumbel=0.0)
            node.state = root_state
            node._terminal_value = 8.0
            root.children[a] = node

        assert gumbel_mcts.config.discount == 1.0
        gumbel_mcts._sequential_halving(root, [0, 1], total_simulations=2, raw_value=0.0)

        for a in (0, 1):
            assert root.children[a].value_sum == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# TestGumbelSearch
# ---------------------------------------------------------------------------


class TestGumbelSearch:
    """Tests for GumbelMCTS.search()."""

    def test_search_returns_gumbel_search_result(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        assert isinstance(result, GumbelSearchResult)

    def test_search_action_is_legal(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        # _MockGame: all 0..9 actions are legal at move 0
        assert 0 <= result.action < ACTION_SPACE

    def test_search_policy_sums_to_one(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        assert abs(result.policy.sum() - 1.0) < 1e-4

    def test_search_policy_size_matches_action_space(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        # Policy size matches game's action space (determined by model output)
        assert result.policy.ndim == 1

    def test_search_value_is_finite(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        assert np.isfinite(result.value)
        assert np.isfinite(result.root_value)

    def test_search_records_correct_n_simulations(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        assert result.n_simulations == gumbel_mcts.config.n_simulations

    def test_search_q_values_are_finite(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        # Non-zero Q values should be finite
        nonzero = result.q_values[result.q_values != 0]
        if len(nonzero) > 0:
            assert np.all(np.isfinite(nonzero))

    def test_search_visit_counts_nonnegative(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result = gumbel_mcts.search(state)
        assert np.all(result.visit_counts >= 0)

    def test_search_smoke_with_use_mixed_value_disabled(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        """End-to-end sanity: use_mixed_value=False still runs to completion."""
        config = GumbelMCTSConfig(
            n_simulations=8,
            max_num_considered_actions=4,
            use_mixed_value=False,
        )
        mcts = GumbelMCTS(config=config, game=mock_game, model=small_model, device="cpu")
        result = mcts.search(_make_state())
        assert isinstance(result, GumbelSearchResult)
        assert np.isfinite(result.q_values).all()

    def test_search_smoke_with_discount_below_one(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        """End-to-end sanity: discount < 1.0 still runs to completion."""
        config = GumbelMCTSConfig(
            n_simulations=8,
            max_num_considered_actions=4,
            discount=0.5,
        )
        mcts = GumbelMCTS(config=config, game=mock_game, model=small_model, device="cpu")
        result = mcts.search(_make_state())
        assert isinstance(result, GumbelSearchResult)
        assert np.isfinite(result.q_values).all()

    def test_search_is_deterministic_with_fixed_seed(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        np.random.seed(0)
        result1 = gumbel_mcts.search(state)
        np.random.seed(0)
        result2 = gumbel_mcts.search(state)
        assert result1.action == result2.action

    def test_search_multiple_calls_dont_share_state(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        result1 = gumbel_mcts.search(state)
        result2 = gumbel_mcts.search(state)
        # Both should return valid results — no shared mutable state
        assert isinstance(result1, GumbelSearchResult)
        assert isinstance(result2, GumbelSearchResult)


# ---------------------------------------------------------------------------
# TestGetImprovedPolicy
# ---------------------------------------------------------------------------


class TestGetImprovedPolicy:
    """Tests for GumbelMCTS.get_improved_policy()."""

    def test_improved_policy_sums_to_one(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        policy = gumbel_mcts.get_improved_policy(state, temperature=1.0)
        assert abs(policy.sum() - 1.0) < 1e-4

    def test_improved_policy_all_nonnegative(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        policy = gumbel_mcts.get_improved_policy(state, temperature=1.0)
        assert np.all(policy >= 0)

    def test_improved_policy_temperature_zero_is_deterministic(
        self, gumbel_mcts: GumbelMCTS
    ) -> None:
        state = _make_state()
        policy = gumbel_mcts.get_improved_policy(state, temperature=0)
        # Exactly one action gets probability 1
        assert policy.sum() == pytest.approx(1.0, abs=1e-4)
        assert (policy == 1.0).sum() == 1

    def test_improved_policy_high_temperature_smoother(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        np.random.seed(42)
        policy_low = gumbel_mcts.get_improved_policy(state, temperature=0.1)
        np.random.seed(42)
        policy_high = gumbel_mcts.get_improved_policy(state, temperature=2.0)
        # Higher temperature → lower max probability (smoother distribution)
        assert policy_high.max() <= policy_low.max() + 1e-4

    def test_improved_policy_ndarray(self, gumbel_mcts: GumbelMCTS) -> None:
        state = _make_state()
        policy = gumbel_mcts.get_improved_policy(state)
        assert isinstance(policy, np.ndarray)


# ---------------------------------------------------------------------------
# TestCreateGumbelMCTS
# ---------------------------------------------------------------------------


class TestCreateGumbelMCTS:
    """Tests for create_gumbel_mcts() factory function."""

    def test_returns_gumbel_mcts_instance(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        mcts = create_gumbel_mcts(
            game=mock_game,
            model=small_model,
            n_simulations=4,
            device="cpu",
        )
        assert isinstance(mcts, GumbelMCTS)

    def test_n_simulations_propagated(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        mcts = create_gumbel_mcts(
            game=mock_game,
            model=small_model,
            n_simulations=16,
            device="cpu",
        )
        assert mcts.config.n_simulations == 16

    def test_kwargs_forwarded_to_config(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        mcts = create_gumbel_mcts(
            game=mock_game,
            model=small_model,
            n_simulations=4,
            device="cpu",
            max_num_considered_actions=8,
            c_visit=25.0,
        )
        assert mcts.config.max_num_considered_actions == 8
        assert mcts.config.c_visit == 25.0

    def test_device_string_accepted(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        mcts = create_gumbel_mcts(
            game=mock_game,
            model=small_model,
            n_simulations=4,
            device="cpu",
        )
        assert mcts.device == torch.device("cpu")

    def test_search_works_after_factory_creation(
        self, mock_game: _MockGame, small_model: AlphaGalerkinModel
    ) -> None:
        mcts = create_gumbel_mcts(
            game=mock_game,
            model=small_model,
            n_simulations=4,
            device="cpu",
        )
        state = _make_state()
        result = mcts.search(state)
        assert isinstance(result, GumbelSearchResult)


# ---------------------------------------------------------------------------
# TestGumbelConstantRuntimeBinding
# ---------------------------------------------------------------------------


class TestGumbelConstantRuntimeBinding:
    """Runtime counterpart to the static site-binding guard in test_gumbel.py.

    ``GUMBEL_NORMALIZATION_EPSILON`` and ``GUMBEL_LOG_PRIOR_FLOOR`` hold the
    same value, so they can only be told apart by *role*. Inflating the
    normalization epsilon must shrink the returned policy mass (it is the
    denominator); inflating the log-prior floor must leave the policy
    normalized (it only shifts Gumbel scores). If the two were swapped at any
    site, one of these two assertions flips.
    """

    SEED = 1234

    @classmethod
    def _policy(cls, mcts: GumbelMCTS) -> np.ndarray:
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        return mcts.get_improved_policy(_make_state(), temperature=1.0)

    @classmethod
    def _visit_counts(cls, mcts: GumbelMCTS) -> np.ndarray:
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        return mcts.search(_make_state()).visit_counts

    def test_baseline_policy_is_normalized(self, gumbel_mcts: GumbelMCTS) -> None:
        """With the shipped 1e-8 epsilon the improved policy sums to 1."""
        assert float(self._policy(gumbel_mcts).sum()) == pytest.approx(1.0, abs=1e-6)

    def test_normalization_epsilon_is_the_policy_denominator(
        self,
        gumbel_mcts: GumbelMCTS,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inflating it shrinks the policy mass to counts/(counts.sum() + eps)."""
        import src.mcts.gumbel as gumbel_module

        baseline_sum = float(self._policy(gumbel_mcts).sum())
        inflated_eps = 1.0
        monkeypatch.setattr(gumbel_module, "GUMBEL_NORMALIZATION_EPSILON", inflated_eps)

        # Both calls run under the patched constant and the same seed, so the
        # visit counts are identical and the only difference is the denominator.
        counts = self._visit_counts(gumbel_mcts)
        inflated = self._policy(gumbel_mcts)
        total = float(counts.sum())

        assert float(inflated.sum()) < baseline_sum
        assert inflated == pytest.approx(counts / (total + inflated_eps))
        assert float(inflated.sum()) == pytest.approx(total / (total + inflated_eps))

    def test_log_prior_floor_does_not_denormalize_the_policy(
        self,
        gumbel_mcts: GumbelMCTS,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inflating the log floor perturbs selection but never the normalizer."""
        import src.mcts.gumbel as gumbel_module

        monkeypatch.setattr(gumbel_module, "GUMBEL_LOG_PRIOR_FLOOR", 1.0)
        assert float(self._policy(gumbel_mcts).sum()) == pytest.approx(1.0, abs=1e-6)
