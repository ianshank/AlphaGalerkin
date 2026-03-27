"""Additional coverage tests for Gumbel MCTS.

Tests cover uncovered paths in src/mcts/gumbel.py:
- GumbelMCTS: Initialization, _sequential_halving, _simulate, _evaluate
- create_gumbel_mcts: Factory function
- get_improved_policy: Temperature handling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from src.mcts.gumbel import (
    GumbelMCTS,
    GumbelMCTSConfig,
    GumbelNode,
    GumbelSearchResult,
    create_gumbel_mcts,
)

SEED = 42
ACTION_SPACE = 10


@dataclass
class FakeGameState:
    """Minimal game state for testing."""

    current_player: int = 1
    board: list[int] = field(default_factory=lambda: [0] * ACTION_SPACE)


class FakeActionMask:
    """Minimal action mask."""

    def __init__(self, size: int):
        self.mask = np.ones(size, dtype=bool)
        self.num_legal = size


class FakeGameResult:
    """Minimal game result."""

    def __init__(self, winner: int):
        self.winner = winner


class FakeGame:
    """Minimal game interface for testing GumbelMCTS."""

    action_space_size = ACTION_SPACE

    def initial_state(self):
        return FakeGameState()

    def get_action_mask(self, state):
        return FakeActionMask(ACTION_SPACE)

    def apply_action(self, state, action):
        new_board = list(state.board)
        new_board[action] = state.current_player
        return FakeGameState(
            current_player=-state.current_player,
            board=new_board,
        )

    def is_terminal(self, state):
        return sum(1 for x in state.board if x != 0) >= 3

    def get_winner(self, state):
        return 1

    def get_result(self, state):
        return FakeGameResult(winner=1)

    def to_tensor(self, state):
        return torch.tensor(state.board, dtype=torch.float32).unsqueeze(0)


class FakeModel(nn.Module):
    """Minimal model that outputs policy and value."""

    def __init__(self, action_size: int = ACTION_SPACE):
        super().__init__()
        self.linear = nn.Linear(action_size, action_size)
        self.value_head = nn.Linear(action_size, 1)
        self.action_size = action_size

    def forward(self, x):
        # x shape: (batch, 1, action_size)
        x = x.squeeze(1)  # (batch, action_size)
        policy_logits = self.linear(x)
        value = torch.tanh(self.value_head(x))

        result = MagicMock()
        result.policy_logits = policy_logits
        result.value = value
        return result


@pytest.fixture
def game() -> FakeGame:
    return FakeGame()


@pytest.fixture
def model() -> FakeModel:
    torch.manual_seed(SEED)
    return FakeModel()


@pytest.fixture
def config() -> GumbelMCTSConfig:
    return GumbelMCTSConfig(
        n_simulations=10,
        max_num_considered_actions=4,
        gumbel_scale=1.0,
        c_visit=50.0,
        c_scale=1.0,
        batch_size=1,
    )


class TestGumbelMCTSInit:
    """Tests for GumbelMCTS initialization."""

    def test_basic_init(self, config, game, model) -> None:
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        assert mcts.config == config
        assert mcts.game is game
        assert mcts.model is model
        assert mcts.device == torch.device("cpu")

    def test_string_device(self, config, game, model) -> None:
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        assert isinstance(mcts.device, torch.device)


class TestGumbelMCTSSearch:
    """Tests for GumbelMCTS search."""

    def test_search_returns_result(self, config, game, model) -> None:
        np.random.seed(SEED)
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        result = mcts.search(state)

        assert isinstance(result, GumbelSearchResult)
        assert 0 <= result.action < ACTION_SPACE
        assert result.policy.shape == (ACTION_SPACE,)
        assert result.n_simulations == config.n_simulations

    def test_search_policy_sums_to_one(self, config, game, model) -> None:
        np.random.seed(SEED)
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        result = mcts.search(state)
        assert abs(result.policy.sum() - 1.0) < 0.1

    def test_search_visit_counts(self, config, game, model) -> None:
        np.random.seed(SEED)
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        result = mcts.search(state)
        assert result.visit_counts.sum() > 0

    def test_search_q_values(self, config, game, model) -> None:
        np.random.seed(SEED)
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        result = mcts.search(state)
        assert result.q_values.shape == (ACTION_SPACE,)


class TestGumbelMCTSGetImprovedPolicy:
    """Tests for get_improved_policy."""

    def test_deterministic_temperature(self, config, game, model) -> None:
        np.random.seed(SEED)
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        policy = mcts.get_improved_policy(state, temperature=0)
        # Should be one-hot
        assert policy.sum() == pytest.approx(1.0)
        assert policy.max() == pytest.approx(1.0)

    def test_temperature_1(self, config, game, model) -> None:
        np.random.seed(SEED)
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        policy = mcts.get_improved_policy(state, temperature=1.0)
        assert policy.shape == (ACTION_SPACE,)
        assert abs(policy.sum() - 1.0) < 0.1


class TestGumbelMCTSSimulate:
    """Tests for _simulate method."""

    def test_simulate_terminal_node(self, config, game, model) -> None:
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        node = GumbelNode()
        node._terminal_value = 1.0
        value = mcts._simulate(node)
        assert value == 1.0

    def test_simulate_terminal_state(self, config, game, model) -> None:
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        # Create a state that is terminal
        state = FakeGameState(board=[1, -1, 1, 0, 0, 0, 0, 0, 0, 0])
        node = GumbelNode(state=state)
        value = mcts._simulate(node)
        assert isinstance(value, float)

    def test_simulate_non_terminal(self, config, game, model) -> None:
        mcts = GumbelMCTS(config=config, game=game, model=model, device="cpu")
        state = game.initial_state()
        node = GumbelNode(state=state)
        value = mcts._simulate(node)
        assert isinstance(value, float)


class TestCreateGumbelMCTS:
    """Tests for create_gumbel_mcts factory."""

    def test_factory_basic(self, game, model) -> None:
        mcts = create_gumbel_mcts(game=game, model=model, n_simulations=10)
        assert isinstance(mcts, GumbelMCTS)
        assert mcts.config.n_simulations == 10

    def test_factory_with_kwargs(self, game, model) -> None:
        mcts = create_gumbel_mcts(
            game=game,
            model=model,
            n_simulations=20,
            max_num_considered_actions=8,
            gumbel_scale=0.5,
        )
        assert mcts.config.n_simulations == 20
        assert mcts.config.max_num_considered_actions == 8
        assert mcts.config.gumbel_scale == 0.5
