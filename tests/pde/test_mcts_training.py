"""Tests for PDE game MCTS training integration.

Validates that:
- PDEGameAdapter creates valid GameInterface for basis selection
- Short self-play episodes generate valid experiences
- MCTS can search the PDE game tree
- Error reduction is tracked per episode
"""

from __future__ import annotations

import numpy as np
import pytest

import src.pde.register_games  # noqa: F401  — ensure PDE games are registered
from src.games.wrapper import StatefulGameWrapper
from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS
from src.pde.config import BasisSelectionConfig, PDEConfig, PDEGameConfig, PDEType
from src.pde.game_interface import PDEGameInterface
from src.pde.games.basis_selection import BasisSelectionGame
from src.pde.mcts_adapter import PDEGameAdapter
from src.pde.operators import PoissonOperator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pde_config() -> PDEConfig:
    """Minimal Poisson PDE config."""
    return PDEConfig(
        name="test_poisson",
        pde_type=PDEType.POISSON,
        domain_dim=2,
        domain_min=[0.0, 0.0],
        domain_max=[1.0, 1.0],
    )


@pytest.fixture()
def game_config(pde_config: PDEConfig) -> PDEGameConfig:
    """Small game config for fast tests."""
    return PDEGameConfig(
        name="test_basis",
        pde_config=pde_config,
        game_mode="basis_selection",
        max_steps=5,
        error_tolerance=0.01,
        computational_budget=1e4,
        basis_config=BasisSelectionConfig(
            name="test_basis_cfg",
            max_basis_functions=8,
            basis_type="fourier",
            max_frequency=3,
            n_collocation_points=50,
            n_boundary_points_per_face=10,
        ),
    )


@pytest.fixture()
def basis_game(pde_config: PDEConfig, game_config: PDEGameConfig) -> BasisSelectionGame:
    operator = PoissonOperator(pde_config)
    return BasisSelectionGame(operator, game_config)


@pytest.fixture()
def adapter(basis_game: BasisSelectionGame) -> PDEGameAdapter:
    return PDEGameAdapter(basis_game)


@pytest.fixture()
def pde_interface(basis_game: BasisSelectionGame) -> PDEGameInterface:
    return PDEGameInterface(pde_game=basis_game, grid_size=8)


# ---------------------------------------------------------------------------
# PDEGameAdapter as valid MCTS GameInterface
# ---------------------------------------------------------------------------


class TestAdapterAsMCTSInterface:
    """Verify PDEGameAdapter satisfies the MCTS GameInterface protocol."""

    def test_get_state_returns_float32_array(self, adapter: PDEGameAdapter) -> None:
        state = adapter.get_state()
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32

    def test_get_legal_actions_non_empty(self, adapter: PDEGameAdapter) -> None:
        actions = adapter.get_legal_actions()
        assert len(actions) > 0

    def test_apply_action_advances_step(self, adapter: PDEGameAdapter) -> None:
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        assert adapter.state.step == 1

    def test_is_terminal_false_initially(self, adapter: PDEGameAdapter) -> None:
        assert not adapter.is_terminal()

    def test_get_winner_valid(self, adapter: PDEGameAdapter) -> None:
        assert adapter.get_winner() in (-1, 0, 1)

    def test_clone_independent(self, adapter: PDEGameAdapter) -> None:
        cloned = adapter.clone()
        actions = cloned.get_legal_actions()
        cloned.apply_action(actions[0])
        assert adapter.state.step == 0
        assert cloned.state.step == 1


# ---------------------------------------------------------------------------
# MCTS search on PDE game tree
# ---------------------------------------------------------------------------


class TestMCTSSearchOnPDE:
    """Verify that MCTS can search the PDE game tree."""

    def test_mcts_search_returns_policy(self, adapter: PDEGameAdapter) -> None:
        """MCTS search should return a non-empty action distribution."""
        n_actions = adapter.pde_game.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=8, c_puct=1.5)

        policy = mcts.search(adapter, add_noise=False)

        assert isinstance(policy, dict)
        assert len(policy) > 0
        # All keys should be valid actions
        legal = set(adapter.get_legal_actions())
        for action in policy:
            assert action in legal
        # Probabilities should sum to ~1
        total = sum(policy.values())
        assert abs(total - 1.0) < 0.01

    def test_mcts_get_action_returns_valid_action(self, adapter: PDEGameAdapter) -> None:
        """MCTS.get_action should return a legal action."""
        n_actions = adapter.pde_game.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=8, c_puct=1.5)

        action = mcts.get_action(adapter, temperature=1.0, add_noise=False)

        assert isinstance(action, int)
        assert action in adapter.get_legal_actions()

    def test_mcts_with_stateful_wrapper(self, pde_interface: PDEGameInterface) -> None:
        """MCTS should work via StatefulGameWrapper (trainer path)."""
        state = pde_interface.initial_state()
        wrapper = StatefulGameWrapper(pde_interface, state)

        n_actions = pde_interface.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=8, c_puct=1.5)

        policy = mcts.search(wrapper, add_noise=False)
        assert len(policy) > 0


# ---------------------------------------------------------------------------
# 2-step self-play generates valid experiences
# ---------------------------------------------------------------------------


class TestSelfPlayExperiences:
    """Verify that short self-play episodes produce valid data."""

    def test_two_step_episode_via_adapter(self, adapter: PDEGameAdapter) -> None:
        """Play 2 steps through the adapter and verify state evolution."""
        n_actions = adapter.pde_game.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=4, c_puct=1.5)

        states: list[np.ndarray] = []
        policies: list[np.ndarray] = []
        actions_taken: list[int] = []

        for _step in range(2):
            if adapter.is_terminal():
                break

            state_arr = adapter.get_state()
            states.append(state_arr)

            policy_dist = mcts.search(adapter, add_noise=False)

            # Convert to full-size policy vector
            policy = np.zeros(n_actions, dtype=np.float32)
            for a, p in policy_dist.items():
                policy[a] = p
            policies.append(policy)

            # Pick best action
            action = max(policy_dist, key=lambda a: policy_dist[a])
            actions_taken.append(action)
            adapter.apply_action(action)
            mcts.advance(action)

        assert len(states) == 2
        assert len(policies) == 2
        assert len(actions_taken) == 2
        # Each policy sums to ~1
        for p in policies:
            assert abs(p.sum() - 1.0) < 0.01

    def test_two_step_episode_via_wrapper(self, pde_interface: PDEGameInterface) -> None:
        """Play 2 steps via StatefulGameWrapper (matches trainer code)."""
        n_actions = pde_interface.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=4, c_puct=1.5)

        state = pde_interface.initial_state()

        states_collected = []
        for _step in range(2):
            if pde_interface.is_terminal(state):
                break

            wrapper = StatefulGameWrapper(pde_interface, state)
            policy_dist = mcts.search(wrapper, add_noise=False)

            policy = np.zeros(n_actions, dtype=np.float32)
            for a, p in policy_dist.items():
                policy[a] = p

            states_collected.append(pde_interface.to_tensor(state).cpu().numpy())

            action = max(policy_dist, key=lambda a: policy_dist[a])
            state = pde_interface.apply_action(state, action)
            mcts.advance(action)

        assert len(states_collected) == 2
        assert state.move_number == 2


# ---------------------------------------------------------------------------
# Error reduction tracking
# ---------------------------------------------------------------------------


class TestErrorReductionLogging:
    """Verify that error reduction is tracked across steps."""

    def test_error_history_grows(self, adapter: PDEGameAdapter) -> None:
        """error_history should grow by 1 per action."""
        assert len(adapter.error_history) == 1

        actions = adapter.get_legal_actions()
        for i, action in enumerate(actions[:3]):
            if adapter.is_terminal():
                break
            adapter.apply_action(action)
            assert len(adapter.error_history) == 2 + i

    def test_error_reduction_computed(self, adapter: PDEGameAdapter) -> None:
        """error_reduction should be a float after actions."""
        actions = adapter.get_legal_actions()
        adapter.apply_action(actions[0])
        reduction = adapter.error_reduction
        assert isinstance(reduction, float)

    def test_error_tracked_in_game_state_metadata(
        self, pde_interface: PDEGameInterface,
    ) -> None:
        """Error should be recorded in GameState.metadata after each step."""
        state = pde_interface.initial_state()
        assert "error_estimate" in state.metadata
        initial_error = state.metadata["error_estimate"]
        assert initial_error > 0

        actions = pde_interface.get_legal_actions(state)
        new_state = pde_interface.apply_action(state, actions[0])
        assert "error_estimate" in new_state.metadata

    def test_play_episode_logs_errors(self, adapter: PDEGameAdapter) -> None:
        """A complete mini-episode should produce a full error trajectory."""
        max_steps = 5
        for _ in range(max_steps):
            if adapter.is_terminal():
                break
            actions = adapter.get_legal_actions()
            if not actions:
                break
            adapter.apply_action(actions[0])

        # Error history should have initial + actions taken
        assert len(adapter.error_history) >= 2
        # All error values should be finite non-negative
        for err in adapter.error_history:
            assert err >= 0.0
            assert np.isfinite(err)
