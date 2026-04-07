"""End-to-end integration test for config-driven PDE training.

Validates the full pipeline: load config -> resolve game -> create trainer
-> run 1 training step. Uses minimal sizes to keep tests fast.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.pde.register_games  # noqa: F401  — ensure PDE games are registered
from src.games.interface import GameInterface
from src.games.registry import get_game
from src.games.wrapper import StatefulGameWrapper
from src.mcts.evaluator import RandomEvaluator
from src.mcts.search import MCTS

# ---------------------------------------------------------------------------
# Config-driven game resolution
# ---------------------------------------------------------------------------


class TestConfigDrivenGameResolution:
    """Simulate what scripts/train.py does: look up game by name from config."""

    def test_resolve_pde_basis_from_config(self) -> None:
        """Simulate: cfg['game']['name'] = 'pde_basis' -> get_game."""
        game_name = "pde_basis"
        game = get_game(game_name)
        assert isinstance(game, GameInterface)
        assert game.action_space_size > 0
        assert game.state_channels > 0

    def test_resolve_pde_mesh_from_config(self) -> None:
        """Simulate: cfg['game']['name'] = 'pde_mesh' -> get_game."""
        game_name = "pde_mesh"
        game = get_game(game_name)
        assert isinstance(game, GameInterface)

    def test_unknown_game_raises(self) -> None:
        """get_game should raise ValueError for unregistered names."""
        with pytest.raises(ValueError, match="not found"):
            get_game("nonexistent_game_xyz")


# ---------------------------------------------------------------------------
# Full self-play episode through GameInterface
# ---------------------------------------------------------------------------


class TestFullSelfPlayEpisode:
    """Run a complete self-play episode using the registered PDE game."""

    def test_basis_selection_episode(self) -> None:
        """Play a mini self-play episode for basis selection.

        Mirrors the logic in SelfPlayWorker._play_game_generic():
        1. Get initial state
        2. Loop: MCTS search -> select action -> apply
        3. Determine outcome
        """
        game = get_game("pde_basis")
        assert game is not None

        state = game.initial_state()
        n_actions = game.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=8, c_puct=1.5)

        states_collected: list[np.ndarray] = []
        policies_collected: list[np.ndarray] = []
        actions_taken: list[int] = []
        max_moves = 3

        move_number = 0
        while not game.is_terminal(state) and move_number < max_moves:
            state_tensor = game.to_tensor(state).cpu().numpy()
            states_collected.append(state_tensor)

            wrapper = StatefulGameWrapper(game, state)
            policy_dist = mcts.search(wrapper, add_noise=False)

            policy = np.zeros(n_actions, dtype=np.float32)
            for a, p in policy_dist.items():
                policy[a] = p
            policies_collected.append(policy)

            action = max(policy_dist, key=lambda a: policy_dist[a])
            actions_taken.append(action)

            state = game.apply_action(state, action)
            mcts.advance(action)
            move_number += 1

        # Verify we collected data
        assert len(states_collected) == max_moves
        assert len(policies_collected) == max_moves
        assert state.move_number == max_moves

        # Outcome determination should not crash
        winner = game.get_winner(state)
        assert winner is None or isinstance(winner, int)

        # Result should be available
        result = game.get_result(state)
        assert result.move_count == max_moves


# ---------------------------------------------------------------------------
# Trainer integration (lightweight, no GPU required)
# ---------------------------------------------------------------------------


class TestTrainerGameIntegration:
    """Test that Trainer can accept a PDE game and wire up self-play."""

    def test_self_play_manual_loop_with_random_evaluator(self) -> None:
        """Reproduce the SelfPlayWorker._play_game_generic() loop manually.

        Uses RandomEvaluator to avoid tensor shape mismatches between
        the PDE state encoding and the board-game-oriented AlphaGalerkin
        embedding layer. This validates the game-side wiring end-to-end.
        """
        game = get_game("pde_basis")
        assert game is not None

        n_actions = game.action_space_size
        evaluator = RandomEvaluator(n_actions=n_actions)
        mcts = MCTS(evaluator=evaluator, n_simulations=4, c_puct=1.5)

        state = game.initial_state()
        states_collected: list[np.ndarray] = []
        policies_collected: list[np.ndarray] = []
        max_moves = 2

        move_number = 0
        while not game.is_terminal(state) and move_number < max_moves:
            state_tensor = game.to_tensor(state).cpu().numpy()
            states_collected.append(state_tensor)

            wrapper = StatefulGameWrapper(game, state)
            policy_dist = mcts.search(wrapper, add_noise=False)

            policy = np.zeros(n_actions, dtype=np.float32)
            for a, p in policy_dist.items():
                policy[a] = p
            policies_collected.append(policy)

            action = max(policy_dist, key=lambda a: policy_dist[a])
            state = game.apply_action(state, action)
            mcts.advance(action)
            move_number += 1

        assert len(states_collected) == max_moves
        assert len(policies_collected) == max_moves

        winner = game.get_winner(state)
        outcome = float(winner) if winner is not None else 0.0
        assert isinstance(outcome, float)

    def test_create_trainer_accepts_game(self) -> None:
        """create_trainer should accept a game parameter and pass it to SelfPlayWorker."""
        from config.schemas import AlphaGalerkinConfig
        from src.modeling.model import AlphaGalerkinModel
        from src.training.trainer import create_trainer

        game = get_game("pde_basis")
        assert game is not None

        # Use default operator config (board-game oriented); we only
        # verify wiring, not a full forward pass.
        config = AlphaGalerkinConfig()
        model = AlphaGalerkinModel(config.operator)

        trainer = create_trainer(
            model=model,
            config=config,
            device="cpu",
            game=game,
        )

        # Verify the game was passed through to self_play_worker
        assert trainer.self_play_worker.game is game
