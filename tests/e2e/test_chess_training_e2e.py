"""End-to-end chess training smoke test.

Validates the full pipeline: model → self-play → replay buffer → training step.
Uses minimal configuration to run fast while exercising the entire code path.
"""

from __future__ import annotations

import pytest
import torch

from config.schemas import MCTSConfig, OperatorConfig, TrainingConfig
from src.games.chess import ChessGame
from src.modeling.model import AlphaGalerkinModel
from src.training.replay_buffer import UniformReplayBuffer
from src.training.self_play import SelfPlayWorker


@pytest.fixture
def chess_config() -> tuple[OperatorConfig, MCTSConfig, TrainingConfig]:
    """Minimal chess training configuration for smoke test."""
    op = OperatorConfig(
        d_model=32,
        d_key=8,
        d_value=8,
        d_ffn=64,
        n_heads=2,
        n_galerkin_layers=1,
        n_softmax_layers=1,
        n_fourier_features=16,
        input_channels=119,
        game_type="chess",
        action_space_size=4672,
    )
    mcts = MCTSConfig(n_simulations=4, c_puct=2.5, dirichlet_alpha=0.3)
    training = TrainingConfig(
        total_steps=2,
        batch_size=8,
        n_self_play_games=2,
        replay_buffer_size=1000,
        learning_rate=0.001,
    )
    return op, mcts, training


class TestChessTrainingE2E:
    """End-to-end chess training smoke tests."""

    def test_self_play_to_training_step(self, chess_config: tuple) -> None:
        """Full pipeline: self-play → buffer → training step."""
        op_config, mcts_config, train_config = chess_config
        game = ChessGame()

        # 1. Create model
        model = AlphaGalerkinModel(op_config)
        model.eval()

        # 2. Create self-play worker
        worker = SelfPlayWorker(
            model=model,
            mcts_config=mcts_config,
            device=torch.device("cpu"),
            game=game,
        )

        # 3. Play 2 games
        records = worker.generate_games(n_games=2, board_size=8)
        assert len(records) == 2

        # 4. Convert to experiences
        experiences = worker.generate_experiences(n_games=2, board_size=8)
        assert len(experiences) > 0

        for exp in experiences:
            assert exp.board_state.shape == (119, 8, 8)
            assert exp.target_policy.shape == (4672,)
            assert -1.0 <= exp.target_value <= 1.0
            assert exp.board_size == 8

        # 5. Add to replay buffer
        buffer = UniformReplayBuffer(capacity=train_config.replay_buffer_size)
        for exp in experiences:
            buffer.add(exp)

        assert len(buffer) == len(experiences)

        # 6. Sample a batch
        batch_exps = buffer.sample(min(8, len(experiences)))
        assert len(batch_exps) <= 8

    def test_model_gradient_flow_chess(self, chess_config: tuple) -> None:
        """Verify gradients flow through the chess model end-to-end."""
        op_config, _, _ = chess_config
        model = AlphaGalerkinModel(op_config)
        model.train()

        # Forward pass with chess input
        x = torch.randn(2, 119, 8, 8)
        policy, value, lbb = model(x)

        assert policy.shape == (2, 4672)
        assert value.shape == (2, 1)

        # Backward pass
        loss = policy.sum() + value.sum()
        if lbb is not None:
            loss = loss + lbb
        loss.backward()

        # Verify gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_self_play_game_terminates(self, chess_config: tuple) -> None:
        """Verify self-play games terminate (no infinite loops)."""
        op_config, mcts_config, _ = chess_config
        game = ChessGame()
        model = AlphaGalerkinModel(op_config)
        model.eval()

        worker = SelfPlayWorker(
            model=model,
            mcts_config=mcts_config,
            device=torch.device("cpu"),
            game=game,
        )

        # Should terminate within 500 moves (game has max_moves limit)
        records = worker.generate_games(n_games=1, board_size=8)
        assert len(records) == 1
        assert len(records[0].states) > 0
        assert len(records[0].policies) > 0
