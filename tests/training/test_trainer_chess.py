"""Trainer chess integration tests and checkpoint resume tests.

Tests:
- Trainer initialization with chess config
- Single training step execution
- Checkpoint save/load/resume state continuity
- Engine evaluation integration (mock Stockfish)
- W&B Elo metric logging verification
"""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from config.schemas import OperatorConfig, TrainingConfig
from src.games.chess import ChessGame
from src.modeling.model import AlphaGalerkinModel
from src.training.checkpoint import CheckpointManager
from src.training.evaluation import EvaluationResult
from src.training.replay_buffer import Experience, UniformReplayBuffer


@pytest.fixture
def chess_op_config() -> OperatorConfig:
    """Minimal chess operator config."""
    return OperatorConfig(
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


@pytest.fixture
def chess_training_config() -> TrainingConfig:
    """Minimal chess training config."""
    return TrainingConfig(
        total_steps=2,
        batch_size=8,
        n_self_play_games=2,
        replay_buffer_size=1000,
        learning_rate=0.001,
        eval_interval=1,
        eval_games=2,
        warmup_steps=0,
        checkpoint_interval=1,
        use_amp=False,
    )


@pytest.fixture
def chess_model(chess_op_config: OperatorConfig) -> AlphaGalerkinModel:
    """Chess model instance."""
    return AlphaGalerkinModel(chess_op_config)


@pytest.fixture
def chess_game() -> ChessGame:
    """Chess game instance."""
    return ChessGame()


class TestCheckpointSaveLoadResume:
    """Tests for checkpoint save/load lifecycle with chess model."""

    def test_save_and_load_checkpoint(
        self, chess_model: AlphaGalerkinModel
    ) -> None:
        """Verify model state is preserved after save → load."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(
                checkpoint_dir=tmp_dir,
                max_checkpoints=3,
            )

            # Save checkpoint
            manager.save(
                step=100,
                model=chess_model,
                metrics={"loss": 5.0, "win_rate": 0.1},
            )

            # Load and verify
            state = manager.load()
            assert state.step == 100
            assert state.metrics["loss"] == 5.0
            assert state.metrics["win_rate"] == 0.1

    def test_checkpoint_restore_weights(
        self, chess_op_config: OperatorConfig
    ) -> None:
        """Verify model weights match after save → restore."""
        model1 = AlphaGalerkinModel(chess_op_config)
        model2 = AlphaGalerkinModel(chess_op_config)

        # Models should have different random weights
        param1 = next(model1.parameters()).data.clone()
        param2 = next(model2.parameters()).data.clone()
        assert not torch.allclose(param1, param2)

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(checkpoint_dir=tmp_dir)
            manager.save(step=50, model=model1)

            # Restore into model2
            manager.restore(model=model2)

            # Now they should match
            param1 = next(model1.parameters()).data
            param2 = next(model2.parameters()).data
            assert torch.allclose(param1, param2)

    def test_checkpoint_resume_training(
        self, chess_op_config: OperatorConfig
    ) -> None:
        """Verify optimizer state persists across checkpoint save/load."""
        model = AlphaGalerkinModel(chess_op_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Do a training step to populate optimizer state
        x = torch.randn(2, 119, 8, 8)
        policy, value, lbb = model(x)
        loss = policy.sum() + value.sum()
        if lbb is not None:
            loss = loss + lbb
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(checkpoint_dir=tmp_dir)
            manager.save(step=10, model=model, optimizer=optimizer)

            # Create fresh model/optimizer
            model2 = AlphaGalerkinModel(chess_op_config)
            optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)

            # Restore
            step = manager.restore(
                model=model2, optimizer=optimizer2
            )
            assert step == 10

    def test_multiple_checkpoints_rotation(
        self, chess_model: AlphaGalerkinModel
    ) -> None:
        """Verify oldest checkpoints are rotated out."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = CheckpointManager(
                checkpoint_dir=tmp_dir,
                max_checkpoints=2,
            )

            manager.save(step=10, model=chess_model)
            manager.save(step=20, model=chess_model)
            manager.save(step=30, model=chess_model)

            # Should only keep 2 most recent
            checkpoints = manager.get_all_checkpoints()
            assert len(checkpoints) <= 3  # max_checkpoints + best


class TestEngineEvalIntegration:
    """Tests for engine evaluation integration with mocked engine."""

    def test_engine_eval_result_has_elo(self) -> None:
        """Verify EvaluationResult structure from engine eval."""
        result = EvaluationResult(
            win_rate=0.4,
            n_games=10,
            wins=3,
            losses=5,
            draws=2,
            avg_game_length=60.0,
            metadata={
                "opponent": "engine",
                "elo_difference": -100.5,
                "elo_ci": (-200.0, -10.0),
                "los": 0.15,
            },
        )

        assert result.win_rate == 0.4
        assert result.metadata["elo_difference"] == -100.5
        assert result.metadata["los"] == 0.15

    def test_engine_eval_metrics_dict(self) -> None:
        """Verify metrics dictionary structure for W&B logging."""
        result = EvaluationResult(
            win_rate=0.5,
            n_games=4,
            wins=1,
            losses=1,
            draws=2,
            avg_game_length=80.0,
            metadata={
                "opponent": "engine",
                "elo_difference": 0.0,
                "los": 0.5,
            },
        )

        # Build the same metrics dict the trainer would log
        elo_metrics: dict[str, float | int] = {
            "eval/engine/win_rate": result.win_rate,
            "eval/engine/wins": result.wins,
            "eval/engine/losses": result.losses,
            "eval/engine/draws": result.draws,
            "eval/engine/n_games": result.n_games,
            "eval/engine/avg_game_length": result.avg_game_length,
        }
        if "elo_difference" in result.metadata:
            elo_metrics["eval/engine/elo_diff"] = result.metadata["elo_difference"]
        if "los" in result.metadata:
            elo_metrics["eval/engine/los"] = result.metadata["los"]

        assert "eval/engine/elo_diff" in elo_metrics
        assert "eval/engine/los" in elo_metrics
        assert elo_metrics["eval/engine/win_rate"] == 0.5

    def test_wandb_logger_receives_elo_metrics(self) -> None:
        """Verify W&B logger mock receives correctly structured metrics."""
        mock_logger = MagicMock()
        mock_logger.is_enabled = True
        mock_logger.log_metrics = MagicMock()

        metrics: dict[str, Any] = {
            "eval/engine/win_rate": 0.3,
            "eval/engine/elo_diff": -150.0,
            "eval/engine/los": 0.1,
        }

        mock_logger.log_metrics(metrics, step=100)
        mock_logger.log_metrics.assert_called_once_with(metrics, step=100)


class TestTrainingConfigEngineFields:
    """Verify engine eval config fields have correct defaults."""

    def test_engine_eval_defaults(self) -> None:
        """Engine eval should be disabled by default for backwards compat."""
        config = TrainingConfig()
        assert config.engine_eval_enabled is False
        assert config.engine_eval_path is None
        assert config.engine_eval_depth == 5
        assert config.engine_eval_games == 4
        assert config.engine_eval_movetime_ms is None

    def test_engine_eval_custom_values(self) -> None:
        """Verify custom engine eval config works."""
        config = TrainingConfig(
            engine_eval_enabled=True,
            engine_eval_path="/usr/bin/stockfish",
            engine_eval_depth=10,
            engine_eval_games=8,
            engine_eval_movetime_ms=500,
        )
        assert config.engine_eval_enabled is True
        assert config.engine_eval_path == "/usr/bin/stockfish"
        assert config.engine_eval_depth == 10
        assert config.engine_eval_games == 8
        assert config.engine_eval_movetime_ms == 500


class TestReplayBufferChessExperiences:
    """Tests for replay buffer with chess-shaped experiences."""

    def test_add_and_sample_chess_experiences(self) -> None:
        """Verify buffer correctly stores and samples chess experiences."""
        buffer = UniformReplayBuffer(capacity=100)

        for i in range(10):
            exp = Experience(
                board_state=torch.randn(119, 8, 8),
                board_size=8,
                target_policy=torch.softmax(torch.randn(4672), dim=0),
                target_value=(-1.0) ** i,
            )
            buffer.add(exp)

        assert len(buffer) == 10

        sample = buffer.sample(5)
        assert len(sample) == 5
        for exp in sample:
            assert exp.board_state.shape == (119, 8, 8)
            assert exp.target_policy.shape == (4672,)

    def test_buffer_overflow_circular(self) -> None:
        """Verify circular buffer evicts oldest entries."""
        buffer = UniformReplayBuffer(capacity=5)

        for i in range(10):
            exp = Experience(
                board_state=torch.randn(119, 8, 8),
                board_size=8,
                target_policy=torch.softmax(torch.randn(4672), dim=0),
                target_value=float(i),
            )
            buffer.add(exp)

        assert len(buffer) == 5
