"""Integration tests for multi-game switching.

Verifies that the training pipeline works correctly when switching
between different game types (Go and Chess) without crashes.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

from config.schemas import (
    AlphaGalerkinConfig,
    DomainConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)
from src.games.chess import ChessGame
from src.games.go import GoGame
from src.modeling.model import AlphaGalerkinModel
from src.training.trainer import Trainer


def _make_go_config() -> AlphaGalerkinConfig:
    """Create a minimal Go training configuration."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=OperatorConfig(
            d_model=32,
            d_key=16,
            d_value=16,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
            # Go: 17 input channels, position-based policy head
            input_channels=17,
            game_type="go",
            action_space_size=None,
        ),
        mcts=MCTSConfig(
            n_simulations=5,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=4,
            gradient_clip=1.0,
            lr_scheduler="constant",
            warmup_steps=0,
            total_steps=10,
            n_self_play_games=2,
            replay_buffer_size=100,
            checkpoint_interval=100,
            use_amp=False,
        ),
        experiment_name="multi_game_go",
        seed=42,
        board_sizes=[9],
    )


def _make_chess_config() -> AlphaGalerkinConfig:
    """Create a minimal Chess training configuration."""
    return AlphaGalerkinConfig(
        domain=DomainConfig(),
        operator=OperatorConfig(
            d_model=32,
            d_key=16,
            d_value=16,
            d_ffn=64,
            n_heads=2,
            n_galerkin_layers=1,
            n_softmax_layers=1,
            n_fourier_features=16,
            use_fnet_mixing=False,
            # Chess: 119 input channels, dense action-space policy head
            input_channels=119,
            game_type="chess",
            action_space_size=4672,
        ),
        mcts=MCTSConfig(
            n_simulations=5,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        ),
        training=TrainingConfig(
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=4,
            gradient_clip=1.0,
            lr_scheduler="constant",
            warmup_steps=0,
            total_steps=10,
            n_self_play_games=2,
            replay_buffer_size=100,
            checkpoint_interval=100,
            use_amp=False,
        ),
        experiment_name="multi_game_chess",
        seed=42,
        board_sizes=[8],
    )


class TestMultiGameSwitching:
    """Tests for sequential multi-game training without crashes."""

    def test_multi_game_sequential_no_crash(self) -> None:
        """Train Go for 3 steps, then Chess for 3 steps.

        Verifies:
        - Go trainer completes 3 steps with finite loss
        - Chess trainer completes 3 steps with finite loss
        - Both trainers reach global_step=3
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # ----- Go training -----
            go_config = _make_go_config()
            go_game = GoGame()
            go_model = AlphaGalerkinModel(go_config.operator)
            go_trainer = Trainer(
                model=go_model,
                config=go_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir) / "go",
                game=go_game,
            )
            go_trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

            assert go_trainer.global_step == 3
            go_history = go_trainer.get_metrics_history()
            assert len(go_history) == 3
            for m in go_history:
                assert math.isfinite(m["total_loss"]), (
                    f"Go: non-finite loss at step {m['step']}"
                )

            # ----- Chess training -----
            chess_config = _make_chess_config()
            chess_game = ChessGame()
            chess_model = AlphaGalerkinModel(chess_config.operator)
            chess_trainer = Trainer(
                model=chess_model,
                config=chess_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir) / "chess",
                game=chess_game,
            )
            chess_trainer.train(n_steps=3, log_interval=1, checkpoint_interval=100)

            assert chess_trainer.global_step == 3
            chess_history = chess_trainer.get_metrics_history()
            assert len(chess_history) == 3
            for m in chess_history:
                assert math.isfinite(m["total_loss"]), (
                    f"Chess: non-finite loss at step {m['step']}"
                )

    def test_go_training_produces_valid_metrics(self) -> None:
        """Smoke test that Go training produces valid metric keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            go_config = _make_go_config()
            go_game = GoGame()
            go_model = AlphaGalerkinModel(go_config.operator)
            go_trainer = Trainer(
                model=go_model,
                config=go_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
                game=go_game,
            )
            go_trainer.train(n_steps=2, log_interval=1, checkpoint_interval=100)

            history = go_trainer.get_metrics_history()
            assert len(history) == 2
            # Verify expected metric keys are present
            expected_keys = {
                "step",
                "total_loss",
                "policy_loss",
                "value_loss",
                "lbb_loss",
                "learning_rate",
            }
            for m in history:
                assert expected_keys.issubset(m.keys())

    def test_chess_training_produces_valid_metrics(self) -> None:
        """Smoke test that Chess training produces valid metric keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chess_config = _make_chess_config()
            chess_game = ChessGame()
            chess_model = AlphaGalerkinModel(chess_config.operator)
            chess_trainer = Trainer(
                model=chess_model,
                config=chess_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
                game=chess_game,
            )
            chess_trainer.train(n_steps=2, log_interval=1, checkpoint_interval=100)

            history = chess_trainer.get_metrics_history()
            assert len(history) == 2
            expected_keys = {
                "step",
                "total_loss",
                "policy_loss",
                "value_loss",
                "lbb_loss",
                "learning_rate",
            }
            for m in history:
                assert expected_keys.issubset(m.keys())
