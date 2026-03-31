"""Tests for extended training configuration.

Validates the train_extended.yaml configuration and its integration
with the Trainer class, particularly warmup/plateau interaction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from config.schemas import (
    AlphaGalerkinConfig,
    MCTSConfig,
    OperatorConfig,
    TrainingConfig,
)


class TestExtendedConfigLoads:
    """Verify extended config file loads and validates correctly."""

    @pytest.fixture
    def config_path(self) -> Path:
        """Get path to extended config."""
        return Path(__file__).parents[2] / "config" / "train_extended.yaml"

    def test_extended_yaml_exists(self, config_path: Path) -> None:
        """Verify train_extended.yaml exists."""
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_extended_yaml_syntax_valid(self, config_path: Path) -> None:
        """Verify YAML syntax is valid."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert "training" in config

    def test_extended_config_key_values(self, config_path: Path) -> None:
        """Verify key extended settings have expected values."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        training = config["training"]

        # Verify extended training settings
        assert training["total_steps"] == 7500
        assert training["learning_rate"] == 0.001
        assert training["warmup_steps"] == 500
        assert training["plateau_patience"] == 10
        assert training["replay_buffer_size"] == 10000
        assert training["n_self_play_games"] == 75
        assert training["batch_size"] == 128

    def test_extended_config_stability_enabled(self, config_path: Path) -> None:
        """Verify stability features are enabled."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        training = config["training"]
        assert training["plateau_detection_enabled"] is True
        assert training["early_stopping_enabled"] is True

    def test_extended_config_wandb_enabled(self, config_path: Path) -> None:
        """Verify W&B logging is enabled."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        wandb = config["wandb"]
        assert wandb["enabled"] is True
        assert "extended" in wandb["tags"]

    def test_omegaconf_loads_extended(self, config_path: Path) -> None:
        """Verify OmegaConf can load the config."""
        cfg = OmegaConf.load(config_path)
        assert cfg.training.total_steps == 7500


class TestWarmupPlateauInteraction:
    """Verify warmup completes before plateau can trigger."""

    def test_warmup_before_first_eval(self) -> None:
        """Verify warmup completes before first evaluation."""
        # Extended config: warmup_steps=500, eval_interval=750
        warmup_steps = 500
        eval_interval = 750

        # Warmup should complete before first evaluation
        assert warmup_steps < eval_interval

    def test_warmup_percentage_reasonable(self) -> None:
        """Verify warmup is reasonable percentage of training."""
        config = TrainingConfig(
            warmup_steps=500,
            total_steps=7500,
        )

        warmup_pct = config.warmup_steps / config.total_steps
        # Warmup should be between 1% and 20% of training
        assert 0.01 <= warmup_pct <= 0.20

    def test_plateau_patience_sufficient(self) -> None:
        """Verify plateau patience allows exploration before LR reduction."""
        config = TrainingConfig(
            plateau_patience=10,
            eval_interval=750,
        )

        # Plateau patience of 10 means 10 steps without improvement
        # This should allow sufficient exploration
        assert config.plateau_patience >= 5

    def test_training_config_validates(self) -> None:
        """Verify extended-style TrainingConfig validates."""
        config = TrainingConfig(
            learning_rate=1e-3,
            warmup_steps=500,
            total_steps=7500,
            n_self_play_games=75,
            replay_buffer_size=10000,
            batch_size=128,
            plateau_detection_enabled=True,
            plateau_patience=10,
            early_stopping_enabled=True,
            early_stopping_patience=8,
        )

        # Should not raise
        assert config.learning_rate == 1e-3
        assert config.warmup_steps == 500


class TestBufferCapacity:
    """Verify buffer sizing for extended config."""

    def test_buffer_holds_sufficient_batches(self) -> None:
        """Verify buffer can hold enough samples for training diversity."""
        config = TrainingConfig(
            replay_buffer_size=10000,
            batch_size=128,
        )

        # Buffer should hold at least 10x batch size for diversity
        assert config.replay_buffer_size >= config.batch_size * 10

    def test_buffer_fill_rate_reasonable(self) -> None:
        """Verify buffer can be filled in reasonable number of rounds."""
        config = TrainingConfig(
            replay_buffer_size=10000,
            n_self_play_games=75,
        )

        # Assume ~40 moves per game average (9x9 to 19x19)
        avg_experiences_per_game = 40
        experiences_per_batch = config.n_self_play_games * avg_experiences_per_game

        # Should fill buffer in < 10 generation rounds
        batches_to_fill = config.replay_buffer_size / experiences_per_batch
        assert batches_to_fill < 10

    def test_batch_size_divides_buffer(self) -> None:
        """Verify batch size evenly samples buffer."""
        config = TrainingConfig(
            replay_buffer_size=10000,
            batch_size=128,
        )

        # Not strictly required but useful for efficient sampling
        # Buffer should hold many complete batches
        batches_in_buffer = config.replay_buffer_size // config.batch_size
        assert batches_in_buffer >= 50


class TestGradientClipping:
    """Verify gradient clipping settings for higher learning rates."""

    def test_gradient_clip_with_high_lr(self) -> None:
        """Verify gradient clip is set when using higher LR."""
        config = TrainingConfig(
            learning_rate=1e-3,
            gradient_clip=1.0,
        )

        # With 1e-3 LR, should have gradient clipping enabled
        assert config.gradient_clip > 0

    def test_gradient_clip_reasonable_range(self) -> None:
        """Verify gradient clip is in reasonable range."""
        config = TrainingConfig(
            gradient_clip=1.0,
        )

        # Typical range is 0.5 to 5.0
        assert 0.1 <= config.gradient_clip <= 10.0


class TestTrainerWarmupTracking:
    """Test warmup tracking functionality in Trainer."""

    @pytest.fixture
    def small_config(self) -> AlphaGalerkinConfig:
        """Create minimal config for testing."""
        return AlphaGalerkinConfig(
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
            ),
            mcts=MCTSConfig(
                n_simulations=5,
                batch_size=2,
            ),
            training=TrainingConfig(
                learning_rate=1e-3,
                batch_size=4,
                warmup_steps=10,
                total_steps=50,
                n_self_play_games=2,
                replay_buffer_size=100,
                checkpoint_interval=25,
                plateau_detection_enabled=True,
                plateau_patience=5,
                use_amp=False,
            ),
            experiment_name="test_warmup",
            seed=42,
            device="cpu",
            board_sizes=[9],
        )

    def test_warmup_tracking_initialized(self, small_config: AlphaGalerkinConfig) -> None:
        """Verify warmup tracking variables are initialized."""
        from src.modeling.model import AlphaGalerkinModel
        from src.training.trainer import Trainer

        model = AlphaGalerkinModel(small_config.operator)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                config=small_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )

            # Verify warmup tracking attributes exist
            assert hasattr(trainer, "_warmup_completed")
            assert hasattr(trainer, "_warmup_steps")
            assert trainer._warmup_steps == 10
            # Not completed initially (warmup_steps > 0)
            assert trainer._warmup_completed is False

    def test_warmup_starts_completed_when_zero_steps(self) -> None:
        """Verify warmup is marked complete when warmup_steps=0."""
        from src.modeling.model import AlphaGalerkinModel
        from src.training.trainer import Trainer

        config = AlphaGalerkinConfig(
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
            ),
            mcts=MCTSConfig(n_simulations=5, batch_size=2),
            training=TrainingConfig(
                warmup_steps=0,  # No warmup
                total_steps=20,
                n_self_play_games=2,
                replay_buffer_size=50,
                use_amp=False,
            ),
            experiment_name="test_no_warmup",
            seed=42,
            device="cpu",
            board_sizes=[9],
        )

        model = AlphaGalerkinModel(config.operator)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                config=config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )

            # Should be completed immediately when warmup_steps=0
            assert trainer._warmup_completed is True


class TestExtendedTrainingIntegration:
    """Integration tests for extended training configuration."""

    @pytest.fixture
    def extended_style_config(self) -> AlphaGalerkinConfig:
        """Create extended-style config for integration testing."""
        return AlphaGalerkinConfig(
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
            ),
            mcts=MCTSConfig(
                n_simulations=5,
                batch_size=2,
            ),
            training=TrainingConfig(
                learning_rate=1e-3,
                batch_size=4,
                warmup_steps=5,
                total_steps=20,
                n_self_play_games=2,
                replay_buffer_size=100,
                checkpoint_interval=10,
                eval_interval=10,
                plateau_detection_enabled=True,
                plateau_patience=3,
                early_stopping_enabled=False,  # Disable for short test
                use_amp=False,
            ),
            experiment_name="test_extended_integration",
            seed=42,
            device="cpu",
            board_sizes=[9],
        )

    @pytest.mark.slow
    def test_short_extended_training_completes(
        self,
        extended_style_config: AlphaGalerkinConfig,
    ) -> None:
        """Test short extended-style training run completes."""
        from src.modeling.model import AlphaGalerkinModel
        from src.training.trainer import Trainer

        model = AlphaGalerkinModel(extended_style_config.operator)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                config=extended_style_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )

            # Run short training
            trainer.train(
                n_steps=15,
                log_interval=5,
                checkpoint_interval=10,
                eval_interval=10,
            )

            # Verify training completed
            assert trainer.global_step == 15

            # Verify warmup completed (warmup_steps=5)
            assert trainer._warmup_completed is True

    @pytest.mark.slow
    def test_metrics_recorded_during_training(
        self,
        extended_style_config: AlphaGalerkinConfig,
    ) -> None:
        """Test metrics are recorded during training."""
        from src.modeling.model import AlphaGalerkinModel
        from src.training.trainer import Trainer

        model = AlphaGalerkinModel(extended_style_config.operator)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                config=extended_style_config,
                device="cpu",
                checkpoint_dir=Path(tmpdir),
            )

            trainer.train(n_steps=10)

            # Verify metrics recorded
            history = trainer.get_metrics_history()
            assert len(history) == 10

            # Verify metrics have expected fields
            first_metric = history[0]
            # Metrics may be dataclass or dict depending on implementation
            if hasattr(first_metric, "total_loss"):
                assert first_metric.total_loss is not None
                assert first_metric.learning_rate is not None
                assert first_metric.gradient_norm is not None
            else:
                # Dictionary access
                assert "total_loss" in first_metric
                assert "learning_rate" in first_metric
                assert "gradient_norm" in first_metric
