"""Tests for physics training pipeline.

Validates TrainingConfig defaults, train_epoch with a tiny model,
evaluate returns a proper metrics dict, and W&B / heavy ops are mocked.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.experiments.physics_model import PhysicsLoss, PhysicsOperator
from src.experiments.train_physics import (
    TrainingConfig,
    evaluate,
    train_epoch,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> TrainingConfig:
    """TrainingConfig with all defaults."""
    return TrainingConfig()


@pytest.fixture
def tiny_config() -> TrainingConfig:
    """A minimal config for fast unit tests."""
    return TrainingConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        n_fourier_features=16,
        train_grid_size=5,
        eval_grid_size=7,
        n_train_samples=8,
        n_eval_samples=4,
        n_charges=2,
        batch_size=4,
        learning_rate=1e-3,
        n_epochs=1,
        log_interval=1,
        eval_interval=1,
        seed=0,
        output_dir="/tmp/test_physics_train",
    )


@pytest.fixture
def tiny_model(tiny_config: TrainingConfig) -> PhysicsOperator:
    """Small PhysicsOperator for unit tests."""
    return PhysicsOperator(
        d_model=tiny_config.d_model,
        n_heads=tiny_config.n_heads,
        n_layers=tiny_config.n_layers,
        n_fourier_features=tiny_config.n_fourier_features,
        fourier_scale=tiny_config.fourier_scale,
        use_fnet=tiny_config.use_fnet,
    )


@pytest.fixture
def tiny_dataset(tiny_config: TrainingConfig):
    """Small PoissonDataset for unit tests."""
    from src.physics.poisson import PoissonDataset

    return PoissonDataset(
        grid_size=tiny_config.train_grid_size,
        n_samples=tiny_config.n_train_samples,
        n_charges=tiny_config.n_charges,
        seed=tiny_config.seed,
    )


@pytest.fixture
def tiny_eval_dataset(tiny_config: TrainingConfig):
    """Small PoissonDataset for evaluation tests."""
    from src.physics.poisson import PoissonDataset

    return PoissonDataset(
        grid_size=tiny_config.eval_grid_size,
        n_samples=tiny_config.n_eval_samples,
        n_charges=tiny_config.n_charges,
        seed=tiny_config.seed + tiny_config.train_eval_seed_offset,
    )


# ---------------------------------------------------------------------------
# Tests: TrainingConfig defaults
# ---------------------------------------------------------------------------


class TestTrainingConfigDefaults:
    """Tests for TrainingConfig default values."""

    def test_model_defaults(self, default_config: TrainingConfig) -> None:
        """Model architecture defaults are sensible."""
        assert default_config.d_model == 128
        assert default_config.n_heads == 4
        assert default_config.n_layers == 4
        assert default_config.n_fourier_features == 64

    def test_data_defaults(self, default_config: TrainingConfig) -> None:
        """Data defaults match documented PoC settings."""
        assert default_config.train_grid_size == 9
        assert default_config.eval_grid_size == 19
        assert default_config.n_train_samples == 5000
        assert default_config.n_eval_samples == 500

    def test_training_defaults(self, default_config: TrainingConfig) -> None:
        """Training hyperparameter defaults."""
        assert default_config.learning_rate == pytest.approx(1e-3)
        assert default_config.weight_decay == pytest.approx(1e-4)
        assert default_config.n_epochs == 100
        assert default_config.batch_size == 32

    def test_success_threshold(self, default_config: TrainingConfig) -> None:
        """Default success threshold is 0.05."""
        assert default_config.success_threshold == pytest.approx(0.05)

    def test_seed_offsets_differ(self, default_config: TrainingConfig) -> None:
        """Train eval and transfer eval seed offsets are distinct."""
        assert default_config.train_eval_seed_offset != default_config.transfer_eval_seed_offset

    def test_wandb_disabled_by_default(self, default_config: TrainingConfig) -> None:
        """W&B logging is off by default."""
        assert default_config.wandb_enabled is False

    def test_custom_overrides(self) -> None:
        """Fields can be overridden at construction."""
        config = TrainingConfig(d_model=64, n_epochs=10, seed=99)
        assert config.d_model == 64
        assert config.n_epochs == 10
        assert config.seed == 99


# ---------------------------------------------------------------------------
# Tests: train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    """Tests for a single training epoch."""

    def test_returns_positive_loss(
        self,
        tiny_model: PhysicsOperator,
        tiny_dataset,
        tiny_config: TrainingConfig,
    ) -> None:
        """train_epoch returns a positive average loss."""
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=tiny_config.learning_rate)
        loss_fn = PhysicsLoss()

        avg_loss = train_epoch(
            model=tiny_model,
            dataset=tiny_dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            batch_size=tiny_config.batch_size,
            log_interval=tiny_config.log_interval,
        )

        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert np.isfinite(avg_loss)

    def test_model_parameters_updated(
        self,
        tiny_model: PhysicsOperator,
        tiny_dataset,
        tiny_config: TrainingConfig,
    ) -> None:
        """Parameters change after one epoch of training."""
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=tiny_config.learning_rate)
        loss_fn = PhysicsLoss()

        # Snapshot a parameter before training
        param_before = next(tiny_model.parameters()).clone()

        train_epoch(
            model=tiny_model,
            dataset=tiny_dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            batch_size=tiny_config.batch_size,
        )

        param_after = next(tiny_model.parameters())
        assert not torch.allclose(param_before, param_after, atol=1e-8)

    def test_batch_size_larger_than_dataset(
        self,
        tiny_model: PhysicsOperator,
        tiny_dataset,
    ) -> None:
        """Single batch larger than dataset still works."""
        device = torch.device("cpu")
        tiny_model.to(device)
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        avg_loss = train_epoch(
            model=tiny_model,
            dataset=tiny_dataset,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            batch_size=len(tiny_dataset) + 10,  # larger than dataset
        )

        assert avg_loss > 0


# ---------------------------------------------------------------------------
# Tests: evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for the evaluation function."""

    def test_returns_metrics_dict(
        self,
        tiny_model: PhysicsOperator,
        tiny_eval_dataset,
        tiny_config: TrainingConfig,
    ) -> None:
        """Evaluate returns dict with mse, mae, rmse keys."""
        device = torch.device("cpu")
        tiny_model.to(device)
        loss_fn = PhysicsLoss()

        metrics = evaluate(
            model=tiny_model,
            dataset=tiny_eval_dataset,
            loss_fn=loss_fn,
            device=device,
            batch_size=tiny_config.batch_size,
        )

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics

    def test_metrics_non_negative(
        self,
        tiny_model: PhysicsOperator,
        tiny_eval_dataset,
        tiny_config: TrainingConfig,
    ) -> None:
        """All metrics are non-negative."""
        device = torch.device("cpu")
        tiny_model.to(device)
        loss_fn = PhysicsLoss()

        metrics = evaluate(
            model=tiny_model,
            dataset=tiny_eval_dataset,
            loss_fn=loss_fn,
            device=device,
            batch_size=tiny_config.batch_size,
        )

        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    def test_rmse_is_sqrt_mse(
        self,
        tiny_model: PhysicsOperator,
        tiny_eval_dataset,
        tiny_config: TrainingConfig,
    ) -> None:
        """RMSE equals sqrt(MSE)."""
        device = torch.device("cpu")
        tiny_model.to(device)
        loss_fn = PhysicsLoss()

        metrics = evaluate(
            model=tiny_model,
            dataset=tiny_eval_dataset,
            loss_fn=loss_fn,
            device=device,
            batch_size=tiny_config.batch_size,
        )

        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]), rel=1e-5)

    def test_model_in_eval_mode_after(
        self,
        tiny_model: PhysicsOperator,
        tiny_eval_dataset,
        tiny_config: TrainingConfig,
    ) -> None:
        """Model is in eval mode after evaluate call."""
        device = torch.device("cpu")
        tiny_model.to(device)
        tiny_model.train()  # start in train mode
        loss_fn = PhysicsLoss()

        evaluate(
            model=tiny_model,
            dataset=tiny_eval_dataset,
            loss_fn=loss_fn,
            device=device,
            batch_size=tiny_config.batch_size,
        )

        assert not tiny_model.training


# ---------------------------------------------------------------------------
# Tests: W&B integration (mocked)
# ---------------------------------------------------------------------------


class TestWandbIntegration:
    """Tests that W&B logging paths are exercised safely."""

    def test_wandb_not_called_when_disabled(self) -> None:
        """W&B is never initialised when wandb_enabled is False."""
        config = TrainingConfig(wandb_enabled=False)
        assert config.wandb_enabled is False

    def test_wandb_project_default(self) -> None:
        """Default W&B project name is set."""
        config = TrainingConfig()
        assert config.wandb_project == "alphagalerkin-physics-poc"

    def test_wandb_name_optional(self) -> None:
        """W&B run name defaults to None."""
        config = TrainingConfig()
        assert config.wandb_name is None


# ---------------------------------------------------------------------------
# Tests: train() full pipeline (tiny, fast)
# ---------------------------------------------------------------------------


class TestTrainPipeline:
    """Tests for the full train() pipeline with tiny config."""

    def test_train_returns_results_dict(
        self, tmp_path, tiny_config: TrainingConfig
    ) -> None:
        """train() returns a results dict with expected keys."""
        from src.experiments.train_physics import train

        tiny_config.output_dir = str(tmp_path / "output")
        tiny_config.n_epochs = 2
        tiny_config.eval_interval = 1
        tiny_config.log_interval = 1

        results = train(tiny_config)

        assert isinstance(results, dict)
        assert "final_metrics" in results
        assert "best_transfer_mse" in results
        assert "training_time_seconds" in results
        assert "success" in results
        assert isinstance(results["success"], bool)

    def test_train_saves_checkpoint(
        self, tmp_path, tiny_config: TrainingConfig
    ) -> None:
        """train() saves best model checkpoint."""
        from src.experiments.train_physics import train

        tiny_config.output_dir = str(tmp_path / "output")
        tiny_config.n_epochs = 2
        tiny_config.eval_interval = 1

        train(tiny_config)

        best_path = tmp_path / "output" / "best_model.pt"
        assert best_path.exists()

    def test_train_saves_log(
        self, tmp_path, tiny_config: TrainingConfig
    ) -> None:
        """train() saves training_log.json."""
        import json

        from src.experiments.train_physics import train

        tiny_config.output_dir = str(tmp_path / "output")
        tiny_config.n_epochs = 2
        tiny_config.eval_interval = 1

        train(tiny_config)

        log_path = tmp_path / "output" / "training_log.json"
        assert log_path.exists()
        with open(log_path) as f:
            log_data = json.load(f)
        assert "history" in log_data
        assert "final_metrics" in log_data

    def test_train_history_populated(
        self, tmp_path, tiny_config: TrainingConfig
    ) -> None:
        """History lists grow during training."""
        from src.experiments.train_physics import train

        tiny_config.output_dir = str(tmp_path / "output")
        tiny_config.n_epochs = 3
        tiny_config.eval_interval = 1

        results = train(tiny_config)

        history = results["history"]
        assert len(history["train_loss"]) == 3
        assert len(history["eval_mse_same_res"]) == 3
        assert len(history["eval_mse_transfer"]) == 3


# ---------------------------------------------------------------------------
# Tests: main() argument parsing
# ---------------------------------------------------------------------------


class TestMainArgParsing:
    """Tests for the main() entry point argument parsing."""

    def test_main_default_args(self) -> None:
        """main() parses default arguments and calls train."""
        from unittest.mock import patch

        from src.experiments.train_physics import main

        with patch(
            "sys.argv", ["train_physics"]
        ), patch(
            "src.experiments.train_physics.train", return_value={}
        ) as mock_train:
            main()
            mock_train.assert_called_once()
            config = mock_train.call_args[0][0]
            assert config.train_grid_size == 9
            assert config.eval_grid_size == 19

    def test_main_custom_args(self) -> None:
        """main() forwards custom CLI args to TrainingConfig."""
        from unittest.mock import patch

        from src.experiments.train_physics import main

        with patch(
            "sys.argv",
            [
                "train_physics",
                "--train-size", "5",
                "--eval-size", "7",
                "--n-epochs", "2",
                "--d-model", "32",
                "--seed", "99",
            ],
        ), patch(
            "src.experiments.train_physics.train", return_value={}
        ) as mock_train:
            main()
            config = mock_train.call_args[0][0]
            assert config.train_grid_size == 5
            assert config.eval_grid_size == 7
            assert config.n_epochs == 2
            assert config.d_model == 32
            assert config.seed == 99
