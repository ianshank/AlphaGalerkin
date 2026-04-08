"""Tests for src/experiments/train_physics.py.

Covers TrainingConfig defaults, argument parsing, train_epoch, evaluate,
and the full train() pipeline with mocked heavy computation.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.experiments.physics_model import PhysicsLoss, PhysicsOperator
from src.experiments.train_physics import (
    TrainingConfig,
    evaluate,
    main,
    train,
    train_epoch,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_GRID = 5
N_SAMPLES = 4
BATCH_SIZE = 2

SMALL_MODEL_KWARGS = {
    "d_model": 16,
    "n_heads": 2,
    "n_layers": 1,
    "n_fourier_features": 8,
    "fourier_scale": 1.0,
    "use_fnet": False,
}


def _make_small_config(tmp_path: Path, **overrides) -> TrainingConfig:
    """Create a minimal TrainingConfig for testing."""
    defaults = {
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 1,
        "n_fourier_features": 8,
        "fourier_scale": 1.0,
        "use_fnet": False,
        "train_grid_size": SMALL_GRID,
        "eval_grid_size": SMALL_GRID,
        "n_train_samples": N_SAMPLES,
        "n_eval_samples": N_SAMPLES,
        "n_charges": 2,
        "batch_size": BATCH_SIZE,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "n_epochs": 2,
        "log_interval": 1,
        "eval_interval": 1,
        "success_threshold": 999.0,  # always pass
        "output_dir": str(tmp_path / "outputs"),
        "seed": 0,
        "wandb_enabled": False,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _make_dataset(grid_size: int = SMALL_GRID, n_samples: int = N_SAMPLES):
    """Build a tiny PoissonDataset."""
    from src.physics.poisson import PoissonDataset

    return PoissonDataset(
        grid_size=grid_size,
        n_samples=n_samples,
        n_charges=2,
        seed=0,
    )


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """Validate TrainingConfig default values and overrides."""

    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.d_model == 128
        assert cfg.n_heads == 4
        assert cfg.n_layers == 4
        assert cfg.n_fourier_features == 64
        assert cfg.fourier_scale == 10.0
        assert cfg.use_fnet is True
        assert cfg.train_grid_size == 9
        assert cfg.eval_grid_size == 19
        assert cfg.n_train_samples == 5000
        assert cfg.n_eval_samples == 500
        assert cfg.n_charges == 5
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 1e-3
        assert cfg.weight_decay == 1e-4
        assert cfg.n_epochs == 100
        assert cfg.log_interval == 10
        assert cfg.eval_interval == 10
        assert cfg.success_threshold == 0.05
        assert cfg.output_dir == "outputs/physics_poc"
        assert cfg.seed == 42
        assert cfg.wandb_enabled is False

    def test_overrides(self) -> None:
        cfg = TrainingConfig(d_model=32, n_epochs=5, seed=99)
        assert cfg.d_model == 32
        assert cfg.n_epochs == 5
        assert cfg.seed == 99

    def test_seed_offsets(self) -> None:
        cfg = TrainingConfig()
        assert cfg.train_eval_seed_offset == 10000
        assert cfg.transfer_eval_seed_offset == 20000

    def test_wandb_fields(self) -> None:
        cfg = TrainingConfig(wandb_enabled=True, wandb_name="run1")
        assert cfg.wandb_enabled is True
        assert cfg.wandb_name == "run1"
        assert cfg.wandb_project == "alphagalerkin-physics-poc"


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_returns_finite_loss(self) -> None:
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        dataset = _make_dataset()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        avg_loss = train_epoch(
            model,
            dataset,
            optimizer,
            loss_fn,
            device,
            batch_size=BATCH_SIZE,
            log_interval=1,
        )
        assert isinstance(avg_loss, float)
        assert np.isfinite(avg_loss)
        assert avg_loss >= 0.0

    def test_model_in_train_mode(self) -> None:
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        model.eval()  # start in eval mode
        dataset = _make_dataset()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        train_epoch(model, dataset, optimizer, loss_fn, device, batch_size=BATCH_SIZE)
        assert model.training

    def test_parameters_updated(self) -> None:
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        dataset = _make_dataset()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        train_epoch(model, dataset, optimizer, loss_fn, device, batch_size=BATCH_SIZE)

        any_changed = any(not torch.equal(params_before[n], p) for n, p in model.named_parameters())
        assert any_changed, "Expected at least one parameter to change after training"

    def test_single_sample_batch(self) -> None:
        """Edge case: batch_size >= dataset size."""
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        dataset = _make_dataset(n_samples=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = PhysicsLoss()

        avg_loss = train_epoch(
            model,
            dataset,
            optimizer,
            loss_fn,
            device,
            batch_size=10,
        )
        assert np.isfinite(avg_loss)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    """Tests for evaluate function."""

    def test_returns_metrics_dict(self) -> None:
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        dataset = _make_dataset()
        loss_fn = PhysicsLoss()

        metrics = evaluate(model, dataset, loss_fn, device, batch_size=BATCH_SIZE)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert all(np.isfinite(v) for v in metrics.values())

    def test_rmse_is_sqrt_mse(self) -> None:
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        dataset = _make_dataset()
        loss_fn = PhysicsLoss()

        metrics = evaluate(model, dataset, loss_fn, device, batch_size=BATCH_SIZE)
        assert abs(metrics["rmse"] - np.sqrt(metrics["mse"])) < 1e-6

    def test_model_left_in_eval_mode(self) -> None:
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        model.train()  # start in train mode
        dataset = _make_dataset()
        loss_fn = PhysicsLoss()

        evaluate(model, dataset, loss_fn, device, batch_size=BATCH_SIZE)
        assert not model.training

    def test_no_grad_context(self) -> None:
        """Evaluate should not accumulate gradients."""
        device = torch.device("cpu")
        model = PhysicsOperator(**SMALL_MODEL_KWARGS).to(device)
        dataset = _make_dataset()
        loss_fn = PhysicsLoss()

        evaluate(model, dataset, loss_fn, device, batch_size=BATCH_SIZE)
        for p in model.parameters():
            assert p.grad is None or torch.all(p.grad == 0)


# ---------------------------------------------------------------------------
# train (full pipeline)
# ---------------------------------------------------------------------------


class TestTrainPipeline:
    """Tests for the top-level train() function."""

    def test_returns_results_dict(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path)
        results = train(config)

        assert isinstance(results, dict)
        assert "config" in results
        assert "history" in results
        assert "final_metrics" in results
        assert "best_transfer_mse" in results
        assert "training_time_seconds" in results
        assert "success" in results
        assert "success_threshold" in results

    def test_history_populated(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path)
        results = train(config)
        history = results["history"]

        assert len(history["train_loss"]) == config.n_epochs
        assert len(history["learning_rate"]) == config.n_epochs
        # With eval_interval=1 and n_epochs=2, expect 2 eval entries
        assert len(history["eval_mse_same_res"]) == config.n_epochs
        assert len(history["eval_mse_transfer"]) == config.n_epochs

    def test_final_metrics_structure(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path)
        results = train(config)
        fm = results["final_metrics"]

        assert "same_resolution" in fm
        assert "zero_shot_transfer" in fm
        for group in fm.values():
            assert "mse" in group
            assert "mae" in group
            assert "rmse" in group

    def test_output_files_created(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path)
        train(config)

        output_dir = Path(config.output_dir)
        assert (output_dir / "training_log.json").exists()
        assert (output_dir / "best_model.pt").exists()

    def test_training_log_json_valid(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path)
        train(config)

        log_path = Path(config.output_dir) / "training_log.json"
        with open(log_path) as f:
            data = json.load(f)
        assert data["success_threshold"] == config.success_threshold

    def test_best_model_checkpoint_loadable(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path)
        train(config)

        ckpt_path = Path(config.output_dir) / "best_model.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "epoch" in ckpt
        assert "transfer_mse" in ckpt

    def test_success_true_with_high_threshold(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path, success_threshold=999.0)
        results = train(config)
        assert results["success"] is True

    def test_success_false_with_zero_threshold(self, tmp_path: Path) -> None:
        config = _make_small_config(tmp_path, success_threshold=0.0)
        results = train(config)
        assert results["success"] is False

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        cfg1 = _make_small_config(tmp_path / "a", seed=7)
        cfg2 = _make_small_config(tmp_path / "b", seed=7)

        r1 = train(cfg1)
        r2 = train(cfg2)

        assert r1["history"]["train_loss"] == r2["history"]["train_loss"]

    def test_wandb_warning_when_not_available(self, tmp_path: Path) -> None:
        """When wandb_enabled=True but wandb unavailable, train still runs."""
        config = _make_small_config(tmp_path, wandb_enabled=True)
        with patch("src.experiments.train_physics.WANDB_AVAILABLE", False):
            results = train(config)
        assert isinstance(results, dict)

    def test_wandb_integration_mocked(self, tmp_path: Path) -> None:
        """When wandb is available and enabled, it is called."""
        config = _make_small_config(tmp_path, wandb_enabled=True)
        mock_wandb = MagicMock()
        with (
            patch("src.experiments.train_physics.WANDB_AVAILABLE", True),
            patch("src.experiments.train_physics.wandb", mock_wandb),
        ):
            results = train(config)

        mock_wandb.init.assert_called_once()
        assert mock_wandb.log.call_count > 0
        mock_wandb.finish.assert_called_once()


# ---------------------------------------------------------------------------
# Argument parsing (main)
# ---------------------------------------------------------------------------


class TestMainArgParsing:
    """Tests for argument parsing in main()."""

    def test_default_args(self, tmp_path: Path) -> None:
        """main() should parse default arguments and run train()."""
        with (
            patch("src.experiments.train_physics.train") as mock_train,
            patch("sys.argv", ["prog", "--output-dir", str(tmp_path)]),
        ):
            mock_train.return_value = {"success": True}
            main()

        mock_train.assert_called_once()
        config = mock_train.call_args[0][0]
        assert isinstance(config, TrainingConfig)
        assert config.train_grid_size == 9
        assert config.eval_grid_size == 19
        assert config.n_epochs == 100

    def test_custom_args(self, tmp_path: Path) -> None:
        with (
            patch("src.experiments.train_physics.train") as mock_train,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--train-size",
                    "5",
                    "--eval-size",
                    "7",
                    "--n-epochs",
                    "3",
                    "--d-model",
                    "16",
                    "--n-layers",
                    "1",
                    "--fourier-scale",
                    "2.0",
                    "--lr",
                    "0.01",
                    "--batch-size",
                    "4",
                    "--success-threshold",
                    "0.1",
                    "--output-dir",
                    str(tmp_path),
                    "--seed",
                    "99",
                ],
            ),
        ):
            mock_train.return_value = {"success": True}
            main()

        config = mock_train.call_args[0][0]
        assert config.train_grid_size == 5
        assert config.eval_grid_size == 7
        assert config.n_epochs == 3
        assert config.d_model == 16
        assert config.n_layers == 1
        assert config.fourier_scale == 2.0
        assert config.learning_rate == 0.01
        assert config.batch_size == 4
        assert config.success_threshold == 0.1
        assert config.seed == 99

    def test_wandb_flag(self, tmp_path: Path) -> None:
        with (
            patch("src.experiments.train_physics.train") as mock_train,
            patch(
                "sys.argv",
                [
                    "prog",
                    "--wandb",
                    "--wandb-project",
                    "my-project",
                    "--wandb-name",
                    "my-run",
                    "--output-dir",
                    str(tmp_path),
                ],
            ),
        ):
            mock_train.return_value = {"success": True}
            main()

        config = mock_train.call_args[0][0]
        assert config.wandb_enabled is True
        assert config.wandb_project == "my-project"
        assert config.wandb_name == "my-run"

    def test_wandb_default_disabled(self, tmp_path: Path) -> None:
        with (
            patch("src.experiments.train_physics.train") as mock_train,
            patch("sys.argv", ["prog", "--output-dir", str(tmp_path)]),
        ):
            mock_train.return_value = {"success": True}
            main()

        config = mock_train.call_args[0][0]
        assert config.wandb_enabled is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and regression tests."""

    def test_eval_interval_larger_than_epochs(self, tmp_path: Path) -> None:
        """No eval performed if eval_interval > n_epochs."""
        config = _make_small_config(
            tmp_path,
            n_epochs=2,
            eval_interval=100,
            log_interval=1,
        )
        results = train(config)
        # No intermediate evals, but final eval always runs
        assert len(results["history"]["eval_mse_transfer"]) == 0
        assert "zero_shot_transfer" in results["final_metrics"]

    def test_different_train_and_eval_grids(self, tmp_path: Path) -> None:
        """Model handles different resolutions for train vs eval."""
        config = _make_small_config(
            tmp_path,
            train_grid_size=5,
            eval_grid_size=7,
        )
        results = train(config)
        assert results["final_metrics"]["zero_shot_transfer"]["mse"] >= 0

    def test_output_dir_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        config = _make_small_config(tmp_path, output_dir=str(nested))
        train(config)
        assert nested.exists()
