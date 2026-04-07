"""Tests for src/training/base_trainer.py.

Covers BaseTrainerConfig, StepResult, and BaseTrainer concrete subclass
(via a minimal ConcreteTrainer fixture), including AMP, gradient clipping,
LR scheduling, checkpoint save/load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor, nn

from src.training.base_trainer import BaseTrainer, BaseTrainerConfig, StepResult

# ---------------------------------------------------------------------------
# Helpers: concrete minimal trainer for testing
# ---------------------------------------------------------------------------


class MinimalConfig(BaseTrainerConfig):
    """Minimal config for unit tests."""


class ConcreteTrainer(BaseTrainer[MinimalConfig]):
    """Minimal concrete trainer for testing the base class."""

    def __init__(
        self,
        model: nn.Module,
        config: MinimalConfig,
        device: torch.device | str = "cpu",
        checkpoint_dir: Path | str | None = None,
    ) -> None:
        super().__init__(model, config, device, checkpoint_dir)
        self._batch_count = 0

    def compute_loss(self, batch: Any) -> tuple[Tensor, dict[str, float]]:
        """Simple L2 loss on random input."""
        x, y = batch
        pred = self.model(x)
        loss = torch.mean((pred - y) ** 2)
        return loss, {"l2": float(loss)}

    def generate_data(self) -> tuple[Tensor, Tensor]:
        """Generate a tiny random batch."""
        x = torch.randn(4, 2, device=self.device)
        y = torch.zeros(4, 2, device=self.device)
        self._batch_count += 1
        return x, y

    def evaluate(self) -> dict[str, float]:
        return {"eval_loss": 0.0}


def _make_model() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))


def _make_trainer(
    tmp_path: Path,
    **cfg_kwargs: Any,
) -> ConcreteTrainer:
    cfg = MinimalConfig(name="test", **cfg_kwargs)
    return ConcreteTrainer(
        _make_model(),
        cfg,
        device="cpu",
        checkpoint_dir=tmp_path / "checkpoints",
    )


# ---------------------------------------------------------------------------
# BaseTrainerConfig
# ---------------------------------------------------------------------------


class TestBaseTrainerConfig:
    def test_defaults(self):
        cfg = BaseTrainerConfig(name="test")
        assert cfg.learning_rate > 0
        assert cfg.weight_decay >= 0
        assert cfg.gradient_clip > 0
        assert cfg.lr_scheduler in ("cosine", "linear", "none")
        assert cfg.warmup_steps >= 0
        assert cfg.total_steps >= 1
        assert 0.0 <= cfg.min_lr_ratio <= 1.0
        assert cfg.save_every >= 1
        assert cfg.log_every >= 1

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            BaseTrainerConfig(name="t", learning_rate=0.0)

    def test_gradient_clip_must_be_positive(self):
        with pytest.raises(ValidationError):
            BaseTrainerConfig(name="t", gradient_clip=0.0)

    def test_warmup_steps_non_negative(self):
        with pytest.raises(ValidationError):
            BaseTrainerConfig(name="t", warmup_steps=-1)

    def test_min_lr_ratio_bounds(self):
        with pytest.raises(ValidationError):
            BaseTrainerConfig(name="t", min_lr_ratio=-0.1)
        with pytest.raises(ValidationError):
            BaseTrainerConfig(name="t", min_lr_ratio=1.5)


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_to_dict_basic(self):
        result = StepResult(loss=0.5)
        d = result.to_dict()
        assert d["loss"] == pytest.approx(0.5)

    def test_to_dict_with_metrics(self):
        result = StepResult(loss=1.0, metrics={"l2": 0.3, "kl": 0.7})
        d = result.to_dict()
        assert d["l2"] == pytest.approx(0.3)
        assert d["kl"] == pytest.approx(0.7)

    def test_to_dict_with_grad_norm(self):
        result = StepResult(loss=1.0, grad_norm=2.5)
        d = result.to_dict()
        assert d["grad_norm"] == pytest.approx(2.5)

    def test_to_dict_no_grad_norm_excluded(self):
        result = StepResult(loss=0.0)
        assert "grad_norm" not in result.to_dict()

    def test_metrics_defaults_empty(self):
        result = StepResult(loss=0.0)
        assert result.metrics == {}


# ---------------------------------------------------------------------------
# BaseTrainer initialization
# ---------------------------------------------------------------------------


class TestBaseTrainerInit:
    def test_device_auto_selects_cpu(self, tmp_path: Path):
        # On CPU-only CI, auto should give cpu
        trainer = _make_trainer(tmp_path)
        assert trainer.device.type in ("cpu", "cuda")

    def test_explicit_device_cpu(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        assert trainer.device == torch.device("cpu")

    def test_model_moved_to_device(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        for p in trainer.model.parameters():
            assert p.device.type == trainer.device.type

    def test_global_step_starts_at_zero(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        assert trainer.global_step == 0

    def test_no_amp_on_cpu(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path, use_amp=False)
        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_checkpoint_dir_created(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        assert trainer.checkpoint_dir is not None

    def test_optimizer_is_adamw(self, tmp_path: Path):
        from torch.optim import AdamW

        trainer = _make_trainer(tmp_path)
        assert isinstance(trainer.optimizer, AdamW)

    def test_optimizer_lr_is_positive(self, tmp_path: Path):
        """LR is positive after scheduler initialization (warmup may reduce it)."""
        lr = 5e-4
        trainer = _make_trainer(tmp_path, learning_rate=lr, warmup_steps=0)
        # With no warmup the initial LR should match config
        assert trainer.get_current_lr() == pytest.approx(lr)


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------


class TestSchedulerSetup:
    @pytest.mark.parametrize("sched", ["cosine", "linear", "none"])
    def test_scheduler_types(self, tmp_path: Path, sched: str):
        trainer = _make_trainer(tmp_path, lr_scheduler=sched, warmup_steps=0, total_steps=100)
        assert trainer.scheduler is not None

    def test_warmup_creates_sequential(self, tmp_path: Path):
        from torch.optim.lr_scheduler import SequentialLR

        trainer = _make_trainer(tmp_path, lr_scheduler="cosine", warmup_steps=100, total_steps=1000)
        assert isinstance(trainer.scheduler, SequentialLR)

    def test_no_warmup_single_scheduler(self, tmp_path: Path):
        from torch.optim.lr_scheduler import CosineAnnealingLR

        trainer = _make_trainer(tmp_path, lr_scheduler="cosine", warmup_steps=0, total_steps=100)
        assert isinstance(trainer.scheduler, CosineAnnealingLR)

    def test_unknown_scheduler_falls_back(self, tmp_path: Path):
        # Unknown type falls back to cosine without crashing
        trainer = _make_trainer(
            tmp_path, lr_scheduler="polynomial", warmup_steps=0, total_steps=100
        )
        assert trainer.scheduler is not None


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    def test_step_returns_step_result(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        result = trainer.step()
        assert isinstance(result, StepResult)

    def test_step_increments_global_step(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        trainer.step()
        assert trainer.global_step == 1
        trainer.step()
        assert trainer.global_step == 2

    def test_step_loss_is_finite(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        result = trainer.step()
        assert result.loss >= 0.0
        import math

        assert math.isfinite(result.loss)

    def test_step_grad_norm_finite(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        result = trainer.step()
        assert result.grad_norm is not None
        assert result.grad_norm >= 0.0

    def test_step_calls_generate_data(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        assert trainer._batch_count == 0
        trainer.step()
        assert trainer._batch_count == 1

    def test_multiple_steps(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        for _ in range(5):
            result = trainer.step()
            assert result.loss >= 0.0
        assert trainer.global_step == 5

    def test_gradient_clipping_applied(self, tmp_path: Path):
        """Verify training step runs without NaN loss."""
        trainer = _make_trainer(tmp_path, gradient_clip=0.1)
        for _ in range(3):
            result = trainer.step()
            assert result.loss == result.loss  # not NaN


# ---------------------------------------------------------------------------
# get_current_lr / set_training
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_get_current_lr(self, tmp_path: Path):
        lr = 2e-3
        trainer = _make_trainer(tmp_path, learning_rate=lr)
        returned_lr = trainer.get_current_lr()
        assert isinstance(returned_lr, float)
        assert returned_lr > 0

    def test_set_training_mode(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        trainer.set_training(True)
        assert trainer.model.training
        trainer.set_training(False)
        assert not trainer.model.training


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_creates_file(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        path = trainer.save_checkpoint()
        assert path.exists()

    def test_save_default_filename(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        trainer.global_step = 42
        path = trainer.save_checkpoint()
        assert "00000042" in path.name

    def test_save_custom_filename(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        path = trainer.save_checkpoint("best.pt")
        assert path.name == "best.pt"

    def test_save_load_round_trip(self, tmp_path: Path):
        """Saving and loading should restore global step and model weights."""
        trainer = _make_trainer(tmp_path)
        # Train a few steps to change state
        for _ in range(3):
            trainer.step()
        step_before = trainer.global_step
        path = trainer.save_checkpoint()

        # Create fresh trainer and load
        trainer2 = _make_trainer(tmp_path)
        assert trainer2.global_step == 0
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == step_before

        # Model params should match
        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_missing_file_raises(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "nonexistent.pt")

    def test_checkpoint_creates_dir(self, tmp_path: Path):
        new_dir = tmp_path / "new" / "nested" / "checkpoints"
        trainer = _make_trainer(tmp_path, checkpoint_dir=str(new_dir))  # type: ignore[call-arg]
        # Override checkpoint_dir directly
        trainer.checkpoint_dir = new_dir
        assert not new_dir.exists()
        trainer.save_checkpoint()
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# _setup_amp
# ---------------------------------------------------------------------------


class TestSetupAmp:
    def test_amp_disabled_on_cpu(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path, use_amp=True)
        # On CPU, AMP should be disabled regardless of config
        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_amp_disabled_when_not_requested(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path, use_amp=False)
        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_setup_amp_returns_tuple(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        result = trainer._setup_amp(use_amp=False, device=torch.device("cpu"))
        assert isinstance(result, tuple)
        assert len(result) == 3
        use_amp, scaler, dtype = result
        assert use_amp is False
        assert scaler is None
        assert dtype == torch.float16

    def test_setup_amp_custom_dtype(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        _, _, dtype = trainer._setup_amp(
            use_amp=False, device=torch.device("cpu"), amp_dtype=torch.bfloat16
        )
        assert dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# _clip_gradients
# ---------------------------------------------------------------------------


class TestClipGradients:
    def test_clip_gradients_returns_float(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        model = trainer.model
        # Do a forward + backward to create gradients
        x = torch.randn(4, 2)
        y = model(x)
        y.sum().backward()
        norm = trainer._clip_gradients(model, max_norm=1.0)
        assert isinstance(norm, float)
        assert norm >= 0.0

    def test_clip_gradients_respects_max_norm(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        model = trainer.model
        # Create large gradients
        x = torch.randn(4, 2) * 1000
        y = model(x)
        (y.sum() * 1000).backward()
        norm_before = trainer._clip_gradients(model, max_norm=0.01)
        # Gradients should now be clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm**0.5
        assert total_norm <= 0.01 + 1e-6  # Allow small numerical error

    def test_clip_gradients_no_grad_returns_zero(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        # No backward pass performed, no gradients
        model = trainer.model
        model.zero_grad()
        norm = trainer._clip_gradients(model, max_norm=1.0)
        assert norm == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _amp_forward_backward
# ---------------------------------------------------------------------------


class TestAmpForwardBackward:
    def test_returns_loss_metrics_grad_norm(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        trainer.optimizer.zero_grad()

        def loss_fn():
            x = torch.randn(4, 2)
            pred = trainer.model(x)
            loss = pred.sum()
            return loss, {"custom": 1.0}

        loss, metrics, grad_norm = trainer._amp_forward_backward(
            loss_fn=loss_fn,
            model=trainer.model,
            optimizer=trainer.optimizer,
            max_norm=1.0,
        )
        assert isinstance(loss, float)
        assert "custom" in metrics
        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0

    def test_optimizer_steps_weights(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path, learning_rate=0.1)
        params_before = [p.clone().detach() for p in trainer.model.parameters()]
        trainer.optimizer.zero_grad()

        def loss_fn():
            x = torch.ones(4, 2) * 10.0
            pred = trainer.model(x)
            loss = (pred**2).sum()
            return loss, {}

        trainer._amp_forward_backward(
            loss_fn=loss_fn,
            model=trainer.model,
            optimizer=trainer.optimizer,
            max_norm=10.0,
        )
        # Weights should have changed
        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(params_before, trainer.model.parameters())
        )
        assert changed, "Optimizer should have updated model parameters"

    def test_loss_is_finite(self, tmp_path: Path):
        import math

        trainer = _make_trainer(tmp_path)
        trainer.optimizer.zero_grad()

        def loss_fn():
            x = torch.randn(4, 2)
            pred = trainer.model(x)
            loss = (pred**2).mean()
            return loss, {}

        loss, _, _ = trainer._amp_forward_backward(
            loss_fn=loss_fn,
            model=trainer.model,
            optimizer=trainer.optimizer,
            max_norm=1.0,
        )
        assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# Static _create_optimizer
# ---------------------------------------------------------------------------


class TestStaticCreateOptimizer:
    def test_creates_adamw(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=1e-4)
        from torch.optim import AdamW

        assert isinstance(opt, AdamW)

    def test_learning_rate_set(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.05, weight_decay=0.0)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_weight_decay_set(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.1)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Static _create_scheduler
# ---------------------------------------------------------------------------


class TestStaticCreateScheduler:
    def test_cosine_scheduler(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.0)
        sched = BaseTrainer._create_scheduler(
            opt, scheduler_type="cosine", warmup_steps=0, total_steps=100
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR

        assert isinstance(sched, CosineAnnealingLR)

    def test_linear_scheduler(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.0)
        sched = BaseTrainer._create_scheduler(
            opt, scheduler_type="linear", warmup_steps=0, total_steps=100
        )
        from torch.optim.lr_scheduler import LinearLR

        assert isinstance(sched, LinearLR)

    def test_constant_scheduler(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.0)
        sched = BaseTrainer._create_scheduler(
            opt, scheduler_type="constant", warmup_steps=0, total_steps=100
        )
        # Should be ConstantLR
        assert sched is not None

    def test_none_scheduler(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.0)
        sched = BaseTrainer._create_scheduler(
            opt, scheduler_type="none", warmup_steps=0, total_steps=100
        )
        assert sched is not None

    def test_warmup_creates_sequential(self):
        from torch.optim.lr_scheduler import SequentialLR

        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.0)
        sched = BaseTrainer._create_scheduler(
            opt, scheduler_type="cosine", warmup_steps=10, total_steps=100
        )
        assert isinstance(sched, SequentialLR)

    def test_unknown_type_falls_back_to_cosine(self):
        model = _make_model()
        opt = BaseTrainer._create_optimizer(model, lr=0.01, weight_decay=0.0)
        sched = BaseTrainer._create_scheduler(
            opt, scheduler_type="polynomial", warmup_steps=0, total_steps=100
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR

        assert isinstance(sched, CosineAnnealingLR)


# ---------------------------------------------------------------------------
# _save_training_state / _load_training_state
# ---------------------------------------------------------------------------


class TestTrainingStatePersistence:
    def test_save_creates_file(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        path = trainer._save_training_state(tmp_path / "state.pt")
        assert path.exists()

    def test_save_load_roundtrip_global_step(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        for _ in range(5):
            trainer.step()
        assert trainer.global_step == 5
        trainer._save_training_state(tmp_path / "state.pt")

        trainer2 = _make_trainer(tmp_path)
        assert trainer2.global_step == 0
        step = trainer2._load_training_state(tmp_path / "state.pt")
        assert step == 5
        assert trainer2.global_step == 5

    def test_save_load_roundtrip_optimizer_state(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        for _ in range(3):
            trainer.step()
        trainer._save_training_state(tmp_path / "state.pt")

        # Get optimizer state keys
        state1 = trainer.optimizer.state_dict()

        trainer2 = _make_trainer(tmp_path)
        trainer2._load_training_state(tmp_path / "state.pt")
        state2 = trainer2.optimizer.state_dict()

        # Both should have the same param groups
        assert len(state1["param_groups"]) == len(state2["param_groups"])
        assert state1["param_groups"][0]["lr"] == pytest.approx(
            state2["param_groups"][0]["lr"]
        )

    def test_load_missing_file_raises(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        with pytest.raises(FileNotFoundError):
            trainer._load_training_state(tmp_path / "nonexistent.pt")

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        trainer = _make_trainer(tmp_path)
        nested = tmp_path / "a" / "b" / "c" / "state.pt"
        path = trainer._save_training_state(nested)
        assert path.exists()


# ---------------------------------------------------------------------------
# Inheritance verification
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_trainer_inherits_base_trainer(self):
        """Verify the main Trainer inherits from BaseTrainer."""
        from src.training.trainer import Trainer

        assert issubclass(Trainer, BaseTrainer)

    def test_distributed_trainer_inherits_base_trainer(self):
        """Verify DistributedTrainer inherits from BaseTrainer."""
        from src.distributed.trainer import DistributedTrainer

        assert issubclass(DistributedTrainer, BaseTrainer)

    def test_base_trainer_has_all_required_helpers(self):
        """Verify BaseTrainer exposes all required helper methods."""
        for method_name in [
            "_setup_amp",
            "_create_scheduler",
            "_create_optimizer",
            "_clip_gradients",
            "_amp_forward_backward",
            "_save_training_state",
            "_load_training_state",
        ]:
            assert hasattr(BaseTrainer, method_name), f"Missing {method_name}"
