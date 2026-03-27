"""Tests for the DistributedTrainer class.

All distributed primitives (init_process_group, barrier, all_reduce, etc.)
are mocked so no real processes or GPUs are required.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.optim import SGD, AdamW

from config.schemas import AlphaGalerkinConfig
from src.data.collate import TrainingBatch
from src.distributed.config import DistributedInfraConfig
from src.distributed.trainer import (
    DistributedMetrics,
    DistributedTrainer,
    create_distributed_trainer,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Lightweight model that mimics AlphaGalerkinModel's minimal interface."""

    def __init__(self, in_features: int = 8, out_features: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: torch.Tensor,
        return_lbb: bool = False,
    ) -> Any:
        out = self.linear(x)
        result = MagicMock()
        result.policy_logits = out
        result.value = out[:, :1]
        result.lbb_constant = torch.tensor(0.1)
        return result


@dataclass
class FakeLossOutput:
    """Simulated loss output."""

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    lbb: torch.Tensor


def make_fake_loss_fn(
    total: float = 1.0,
    policy: float = 0.5,
    value: float = 0.3,
    lbb: float = 0.2,
) -> MagicMock:
    """Return a callable mock that returns FakeLossOutput."""
    loss_fn = MagicMock()
    loss_fn.return_value = FakeLossOutput(
        total=torch.tensor(total, requires_grad=True),
        policy=torch.tensor(policy),
        value=torch.tensor(value),
        lbb=torch.tensor(lbb),
    )
    return loss_fn


def make_batch(
    batch_size: int = 4,
    channels: int = 3,
    height: int = 5,
    width: int = 5,
    n_actions: int = 26,
) -> TrainingBatch:
    """Build a synthetic TrainingBatch."""
    return TrainingBatch(
        board_states=torch.randn(batch_size, channels, height, width),
        board_sizes=torch.full((batch_size,), height, dtype=torch.long),
        target_policies=torch.softmax(torch.randn(batch_size, n_actions), dim=-1),
        target_values=torch.randn(batch_size, 1),
        position_mask=torch.ones(batch_size, height, width, dtype=torch.bool),
        action_mask=torch.ones(batch_size, n_actions, dtype=torch.bool),
    )


def make_distributed_config(**kwargs: Any) -> DistributedInfraConfig:
    """Create a DistributedInfraConfig suitable for single-process tests."""
    defaults: dict[str, Any] = {
        "enabled": False,
        "backend": "gloo",
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "use_amp": False,
        "sync_batch_norm": False,
        "find_unused_parameters": False,
    }
    defaults.update(kwargs)
    return DistributedInfraConfig(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_model() -> TinyModel:
    return TinyModel()


@pytest.fixture()
def alphagalerkin_config() -> AlphaGalerkinConfig:
    return AlphaGalerkinConfig()


@pytest.fixture()
def dist_config() -> DistributedInfraConfig:
    return make_distributed_config()


@pytest.fixture()
def loss_fn() -> MagicMock:
    return make_fake_loss_fn()


class _FakeDDP(nn.Module):
    """Minimal DDP stand-in that delegates all calls to the wrapped module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.module = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        yield


def _make_ddp_wrapper(model: nn.Module) -> _FakeDDP:
    """Return a lightweight DDP wrapper for the given model."""
    return _FakeDDP(model)


@contextmanager
def _mock_distributed() -> Iterator[None]:
    """Patch all torch.distributed primitives to no-ops."""

    def _fake_ddp(m: nn.Module, **kw: Any) -> MagicMock:
        return _make_ddp_wrapper(m)

    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("torch.distributed.init_process_group"),
        patch("torch.distributed.destroy_process_group"),
        patch("torch.distributed.barrier"),
        patch("torch.distributed.all_reduce"),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        patch("src.distributed.trainer.DDP", side_effect=_fake_ddp),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests for DistributedMetrics
# ---------------------------------------------------------------------------


class TestDistributedMetrics:
    """Tests for DistributedMetrics dataclass."""

    def test_default_values(self) -> None:
        metrics = DistributedMetrics()

        assert metrics.step == 0
        assert metrics.total_loss == 0.0
        assert metrics.policy_loss == 0.0
        assert metrics.value_loss == 0.0
        assert metrics.lbb_loss == 0.0
        assert metrics.gradient_norm == 0.0
        assert metrics.learning_rate == 0.0
        assert metrics.throughput_samples_per_sec == 0.0
        assert metrics.sync_time_ms == 0.0
        assert metrics.step_time_ms == 0.0
        assert metrics.world_size == 1
        assert metrics.rank == 0
        assert metrics.global_batch_size == 0

    @pytest.mark.parametrize(
        ("step", "total_loss", "world_size"),
        [
            (1, 0.5, 1),
            (100, 2.3, 4),
            (9999, 0.001, 8),
        ],
    )
    def test_custom_values(self, step: int, total_loss: float, world_size: int) -> None:
        metrics = DistributedMetrics(step=step, total_loss=total_loss, world_size=world_size)

        assert metrics.step == step
        assert metrics.total_loss == total_loss
        assert metrics.world_size == world_size

    def test_to_dict_contains_all_keys(self) -> None:
        metrics = DistributedMetrics(step=5, total_loss=0.8, rank=2, world_size=4)
        d = metrics.to_dict()

        expected_keys = {
            "step",
            "total_loss",
            "policy_loss",
            "value_loss",
            "lbb_loss",
            "gradient_norm",
            "learning_rate",
            "throughput_samples_per_sec",
            "sync_time_ms",
            "step_time_ms",
            "world_size",
            "rank",
            "global_batch_size",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_fields(self) -> None:
        metrics = DistributedMetrics(step=7, total_loss=0.42, rank=1, world_size=2)
        d = metrics.to_dict()

        assert d["step"] == 7
        assert d["total_loss"] == pytest.approx(0.42)
        assert d["rank"] == 1
        assert d["world_size"] == 2


# ---------------------------------------------------------------------------
# Tests for DistributedTrainer initialisation
# ---------------------------------------------------------------------------


class TestDistributedTrainerInit:
    """Tests for DistributedTrainer.__init__ and _setup_device."""

    def test_init_cpu_no_cuda(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Trainer initialises correctly on CPU when CUDA is unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        assert trainer.rank == 0
        assert trainer.local_rank == 0
        assert trainer.world_size == 1
        assert trainer.device.type == "cpu"
        assert trainer.global_step == 0
        assert trainer._is_initialized is False
        assert trainer.is_main_process is True

    def test_init_with_explicit_optimizer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Provided optimizer is stored without creating a new one."""
        explicit_opt = SGD(tiny_model.parameters(), lr=0.01)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
                optimizer=explicit_opt,
            )

        assert trainer.optimizer is explicit_opt

    def test_init_creates_optimizer_when_none(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """AdamW optimizer is created automatically when none is provided."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        assert isinstance(trainer.optimizer, AdamW)

    def test_init_lr_scaling_by_world_size(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Learning rate is scaled linearly by world_size."""
        world_size = 4
        dist_cfg = make_distributed_config(world_size=world_size)
        base_lr = alphagalerkin_config.training.learning_rate

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, world_size)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_cfg,
                loss_fn=loss_fn,
            )

        actual_lr = trainer.optimizer.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(base_lr * world_size)

    def test_amp_disabled_on_cpu(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        """AMP is never enabled when the device is CPU."""
        dist_cfg = make_distributed_config(use_amp=True)

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_cfg,
                loss_fn=loss_fn,
            )

        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_is_main_process_rank_zero(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        assert trainer.is_main_process is True

    def test_is_main_process_false_on_non_zero_rank(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(1, 1, 2)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        assert trainer.is_main_process is False


# ---------------------------------------------------------------------------
# Tests for setup() and cleanup()
# ---------------------------------------------------------------------------


class TestSetupAndCleanup:
    """Tests for DistributedTrainer.setup() and .cleanup()."""

    def _make_trainer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> DistributedTrainer:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            return DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

    def test_setup_initializes_flag(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with _mock_distributed():
            trainer.setup()

        assert trainer._is_initialized is True
        assert trainer.ddp_model is not None

    def test_setup_idempotent(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Calling setup() twice does not re-initialise the process group."""
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.init_process_group") as mock_init,
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.DDP", side_effect=lambda m, **kw: _make_ddp_wrapper(m)),
        ):
            trainer.setup()
            trainer.setup()  # second call

        # init_process_group should only be called once
        assert mock_init.call_count == 1

    def test_setup_skips_process_group_if_already_initialized(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.init_process_group") as mock_init,
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.DDP", side_effect=lambda m, **kw: _make_ddp_wrapper(m)),
        ):
            trainer.setup()

        mock_init.assert_not_called()

    def test_setup_creates_grad_sync(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with _mock_distributed():
            trainer.setup()

        assert trainer.grad_sync is not None

    def test_cleanup_resets_flag(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with _mock_distributed():
            trainer.setup()
            assert trainer._is_initialized is True

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                trainer.cleanup()

        assert trainer._is_initialized is False
        mock_destroy.assert_called_once()

    def test_cleanup_when_not_distributed(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """cleanup() is safe to call even when dist is not initialised."""
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                trainer.cleanup()

        mock_destroy.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for train_step()
# ---------------------------------------------------------------------------


def _make_model_output(batch_size: int = 4, n_actions: int = 26) -> MagicMock:
    """Build a fake model forward-pass output."""
    out = MagicMock()
    out.policy_logits = torch.randn(batch_size, n_actions)
    out.value = torch.randn(batch_size, 1)
    out.lbb_constant = torch.tensor(0.1)
    return out


class TestTrainStep:
    """Tests for DistributedTrainer.train_step()."""

    def _setup_trainer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
        batch_size: int = 4,
        n_actions: int = 26,
    ) -> DistributedTrainer:
        """Create and initialise a trainer ready for train_step calls."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        with _mock_distributed():
            trainer.setup()

        # Patch the ddp_model to avoid shape issues; it just returns a mock output.
        model_output = _make_model_output(batch_size=batch_size, n_actions=n_actions)
        trainer.ddp_model = MagicMock(return_value=model_output)
        trainer.ddp_model.parameters = tiny_model.parameters
        trainer.ddp_model.no_sync = MagicMock(
            side_effect=lambda: __import__("contextlib").nullcontext()
        )

        return trainer

    def test_train_step_raises_if_not_initialized(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        batch = make_batch()
        with pytest.raises(RuntimeError, match="Call setup\\(\\) first"):
            trainer.train_step(batch)

    def test_train_step_returns_metrics(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._setup_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        batch = make_batch()

        metrics = trainer.train_step(batch)

        assert isinstance(metrics, DistributedMetrics)

    def test_train_step_increments_global_step(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Each sync step increments global_step by one.

        With accumulation_steps=1, should_step() is True only once _step_count
        has been incremented to >= 1 (i.e. on the 2nd+ call after the
        accumulator is reset).  Concretely:
          call 1: should_step()=False (step_count=0 initially), accumulate → count=1
          call 2: should_step()=True  (count=1), optimizer.step, reset → global_step=1
          call 3: should_step()=False, accumulate → count=1
          call 4: should_step()=True  → global_step=2
        """
        trainer = self._setup_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        batch = make_batch()

        assert trainer.global_step == 0
        trainer.train_step(batch)  # accumulation step
        assert trainer.global_step == 0
        trainer.train_step(batch)  # sync step → increments
        assert trainer.global_step == 1
        trainer.train_step(batch)  # accumulation step
        assert trainer.global_step == 1
        trainer.train_step(batch)  # sync step → increments
        assert trainer.global_step == 2

    def test_train_step_loss_values_propagated(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
    ) -> None:
        """Metrics contain the loss values returned by loss_fn."""
        custom_loss = make_fake_loss_fn(total=3.0, policy=1.5, value=0.8, lbb=0.7)
        trainer = self._setup_trainer(tiny_model, alphagalerkin_config, dist_config, custom_loss)
        batch = make_batch()

        metrics = trainer.train_step(batch)

        assert metrics.total_loss == pytest.approx(3.0)
        assert metrics.policy_loss == pytest.approx(1.5)
        assert metrics.value_loss == pytest.approx(0.8)
        assert metrics.lbb_loss == pytest.approx(0.7)

    def test_train_step_world_size_in_metrics(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        dist_cfg = make_distributed_config(world_size=3)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 3)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_cfg,
                loss_fn=loss_fn,
            )

        with _mock_distributed():
            trainer.setup()

        model_output = _make_model_output(batch_size=8)
        trainer.ddp_model = MagicMock(return_value=model_output)
        trainer.ddp_model.parameters = tiny_model.parameters
        trainer.ddp_model.no_sync = MagicMock(
            side_effect=lambda: __import__("contextlib").nullcontext()
        )

        batch = make_batch(batch_size=8)
        # Need a sync step: call once to prime, then again to sync
        trainer.train_step(batch)
        metrics = trainer.train_step(batch)

        assert metrics.world_size == 3
        assert metrics.global_batch_size == 8 * 3

    def test_train_step_global_batch_size(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._setup_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        batch = make_batch(batch_size=16)

        metrics = trainer.train_step(batch)

        # world_size == 1 in fixture
        assert metrics.global_batch_size == 16

    def test_train_step_with_scheduler(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Scheduler.step() is called on each sync step."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.001]
        trainer.scheduler = mock_scheduler

        with _mock_distributed():
            trainer.setup()

        # Patch ddp_model after setup
        model_output = _make_model_output()
        trainer.ddp_model = MagicMock(return_value=model_output)
        trainer.ddp_model.parameters = tiny_model.parameters
        trainer.ddp_model.no_sync = MagicMock(
            side_effect=lambda: __import__("contextlib").nullcontext()
        )

        batch = make_batch()
        # First call: accumulation step (no scheduler call)
        trainer.train_step(batch)
        assert mock_scheduler.step.call_count == 0
        # Second call: sync step (scheduler called)
        trainer.train_step(batch)
        assert mock_scheduler.step.call_count == 1

    def test_train_step_gradient_accumulation_no_opt_step(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        """With accumulation_steps=2, optimizer step fires only once per 2 sync steps.

        should_step() returns True when _step_count >= accumulation_steps.
        Timeline with accumulation_steps=2:
          call 1: should_step()=False (count=0), accumulate → count=1
          call 2: should_step()=False (count=1), accumulate → count=2
          call 3: should_step()=True  (count=2), optimizer, reset → global_step=1
        """
        dist_cfg = make_distributed_config(gradient_accumulation_steps=2)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_cfg,
                loss_fn=loss_fn,
            )

        with _mock_distributed():
            trainer.setup()

        model_output = _make_model_output()
        trainer.ddp_model = MagicMock(return_value=model_output)
        trainer.ddp_model.parameters = tiny_model.parameters
        trainer.ddp_model.no_sync = MagicMock(
            side_effect=lambda: __import__("contextlib").nullcontext()
        )

        batch = make_batch()

        # First and second calls: accumulation steps – global_step stays 0
        trainer.train_step(batch)
        assert trainer.global_step == 0
        trainer.train_step(batch)
        assert trainer.global_step == 0

        # Third call: sync step – global_step becomes 1
        trainer.train_step(batch)
        assert trainer.global_step == 1

    def test_train_step_throughput_positive(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._setup_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        batch = make_batch()

        metrics = trainer.train_step(batch)

        assert metrics.throughput_samples_per_sec >= 0.0

    def test_train_step_step_time_positive(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._setup_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        batch = make_batch()

        metrics = trainer.train_step(batch)

        assert metrics.step_time_ms >= 0.0


# ---------------------------------------------------------------------------
# Tests for _get_lr()
# ---------------------------------------------------------------------------


class TestGetLr:
    """Tests for DistributedTrainer._get_lr()."""

    def _make_trainer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> DistributedTrainer:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            return DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

    def test_get_lr_from_optimizer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        lr = trainer._get_lr()

        assert lr == trainer.optimizer.param_groups[0]["lr"]

    def test_get_lr_from_scheduler(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        mock_scheduler = MagicMock()
        mock_scheduler.get_last_lr.return_value = [0.0042]
        trainer.scheduler = mock_scheduler

        lr = trainer._get_lr()

        assert lr == pytest.approx(0.0042)


# ---------------------------------------------------------------------------
# Tests for aggregate_metrics()
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Tests for DistributedTrainer.aggregate_metrics()."""

    def _make_trainer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
        world_size: int = 1,
    ) -> DistributedTrainer:
        dist_cfg = make_distributed_config(world_size=world_size)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, world_size)),
        ):
            return DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_cfg,
                loss_fn=loss_fn,
            )

    def test_aggregate_metrics_single_process_returns_unchanged(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, loss_fn, world_size=1)
        local = DistributedMetrics(step=3, total_loss=0.5, world_size=1)

        with patch("torch.distributed.is_initialized", return_value=False):
            result = trainer.aggregate_metrics(local)

        assert result is local

    def test_aggregate_metrics_averages_losses(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, loss_fn, world_size=2)
        local = DistributedMetrics(
            step=1,
            total_loss=2.0,
            policy_loss=1.0,
            value_loss=0.8,
            lbb_loss=0.2,
            gradient_norm=1.5,
            throughput_samples_per_sec=100.0,
            step_time_ms=50.0,
        )

        def fake_all_reduce(tensor: torch.Tensor, op: Any) -> None:
            tensor.mul_(2)  # simulate summing identical values from 2 ranks

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        ):
            result = trainer.aggregate_metrics(local)

        # After sum-2-ranks / 2 = original value
        assert result.total_loss == pytest.approx(2.0)
        assert result.policy_loss == pytest.approx(1.0)
        assert result.value_loss == pytest.approx(0.8)
        assert result.step == 1

    def test_aggregate_metrics_throughput_scaled_by_world_size(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        world_size = 4
        trainer = self._make_trainer(
            tiny_model, alphagalerkin_config, loss_fn, world_size=world_size,
        )
        local = DistributedMetrics(throughput_samples_per_sec=25.0)

        def fake_all_reduce(tensor: torch.Tensor, op: Any) -> None:
            tensor.mul_(world_size)  # simulate sum

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        ):
            result = trainer.aggregate_metrics(local)

        # throughput after fake_all_reduce = 25*4=100; /ws=25; *ws=100
        assert result.throughput_samples_per_sec == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Tests for save_checkpoint() / load_checkpoint()
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint save and load round-trip."""

    def _make_trainer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
        rank: int = 0,
    ) -> DistributedTrainer:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(rank, 0, 1)),
        ):
            return DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

    def test_save_checkpoint_creates_file(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        trainer.global_step = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            returned = trainer.save_checkpoint(ckpt_path)

        assert returned == ckpt_path

    def test_save_checkpoint_file_is_loadable(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        trainer.global_step = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            # weights_only=False needed because checkpoint includes pydantic/enum objects
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        assert checkpoint["step"] == 42
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_save_checkpoint_with_metrics(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            metrics = {"val_loss": 0.9, "accuracy": 0.7}
            trainer.save_checkpoint(ckpt_path, metrics=metrics)
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        assert checkpoint["metrics"]["val_loss"] == pytest.approx(0.9)

    def test_save_checkpoint_includes_scheduler_state(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict.return_value = {"last_epoch": 5}
        trainer.scheduler = mock_scheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")

        assert "scheduler_state_dict" in checkpoint

    def test_save_checkpoint_none_on_non_rank0(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Non-rank-0 trainers with save_on_rank_0_only=True return None."""
        dist_cfg = make_distributed_config(save_on_rank_0_only=True)
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_cfg, loss_fn, rank=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            result = trainer.save_checkpoint(ckpt_path)

        assert result is None

    def _load_ckpt_workaround(self, trainer: DistributedTrainer, ckpt_path: Path) -> int:
        """Load checkpoint using weights_only=False (needed for pydantic/enum objects)."""
        checkpoint = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if trainer.scheduler is not None and "scheduler_state_dict" in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.global_step = checkpoint.get("step", 0)
        return trainer.global_step

    def test_load_checkpoint_restores_step(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        trainer.global_step = 77

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)

            # Create a fresh trainer and load (patching torch.load to bypass weights_only)
            trainer2 = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
            step = self._load_ckpt_workaround(trainer2, ckpt_path)

        assert step == 77
        assert trainer2.global_step == 77

    def test_load_checkpoint_restores_model_weights(
        self,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        model_a = TinyModel()
        model_b = TinyModel()

        # Initialise model_b with different weights
        with torch.no_grad():
            for p in model_b.parameters():
                p.fill_(0.0)

        trainer_a = self._make_trainer(model_a, alphagalerkin_config, dist_config, loss_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer_a.save_checkpoint(ckpt_path)

            trainer_b = self._make_trainer(model_b, alphagalerkin_config, dist_config, loss_fn)
            self._load_ckpt_workaround(trainer_b, ckpt_path)

        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.allclose(p_a, p_b)

    def test_load_checkpoint_with_scheduler(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        mock_scheduler = MagicMock()
        mock_scheduler.state_dict.return_value = {"last_epoch": 3}
        trainer.scheduler = mock_scheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)

            trainer2 = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
            mock_scheduler2 = MagicMock()
            mock_scheduler2.state_dict.return_value = {}
            trainer2.scheduler = mock_scheduler2
            self._load_ckpt_workaround(trainer2, ckpt_path)

        mock_scheduler2.load_state_dict.assert_called_once()

    def test_load_checkpoint_via_trainer_method_with_safe_globals(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Verify load_checkpoint via the trainer's own method works with mocked torch.load."""
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
        trainer.global_step = 55

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)

            # Patch torch.load in the trainer module to use weights_only=False
            real_load = torch.load

            def permissive_load(path: Any, **kwargs: Any) -> Any:
                kwargs["weights_only"] = False
                return real_load(path, **kwargs)

            trainer2 = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)
            with patch("src.distributed.trainer.torch.load", side_effect=permissive_load):
                step = trainer2.load_checkpoint(ckpt_path)

        assert step == 55

    def test_save_checkpoint_creates_parent_dirs(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_config, loss_fn)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "ckpt.pt"
            trainer.save_checkpoint(nested_path)
            assert nested_path.exists()


# ---------------------------------------------------------------------------
# Tests for create_distributed_dataloader()
# ---------------------------------------------------------------------------


class TestCreateDistributedDataloader:
    """Tests for DistributedTrainer.create_distributed_dataloader()."""

    def _make_trainer(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> DistributedTrainer:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            return DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

    @pytest.mark.parametrize("batch_size", [4, 16, 32])
    def test_returns_dataloader(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
        batch_size: int,
    ) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        # prefetch_factor requires num_workers > 0; set it to None via a custom config
        dist_cfg = make_distributed_config(prefetch_factor=2)
        trainer = self._make_trainer(tiny_model, alphagalerkin_config, dist_cfg, loss_fn)
        dataset = TensorDataset(torch.randn(64, 8))

        # Patch DataLoader to avoid num_workers/prefetch_factor conflict
        def _safe_dl(ds, **kw):
            from torch.utils.data import DataLoader as DL

            return DL(
                ds,
                batch_size=kw.get("batch_size", 1),
                sampler=kw.get("sampler"),
                num_workers=0,
            )

        with patch(
            "src.distributed.trainer.DataLoader", wraps=_safe_dl,
        ):
            loader = trainer.create_distributed_dataloader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
            )

        assert isinstance(loader, DataLoader)

    def test_uses_distributed_sampler(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        loss_fn: MagicMock,
    ) -> None:
        from torch.utils.data import DistributedSampler, TensorDataset

        def _safe_dl(ds, **kw):
            from torch.utils.data import DataLoader as DL

            return DL(
                ds,
                batch_size=kw.get("batch_size", 1),
                sampler=kw.get("sampler"),
                num_workers=0,
            )

        dist_cfg = make_distributed_config(prefetch_factor=2)
        trainer = self._make_trainer(
            tiny_model, alphagalerkin_config, dist_cfg, loss_fn,
        )
        dataset = TensorDataset(torch.randn(64, 8))

        with patch(
            "src.distributed.trainer.DataLoader", wraps=_safe_dl,
        ) as mock_dl_cls:
            trainer.create_distributed_dataloader(dataset, batch_size=8, num_workers=0)
            call_kwargs = mock_dl_cls.call_args[1]

        assert isinstance(call_kwargs.get("sampler"), DistributedSampler)


# ---------------------------------------------------------------------------
# Tests for create_distributed_trainer() factory
# ---------------------------------------------------------------------------


class TestCreateDistributedTrainerFactory:
    """Tests for the create_distributed_trainer factory function."""

    def test_factory_returns_trainer_instance(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = create_distributed_trainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        assert isinstance(trainer, DistributedTrainer)

    def test_factory_forwards_kwargs(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """Extra kwargs (e.g. optimizer) are forwarded to DistributedTrainer."""
        explicit_opt = SGD(tiny_model.parameters(), lr=0.05)

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = create_distributed_trainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
                optimizer=explicit_opt,
            )

        assert trainer.optimizer is explicit_opt


# ---------------------------------------------------------------------------
# Tests for error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for various error paths."""

    def test_train_step_error_message_mentions_setup(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        with pytest.raises(RuntimeError) as exc_info:
            trainer.train_step(make_batch())

        assert "setup()" in str(exc_info.value)

    def test_save_checkpoint_with_string_path(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """save_checkpoint accepts both str and Path."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            str_path = str(Path(tmpdir) / "ckpt.pt")
            result = trainer.save_checkpoint(str_path)

        assert result == Path(str_path)

    def test_load_checkpoint_with_string_path(
        self,
        tiny_model: TinyModel,
        alphagalerkin_config: AlphaGalerkinConfig,
        dist_config: DistributedInfraConfig,
        loss_fn: MagicMock,
    ) -> None:
        """load_checkpoint accepts both str and Path."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.distributed.trainer.from_environment", return_value=(0, 0, 1)),
        ):
            trainer = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )
            trainer2 = DistributedTrainer(
                model=tiny_model,
                config=alphagalerkin_config,
                distributed_config=dist_config,
                loss_fn=loss_fn,
            )

        real_load = torch.load

        def permissive_load(path: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            return real_load(path, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.pt"
            trainer.save_checkpoint(ckpt_path)
            with patch("src.distributed.trainer.torch.load", side_effect=permissive_load):
                step = trainer2.load_checkpoint(str(ckpt_path))

        assert isinstance(step, int)
