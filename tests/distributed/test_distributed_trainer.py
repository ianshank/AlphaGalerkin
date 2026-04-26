"""Tests for DistributedTrainer, DistributedMetrics, and GradientAccumulator.

Covers Phase 2B gaps:
  - DistributedMetrics unit tests (defaults, serialization, custom values)
  - DistributedTrainer mock tests (init, device setup, optimizer LR scaling,
    is_main_process, checkpoint save/load)
  - Multi-process metric aggregation (2-process GLOO spawn)
  - Checkpoint coordination across ranks (2-process GLOO spawn)
  - GradientAccumulator integration (should_step, scale, reset)

Backend: GLOO (works on CPU, no CUDA required).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from src.distributed.config import DistributedBackend, DistributedInfraConfig
from src.distributed.gradient_sync import GradientAccumulator
from src.distributed.trainer import DistributedMetrics, DistributedTrainer

# Allow DistributedBackend enum to be deserialized with weights_only=True
torch.serialization.add_safe_globals([DistributedBackend])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE_2 = 2
INPUT_DIM = 4
OUTPUT_DIM = 2
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
ACCUMULATION_STEPS_1 = 1
ACCUMULATION_STEPS_4 = 4
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class _ToyModel(nn.Module):
    """Minimal linear model for distributed training tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_mock_config(
    learning_rate: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    gradient_clip: float = GRADIENT_CLIP,
) -> MagicMock:
    """Create a mock AlphaGalerkinConfig with training sub-config."""
    config = MagicMock()
    config.training.learning_rate = learning_rate
    config.training.weight_decay = weight_decay
    config.training.gradient_clip = gradient_clip
    config.model_dump.return_value = {"training": {"learning_rate": learning_rate}}
    return config


def _make_distributed_config(**kwargs: Any) -> DistributedInfraConfig:
    """Create a DistributedInfraConfig with GLOO backend and sensible defaults."""
    defaults = {
        "enabled": True,
        "backend": DistributedBackend.GLOO,
        "world_size": 1,
        "use_amp": False,
        "save_on_rank_0_only": True,
        "gradient_accumulation_steps": ACCUMULATION_STEPS_1,
    }
    defaults.update(kwargs)
    return DistributedInfraConfig(**defaults)


def _make_trainer(
    model: nn.Module | None = None,
    config: MagicMock | None = None,
    distributed_config: DistributedInfraConfig | None = None,
    loss_fn: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> DistributedTrainer:
    """Create a DistributedTrainer with mocked environment."""
    model = model or _ToyModel()
    config = config or _make_mock_config()
    distributed_config = distributed_config or _make_distributed_config()
    loss_fn = loss_fn or MagicMock()

    env = {
        "RANK": str(rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
    }
    with patch.dict(os.environ, env):
        return DistributedTrainer(
            model=model,
            config=config,
            distributed_config=distributed_config,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )


# =========================================================================
# 1. DistributedMetrics unit tests
# =========================================================================


class TestDistributedMetrics:
    """Unit tests for the DistributedMetrics dataclass."""

    def test_default_values(self) -> None:
        """Verify all default field values are zero or unity."""
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

    def test_custom_values(self) -> None:
        """Verify custom values are stored correctly."""
        metrics = DistributedMetrics(
            step=100,
            total_loss=1.5,
            policy_loss=0.8,
            value_loss=0.5,
            lbb_loss=0.2,
            gradient_norm=3.14,
            learning_rate=0.001,
            throughput_samples_per_sec=512.0,
            sync_time_ms=2.5,
            step_time_ms=10.0,
            world_size=4,
            rank=2,
            global_batch_size=256,
        )
        assert metrics.step == 100
        assert metrics.total_loss == 1.5
        assert metrics.world_size == 4
        assert metrics.rank == 2
        assert metrics.global_batch_size == 256

    def test_to_dict_serialization(self) -> None:
        """Verify to_dict() returns all fields as a plain dict."""
        metrics = DistributedMetrics(step=42, total_loss=1.23, rank=1)
        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert d["step"] == 42
        assert d["total_loss"] == 1.23
        assert d["rank"] == 1

    def test_to_dict_contains_all_fields(self) -> None:
        """Verify to_dict() includes every dataclass field."""
        metrics = DistributedMetrics()
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

    def test_to_dict_roundtrip(self) -> None:
        """Verify metrics can be reconstructed from dict values."""
        original = DistributedMetrics(step=7, total_loss=0.99, world_size=2)
        d = original.to_dict()
        reconstructed = DistributedMetrics(**d)
        assert reconstructed.step == original.step
        assert reconstructed.total_loss == original.total_loss
        assert reconstructed.world_size == original.world_size


# =========================================================================
# 2. DistributedTrainer mock tests (no real process group)
# =========================================================================


class TestDistributedTrainerInit:
    """Mock tests for DistributedTrainer without a real process group."""

    def test_initialization_with_mock_config(self) -> None:
        """Trainer initializes with mocked config objects."""
        trainer = _make_trainer()
        assert trainer.global_step == 0
        assert not trainer._is_initialized

    def test_setup_device_returns_cpu_when_no_cuda(self) -> None:
        """_setup_device returns CPU device when CUDA is unavailable."""
        trainer = _make_trainer()
        assert trainer.device == torch.device("cpu")

    def test_create_optimizer_scales_lr_by_world_size(self) -> None:
        """_create_optimizer scales LR via DistributedInfraConfig.scale_learning_rate."""
        world_size = 4
        base_lr = DEFAULT_LR
        trainer = _make_trainer(
            world_size=world_size,
            distributed_config=_make_distributed_config(
                world_size=world_size,
                learning_rate_scaling="linear",
            ),
        )
        expected_lr = base_lr * world_size
        actual_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < TOLERANCE

    def test_create_optimizer_weight_decay(self) -> None:
        """_create_optimizer respects weight_decay from config."""
        trainer = _make_trainer()
        actual_wd = trainer.optimizer.param_groups[0]["weight_decay"]
        assert abs(actual_wd - DEFAULT_WEIGHT_DECAY) < TOLERANCE

    def test_is_main_process_rank_zero(self) -> None:
        """is_main_process returns True for rank 0."""
        trainer = _make_trainer(rank=0)
        assert trainer.is_main_process is True

    def test_is_main_process_rank_nonzero(self) -> None:
        """is_main_process returns False for non-zero ranks."""
        trainer = _make_trainer(rank=1, world_size=2)
        assert trainer.is_main_process is False

    def test_external_optimizer_is_used(self) -> None:
        """When an optimizer is provided, trainer uses it instead of creating one."""
        model = _ToyModel()
        custom_optimizer = AdamW(model.parameters(), lr=0.1)
        trainer = _make_trainer(model=model, optimizer=custom_optimizer)
        assert trainer.optimizer is custom_optimizer

    def test_scheduler_stored(self) -> None:
        """Scheduler is stored when provided."""
        model = _ToyModel()
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        trainer = _make_trainer(model=model, optimizer=optimizer, scheduler=scheduler)
        assert trainer.scheduler is scheduler

    def test_amp_disabled_on_cpu(self) -> None:
        """AMP is disabled even if config says use_amp=True when device is CPU."""
        dist_config = _make_distributed_config(use_amp=True)
        trainer = _make_trainer(distributed_config=dist_config)
        assert trainer.use_amp is False
        assert trainer.scaler is None

    def test_gradient_accumulator_created(self) -> None:
        """Gradient accumulator is created with correct steps."""
        dist_config = _make_distributed_config(gradient_accumulation_steps=ACCUMULATION_STEPS_4)
        trainer = _make_trainer(distributed_config=dist_config)
        assert trainer.grad_accumulator.accumulation_steps == ACCUMULATION_STEPS_4


class TestDistributedTrainerCheckpoint:
    """Tests for save/load checkpoint logic (mocked, no process group)."""

    def test_save_checkpoint_rank_zero(self, tmp_path: Path) -> None:
        """Rank 0 saves checkpoint successfully."""
        trainer = _make_trainer(rank=0)
        ckpt_path = tmp_path / "checkpoint.pt"
        result = trainer.save_checkpoint(ckpt_path, metrics={"loss": 0.5})
        assert result == ckpt_path
        assert ckpt_path.exists()

    def test_save_checkpoint_skipped_on_nonzero_rank(self, tmp_path: Path) -> None:
        """Non-zero rank returns None when save_on_rank_0_only is True."""
        trainer = _make_trainer(rank=1, world_size=2)
        ckpt_path = tmp_path / "checkpoint.pt"
        result = trainer.save_checkpoint(ckpt_path)
        assert result is None
        assert not ckpt_path.exists()

    def test_save_checkpoint_all_ranks_when_configured(self, tmp_path: Path) -> None:
        """Non-zero rank can save when save_on_rank_0_only is False."""
        dist_config = _make_distributed_config(save_on_rank_0_only=False)
        trainer = _make_trainer(rank=1, world_size=2, distributed_config=dist_config)
        ckpt_path = tmp_path / "checkpoint.pt"
        result = trainer.save_checkpoint(ckpt_path)
        assert result == ckpt_path
        assert ckpt_path.exists()

    def test_save_checkpoint_contains_expected_keys(self, tmp_path: Path) -> None:
        """Saved checkpoint contains model, optimizer, config, and step."""
        trainer = _make_trainer(rank=0)
        trainer.global_step = 42
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path, metrics={"test": True})

        checkpoint = torch.load(ckpt_path, weights_only=True)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "step" in checkpoint
        assert checkpoint["step"] == 42
        assert checkpoint["metrics"]["test"] is True

    def test_save_checkpoint_includes_scheduler_state(self, tmp_path: Path) -> None:
        """Scheduler state is saved when a scheduler is present."""
        model = _ToyModel()
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=10)
        trainer = _make_trainer(model=model, optimizer=optimizer, scheduler=scheduler, rank=0)
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)

        checkpoint = torch.load(ckpt_path, weights_only=True)
        assert "scheduler_state_dict" in checkpoint

    def test_load_checkpoint_restores_state(self, tmp_path: Path) -> None:
        """load_checkpoint restores model weights, optimizer, and step."""
        model = _ToyModel()
        torch.manual_seed(99)
        nn.init.constant_(model.fc.weight, 3.14)
        optimizer = AdamW(model.parameters(), lr=0.01)
        trainer = _make_trainer(model=model, optimizer=optimizer, rank=0)
        trainer.global_step = 50
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)

        # Create fresh trainer and load
        model2 = _ToyModel()
        optimizer2 = AdamW(model2.parameters(), lr=0.01)
        trainer2 = _make_trainer(model=model2, optimizer=optimizer2, rank=0)
        step = trainer2.load_checkpoint(ckpt_path)

        assert step == 50
        assert trainer2.global_step == 50
        assert torch.allclose(model.fc.weight, model2.fc.weight, atol=TOLERANCE)

    def test_load_checkpoint_restores_scheduler(self, tmp_path: Path) -> None:
        """Scheduler state is restored from checkpoint."""
        model = _ToyModel()
        optimizer = AdamW(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        # Advance scheduler a few steps
        for _ in range(3):
            scheduler.step()

        trainer = _make_trainer(model=model, optimizer=optimizer, scheduler=scheduler, rank=0)
        ckpt_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(ckpt_path)
        saved_last_epoch = scheduler.last_epoch

        # Fresh trainer with fresh scheduler
        model2 = _ToyModel()
        optimizer2 = AdamW(model2.parameters(), lr=0.01)
        scheduler2 = StepLR(optimizer2, step_size=5, gamma=0.5)
        trainer2 = _make_trainer(
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            rank=0,
        )
        trainer2.load_checkpoint(ckpt_path)

        assert scheduler2.last_epoch == saved_last_epoch

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """save_checkpoint creates parent dirs if they don't exist."""
        trainer = _make_trainer(rank=0)
        ckpt_path = tmp_path / "subdir" / "deep" / "checkpoint.pt"
        result = trainer.save_checkpoint(ckpt_path)
        assert result == ckpt_path
        assert ckpt_path.exists()


# =========================================================================
# 3. Multi-process metric aggregation (2-process GLOO spawn)
# =========================================================================


def _metric_aggregation_worker(
    rank: int,
    world_size: int,
    port: int,
    result_dict: Any,
) -> None:
    """Worker that creates rank-specific metrics and all-reduces them."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

    try:
        # Each rank has different metric values
        local_total_loss = float(rank + 1)  # rank 0 -> 1.0, rank 1 -> 2.0
        local_policy_loss = float(rank + 1) * 0.5
        local_value_loss = float(rank + 1) * 0.3
        local_lbb_loss = float(rank + 1) * 0.2
        local_grad_norm = float(rank + 1) * 10.0
        local_throughput = float(rank + 1) * 100.0
        local_step_time = float(rank + 1) * 5.0

        metrics_tensor = torch.tensor(
            [
                local_total_loss,
                local_policy_loss,
                local_value_loss,
                local_lbb_loss,
                local_grad_norm,
                local_throughput,
                local_step_time,
            ]
        )

        torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
        metrics_tensor /= world_size

        # Expected averages: (1+2)/2 = 1.5 for total_loss, etc.
        expected_total_loss = 1.5
        expected_policy_loss = 0.75
        expected_value_loss = 0.45
        expected_lbb_loss = 0.30
        expected_grad_norm = 15.0
        expected_throughput = 150.0
        expected_step_time = 7.5

        checks_passed = (
            abs(metrics_tensor[0].item() - expected_total_loss) < 1e-5
            and abs(metrics_tensor[1].item() - expected_policy_loss) < 1e-5
            and abs(metrics_tensor[2].item() - expected_value_loss) < 1e-5
            and abs(metrics_tensor[3].item() - expected_lbb_loss) < 1e-5
            and abs(metrics_tensor[4].item() - expected_grad_norm) < 1e-5
            and abs(metrics_tensor[5].item() - expected_throughput) < 1e-5
            and abs(metrics_tensor[6].item() - expected_step_time) < 1e-5
        )

        result_dict[rank] = "ok" if checks_passed else f"mismatch: {metrics_tensor.tolist()}"
    finally:
        torch.distributed.destroy_process_group()


class TestMultiProcessMetricAggregation:
    """Multi-process tests for metric aggregation with GLOO backend."""

    def test_metric_all_reduce_averages_correctly(self) -> None:
        """Two-process all-reduce averages rank-specific metrics."""
        port = _find_free_port()
        ctx = torch.multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

        try:
            torch.multiprocessing.spawn(
                _metric_aggregation_worker,
                args=(WORLD_SIZE_2, port, result_dict),
                nprocs=WORLD_SIZE_2,
                join=True,
            )
        except RuntimeError as exc:
            pytest.skip(f"torch.multiprocessing.spawn not supported: {exc}")

        assert len(result_dict) == WORLD_SIZE_2
        for rank in range(WORLD_SIZE_2):
            assert result_dict[rank] == "ok", f"Rank {rank}: {result_dict[rank]}"


# =========================================================================
# 4. Checkpoint coordination (2-process GLOO spawn)
# =========================================================================


def _checkpoint_coordination_worker(
    rank: int,
    world_size: int,
    checkpoint_dir: str,
    port: int,
    result_dict: Any,
) -> None:
    """Worker that tests checkpoint save on rank 0 and load on rank 1."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
    )

    try:
        torch.manual_seed(42)
        model = _ToyModel()
        optimizer = AdamW(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # Simulate a training step to populate optimizer state
        x = torch.randn(2, INPUT_DIM)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        ckpt_path = os.path.join(checkpoint_dir, "trainer_ckpt.pt")

        if rank == 0:
            checkpoint = {
                "step": 10,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {},
                "distributed_config": {},
                "metrics": {"total_loss": 0.42},
            }
            torch.save(checkpoint, ckpt_path)

        torch.distributed.barrier()

        if rank == 1:
            loaded = torch.load(ckpt_path, weights_only=True)

            # Verify step
            assert loaded["step"] == 10, f"Step mismatch: {loaded['step']}"

            # Verify model state matches
            loaded_model = _ToyModel()
            loaded_model.load_state_dict(loaded["model_state_dict"])
            for (name, p1), (_, p2) in zip(
                model.named_parameters(),
                loaded_model.named_parameters(),
            ):
                assert torch.allclose(p1, p2, atol=1e-6), f"Weight mismatch: {name}"

            # Verify optimizer state can be loaded
            loaded_optimizer = AdamW(loaded_model.parameters(), lr=0.001)
            loaded_optimizer.load_state_dict(loaded["optimizer_state_dict"])

            # Verify scheduler state
            loaded_scheduler = StepLR(loaded_optimizer, step_size=5, gamma=0.5)
            loaded_scheduler.load_state_dict(loaded["scheduler_state_dict"])
            assert loaded_scheduler.last_epoch == scheduler.last_epoch

            # Verify metrics
            assert loaded["metrics"]["total_loss"] == 0.42

        result_dict[rank] = "ok"
    finally:
        torch.distributed.destroy_process_group()


class TestCheckpointCoordination:
    """Multi-process tests for checkpoint save/load coordination."""

    def test_rank0_save_rank1_load(self, tmp_path: Path) -> None:
        """Rank 0 saves full checkpoint; rank 1 loads and verifies all states."""
        port = _find_free_port()
        ctx = torch.multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        result_dict = manager.dict()

        try:
            torch.multiprocessing.spawn(
                _checkpoint_coordination_worker,
                args=(WORLD_SIZE_2, str(tmp_path), port, result_dict),
                nprocs=WORLD_SIZE_2,
                join=True,
            )
        except RuntimeError as exc:
            pytest.skip(f"torch.multiprocessing.spawn not supported: {exc}")

        assert len(result_dict) == WORLD_SIZE_2
        for rank in range(WORLD_SIZE_2):
            assert result_dict[rank] == "ok", f"Rank {rank}: {result_dict[rank]}"


# =========================================================================
# 5. GradientAccumulator integration tests
# =========================================================================


class TestGradientAccumulator:
    """Tests for the GradientAccumulator dataclass."""

    def test_should_step_single_accumulation(self) -> None:
        """With accumulation_steps=1, should_step is False initially."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_1)
        # Before any accumulation, step_count is 0 which is < 1
        assert not acc.should_step()

    def test_should_step_after_accumulate(self) -> None:
        """should_step returns True after enough accumulate calls."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4)
        for i in range(ACCUMULATION_STEPS_4):
            acc.accumulate(1.0)
        assert acc.should_step()

    def test_should_step_not_ready(self) -> None:
        """should_step returns False when fewer than accumulation_steps calls."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4)
        acc.accumulate(1.0)
        acc.accumulate(1.0)
        assert not acc.should_step()

    def test_scale_divides_loss_correctly(self) -> None:
        """scale() divides loss by accumulation_steps when steps > 1."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4)
        loss = torch.tensor(4.0)
        scaled = acc.scale(loss)
        assert abs(scaled.item() - 1.0) < TOLERANCE

    def test_scale_no_division_when_steps_is_one(self) -> None:
        """scale() returns loss unchanged when accumulation_steps is 1."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_1)
        loss = torch.tensor(4.0)
        scaled = acc.scale(loss)
        assert abs(scaled.item() - 4.0) < TOLERANCE

    def test_scale_no_division_when_scale_loss_disabled(self) -> None:
        """scale() returns loss unchanged when scale_loss is False."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4, scale_loss=False)
        loss = torch.tensor(4.0)
        scaled = acc.scale(loss)
        assert abs(scaled.item() - 4.0) < TOLERANCE

    def test_reset_clears_state(self) -> None:
        """reset() zeros accumulated_loss and step count."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4)
        acc.accumulate(2.0)
        acc.accumulate(3.0)
        acc.reset()
        assert acc.accumulated_loss == 0.0
        assert acc._step_count == 0
        assert not acc.should_step()

    def test_get_average_loss(self) -> None:
        """get_average_loss returns mean of accumulated losses."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4)
        acc.accumulate(2.0)
        acc.accumulate(4.0)
        assert abs(acc.get_average_loss() - 3.0) < TOLERANCE

    def test_get_average_loss_zero_steps(self) -> None:
        """get_average_loss returns 0.0 when no steps have been taken."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_1)
        assert acc.get_average_loss() == 0.0

    def test_accumulate_tracks_running_sum(self) -> None:
        """accumulate() adds to running total correctly."""
        acc = GradientAccumulator(accumulation_steps=ACCUMULATION_STEPS_4)
        acc.accumulate(1.5)
        acc.accumulate(2.5)
        acc.accumulate(3.0)
        assert abs(acc.accumulated_loss - 7.0) < TOLERANCE
        assert acc._step_count == 3
