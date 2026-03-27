"""Coverage tests for Vertex AI trainer wrapper.

Targets uncovered lines in src/vertex/trainer.py:
    - VertexTrainer init
    - VertexTrainingResult
    - _train_step (default and custom)
    - _load_state (DDP path, optimizer/scheduler restore)
    - _save_checkpoint (DDP model unwrap, non-main skip)
    - _emergency_checkpoint (success and failure paths)
    - _log_metrics
    - _get_latest_metrics
    - setup() (mocked dependencies)
    - train() loop (preemption, exception, auto-resume, checkpoint trigger)
    - _resume_from_checkpoint (found and not-found paths)
    - Properties: is_main_process, current_step, distributed_context
    - create_vertex_trainer factory
    - _get_torch / _get_dist lazy imports
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.vertex.config import VertexStorageConfig, VertexTrainingConfig
from src.vertex.trainer import (
    VertexTrainer,
    VertexTrainingResult,
    _get_dist,
    _get_torch,
    create_vertex_trainer,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _simple_model() -> nn.Module:
    return nn.Linear(4, 2)


def _vertex_config(
    bucket: str = "test-bucket",
    project: str = "test-project",
) -> VertexTrainingConfig:
    return VertexTrainingConfig(
        project_id=project,
        staging_bucket=f"gs://{bucket}",
        storage=VertexStorageConfig(bucket_name=bucket),
    )


def _make_trainer(
    config: dict[str, Any] | None = None,
    optimizer: Any = None,
    scheduler: Any = None,
    train_step_fn: Any = None,
) -> VertexTrainer:
    model = _simple_model()
    vc = _vertex_config()
    return VertexTrainer(
        model=model,
        config=config or {},
        vertex_config=vc,
        optimizer=optimizer,
        scheduler=scheduler,
        train_step_fn=train_step_fn,
    )


def _mock_distributed_ctx(world_size: int = 1, rank: int = 0) -> MagicMock:
    ctx = MagicMock()
    ctx.world_size = world_size
    ctx.rank = rank
    ctx.local_rank = rank
    ctx.is_main_process.return_value = rank == 0
    return ctx


def _make_preemption_handler(
    is_preempted: bool = False,
    should_checkpoint: bool = False,
    preemption_event: Any = None,
) -> MagicMock:
    ph = MagicMock()
    ph.is_preempted = is_preempted
    ph.should_save_checkpoint.return_value = should_checkpoint
    ph.preemption_event = preemption_event
    return ph


# ---------------------------------------------------------------------------
# Tests: lazy imports
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_get_torch_returns_torch(self) -> None:
        t = _get_torch()
        assert t is torch

    def test_get_dist_returns_dist(self) -> None:
        import torch.distributed as dist_module

        d = _get_dist()
        assert d is dist_module


# ---------------------------------------------------------------------------
# Tests: VertexTrainingResult
# ---------------------------------------------------------------------------


class TestVertexTrainingResult:
    def test_to_dict_minimal(self) -> None:
        result = VertexTrainingResult(status="completed", final_step=1000)
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["final_step"] == 1000
        assert d["final_checkpoint"] is None
        assert d["metrics"] == {}
        assert d["cost_estimate"] is None
        assert d["preemption_event"] is None

    def test_to_dict_with_all_fields(self) -> None:
        result = VertexTrainingResult(
            status="preempted",
            final_step=500,
            final_checkpoint="gs://bucket/ckpt.pt",
            metrics={"loss": 0.01},
            cost_estimate={"total": 10.0},
            preemption_event={"time": "12:00"},
        )
        d = result.to_dict()
        assert d["status"] == "preempted"
        assert d["final_step"] == 500
        assert d["final_checkpoint"] == "gs://bucket/ckpt.pt"
        assert d["metrics"]["loss"] == 0.01
        assert d["cost_estimate"]["total"] == 10.0
        assert d["preemption_event"]["time"] == "12:00"

    @pytest.mark.parametrize("status", ["completed", "preempted", "failed"])
    def test_various_statuses(self, status: str) -> None:
        result = VertexTrainingResult(status=status, final_step=0)
        assert result.to_dict()["status"] == status


# ---------------------------------------------------------------------------
# Tests: VertexTrainer init
# ---------------------------------------------------------------------------


class TestVertexTrainerInit:
    def test_basic_init(self) -> None:
        model = _simple_model()
        config = {"training": {"total_steps": 10}}
        vc = _vertex_config()

        trainer = VertexTrainer(model=model, config=config, vertex_config=vc)

        assert trainer.model is model
        assert trainer.config is config
        assert trainer.vertex_config is vc
        assert trainer._current_step == 0
        assert trainer._best_metric is None
        assert trainer._metrics_history == []
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer._train_step_fn is None
        assert trainer._distributed_ctx is None
        assert trainer._checkpoint_manager is None
        assert trainer._preemption_handler is None
        assert trainer._cost_tracker is None

    def test_init_with_optimizer(self) -> None:
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        vc = _vertex_config()

        trainer = VertexTrainer(
            model=model, config={}, vertex_config=vc, optimizer=opt
        )
        assert trainer.optimizer is opt

    def test_init_with_scheduler(self) -> None:
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        vc = _vertex_config()

        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            optimizer=opt,
            scheduler=sched,
        )
        assert trainer.scheduler is sched

    def test_init_with_train_step_fn(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        step_fn = MagicMock(return_value={"loss": 0.0})

        trainer = VertexTrainer(
            model=model, config={}, vertex_config=vc, train_step_fn=step_fn
        )
        assert trainer._train_step_fn is step_fn

    @pytest.mark.parametrize("project", ["proj-a", "proj-b", "my-project"])
    def test_init_with_various_projects(self, project: str) -> None:
        model = _simple_model()
        vc = _vertex_config(project=project)
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        assert trainer.vertex_config.project_id == project


# ---------------------------------------------------------------------------
# Tests: _train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_default_train_step(self) -> None:
        trainer = _make_trainer()
        metrics = trainer._train_step()
        assert metrics["loss"] == 0.0
        assert "step" in metrics

    def test_custom_train_step(self) -> None:
        step_fn = MagicMock(return_value={"loss": 0.5, "acc": 0.9})
        trainer = _make_trainer(train_step_fn=step_fn)
        metrics = trainer._train_step()
        assert metrics["loss"] == 0.5
        assert metrics["acc"] == 0.9
        step_fn.assert_called_once()

    def test_custom_train_step_called_multiple_times(self) -> None:
        step_fn = MagicMock(return_value={"loss": 1.0})
        trainer = _make_trainer(train_step_fn=step_fn)
        for _ in range(3):
            trainer._train_step()
        assert step_fn.call_count == 3

    def test_default_step_increments_indirectly(self) -> None:
        trainer = _make_trainer()
        result = trainer._train_step()
        # step in result matches current (not yet incremented by caller)
        assert result["step"] == 0


# ---------------------------------------------------------------------------
# Tests: _load_state
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_load_model_state(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        state = {
            "model_state_dict": model.state_dict(),
            "step": 50,
        }

        trainer._load_state(state)
        assert trainer._current_step == 50

    def test_load_state_with_optimizer(self) -> None:
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model, config={}, vertex_config=vc, optimizer=opt
        )

        opt_state = opt.state_dict()
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt_state,
            "step": 100,
        }

        trainer._load_state(state)
        assert trainer._current_step == 100

    def test_load_state_with_scheduler(self) -> None:
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10)
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            optimizer=opt,
            scheduler=sched,
        )

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "step": 200,
        }

        trainer._load_state(state)
        assert trainer._current_step == 200

    def test_load_state_ddp_model(self) -> None:
        """With DDP-wrapped model, loads into .module."""
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        # Simulate DDP wrapper
        ddp_mock = MagicMock()
        ddp_mock.module = model
        trainer.model = ddp_mock

        state = {
            "model_state_dict": model.state_dict(),
            "step": 42,
        }
        trainer._load_state(state)
        assert trainer._current_step == 42

    def test_load_state_no_step_key(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        trainer._current_step = 10

        state = {"model_state_dict": model.state_dict()}
        trainer._load_state(state)
        assert trainer._current_step == 0

    def test_load_state_no_optimizer_state(self) -> None:
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model, config={}, vertex_config=vc, optimizer=opt
        )
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "step": 5,
        }
        trainer._load_state(state)
        assert trainer._current_step == 5


# ---------------------------------------------------------------------------
# Tests: _save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_save_checkpoint_main_process(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.save.return_value = "gs://bucket/ckpt_0.pt"

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm

        path = trainer._save_checkpoint(metrics={"loss": 0.1})
        assert path == "gs://bucket/ckpt_0.pt"
        mock_cm.save.assert_called_once()

    def test_save_checkpoint_non_main_skips(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_ctx = _mock_distributed_ctx(world_size=2, rank=1)
        mock_cm = MagicMock()

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm

        path = trainer._save_checkpoint(metrics={"loss": 0.1})
        assert path is None
        mock_cm.save.assert_not_called()

    def test_save_checkpoint_force_non_main(self) -> None:
        """force=True overrides non-main-process check."""
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_ctx = _mock_distributed_ctx(world_size=2, rank=1)
        mock_cm = MagicMock()
        mock_cm.save.return_value = "gs://bucket/ckpt_forced.pt"

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm

        path = trainer._save_checkpoint(metrics={}, force=True)
        assert path == "gs://bucket/ckpt_forced.pt"
        mock_cm.save.assert_called_once()

    def test_save_checkpoint_ddp_unwrap(self) -> None:
        """DDP-wrapped model is unwrapped before saving."""
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        ddp_mock = MagicMock()
        ddp_mock.module = model
        trainer.model = ddp_mock

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.save.return_value = "gs://bucket/ddp_ckpt.pt"

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm

        trainer._save_checkpoint(metrics={})
        call_kwargs = mock_cm.save.call_args[1]
        assert call_kwargs["model"] is model

    def test_save_checkpoint_none_metrics_uses_empty(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.save.return_value = "gs://bucket/ckpt.pt"

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm

        trainer._save_checkpoint(metrics=None)
        call_kwargs = mock_cm.save.call_args[1]
        assert call_kwargs["metrics"] == {}


# ---------------------------------------------------------------------------
# Tests: _emergency_checkpoint
# ---------------------------------------------------------------------------


class TestEmergencyCheckpoint:
    def test_emergency_checkpoint_calls_save(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.save.return_value = "gs://bucket/emergency.pt"

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm
        trainer._current_step = 99

        trainer._emergency_checkpoint()
        mock_cm.save.assert_called_once()

    def test_emergency_checkpoint_exception_handled(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.save.side_effect = OSError("GCS unavailable")

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm

        # Should not raise
        trainer._emergency_checkpoint()


# ---------------------------------------------------------------------------
# Tests: _log_metrics
# ---------------------------------------------------------------------------


class TestLogMetrics:
    def test_log_metrics_appends_to_history(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_current_cost.return_value = None
        trainer._cost_tracker = mock_cost_tracker

        metrics = {"loss": 0.5, "acc": 0.8}
        trainer._log_metrics(metrics)

        assert len(trainer._metrics_history) == 1
        assert trainer._metrics_history[0]["loss"] == 0.5

    def test_log_metrics_adds_cost(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_cost = MagicMock()
        mock_cost.estimated_total_cost = 7.5

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_current_cost.return_value = mock_cost
        trainer._cost_tracker = mock_cost_tracker

        metrics: dict[str, float] = {"loss": 1.0}
        trainer._log_metrics(metrics)

        assert trainer._metrics_history[-1]["cost_usd"] == 7.5

    def test_log_metrics_no_cost(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_current_cost.return_value = None
        trainer._cost_tracker = mock_cost_tracker

        trainer._log_metrics({"loss": 0.2})
        assert "cost_usd" not in trainer._metrics_history[-1]

    def test_log_metrics_multiple_entries(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_cost_tracker = MagicMock()
        mock_cost_tracker.get_current_cost.return_value = None
        trainer._cost_tracker = mock_cost_tracker

        for i in range(3):
            trainer._log_metrics({"loss": float(i)})

        assert len(trainer._metrics_history) == 3
        assert trainer._metrics_history[-1]["loss"] == 2.0


# ---------------------------------------------------------------------------
# Tests: _get_latest_metrics
# ---------------------------------------------------------------------------


class TestGetLatestMetrics:
    def test_empty_history(self) -> None:
        trainer = _make_trainer()
        assert trainer._get_latest_metrics() == {}

    def test_returns_last_entry(self) -> None:
        trainer = _make_trainer()
        trainer._metrics_history = [{"loss": 1.0}, {"loss": 0.5}, {"loss": 0.1}]
        assert trainer._get_latest_metrics() == {"loss": 0.1}

    def test_single_entry(self) -> None:
        trainer = _make_trainer()
        trainer._metrics_history = [{"loss": 0.42}]
        assert trainer._get_latest_metrics() == {"loss": 0.42}


# ---------------------------------------------------------------------------
# Tests: properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_main_process_no_ctx(self) -> None:
        trainer = _make_trainer()
        assert trainer.is_main_process is True

    def test_is_main_process_with_ctx_rank_0(self) -> None:
        trainer = _make_trainer()
        trainer._distributed_ctx = _mock_distributed_ctx(rank=0)
        assert trainer.is_main_process is True

    def test_is_main_process_with_ctx_rank_1(self) -> None:
        trainer = _make_trainer()
        trainer._distributed_ctx = _mock_distributed_ctx(world_size=2, rank=1)
        assert trainer.is_main_process is False

    def test_current_step_initial(self) -> None:
        trainer = _make_trainer()
        assert trainer.current_step == 0

    def test_current_step_after_update(self) -> None:
        trainer = _make_trainer()
        trainer._current_step = 42
        assert trainer.current_step == 42

    def test_distributed_context_initially_none(self) -> None:
        trainer = _make_trainer()
        assert trainer.distributed_context is None

    def test_distributed_context_after_setup(self) -> None:
        trainer = _make_trainer()
        mock_ctx = _mock_distributed_ctx()
        trainer._distributed_ctx = mock_ctx
        assert trainer.distributed_context is mock_ctx


# ---------------------------------------------------------------------------
# Tests: setup()
# ---------------------------------------------------------------------------


class TestSetup:
    @patch("src.vertex.trainer.create_preemption_handler")
    @patch("src.vertex.trainer.CostTracker")
    @patch("src.vertex.trainer.GCSCheckpointManager")
    @patch("src.vertex.trainer.setup_distributed_training")
    @patch("src.vertex.trainer._get_torch")
    def test_setup_initializes_components(
        self,
        mock_get_torch: MagicMock,
        mock_setup_dist: MagicMock,
        mock_cm_cls: MagicMock,
        mock_cost_cls: MagicMock,
        mock_preemption: MagicMock,
    ) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_get_torch.return_value = mock_torch

        mock_ctx = _mock_distributed_ctx()
        mock_setup_dist.return_value = mock_ctx

        mock_cm = MagicMock()
        mock_cm.get_latest_step.return_value = None
        mock_cm_cls.return_value = mock_cm

        mock_cost = MagicMock()
        mock_cost_cls.return_value = mock_cost

        mock_ph = _make_preemption_handler()
        mock_preemption.return_value = mock_ph

        model = _simple_model()
        vc = _vertex_config()
        # patch model.parameters() for device detection
        with patch.object(model, "parameters") as mock_params:
            mock_params.return_value = iter([MagicMock(device=MagicMock())])
            trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
            trainer.setup()

        assert trainer._distributed_ctx is mock_ctx
        assert trainer._checkpoint_manager is mock_cm
        assert trainer._preemption_handler is mock_ph
        assert trainer._cost_tracker is mock_cost
        mock_cost.start.assert_called_once()

    @patch("src.vertex.trainer.create_preemption_handler")
    @patch("src.vertex.trainer.CostTracker")
    @patch("src.vertex.trainer.GCSCheckpointManager")
    @patch("src.vertex.trainer.setup_distributed_training")
    @patch("src.vertex.trainer._get_torch")
    def test_setup_no_cuda(
        self,
        mock_get_torch: MagicMock,
        mock_setup_dist: MagicMock,
        mock_cm_cls: MagicMock,
        mock_cost_cls: MagicMock,
        mock_preemption: MagicMock,
    ) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_get_torch.return_value = mock_torch

        mock_ctx = _mock_distributed_ctx()
        mock_setup_dist.return_value = mock_ctx

        mock_cm = MagicMock()
        mock_cm_cls.return_value = mock_cm
        mock_cost_cls.return_value = MagicMock()
        mock_preemption.return_value = _make_preemption_handler()

        model = _simple_model()
        vc = _vertex_config()
        with patch.object(model, "parameters") as mock_params:
            mock_params.return_value = iter([MagicMock(device=MagicMock())])
            trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
            trainer.setup()

        # CUDA not called to set device since cuda unavailable
        mock_torch.cuda.set_device.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: _resume_from_checkpoint
# ---------------------------------------------------------------------------


class TestResumeFromCheckpoint:
    def test_resume_loads_state(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_cm = MagicMock()
        mock_cm.load.return_value = {
            "model_state_dict": model.state_dict(),
            "step": 77,
        }
        trainer._checkpoint_manager = mock_cm

        trainer._resume_from_checkpoint("gs://bucket/ckpt.pt")
        assert trainer._current_step == 77
        mock_cm.load.assert_called_once_with(gcs_path="gs://bucket/ckpt.pt")

    def test_resume_file_not_found_raises(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        mock_cm = MagicMock()
        mock_cm.load.side_effect = FileNotFoundError("not found")
        trainer._checkpoint_manager = mock_cm

        with pytest.raises(FileNotFoundError):
            trainer._resume_from_checkpoint("gs://bucket/missing.pt")


# ---------------------------------------------------------------------------
# Tests: train() loop
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def _setup_trainer_for_train(
        self,
        total_steps: int = 3,
        step_fn: Any = None,
        is_preempted: bool = False,
        should_checkpoint: bool = False,
        preemption_event: Any = None,
        auto_resume: bool = False,
    ) -> VertexTrainer:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model,
            config={"training": {"total_steps": total_steps}},
            vertex_config=vc,
            train_step_fn=step_fn,
        )

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.save.return_value = "gs://bucket/ckpt.pt"
        if auto_resume:
            mock_cm.get_latest_step.return_value = 1
            mock_cm.load_latest.return_value = {
                "model_state_dict": model.state_dict(),
                "step": 1,
            }
        else:
            mock_cm.get_latest_step.return_value = None

        mock_ph = _make_preemption_handler(
            is_preempted=is_preempted,
            should_checkpoint=should_checkpoint,
            preemption_event=preemption_event,
        )

        mock_cost = MagicMock()
        mock_cost.get_current_cost.return_value = None

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm
        trainer._preemption_handler = mock_ph
        trainer._cost_tracker = mock_cost

        return trainer

    def test_train_completes_n_steps(self) -> None:
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer = self._setup_trainer_for_train(total_steps=5, step_fn=step_fn)

        result = trainer.train(total_steps=5)

        assert result.status == "completed"
        assert result.final_step == 5
        assert step_fn.call_count == 5

    def test_train_uses_config_total_steps(self) -> None:
        step_fn = MagicMock(return_value={"loss": 0.2})
        trainer = self._setup_trainer_for_train(total_steps=3, step_fn=step_fn)

        result = trainer.train()  # no total_steps arg
        assert result.final_step == 3

    def test_train_preemption_stops_loop(self) -> None:
        trainer = self._setup_trainer_for_train(
            total_steps=100,
            is_preempted=True,
        )

        result = trainer.train(total_steps=100)

        assert result.status == "preempted"
        assert result.final_step == 0  # Stops before any step

    def test_train_final_checkpoint_saved(self) -> None:
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer = self._setup_trainer_for_train(total_steps=2, step_fn=step_fn)

        trainer.train(total_steps=2)

        # Final checkpoint should be called with force=True
        save_calls = trainer._checkpoint_manager.save.call_args_list
        assert len(save_calls) >= 1

    def test_train_with_resume(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            train_step_fn=MagicMock(return_value={"loss": 0.1}),
        )

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.get_latest_step.return_value = None
        mock_cm.save.return_value = "gs://bucket/ckpt.pt"
        mock_cm.load.return_value = {
            "model_state_dict": model.state_dict(),
            "step": 10,
        }
        mock_ph = _make_preemption_handler()
        mock_cost = MagicMock()
        mock_cost.get_current_cost.return_value = None

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm
        trainer._preemption_handler = mock_ph
        trainer._cost_tracker = mock_cost

        result = trainer.train(total_steps=12, resume_from="gs://bucket/ckpt.pt")
        assert result.final_step == 12
        mock_cm.load.assert_called_once()

    def test_train_auto_resume(self) -> None:
        trainer = self._setup_trainer_for_train(total_steps=5, auto_resume=True)
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer._train_step_fn = step_fn

        result = trainer.train(total_steps=5)
        # Auto-resume loads step=1, so only 4 more steps needed
        assert result.final_step == 5

    def test_train_auto_resume_failure_continues(self) -> None:
        """Auto-resume failure is caught and training continues from step 0."""
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            train_step_fn=MagicMock(return_value={"loss": 0.1}),
        )

        mock_ctx = _mock_distributed_ctx()
        mock_cm = MagicMock()
        mock_cm.get_latest_step.return_value = 5
        mock_cm.load_latest.side_effect = RuntimeError("load failed")
        mock_cm.save.return_value = "gs://bucket/ckpt.pt"
        mock_ph = _make_preemption_handler()
        mock_cost = MagicMock()
        mock_cost.get_current_cost.return_value = None

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm
        trainer._preemption_handler = mock_ph
        trainer._cost_tracker = mock_cost

        result = trainer.train(total_steps=3)
        assert result.final_step == 3

    def test_train_calls_setup_when_not_ready(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            train_step_fn=MagicMock(return_value={"loss": 0.1}),
        )

        with patch.object(trainer, "setup") as mock_setup:
            # After setup is called, inject mocks
            def _inject_mocks() -> None:
                mock_ctx = _mock_distributed_ctx()
                mock_cm = MagicMock()
                mock_cm.get_latest_step.return_value = None
                mock_cm.save.return_value = "gs://bucket/ckpt.pt"
                mock_ph = _make_preemption_handler()
                mock_cost = MagicMock()
                mock_cost.get_current_cost.return_value = None

                trainer._distributed_ctx = mock_ctx
                trainer._checkpoint_manager = mock_cm
                trainer._preemption_handler = mock_ph
                trainer._cost_tracker = mock_cost

            mock_setup.side_effect = _inject_mocks

            result = trainer.train(total_steps=2)
            mock_setup.assert_called_once()

    def test_train_checkpoint_triggered_by_handler(self) -> None:
        """When preemption handler says to checkpoint, _save_checkpoint is called."""
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer = self._setup_trainer_for_train(
            total_steps=3,
            step_fn=step_fn,
            should_checkpoint=True,
        )

        with patch.object(trainer, "_save_checkpoint", return_value="gs://ckpt") as mock_save:
            trainer.train(total_steps=3)
            # Should be called at each step + final checkpoint
            assert mock_save.call_count >= 1

    def test_train_metrics_logged_at_100_steps(self) -> None:
        """_log_metrics is called every 100 steps."""
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer = self._setup_trainer_for_train(
            total_steps=200,
            step_fn=step_fn,
        )

        with patch.object(trainer, "_log_metrics") as mock_log:
            trainer.train(total_steps=200)
            assert mock_log.call_count == 2  # at step 100 and 200

    def test_train_exception_sets_failed_status_and_reraises(self) -> None:
        step_fn = MagicMock(side_effect=RuntimeError("boom"))
        trainer = self._setup_trainer_for_train(total_steps=5, step_fn=step_fn)

        with pytest.raises(RuntimeError, match="boom"):
            trainer.train(total_steps=5)

    def test_train_result_contains_cost_estimate(self) -> None:
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer = self._setup_trainer_for_train(total_steps=2, step_fn=step_fn)

        mock_cost_obj = MagicMock()
        mock_cost_obj.to_dict.return_value = {"total": 3.50}
        trainer._cost_tracker.get_current_cost.return_value = mock_cost_obj

        result = trainer.train(total_steps=2)
        assert result.cost_estimate == {"total": 3.50}

    def test_train_result_no_cost(self) -> None:
        step_fn = MagicMock(return_value={"loss": 0.1})
        trainer = self._setup_trainer_for_train(total_steps=2, step_fn=step_fn)
        trainer._cost_tracker.get_current_cost.return_value = None

        result = trainer.train(total_steps=2)
        assert result.cost_estimate is None

    def test_train_result_preemption_event(self) -> None:
        mock_event = MagicMock()
        mock_event.to_dict.return_value = {"reason": "spot"}
        trainer = self._setup_trainer_for_train(
            total_steps=100,
            is_preempted=True,
            preemption_event=mock_event,
        )

        result = trainer.train(total_steps=100)
        assert result.preemption_event == {"reason": "spot"}

    def test_train_final_path_none_non_main(self) -> None:
        """Non-main process should have final_path = None."""
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            train_step_fn=MagicMock(return_value={"loss": 0.1}),
        )

        mock_ctx = _mock_distributed_ctx(world_size=2, rank=1)
        mock_cm = MagicMock()
        mock_cm.get_latest_step.return_value = None
        mock_ph = _make_preemption_handler()
        mock_cost = MagicMock()
        mock_cost.get_current_cost.return_value = None

        trainer._distributed_ctx = mock_ctx
        trainer._checkpoint_manager = mock_cm
        trainer._preemption_handler = mock_ph
        trainer._cost_tracker = mock_cost

        result = trainer.train(total_steps=2)
        assert result.final_checkpoint is None


# ---------------------------------------------------------------------------
# Tests: create_vertex_trainer factory
# ---------------------------------------------------------------------------


class TestCreateVertexTrainer:
    def test_factory_returns_vertex_trainer(self) -> None:
        model = _simple_model()
        vc = _vertex_config()

        trainer = create_vertex_trainer(
            model=model,
            config={"training": {"total_steps": 5}},
            vertex_config=vc,
        )

        assert isinstance(trainer, VertexTrainer)
        assert trainer.model is model

    def test_factory_passes_kwargs(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        opt = torch.optim.Adam(model.parameters())
        step_fn = MagicMock(return_value={"loss": 0.0})

        trainer = create_vertex_trainer(
            model=model,
            config={},
            vertex_config=vc,
            optimizer=opt,
            train_step_fn=step_fn,
        )

        assert trainer.optimizer is opt
        assert trainer._train_step_fn is step_fn

    @pytest.mark.parametrize("total_steps", [100, 1000, 50000])
    def test_factory_various_total_steps(self, total_steps: int) -> None:
        model = _simple_model()
        vc = _vertex_config()

        trainer = create_vertex_trainer(
            model=model,
            config={"training": {"total_steps": total_steps}},
            vertex_config=vc,
        )
        assert trainer.config["training"]["total_steps"] == total_steps
