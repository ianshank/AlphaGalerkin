"""Coverage tests for Vertex AI trainer wrapper.

Targets uncovered lines in src/vertex/trainer.py:
    - VertexTrainer init
    - VertexTrainingResult
    - _train_step (default and custom)
    - _is_better
    - _log_metrics
    - _get_latest_metrics
    - Properties: is_main_process, current_step, distributed_context
    - create_vertex_trainer factory
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
    create_vertex_trainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_model() -> nn.Module:
    return nn.Linear(4, 2)


def _vertex_config() -> VertexTrainingConfig:
    return VertexTrainingConfig(
        project_id="test-project",
        staging_bucket="gs://test-bucket",
        storage=VertexStorageConfig(bucket_name="test-bucket"),
    )


# ---------------------------------------------------------------------------
# Tests: VertexTrainingResult
# ---------------------------------------------------------------------------


class TestVertexTrainingResult:
    def test_to_dict(self) -> None:
        result = VertexTrainingResult(
            status="completed",
            final_step=1000,
            metrics={"loss": 0.01},
        )
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["final_step"] == 1000
        assert d["metrics"]["loss"] == 0.01
        assert d["final_checkpoint"] is None

    def test_with_cost_and_preemption(self) -> None:
        result = VertexTrainingResult(
            status="preempted",
            final_step=500,
            cost_estimate={"total": 10.0},
            preemption_event={"time": "12:00"},
        )
        d = result.to_dict()
        assert d["cost_estimate"]["total"] == 10.0


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
        assert trainer._current_step == 0
        assert trainer._best_metric is None

    def test_init_with_optimizer(self) -> None:
        model = _simple_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        vc = _vertex_config()

        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            optimizer=opt,
        )
        assert trainer.optimizer is opt

    def test_init_with_train_step_fn(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        step_fn = MagicMock(return_value={"loss": 0.0})

        trainer = VertexTrainer(
            model=model,
            config={},
            vertex_config=vc,
            train_step_fn=step_fn,
        )
        assert trainer._train_step_fn is step_fn


# ---------------------------------------------------------------------------
# Tests: _train_step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_default_train_step(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        metrics = trainer._train_step()
        assert metrics["loss"] == 0.0

    def test_custom_train_step(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        step_fn = MagicMock(return_value={"loss": 0.5, "acc": 0.9})

        trainer = VertexTrainer(
            model=model, config={}, vertex_config=vc, train_step_fn=step_fn
        )
        metrics = trainer._train_step()
        assert metrics["loss"] == 0.5
        step_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_main_process_no_ctx(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        assert trainer.is_main_process is True

    def test_current_step(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        assert trainer.current_step == 0
        trainer._current_step = 42
        assert trainer.current_step == 42

    def test_distributed_context_none(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        assert trainer.distributed_context is None


# ---------------------------------------------------------------------------
# Tests: _get_latest_metrics
# ---------------------------------------------------------------------------


class TestGetLatestMetrics:
    def test_empty_history(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        assert trainer._get_latest_metrics() == {}

    def test_with_history(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)
        trainer._metrics_history = [
            {"loss": 1.0},
            {"loss": 0.5},
        ]
        assert trainer._get_latest_metrics() == {"loss": 0.5}


# ---------------------------------------------------------------------------
# Tests: _load_state
# ---------------------------------------------------------------------------


class TestLoadState:
    def test_load_state(self) -> None:
        model = _simple_model()
        vc = _vertex_config()
        trainer = VertexTrainer(model=model, config={}, vertex_config=vc)

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None,
            "scheduler_state_dict": None,
            "step": 50,
        }

        trainer._load_state(state)
        assert trainer._current_step == 50


# ---------------------------------------------------------------------------
# Tests: create_vertex_trainer factory
# ---------------------------------------------------------------------------


class TestCreateVertexTrainer:
    def test_factory(self) -> None:
        model = _simple_model()
        vc = _vertex_config()

        trainer = create_vertex_trainer(
            model=model,
            config={"training": {"total_steps": 5}},
            vertex_config=vc,
        )

        assert isinstance(trainer, VertexTrainer)
