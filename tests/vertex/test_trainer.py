"""Tests for VertexTrainer._train_step delegation paths.

These call ``_train_step()`` directly without ``setup()`` (no GCS / distributed
init needed — the step touches only model/optimizer/data attributes set in
``__init__``).
"""

from __future__ import annotations

import torch
from torch import nn

from src.vertex.trainer import VertexTrainer
from src.vertex.training_step import VertexHyperparams


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(1, 1)


def _data() -> list[tuple[torch.Tensor, torch.Tensor]]:
    x = torch.linspace(-1, 1, 16).unsqueeze(1)
    return [(x, 3.0 * x)]


class TestTrainStepFnOverride:
    def test_custom_train_step_fn_wins(self, sample_vertex_config) -> None:
        trainer = VertexTrainer(
            model=_model(),
            config={},
            vertex_config=sample_vertex_config,
            train_step_fn=lambda: {"loss": 1.23},
        )
        assert trainer._train_step() == {"loss": 1.23}


class TestDataSourceTraining:
    def test_optimizer_created_eagerly(self, sample_vertex_config) -> None:
        trainer = VertexTrainer(
            model=_model(),
            config={"training": {"learning_rate": 1e-2}},
            vertex_config=sample_vertex_config,
            data_source=_data(),
        )
        assert trainer.optimizer is not None
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-2

    def test_real_step_returns_metrics(self, sample_vertex_config) -> None:
        trainer = VertexTrainer(
            model=_model(),
            config={"training": {"learning_rate": 1e-1}},
            vertex_config=sample_vertex_config,
            data_source=_data(),
        )
        metrics = trainer._train_step()
        assert "loss" in metrics and "lr" in metrics and "step" in metrics
        assert metrics["lr"] == 1e-1

    def test_loss_decreases_over_steps(self, sample_vertex_config) -> None:
        trainer = VertexTrainer(
            model=_model(),
            config={"training": {"learning_rate": 1e-1}},
            vertex_config=sample_vertex_config,
            data_source=_data(),
            hyperparams=VertexHyperparams(learning_rate=1e-1),
        )
        first = trainer._train_step()["loss"]
        for _ in range(40):
            last = trainer._train_step()["loss"]
        assert last < first


class TestNoOpFallback:
    def test_noop_returns_zero_and_warns_once(self, sample_vertex_config) -> None:
        trainer = VertexTrainer(
            model=_model(),
            config={},
            vertex_config=sample_vertex_config,
        )
        assert trainer._batch_source is None
        m1 = trainer._train_step()
        assert m1["loss"] == 0.0
        assert trainer._warned_train_step_noop is True
        # Second call still works and stays no-op (warning already emitted once).
        m2 = trainer._train_step()
        assert m2["loss"] == 0.0
