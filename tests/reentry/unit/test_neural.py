"""Tests for reentry neural encoder and trainer."""

from __future__ import annotations

import numpy as np
import torch

from src.reentry.neural.encoder import FlowFieldEncoder
from src.reentry.neural.training import ReentryEvaluatorTrainer, TrainingEpisode


class TestFlowFieldEncoder:
    def test_forward_pass(self) -> None:
        encoder = FlowFieldEncoder(encode_size=16, latent_dim=64)
        x = torch.randn(2, 7, 20, 20)
        latent = encoder(x)
        assert latent.shape == (2, 64)

    def test_encode_flow_field(self) -> None:
        ny, nx = 10, 15
        encoding = FlowFieldEncoder.encode_flow_field(
            density=np.ones((ny, nx)),
            velocity_x=np.ones((ny, nx)) * 100,
            velocity_y=np.zeros((ny, nx)),
            pressure=np.ones((ny, nx)) * 1e5,
            mach=np.ones((ny, nx)) * 2.0,
            freestream_rho=1.0,
            freestream_u=100.0,
            freestream_p=1e5,
        )
        assert encoding.tensor.shape == (1, 7, ny, nx)
        assert len(encoding.channels) == 7


class TestReentryTrainer:
    def test_predict(self) -> None:
        trainer = ReentryEvaluatorTrainer(n_actions=5, latent_dim=32)
        x = torch.randn(1, 7, 16, 16)
        value, policy = trainer.predict(x)
        assert -1 <= value <= 1
        assert policy.shape == (5,)

    def test_train_step(self) -> None:
        trainer = ReentryEvaluatorTrainer(n_actions=5, latent_dim=32)
        ep = TrainingEpisode(
            states=[torch.randn(7, 16, 16)],
            actions=[0],
            rewards=[0.5],
            value_target=0.8,
            policy_target=[torch.softmax(torch.randn(5), dim=0)],
        )
        metrics = trainer.train_step([ep])
        assert metrics["n_samples"] == 1
        assert metrics["total_loss"] > 0

    def test_train_empty(self) -> None:
        trainer = ReentryEvaluatorTrainer(n_actions=5)
        metrics = trainer.train_step([])
        assert metrics["n_samples"] == 0
