"""Tests for firefighting neural encoder and trainer."""

from __future__ import annotations

import numpy as np
import torch

from src.firefighting.neural.encoder import FireStateEncoder
from src.firefighting.neural.training import FireEvaluatorTrainer, FireTrainingEpisode


class TestFireStateEncoder:
    def test_forward_pass(self) -> None:
        encoder = FireStateEncoder(encode_size=16, latent_dim=64)
        x = torch.randn(2, 7, 20, 20)
        latent = encoder(x)
        assert latent.shape == (2, 64)

    def test_encode_fire_state(self) -> None:
        ny, nx = 10, 15
        encoding = FireStateEncoder.encode_fire_state(
            temperature=np.ones((ny, nx)) * 500.0,
            fuel_remaining=np.ones((ny, nx)) * 0.8,
            wind_u=np.ones((ny, nx)) * 3.0,
            wind_v=np.zeros((ny, nx)),
        )
        assert encoding.tensor.shape == (1, 7, ny, nx)
        assert len(encoding.channels) == 7


class TestFireTrainer:
    def test_predict(self) -> None:
        trainer = FireEvaluatorTrainer(n_actions=5, latent_dim=32)
        x = torch.randn(1, 7, 16, 16)
        value, policy = trainer.predict(x)
        assert -1 <= value <= 1
        assert policy.shape == (5,)

    def test_train_step(self) -> None:
        trainer = FireEvaluatorTrainer(n_actions=5, latent_dim=32)
        ep = FireTrainingEpisode(
            states=[torch.randn(7, 16, 16)],
            actions=[0],
            rewards=[0.3],
            value_target=0.6,
            policy_target=[torch.softmax(torch.randn(5), dim=0)],
        )
        metrics = trainer.train_step([ep])
        assert metrics["n_samples"] == 1
        assert metrics["total_loss"] > 0
