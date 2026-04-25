"""Regression tests for ``DistributedTrainer._create_optimizer`` LR scaling.

PR #53 review (gemini ``src/distributed/trainer.py:193``):
``_create_optimizer`` previously hardcoded ``base_lr * world_size``,
ignoring :attr:`DistributedInfraConfig.learning_rate_scaling`. Verify
that all three strategies (``linear``/``sqrt``/``none``) produce the
expected scaled LR via :meth:`DistributedInfraConfig.scale_learning_rate`.
"""

from __future__ import annotations

import math
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from src.distributed.config import DistributedBackend, DistributedInfraConfig
from src.distributed.trainer import DistributedTrainer

BASE_LR = 1e-3
WORLD_SIZE = 4


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2, bias=False)

    def forward(self, x: Any) -> Any:  # pragma: no cover - unused
        return self.fc(x)


def _make_config(lr: float = BASE_LR) -> MagicMock:
    cfg = MagicMock()
    cfg.training.learning_rate = lr
    cfg.training.weight_decay = 0.0
    cfg.training.gradient_clip = 1.0
    cfg.model_dump.return_value = {"training": {"learning_rate": lr}}
    return cfg


def _make_trainer(strategy: str) -> DistributedTrainer:
    distributed_cfg = DistributedInfraConfig(
        enabled=True,
        backend=DistributedBackend.GLOO,
        world_size=WORLD_SIZE,
        learning_rate_scaling=strategy,  # type: ignore[arg-type]
        use_amp=False,
        save_on_rank_0_only=True,
    )
    env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": str(WORLD_SIZE)}
    with patch.dict(os.environ, env):
        return DistributedTrainer(
            model=_Tiny(),
            config=_make_config(),
            distributed_config=distributed_cfg,
            loss_fn=MagicMock(),
        )


@pytest.mark.parametrize(
    ("strategy", "expected_lr"),
    [
        ("linear", BASE_LR * WORLD_SIZE),
        ("sqrt", BASE_LR * math.sqrt(WORLD_SIZE)),
        ("none", BASE_LR),
    ],
)
def test_optimizer_uses_configured_lr_strategy(strategy: str, expected_lr: float) -> None:
    """``_create_optimizer`` must respect the configured LR strategy."""
    trainer = _make_trainer(strategy)
    optimizer = trainer.optimizer
    assert optimizer is not None
    actual_lr = optimizer.param_groups[0]["lr"]
    assert math.isclose(actual_lr, expected_lr, rel_tol=1e-9), (
        f"strategy={strategy} expected={expected_lr} actual={actual_lr}"
    )


def test_lr_scaling_matches_config_helper() -> None:
    """Trainer scaling must agree with ``scale_learning_rate`` directly."""
    for strategy in ("linear", "sqrt", "none"):
        trainer = _make_trainer(strategy)
        expected = trainer.distributed_config.scale_learning_rate(BASE_LR)
        actual = trainer.optimizer.param_groups[0]["lr"]
        assert math.isclose(actual, expected, rel_tol=1e-9)
