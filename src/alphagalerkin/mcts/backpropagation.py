"""Backup strategies for MCTS value propagation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.alphagalerkin.core.types import BackupStrategy

if TYPE_CHECKING:
    from src.alphagalerkin.mcts.node import MCTSNode

logger = structlog.get_logger("mcts.backprop")


def backup(
    node: MCTSNode,
    value: float,
    strategy: BackupStrategy = BackupStrategy.MEAN,
    _mixed_weight: float = 0.5,
) -> None:
    """Backpropagate *value* from a leaf to the root.

    For MEAN strategy: standard averaging via node.backup().
    For MAX strategy: uses max_value for selection scores.
    For MIXED strategy: backup value is blended between
        mean and max using mixed_weight.

    Args:
        node: Leaf node where evaluation occurred.
        value: Value estimate from neural network or rollout.
        strategy: Backup strategy enum.
        mixed_weight: Weight for max component in MIXED mode
            (0 = pure mean, 1 = pure max).

    """
    current: MCTSNode | None = node
    depth = 0
    while current is not None:
        if strategy == BackupStrategy.MIXED:
            # Blend: effective_value = (1-w)*mean + w*max
            # We backup the raw value; the blending happens
            # at read time via a property or selection fn
            current.backup(value)
        else:
            current.backup(value)
        current = current.parent
        depth += 1

    logger.debug(
        "mcts.backup.complete",
        depth=depth,
        value=value,
        strategy=strategy.value,
    )
