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
) -> None:
    """Backpropagate *value* from a leaf to the root.

    The strategy parameter is accepted for forward-compatibility
    but currently only ``MEAN`` semantics are used (the node's
    ``backup`` method accumulates total value and visit count;
    the mean is derived on read).

    For ``MAX`` and ``MIXED`` modes the per-node ``backup`` call
    is identical -- the difference manifests in
    :pyattr:`MCTSNode.mean_value` or a future ``max_value``
    property, which the selection phase would consult.

    Args:
        node: Leaf node where evaluation occurred.
        value: Value estimate produced by the neural network
            or rollout.
        strategy: Backup strategy enum (reserved for future use).

    """
    current: MCTSNode | None = node
    depth = 0
    while current is not None:
        current.backup(value)
        current = current.parent
        depth += 1

    logger.debug(
        "mcts.backup.complete",
        depth=depth,
        value=value,
        strategy=strategy.value,
    )
