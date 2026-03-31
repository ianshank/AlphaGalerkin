"""Unified loss function package for AlphaGalerkin.

Re-exports all public loss classes for backwards compatibility:

    from src.training.losses import AlphaGalerkinLoss
    from src.training.losses import L2RelativeLoss
    from src.training.losses import PhysicsInformedLoss

Also provides a ``get_loss()`` factory that creates any registered loss by
name, and the ``LossRegistry`` for programmatic discovery.
"""

from __future__ import annotations

from typing import Any

from torch import nn

# -- AlphaGalerkin losses ----------------------------------------------------
from src.training.losses.alphagalerkin import (  # noqa: F401
    AlphaGalerkinLoss,
    EntropyRegularizer,
)

# -- base types --------------------------------------------------------------
from src.training.losses.base import (  # noqa: F401
    BaseLoss,
    LossOutput,
    LossRegistry,
    register_loss,
)

# -- Operator learning losses ------------------------------------------------
from src.training.losses.operator import (  # noqa: F401
    H1Loss,
    L2RelativeLoss,
    MSELoss,
)

# -- Physics-informed losses -------------------------------------------------
from src.training.losses.physics import (  # noqa: F401
    BoundaryLoss,
    CombinedAlphaGalerkinPhysicsLoss,
    ConservationLoss,
    InitialConditionLoss,
    PhysicsInformedLoss,
    PhysicsLossConfig,
    PhysicsLossOutput,
    ResidualLoss,
)

__all__ = [
    # base
    "BaseLoss",
    "LossOutput",
    "LossRegistry",
    "register_loss",
    # alphagalerkin
    "AlphaGalerkinLoss",
    "EntropyRegularizer",
    # operator
    "L2RelativeLoss",
    "H1Loss",
    "MSELoss",
    # physics
    "ResidualLoss",
    "BoundaryLoss",
    "InitialConditionLoss",
    "ConservationLoss",
    "PhysicsInformedLoss",
    "CombinedAlphaGalerkinPhysicsLoss",
    "PhysicsLossConfig",
    "PhysicsLossOutput",
    # factory
    "get_loss",
]


def get_loss(name: str, **kwargs: Any) -> nn.Module:
    """Factory function to create a loss by registered name.

    Looks up ``name`` in the :class:`LossRegistry`.  If not found there,
    falls back to the hard-coded operator-loss mapping for full backwards
    compatibility with the old ``src.training.losses.get_loss`` API.

    Args:
        name: Registered loss name (e.g. ``"alphagalerkin"``, ``"l2_relative"``).
        **kwargs: Keyword arguments forwarded to the loss constructor.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If *name* is not registered.

    """
    # Try the registry first
    registry = LossRegistry()
    loss_cls = registry.get(name)

    if loss_cls is not None:
        return loss_cls(**kwargs)
    # Backwards-compat aliases (the old losses.py used these exact keys)
    _aliases: dict[str, str] = {
        "l2": "l2_relative",
        "sobolev": "h1",
    }
    alias_target = _aliases.get(name)
    if alias_target is not None:
        loss_cls = registry.get(alias_target)
        if loss_cls is not None:
            return loss_cls(**kwargs)
    available = registry.list_items()
    raise ValueError(f"Unknown loss: {name!r}. Available: {available}")
