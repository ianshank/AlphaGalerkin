"""Physics-informed loss components for AlphaGalerkin.

Backwards-compatibility wrapper. All implementations now live in
``src.training.losses.physics``.
"""

from src.training.losses.physics import (
    BoundaryLoss,
    CombinedAlphaGalerkinPhysicsLoss,
    ConservationLoss,
    InitialConditionLoss,
    PhysicsInformedLoss,
    PhysicsLossConfig,
    PhysicsLossOutput,
    ResidualLoss,
    _get_device_from_model,
)

__all__ = [
    "BoundaryLoss",
    "CombinedAlphaGalerkinPhysicsLoss",
    "ConservationLoss",
    "InitialConditionLoss",
    "PhysicsInformedLoss",
    "PhysicsLossConfig",
    "PhysicsLossOutput",
    "ResidualLoss",
    "_get_device_from_model",
]
