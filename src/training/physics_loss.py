"""Physics-informed loss components for AlphaGalerkin.

Backwards-compatibility wrapper. All implementations now live in
``src.training.losses.physics``.
"""

from src.training.losses.physics import (
    BoundaryLoss,  # noqa: F401
    CombinedAlphaGalerkinPhysicsLoss,  # noqa: F401
    ConservationLoss,  # noqa: F401
    InitialConditionLoss,  # noqa: F401
    PhysicsInformedLoss,  # noqa: F401
    PhysicsLossConfig,  # noqa: F401
    PhysicsLossOutput,  # noqa: F401
    ResidualLoss,  # noqa: F401
    _get_device_from_model,  # noqa: F401
)
