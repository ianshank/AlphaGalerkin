"""Physics-informed loss components for AlphaGalerkin.

Backwards-compatibility wrapper. All implementations now live in
``src.training.losses.physics``.
"""

from src.training.losses.physics import BoundaryLoss  # noqa: F401
from src.training.losses.physics import CombinedAlphaGalerkinPhysicsLoss  # noqa: F401
from src.training.losses.physics import ConservationLoss  # noqa: F401
from src.training.losses.physics import InitialConditionLoss  # noqa: F401
from src.training.losses.physics import PhysicsInformedLoss  # noqa: F401
from src.training.losses.physics import PhysicsLossConfig  # noqa: F401
from src.training.losses.physics import PhysicsLossOutput  # noqa: F401
from src.training.losses.physics import ResidualLoss  # noqa: F401
from src.training.losses.physics import _get_device_from_model  # noqa: F401
