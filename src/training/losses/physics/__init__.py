"""Physics-informed loss components for AlphaGalerkin.

This module provides physics-informed loss terms that can be
combined with the standard policy/value losses. These losses
encode physical constraints and PDE residuals to provide
denser training signal.

Components:
- ResidualLoss: PDE residual minimization
- BoundaryLoss: Boundary condition enforcement
- InitialConditionLoss: Initial condition enforcement
- ConservationLoss: Conservation law satisfaction
- PhysicsInformedLoss: Combined physics loss with balancing

Reference:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).
    Physics-informed neural networks: A deep learning framework
    for solving forward and inverse problems.

This module was originally a single flat file
(``src/training/losses/physics.py``). It is now a package with one
``@register_loss`` class per file. This file re-exports every name the
old module exposed, so ``from src.training.losses.physics import ResidualLoss``
continues to work. See
``tests/training/test_losses_physics.py::TestPhysicsPackagePublicAPI``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor, nn

from src.templates.config import BaseModuleConfig
from src.training.loss_balancing import (
    LossBalancingConfig,
    create_loss_balancer,
)
from src.training.losses.base import register_loss

logger = structlog.get_logger(__name__)

from src.training.losses.physics.boundary import BoundaryLoss  # noqa: E402
from src.training.losses.physics.combined import CombinedAlphaGalerkinPhysicsLoss  # noqa: E402
from src.training.losses.physics.config import (  # noqa: E402
    PhysicsLossConfig,
    PhysicsLossOutput,
)
from src.training.losses.physics.conservation import ConservationLoss  # noqa: E402
from src.training.losses.physics.device import _get_device_from_model  # noqa: E402
from src.training.losses.physics.informed import PhysicsInformedLoss  # noqa: E402
from src.training.losses.physics.initial_condition import InitialConditionLoss  # noqa: E402
from src.training.losses.physics.residual import ResidualLoss  # noqa: E402

# ``from src.training.losses.physics.<submodule> import ...`` binds each
# submodule as an attribute of this package. The old flat module never
# exposed those names. Delete the bindings; submodules stay in sys.modules.
del (
    boundary,
    combined,
    config,
    conservation,
    device,
    informed,
    initial_condition,
    residual,
)

__all__ = [
    "BaseModuleConfig",
    "BoundaryLoss",
    "Callable",
    "CombinedAlphaGalerkinPhysicsLoss",
    "ConservationLoss",
    "Field",
    "Float",
    "InitialConditionLoss",
    "Literal",
    "LossBalancingConfig",
    "PhysicsInformedLoss",
    "PhysicsLossConfig",
    "PhysicsLossOutput",
    "ResidualLoss",
    "TYPE_CHECKING",
    "Tensor",
    "_get_device_from_model",
    "create_loss_balancer",
    "dataclass",
    "logger",
    "nn",
    "np",
    "register_loss",
    "structlog",
    "torch",
]
