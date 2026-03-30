"""Training loss functions for AlphaGalerkin.

Backwards-compatibility wrapper. All implementations now live in
``src.training.losses.alphagalerkin`` and ``src.training.losses.base``.
"""

from src.training.losses.alphagalerkin import AlphaGalerkinLoss  # noqa: F401
from src.training.losses.alphagalerkin import EntropyRegularizer  # noqa: F401
from src.training.losses.base import LossOutput  # noqa: F401
