"""Training loss functions for AlphaGalerkin.

Backwards-compatibility wrapper. All implementations now live in
``src.training.losses.alphagalerkin`` and ``src.training.losses.base``.
"""

from src.training.losses.alphagalerkin import (
    AlphaGalerkinLoss,
    EntropyRegularizer,
)
from src.training.losses.base import LossOutput

__all__ = ["AlphaGalerkinLoss", "EntropyRegularizer", "LossOutput"]
