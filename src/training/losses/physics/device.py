"""Device helper for physics-informed losses."""

from __future__ import annotations

import structlog
import torch
from torch import nn

logger = structlog.get_logger(__name__)


def _get_device_from_model(model: nn.Module) -> torch.device:
    """Safely get device from model parameters.

    Args:
        model: The neural network model.

    Returns:
        Device the model is on, or CPU if model has no parameters.

    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters (e.g., DataParallel wrapper or frozen model)
        logger.warning("model_has_no_parameters", defaulting_to="cpu")
        return torch.device("cpu")
