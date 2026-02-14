"""Reproducible RNG management."""

from __future__ import annotations

import random

import numpy as np
import structlog
import torch

from src.alphagalerkin.core.constants import DEFAULT_SEED

logger = structlog.get_logger("seeding")


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("seeding.complete", seed=seed)


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Get a numpy random generator with optional seed."""
    return np.random.default_rng(seed)
