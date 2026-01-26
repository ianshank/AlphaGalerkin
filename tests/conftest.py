"""Pytest configuration and fixtures."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def device() -> torch.device:
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
