"""Pytest configuration and fixtures."""

from __future__ import annotations

import pytest

# Optional torch import - not all tests need it
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    
# Configure matplotlib for headless testing
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set random seed for reproducibility."""
    if HAS_TORCH:
        torch.manual_seed(42)


@pytest.fixture
def device():
    """Get device for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
