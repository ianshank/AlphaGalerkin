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


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers for video and other test categories."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "e2e: end-to-end workflow tests")
    config.addinivalue_line("markers", "video: video compression tests")
    config.addinivalue_line("markers", "requires_video: tests requiring real video files")
    config.addinivalue_line("markers", "integration: integration tests")


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

