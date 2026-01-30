"""Pytest configuration for video compression tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_image():
    """Create small test image."""
    return torch.rand(2, 3, 64, 64)


@pytest.fixture
def medium_image():
    """Create medium test image."""
    return torch.rand(2, 3, 128, 128)


@pytest.fixture
def large_image():
    """Create large test image."""
    return torch.rand(1, 3, 256, 256)
