"""Pytest configuration and fixtures."""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck
from hypothesis import settings as hypothesis_settings

# Register a CI-friendly hypothesis profile: no deadline, lower example count to
# stay within the CI job timeout across all Python versions.
# This is loaded whenever the CI env var is set (GitHub Actions sets CI=true).
hypothesis_settings.register_profile(
    "ci",
    deadline=None,
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow],
)
hypothesis_settings.register_profile(
    "default",
    deadline=None,  # keep deadline off locally too to avoid false flakes
)
hypothesis_settings.load_profile("ci" if os.environ.get("CI") else "default")

# Re-export video compression fixtures for test discovery
pytest_plugins = ["tests.video_compression.video_fixtures"]

# Optional torch import - not all tests need it
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

# Optional JAX import - not all tests need it
try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Configure matplotlib for headless testing
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest markers for video and other test categories."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "e2e: end-to-end workflow tests")
    config.addinivalue_line("markers", "video: video compression tests")
    config.addinivalue_line("markers", "requires_video: tests requiring real video files")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "jax: JAX-specific tests")
    config.addinivalue_line("markers", "cross_backend: cross-backend equivalence tests")


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


@pytest.fixture
def torch_backend():
    """Get the PyTorch backend for testing."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from src.backend import get_backend

    return get_backend("torch")


@pytest.fixture
def jax_backend():
    """Get the JAX backend for testing."""
    if not HAS_JAX:
        pytest.skip("jax not available")
    from src.backend import get_backend

    return get_backend("jax")


@pytest.fixture(params=["torch", "jax"])
def backend(request):
    """Parametrized fixture providing both backends.

    Tests using this fixture will run once per available backend.
    Backends that are not installed will be skipped.
    """
    backend_name = request.param
    if backend_name == "torch" and not HAS_TORCH:
        pytest.skip("torch not available")
    if backend_name == "jax" and not HAS_JAX:
        pytest.skip("jax not available")
    from src.backend import get_backend

    return get_backend(backend_name)
