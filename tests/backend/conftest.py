"""Backend-specific pytest fixtures."""

from __future__ import annotations

import pytest

from tests.conftest import HAS_JAX, HAS_TORCH


@pytest.fixture
def torch_backend():
    """Provide a TorchBackend instance, skipping if PyTorch is unavailable."""
    if not HAS_TORCH:
        pytest.skip("torch not available")
    from src.backend import get_backend

    return get_backend("torch")


@pytest.fixture
def jax_backend():
    """Provide a JaxBackend instance, skipping if JAX is unavailable."""
    if not HAS_JAX:
        pytest.skip("jax not available")
    from src.backend import get_backend

    return get_backend("jax")
