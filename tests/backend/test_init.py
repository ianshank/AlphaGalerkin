"""Tests for the backend __init__.py public API."""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")


import torch

from src.backend import (
    BackendConfig,
    BackendType,
    clear_cache,
    get_backend,
    get_device,
    set_default_backend,
)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear backend cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


class TestGetBackend:
    """Test get_backend factory function."""

    def test_get_backend_string(self):
        b = get_backend("torch")
        assert b.name == BackendType.TORCH

    def test_get_backend_config(self):
        config = BackendConfig(backend=BackendType.TORCH)
        b = get_backend(config)
        assert b.name == BackendType.TORCH

    def test_get_backend_keyword_config(self):
        config = BackendConfig(backend=BackendType.TORCH)
        b = get_backend(config=config)
        assert b.name == BackendType.TORCH

    def test_get_backend_default(self):
        b = get_backend()
        assert b.name == BackendType.TORCH

    def test_get_backend_caching(self):
        b1 = get_backend("torch")
        b2 = get_backend("torch")
        assert b1 is b2

    def test_get_backend_different_configs(self):
        from src.backend.types import Precision

        b1 = get_backend(BackendConfig(backend=BackendType.TORCH, precision=Precision.FLOAT32))
        b2 = get_backend(BackendConfig(backend=BackendType.TORCH, precision=Precision.FLOAT64))
        assert b1 is not b2

    def test_get_backend_invalid_type(self):
        with pytest.raises(TypeError):
            get_backend(42)  # type: ignore[arg-type]


class TestSetDefaultBackend:
    """Test set_default_backend."""

    def test_set_default_string(self):
        from src.backend import default_backend

        set_default_backend("torch")
        b = default_backend()
        assert b.name == BackendType.TORCH

    def test_set_default_config(self):
        from src.backend import default_backend

        config = BackendConfig(backend=BackendType.TORCH)
        set_default_backend(config)
        b = default_backend()
        assert b.name == BackendType.TORCH


class TestClearCache:
    """Test cache clearing."""

    def test_clear_cache_resets(self):
        b1 = get_backend("torch")
        clear_cache()
        b2 = get_backend("torch")
        assert b1 is not b2


class TestGetDevice:
    """Tests for the get_device() utility function."""

    def test_auto_returns_device(self):
        dev = get_device()
        assert isinstance(dev, torch.device)

    def test_auto_is_cuda_or_cpu(self):
        dev = get_device("auto")
        assert dev.type in {"cuda", "cpu"}

    def test_explicit_cpu(self):
        dev = get_device("cpu")
        assert dev.type == "cpu"

    def test_explicit_cuda_string(self):
        # String forwarded directly to torch.device — no availability check
        dev = get_device("cpu")  # always available
        assert dev.type == "cpu"

    def test_auto_matches_cuda_availability(self):
        dev = get_device()
        expected_type = "cuda" if torch.cuda.is_available() else "cpu"
        assert dev.type == expected_type

    def test_result_is_torch_device_instance(self):
        for arg in ("auto", "cpu"):
            assert isinstance(get_device(arg), torch.device)
