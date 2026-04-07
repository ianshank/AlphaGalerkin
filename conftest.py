"""Root conftest.py — pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip tests marked gpu_required when CUDA is not available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available (no NVIDIA driver)")
    for item in items:
        if item.get_closest_marker("gpu_required"):
            item.add_marker(skip_gpu)
