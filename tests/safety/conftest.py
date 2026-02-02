"""Pytest fixtures for safety module tests."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch

from src.safety.config import AllowlistConfig, ValidationConfig, ValidationLevel


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_checkpoint(temp_dir: Path) -> Path:
    """Create a valid PyTorch checkpoint for testing."""
    checkpoint_path = temp_dir / "valid_checkpoint.pt"

    # Create a simple model state dict
    state_dict = {
        "layer1.weight": torch.randn(64, 32),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(10, 64),
        "layer2.bias": torch.randn(10),
    }

    # Create full checkpoint format
    checkpoint = {
        "model_state_dict": state_dict,
        "step": 1000,
        "version": "1.0.0",
        "config": {"d_model": 64, "n_layers": 2},
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def state_dict_only_checkpoint(temp_dir: Path) -> Path:
    """Create a checkpoint with just state dict (no wrapper)."""
    checkpoint_path = temp_dir / "state_dict_only.pt"

    state_dict = {
        "weight": torch.randn(32, 32),
        "bias": torch.randn(32),
    }

    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def checkpoint_with_nan(temp_dir: Path) -> Path:
    """Create a checkpoint containing NaN values."""
    checkpoint_path = temp_dir / "nan_checkpoint.pt"

    tensor = torch.randn(10, 10)
    tensor[0, 0] = float("nan")

    state_dict = {"corrupted": tensor}
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def checkpoint_with_inf(temp_dir: Path) -> Path:
    """Create a checkpoint containing Inf values."""
    checkpoint_path = temp_dir / "inf_checkpoint.pt"

    tensor = torch.randn(10, 10)
    tensor[0, 0] = float("inf")

    state_dict = {"corrupted": tensor}
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def large_tensor_checkpoint(temp_dir: Path) -> Path:
    """Create a checkpoint with a large tensor (for size limit testing)."""
    checkpoint_path = temp_dir / "large_checkpoint.pt"

    # Create a moderately large tensor (not actually huge to avoid memory issues)
    large_tensor = torch.randn(1000, 1000)  # ~4MB

    state_dict = {"large": large_tensor}
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def empty_checkpoint(temp_dir: Path) -> Path:
    """Create an empty checkpoint (no tensors)."""
    checkpoint_path = temp_dir / "empty_checkpoint.pt"

    checkpoint = {
        "step": 0,
        "config": {},
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def permissive_config() -> ValidationConfig:
    """Create a permissive validation config for testing."""
    return ValidationConfig(
        name="test_permissive",
        level=ValidationLevel.PERMISSIVE,
        max_file_size_gb=100.0,
        check_nan_inf=False,
    )


@pytest.fixture
def standard_config() -> ValidationConfig:
    """Create a standard validation config for testing."""
    return ValidationConfig(
        name="test_standard",
        level=ValidationLevel.STANDARD,
        max_file_size_gb=50.0,
        check_nan_inf=True,
    )


@pytest.fixture
def strict_config() -> ValidationConfig:
    """Create a strict validation config for testing."""
    return ValidationConfig(
        name="test_strict",
        level=ValidationLevel.STRICT,
        max_file_size_gb=10.0,
        max_tensor_size_gb=5.0,
        check_nan_inf=True,
        require_hash_verification=True,
    )


@pytest.fixture
def default_allowlist() -> AllowlistConfig:
    """Create default allowlist config for testing."""
    return AllowlistConfig(name="test_allowlist")


@pytest.fixture
def custom_allowlist() -> AllowlistConfig:
    """Create a custom allowlist with additional allowed classes."""
    return AllowlistConfig(
        name="custom_allowlist",
        custom_allowlist=["my_module.MyClass"],
        custom_denylist=["dangerous.Evil"],
    )
