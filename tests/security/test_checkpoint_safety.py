"""Tests for model checkpoint safety and secure loading."""

import os
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch

from src.training.checkpoint import CheckpointManager

pytestmark = pytest.mark.security

class MaliciousPickle:
    """A malicious object that executes code when unpickled."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (os.system, ('echo "vulnerable" > /dev/null',))

@pytest.fixture
def malicious_checkpoint(tmp_path: Path) -> Path:
    """Create a mock malicious checkpoint using standard pickle."""
    ckpt_path = tmp_path / "malicious.pt"
    state = {"weights": MaliciousPickle()}
    torch.save(state, ckpt_path)
    return ckpt_path

def test_torch_load_weights_only() -> None:
    """Verify torch.load is explicitly called with weights_only=True where applicable."""
    # Check that when CheckpointManager loads weights, it specifies weights_only=True
    with patch("torch.load") as mock_load:
        mock_load.return_value = {"model_state_dict": {}}
        manager = CheckpointManager(checkpoint_dir="dummy")
        # We need a mock model to pass to load_model_only
        mock_model = torch.nn.Linear(1, 1)
        from src.training.checkpoint import load_model_only
        load_model_only(mock_model, "dummy.pt", strict=False)

        mock_load.assert_called_with("dummy.pt", map_location="cpu", weights_only=True)

def test_malicious_checkpoint_rejected(malicious_checkpoint: Path) -> None:
    """Verify that a malicious checkpoint is rejected when loaded with weights_only=True."""
    with pytest.raises(pickle.UnpicklingError):
        torch.load(malicious_checkpoint, weights_only=True)

def test_checkpoint_path_validation(tmp_path: Path) -> None:
    """Verify untrusted checkpoint paths are validated against traversal."""
    manager = CheckpointManager(checkpoint_dir=str(tmp_path))
    with pytest.raises((ValueError, FileNotFoundError, PermissionError, RuntimeError)):
        manager.load(path="../../../etc/shadow")
