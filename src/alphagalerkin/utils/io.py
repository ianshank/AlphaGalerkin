"""I/O utilities for configuration and checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger("utils.io")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.exists():
        msg = f"Configuration file not found: {path}"
        raise FileNotFoundError(msg)
    with open(path) as f:
        data = yaml.safe_load(f)
    logger.debug("io.yaml_loaded", path=str(path))
    return data or {}


def save_yaml(data: dict[str, Any], path: Path) -> None:
    """Save data to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.debug("io.yaml_saved", path=str(path))


def resolve_device(device: str) -> str:
    """Resolve 'auto' device to actual device string."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
