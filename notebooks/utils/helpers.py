"""Helper utilities for AlphaGalerkin demo notebooks.

Provides environment setup, sample data generation, and safe model operations.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    # Fallback to standard logging if structlog not available
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import nn

    from notebooks.utils.config import DemoConfig, GoBoardConfig


@dataclass
class EnvironmentInfo:
    """Information about the runtime environment."""

    device: torch.device
    cuda_available: bool
    cuda_device_name: str | None
    python_version: str
    torch_version: str
    project_root: Path


def setup_environment(
    random_seed: int = 42,
    project_root: Path | str | None = None,
) -> EnvironmentInfo:
    """Set up the notebook environment with proper paths and seeds.

    Args:
        random_seed: Random seed for reproducibility.
        project_root: Path to project root (auto-detected if None).

    Returns:
        EnvironmentInfo with device and version details.

    Raises:
        RuntimeError: If project root cannot be found.

    """
    import numpy as np

    # Detect project root
    if project_root is None:
        # Try to find project root from notebook location
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "src").exists() and (parent / "config").exists():
                project_root = parent
                break

        if project_root is None:
            # Fallback: assume we're in notebooks/
            project_root = current.parent

    project_root = Path(project_root)

    # Add project root to path
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        logger.debug("added_to_path", path=root_str)

    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    logger.debug("seeds_set", seed=random_seed)

    # Detect device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None

    env_info = EnvironmentInfo(
        device=device,
        cuda_available=cuda_available,
        cuda_device_name=cuda_name,
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        project_root=project_root,
    )

    logger.info(
        "environment_setup",
        device=str(device),
        cuda=cuda_available,
        project_root=str(project_root),
    )

    return env_info


def create_sample_board(
    size: int,
    black_positions: Sequence[tuple[int, int]] | None = None,
    white_positions: Sequence[tuple[int, int]] | None = None,
    n_channels: int = 17,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Create a sample Go board with stones.

    Args:
        size: Board size (height/width).
        black_positions: List of (row, col) for black stones.
        white_positions: List of (row, col) for white stones.
        n_channels: Number of input channels.
        device: Device for the tensor.

    Returns:
        Board tensor of shape (1, n_channels, size, size).

    """
    # Default positions if not specified
    if black_positions is None:
        black_positions = [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
    if white_positions is None:
        white_positions = [(3, 4), (4, 3), (4, 4), (5, 5)]

    board = torch.zeros(1, n_channels, size, size, device=device)

    # Channel 0: Black stones
    for r, c in black_positions:
        if 0 <= r < size and 0 <= c < size:
            board[0, 0, r, c] = 1

    # Channel 1: White stones
    for r, c in white_positions:
        if 0 <= r < size and 0 <= c < size:
            board[0, 1, r, c] = 1

    logger.debug(
        "created_sample_board",
        size=size,
        n_black=len([p for p in black_positions if p[0] < size and p[1] < size]),
        n_white=len([p for p in white_positions if p[0] < size and p[1] < size]),
    )

    return board


def create_sample_board_from_config(
    size: int,
    config: "GoBoardConfig",
    n_channels: int = 17,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Create a sample board using configuration.

    Args:
        size: Board size.
        config: GoBoardConfig with stone positions.
        n_channels: Number of input channels.
        device: Device for the tensor.

    Returns:
        Board tensor.

    """
    return create_sample_board(
        size=size,
        black_positions=config.black_stone_positions,
        white_positions=config.white_stone_positions,
        n_channels=n_channels,
        device=device,
    )


@dataclass
class ModelForwardResult:
    """Result of a safe model forward pass."""

    success: bool
    policy_logits: Tensor | None
    value: Tensor | None
    lbb_constant: Tensor | None
    error: str | None


def safe_model_forward(
    model: "nn.Module",
    x: Tensor,
    return_lbb: bool = False,
) -> ModelForwardResult:
    """Safely execute model forward pass with error handling.

    Args:
        model: Model to execute.
        x: Input tensor.
        return_lbb: Whether to request LBB constant.

    Returns:
        ModelForwardResult with outputs or error information.

    """
    try:
        model.eval()
        with torch.no_grad():
            output = model(x, return_lbb=return_lbb) if return_lbb else model(x)

        return ModelForwardResult(
            success=True,
            policy_logits=output.policy_logits,
            value=output.value,
            lbb_constant=getattr(output, "lbb_constant", None),
            error=None,
        )

    except Exception as e:
        logger.error(
            "model_forward_failed",
            error=str(e),
            input_shape=list(x.shape),
        )
        return ModelForwardResult(
            success=False,
            policy_logits=None,
            value=None,
            lbb_constant=None,
            error=str(e),
        )


def format_model_summary(model: "nn.Module") -> str:
    """Format a model summary with parameter counts.

    Args:
        model: PyTorch model.

    Returns:
        Formatted summary string.

    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        f"Model: {model.__class__.__name__}",
        f"Total parameters: {total_params:,}",
        f"Trainable parameters: {trainable_params:,}",
    ]

    return "\n".join(lines)


def validate_board_sizes(sizes: Sequence[int], min_size: int = 3, max_size: int = 25) -> bool:
    """Validate board sizes are within acceptable range.

    Args:
        sizes: Board sizes to validate.
        min_size: Minimum allowed size.
        max_size: Maximum allowed size.

    Returns:
        True if all sizes are valid.

    Raises:
        ValueError: If any size is invalid.

    """
    for size in sizes:
        if not isinstance(size, int):
            raise ValueError(f"Board size must be int, got {type(size)}")
        if size < min_size or size > max_size:
            raise ValueError(
                f"Board size {size} out of range [{min_size}, {max_size}]"
            )

    logger.debug("validated_board_sizes", sizes=list(sizes))
    return True
