"""Pytest configuration for video compression tests.

Provides shared fixtures for:
- Random seed management and device selection
- Image and video test tensors
- MCTS rate control configuration
- Mock codec components
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import Tensor

# Import fixtures from video_fixtures for test discovery
from tests.video_compression.video_fixtures import *  # noqa: F401, F403

# --------------------------------------------------------------------------
# Core Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device() -> torch.device:
    """Get device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Image Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def small_image() -> Tensor:
    """Create small test image (batch=2, 64x64)."""
    return torch.rand(2, 3, 64, 64)


@pytest.fixture
def medium_image() -> Tensor:
    """Create medium test image (batch=2, 128x128)."""
    return torch.rand(2, 3, 128, 128)


@pytest.fixture
def large_image() -> Tensor:
    """Create large test image (batch=1, 256x256)."""
    return torch.rand(1, 3, 256, 256)


# --------------------------------------------------------------------------
# Video Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def video_sequence_short() -> Tensor:
    """Create short video sequence (4 frames)."""
    return torch.rand(4, 3, 128, 128)


@pytest.fixture
def video_sequence_gop() -> Tensor:
    """Create GOP-sized video sequence (16 frames)."""
    return torch.rand(16, 3, 128, 128)


# --------------------------------------------------------------------------
# MCTS Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def mcts_config():
    """Create MCTS rate control configuration."""
    from src.video_compression.config import MCTSRateControlConfig, RateControlMode

    return MCTSRateControlConfig(
        name="test_mcts",
        num_simulations=10,  # Reduced for test speed
        c_puct=1.25,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        discount=0.99,
        gop_size=8,
        qp_min=0,
        qp_max=51,
        target_bitrate_kbps=2000.0,
        fps=30.0,
        rate_control_mode=RateControlMode.VBR,
    )


@pytest.fixture
def sample_latent() -> Tensor:
    """Create sample frame latent for MCTS testing."""
    return torch.randn(1, 192, 16, 16)


@pytest.fixture
def sample_state() -> Tensor:
    """Create sample hidden state for MCTS testing."""
    return torch.randn(1, 256)


# --------------------------------------------------------------------------
# Codec Configuration Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def codec_config():
    """Create default codec configuration."""
    from src.video_compression.config import CodecConfig

    return CodecConfig(name="test_codec")


@pytest.fixture
def dataset_config():
    """Create default dataset configuration."""
    from src.video_compression.data.dataset import DatasetConfig

    return DatasetConfig(
        root_dir="/test/data",
        patch_size=128,
        clip_length=8,
        random_crop=True,
        random_flip=True,
    )


# --------------------------------------------------------------------------
# File System Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_video_file(temp_dir: Path) -> Path:
    """Create temporary video file path."""
    path = temp_dir / "test_video.mp4"
    path.touch()
    return path


@pytest.fixture
def temp_bitstream_file(temp_dir: Path) -> Path:
    """Create temporary bitstream file path."""
    path = temp_dir / "test.agk"
    return path


# --------------------------------------------------------------------------
# Bitstream Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def sample_bitstream_header():
    """Create sample bitstream header."""
    from src.video_compression.utils.bitstream import BitstreamHeader

    return BitstreamHeader(
        width=256,
        height=256,
        num_frames=4,
        frame_rate=30.0,
        gop_size=16,
        downsample_factor=16,
        latent_channels=192,
        padded_width=256,
        padded_height=256,
        lambda_rd=0.01,
    )
