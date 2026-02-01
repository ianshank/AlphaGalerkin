"""Video test fixtures for E2E workflow validation.

Provides reusable pytest fixtures for video compression testing,
including clip extraction, synthetic video generation, and codec setup.

All fixtures are parametrizable and use configuration from video_config.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.video_compression.video_config import (
    VideoTestConfig,
    get_video_test_config,
)

if TYPE_CHECKING:
    pass

# Configure logging for test fixtures
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def video_test_config() -> VideoTestConfig:
    """Get video test configuration (session-scoped for efficiency)."""
    return get_video_test_config()


@pytest.fixture(scope="session")
def has_real_videos(video_test_config: VideoTestConfig) -> bool:
    """Check if real test videos are available."""
    return video_test_config.has_real_videos


# ============================================================================
# Video Path Fixtures
# ============================================================================


@pytest.fixture
def sample_video_mp4(video_test_config: VideoTestConfig) -> Path | None:
    """Get path to a sample MP4 video, if available."""
    for key, path in video_test_config.sample_videos.items():
        if path.suffix.lower() == ".mp4":
            return path
    return None


@pytest.fixture
def sample_video_mov(video_test_config: VideoTestConfig) -> Path | None:
    """Get path to a sample MOV video, if available."""
    for key, path in video_test_config.sample_videos.items():
        if path.suffix.lower() == ".mov":
            return path
    return None


@pytest.fixture
def all_sample_videos(video_test_config: VideoTestConfig) -> dict[str, Path]:
    """Get all available sample videos."""
    return video_test_config.sample_videos


# ============================================================================
# Clip Extraction Fixtures
# ============================================================================


@pytest.fixture
def short_clip_factory(
    video_test_config: VideoTestConfig,
    tmp_path: Path,
):
    """Factory fixture for creating short video clips.

    Returns a callable that extracts short clips from source videos.

    Usage:
        def test_something(short_clip_factory):
            clip_path = short_clip_factory(source_video, num_frames=10, resolution=(360, 640))
    """
    import cv2

    def _create_clip(
        source_path: Path,
        num_frames: int | None = None,
        resolution: tuple[int, int] | None = None,
        output_name: str | None = None,
    ) -> Path:
        """Extract a short clip from source video.

        Args:
            source_path: Source video file path.
            num_frames: Number of frames to extract (default from config).
            resolution: Target (height, width) or None for original.
            output_name: Output filename (auto-generated if None).

        Returns:
            Path to the extracted clip.
        """
        num_frames = num_frames or video_test_config.default_clip_frames
        resolution = resolution or video_test_config.default_clip_resolution

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {source_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine output dimensions
        target_h, target_w = resolution
        if resolution == (0, 0):
            target_h, target_w = orig_height, orig_width

        # Output path
        if output_name is None:
            output_name = f"clip_{source_path.stem}_{target_h}p_{num_frames}f.mp4"
        output_path = tmp_path / output_name

        # Initialize writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_w, target_h))

        frames_written = 0
        try:
            while frames_written < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize if needed
                if (frame.shape[0], frame.shape[1]) != (target_h, target_w):
                    frame = cv2.resize(frame, (target_w, target_h))

                writer.write(frame)
                frames_written += 1
        finally:
            cap.release()
            writer.release()

        logger.debug(f"Created clip: {output_path} ({frames_written} frames, {target_w}x{target_h})")
        return output_path

    return _create_clip


@pytest.fixture
def short_clip_from_real_video(
    video_test_config: VideoTestConfig,
    short_clip_factory,
    sample_video_mp4: Path | None,
) -> Path | None:
    """Get a short clip extracted from a real video source.

    Returns None if no real videos are available.
    Prefer using short_clip_factory directly for more control.
    """
    if sample_video_mp4 is None:
        return None
    return short_clip_factory(sample_video_mp4)


# ============================================================================
# Synthetic Video Fixtures
# ============================================================================


@pytest.fixture
def synthetic_video_factory(tmp_path: Path):
    """Factory fixture for creating synthetic test videos.

    Creates videos with known patterns for deterministic testing.

    Usage:
        def test_something(synthetic_video_factory):
            video_path = synthetic_video_factory(height=64, width=64, num_frames=10)
    """
    import cv2
    import numpy as np

    def _create_synthetic(
        height: int = 64,
        width: int = 64,
        num_frames: int = 10,
        fps: float = 30.0,
        pattern: str = "gradient",  # "gradient", "random", "checkerboard"
        output_name: str | None = None,
        extension: str = ".mp4",
    ) -> Path:
        """Create a synthetic video with known patterns.

        Args:
            height: Frame height.
            width: Frame width.
            num_frames: Number of frames.
            fps: Frame rate.
            pattern: Frame pattern type.
            output_name: Output filename.
            extension: Output file extension.

        Returns:
            Path to created video.
        """
        if output_name is None:
            output_name = f"synthetic_{pattern}_{height}x{width}_{num_frames}f{extension}"
        output_path = tmp_path / output_name

        fourcc_map = {
            ".mp4": "mp4v",
            ".avi": "XVID",
            ".mov": "mp4v",
        }
        fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(extension, "mp4v"))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        try:
            for i in range(num_frames):
                if pattern == "gradient":
                    # Smooth gradient that changes over time
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    for c in range(3):
                        # Create gradient based on frame index and channel
                        base_val = int(255 * (i / max(num_frames - 1, 1)))
                        grad = np.linspace(0, 255, width).astype(np.uint8)
                        shift = (c * 85 + base_val) % 256
                        frame[:, :, c] = (grad + shift) % 256

                elif pattern == "random":
                    # Random noise
                    frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

                elif pattern == "checkerboard":
                    # Animated checkerboard
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    block_size = max(8, height // 8)
                    for y in range(0, height, block_size):
                        for x in range(0, width, block_size):
                            if ((y // block_size) + (x // block_size) + i) % 2 == 0:
                                frame[y:y+block_size, x:x+block_size] = 255
                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                writer.write(frame)
        finally:
            writer.release()

        return output_path

    return _create_synthetic


@pytest.fixture
def synthetic_test_video(synthetic_video_factory) -> Path:
    """Get a default synthetic test video (64x64, 10 frames)."""
    return synthetic_video_factory()


# ============================================================================
# Codec Fixtures
# ============================================================================


@pytest.fixture
def dummy_codec_checkpoint(tmp_path: Path):
    """Create a dummy codec checkpoint for testing.

    Returns path to a minimal valid checkpoint file.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2]))

    import torch
    from src.video_compression.config import CodecConfig
    from src.video_compression.codec.codec import create_codec

    config = CodecConfig(name="test_codec")
    codec = create_codec(config)

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": codec.state_dict(),
            "config": config.model_dump() if hasattr(config, "model_dump") else {},
        },
        checkpoint_path,
    )

    return checkpoint_path


# ============================================================================
# Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def auto_cleanup_cuda():
    """Automatically clean up CUDA memory after each test."""
    yield
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ============================================================================
# Skip Conditions
# ============================================================================


# Decorator for tests requiring real videos
requires_real_video = pytest.mark.skipif(
    not get_video_test_config().has_real_videos,
    reason="Real test videos not available",
)

# Decorator for slow tests
slow_test = pytest.mark.slow
