"""E2E tests for video compression workflow with real content.

Tests the complete encode-decode cycle using real video test content.
Validates quality metrics, extension preservation, and GOP handling.

Usage:
    pytest tests/e2e/test_real_video_e2e.py -v
    pytest tests/e2e/test_real_video_e2e.py -v -m "not slow"  # Skip slow tests
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from tests.video_compression.video_config import (
    VideoTestConfig,
    get_video_test_config,
)
from tests.video_compression.video_fixtures import (
    requires_real_video,
    slow_test,
)

if TYPE_CHECKING:
    pass

# Configure logging
logger = logging.getLogger(__name__)

# Resolution presets for parametrized tests
RESOLUTIONS = {
    "360p": (360, 640),
    "480p": (480, 854),
    "720p": (720, 1280),
}


# ============================================================================
# Fixtures (import from video_fixtures for session-level ones)
# ============================================================================


@pytest.fixture
def video_config() -> VideoTestConfig:
    """Get video test configuration."""
    return get_video_test_config()


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Get output directory for test artifacts."""
    output = tmp_path / "video_outputs"
    output.mkdir(parents=True, exist_ok=True)
    return output


# ============================================================================
# Utility Functions
# ============================================================================


def run_script(script: str, args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a Python script with arguments.

    Args:
        script: Script path relative to project root.
        args: Command-line arguments.
        timeout: Timeout in seconds.

    Returns:
        CompletedProcess result.

    """
    cmd = [sys.executable, script, *args]
    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(get_video_test_config().project_root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        logger.error(f"Script failed: {result.stderr}")

    return result


# ============================================================================
# E2E Tests with Synthetic Videos (Always Run)
# ============================================================================


@pytest.mark.e2e
@pytest.mark.video
class TestSyntheticVideoE2E:
    """E2E tests using synthetic video content (no external dependencies)."""

    def test_encode_decode_synthetic_video(
        self,
        synthetic_video_factory,
        dummy_codec_checkpoint: Path,
        output_dir: Path,
    ) -> None:
        """Test full encode-decode cycle with synthetic video."""
        # Create synthetic test video
        input_video = synthetic_video_factory(
            height=64,
            width=64,
            num_frames=8,
            pattern="gradient",
        )

        bitstream_path = output_dir / "synthetic.agk"
        decoded_path = output_dir / "synthetic_decoded.mp4"

        # Run encode
        result = run_script(
            "scripts/encode_video.py",
            [
                str(input_video),
                str(bitstream_path),
                "--qp",
                "32",
                "--model",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
                "--gop-size",
                "4",
            ],
        )

        assert result.returncode == 0, f"Encode failed: {result.stderr}"
        assert bitstream_path.exists(), "Bitstream not created"
        assert bitstream_path.stat().st_size > 0, "Empty bitstream"

        # Run decode
        result = run_script(
            "scripts/decode_video.py",
            [
                "--input",
                str(bitstream_path),
                "--output",
                str(decoded_path),
                "--checkpoint",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )

        assert result.returncode == 0, f"Decode failed: {result.stderr}"
        assert decoded_path.exists(), "Decoded video not created"
        assert decoded_path.stat().st_size > 0, "Empty decoded video"

    @pytest.mark.parametrize("extension", [".mp4", ".mov", ".avi"])
    def test_extension_preservation(
        self,
        synthetic_video_factory,
        dummy_codec_checkpoint: Path,
        output_dir: Path,
        extension: str,
    ) -> None:
        """Test that output extension matches input extension."""
        input_video = synthetic_video_factory(
            height=64,
            width=64,
            num_frames=5,
            extension=extension,
        )

        bitstream_path = output_dir / f"test{extension}.agk"
        decoded_path = output_dir / f"decoded{extension}"

        # Encode
        result = run_script(
            "scripts/encode_video.py",
            [
                str(input_video),
                str(bitstream_path),
                "--qp",
                "32",
                "--model",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )
        assert result.returncode == 0, f"Encode failed: {result.stderr}"

        # Decode with matching extension
        result = run_script(
            "scripts/decode_video.py",
            [
                "--input",
                str(bitstream_path),
                "--output",
                str(decoded_path),
                "--checkpoint",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )
        assert result.returncode == 0, f"Decode failed: {result.stderr}"
        assert decoded_path.suffix.lower() == extension.lower()


# ============================================================================
# E2E Tests with Real Videos
# ============================================================================


@pytest.mark.e2e
@pytest.mark.video
@requires_real_video
class TestRealVideoE2E:
    """E2E tests using real video content."""

    def test_encode_decode_with_short_clip(
        self,
        short_clip_factory,
        sample_video_mp4: Path | None,
        dummy_codec_checkpoint: Path,
        output_dir: Path,
    ) -> None:
        """Test encode-decode with a short clip from real video."""
        if sample_video_mp4 is None:
            pytest.skip("No sample MP4 video available")

        # Extract short clip
        clip_path = short_clip_factory(
            sample_video_mp4,
            num_frames=10,
            resolution=(360, 640),
        )

        bitstream_path = output_dir / "real_clip.agk"
        decoded_path = output_dir / "real_decoded.mp4"
        metrics_path = output_dir / "metrics.json"

        # Encode
        result = run_script(
            "scripts/encode_video.py",
            [
                str(clip_path),
                str(bitstream_path),
                "--qp",
                "28",
                "--model",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )
        assert result.returncode == 0, f"Encode failed: {result.stderr}"

        # Decode with quality metrics
        result = run_script(
            "scripts/decode_video.py",
            [
                "--input",
                str(bitstream_path),
                "--output",
                str(decoded_path),
                "--checkpoint",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
                "--quality-report",
                str(metrics_path),
                "--reference-video",
                str(clip_path),
            ],
        )
        assert result.returncode == 0, f"Decode failed: {result.stderr}"

        # Verify outputs
        assert decoded_path.exists()
        assert decoded_path.stat().st_size > 0

        # Verify quality metrics if available
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            logger.info(
                f"Quality metrics: PSNR={metrics.get('avg_psnr')}, SSIM={metrics.get('avg_ssim')}"
            )

            # Check quality thresholds (relaxed for CPU/random model)
            avg_psnr = metrics.get("avg_psnr", 0)
            assert avg_psnr > 10.0, f"PSNR too low: {avg_psnr}"  # Relaxed for random model

    @slow_test
    @pytest.mark.parametrize("resolution", ["360p", "480p"])
    def test_multiple_resolutions(
        self,
        short_clip_factory,
        sample_video_mp4: Path | None,
        dummy_codec_checkpoint: Path,
        output_dir: Path,
        resolution: str,
    ) -> None:
        """Test encode-decode at multiple resolutions."""
        if sample_video_mp4 is None:
            pytest.skip("No sample MP4 video available")

        height, width = RESOLUTIONS[resolution]

        # Extract clip at target resolution
        clip_path = short_clip_factory(
            sample_video_mp4,
            num_frames=15,
            resolution=(height, width),
        )

        bitstream_path = output_dir / f"real_{resolution}.agk"
        decoded_path = output_dir / f"decoded_{resolution}.mp4"

        # Encode
        result = run_script(
            "scripts/encode_video.py",
            [
                str(clip_path),
                str(bitstream_path),
                "--qp",
                "32",
                "--model",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )
        assert result.returncode == 0, f"Encode failed at {resolution}: {result.stderr}"

        # Decode
        result = run_script(
            "scripts/decode_video.py",
            [
                "--input",
                str(bitstream_path),
                "--output",
                str(decoded_path),
                "--checkpoint",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )
        assert result.returncode == 0, f"Decode failed at {resolution}: {result.stderr}"

        # Verify frame dimensions
        import cv2

        cap = cv2.VideoCapture(str(decoded_path))
        try:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # May differ due to padding, check approximate
            assert abs(frame_height - height) <= 16
            assert abs(frame_width - width) <= 16
        finally:
            cap.release()


# ============================================================================
# GOP Boundary Tests
# ============================================================================


@pytest.mark.e2e
@pytest.mark.video
class TestGOPHandling:
    """Tests for GOP (Group of Pictures) boundary handling."""

    @pytest.mark.parametrize("gop_size", [4, 8, 16])
    def test_gop_sizes(
        self,
        synthetic_video_factory,
        dummy_codec_checkpoint: Path,
        output_dir: Path,
        gop_size: int,
    ) -> None:
        """Test encoding with different GOP sizes."""
        # Create video longer than GOP size
        num_frames = gop_size * 2 + 1
        input_video = synthetic_video_factory(
            height=64,
            width=64,
            num_frames=num_frames,
        )

        bitstream_path = output_dir / f"gop_{gop_size}.agk"
        decoded_path = output_dir / f"gop_{gop_size}_decoded.mp4"

        # Encode with specific GOP size
        result = run_script(
            "scripts/encode_video.py",
            [
                str(input_video),
                str(bitstream_path),
                "--qp",
                "32",
                "--model",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
                "--gop-size",
                str(gop_size),
            ],
        )
        assert result.returncode == 0, f"Encode failed with GOP {gop_size}: {result.stderr}"

        # Decode
        result = run_script(
            "scripts/decode_video.py",
            [
                "--input",
                str(bitstream_path),
                "--output",
                str(decoded_path),
                "--checkpoint",
                str(dummy_codec_checkpoint),
                "--device",
                "cpu",
            ],
        )
        assert result.returncode == 0, f"Decode failed with GOP {gop_size}: {result.stderr}"

        # Verify frame count
        import cv2

        cap = cv2.VideoCapture(str(decoded_path))
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Allow some tolerance
            assert (
                frame_count >= num_frames - 1
            ), f"Frame count mismatch: expected {num_frames}, got {frame_count}"
        finally:
            cap.release()


# ============================================================================
# Fixtures from video_fixtures.py (re-export for pytest discovery)
# ============================================================================

# These fixtures are imported via conftest pattern
from tests.video_compression.video_fixtures import (
    all_sample_videos,
    auto_cleanup_cuda,
    dummy_codec_checkpoint,
    has_real_videos,
    sample_video_mov,
    sample_video_mp4,
    short_clip_factory,
    short_clip_from_real_video,
    synthetic_test_video,
    synthetic_video_factory,
    video_test_config,
)

# Re-declare for pytest fixture discovery
__all__ = [
    "video_test_config",
    "has_real_videos",
    "sample_video_mp4",
    "sample_video_mov",
    "all_sample_videos",
    "short_clip_factory",
    "short_clip_from_real_video",
    "synthetic_video_factory",
    "synthetic_test_video",
    "dummy_codec_checkpoint",
    "auto_cleanup_cuda",
]
