"""Video test configuration module.

Provides dynamic, environment-configurable settings for video compression tests.
No hardcoded paths - all values configurable via environment variables or defaults.

Example usage:
    from tests.video_compression.video_config import VideoTestConfig

    config = VideoTestConfig()
    if config.has_real_videos:
        video_path = config.sample_videos["4k_mp4"]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Constants - thresholds and limits (not paths)
DEFAULT_CLIP_FRAMES: Final[int] = 10
DEFAULT_CLIP_RESOLUTION: Final[tuple[int, int]] = (360, 640)  # height, width (360p)
QUALITY_THRESHOLD_PSNR: Final[float] = 25.0
QUALITY_THRESHOLD_SSIM: Final[float] = 0.8
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})


def _get_project_root() -> Path:
    """Get project root directory dynamically."""
    # Navigate up from this file to find project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
            return parent
    # Fallback: assume tests is directly under project root
    return current.parent.parent.parent


@dataclass
class VideoTestConfig:
    """Configuration for video compression tests.

    All paths are resolved relative to project root or from environment variables.
    This ensures the tests work in any environment (local, CI, container).
    """

    # Project root (auto-detected or from env var)
    project_root: Path = field(default_factory=_get_project_root)

    # Test video directory - configurable via env var
    _test_video_dir_override: str | None = field(
        default_factory=lambda: os.environ.get("TEST_VIDEO_DIR"),
        repr=False,
    )

    # Quality thresholds
    psnr_threshold: float = QUALITY_THRESHOLD_PSNR
    ssim_threshold: float = QUALITY_THRESHOLD_SSIM

    # Clip extraction settings
    default_clip_frames: int = DEFAULT_CLIP_FRAMES
    default_clip_resolution: tuple[int, int] = DEFAULT_CLIP_RESOLUTION

    # Test execution settings
    slow_test_timeout: int = field(
        default_factory=lambda: int(os.environ.get("VIDEO_TEST_TIMEOUT", "300"))
    )

    @property
    def test_video_dir(self) -> Path:
        """Get test video directory path."""
        if self._test_video_dir_override:
            return Path(self._test_video_dir_override)
        return self.project_root / "Test_Video_Materals"

    @property
    def has_real_videos(self) -> bool:
        """Check if real test videos are available."""
        if not self.test_video_dir.exists():
            return False
        return any(
            f.suffix.lower() in SUPPORTED_EXTENSIONS
            for f in self.test_video_dir.iterdir()
            if f.is_file()
        )

    @property
    def sample_videos(self) -> dict[str, Path]:
        """Get mapping of sample video identifiers to paths."""
        videos: dict[str, Path] = {}
        if not self.test_video_dir.exists():
            return videos

        for video_file in self.test_video_dir.iterdir():
            if video_file.is_file() and video_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                # Create identifier from filename
                ext = video_file.suffix.lower().lstrip(".")
                # Find unique key
                base_key = f"sample_{ext}"
                key = base_key
                counter = 1
                while key in videos:
                    key = f"{base_key}_{counter}"
                    counter += 1
                videos[key] = video_file

        return videos

    @property
    def output_dir(self) -> Path:
        """Get output directory for test artifacts."""
        path = self.project_root / "test_outputs" / "video_compression"
        return path

    def get_temp_clip_path(self, base_name: str = "test_clip") -> Path:
        """Get path for temporary clip file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{base_name}.mp4"


# Singleton instance for easy import
_config: VideoTestConfig | None = None


def get_video_test_config() -> VideoTestConfig:
    """Get or create the video test configuration singleton."""
    global _config
    if _config is None:
        _config = VideoTestConfig()
    return _config


# Pytest markers for video tests
VIDEO_TEST_MARKERS = {
    "video": "Video compression tests",
    "slow": "Slow tests (full resolution, many frames)",
    "e2e": "End-to-end workflow tests",
    "requires_video": "Tests requiring real video files",
}
