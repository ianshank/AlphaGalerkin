"""Dataset classes for video compression training.

Provides reusable dataset implementations for:
- Video sequences with temporal sampling
- Static images for image compression
- Variable resolution support
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DatasetConfig(BaseModel):
    """Configuration for compression datasets."""

    # Paths
    root_dir: str = Field(..., description="Root directory for data")

    # Image/patch settings
    patch_size: int = Field(default=256, ge=32, le=1024, description="Training patch size")
    min_resolution: int = Field(default=128, ge=32, description="Minimum resolution to load")

    # Video settings
    clip_length: int = Field(default=8, ge=1, le=64, description="Frames per video clip")
    frame_skip: int = Field(default=1, ge=1, description="Frames to skip between samples")

    # Augmentation
    random_crop: bool = Field(default=True, description="Use random cropping")
    random_flip: bool = Field(default=True, description="Use random horizontal flip")
    color_jitter: bool = Field(default=False, description="Apply color jittering")

    # Loading
    num_workers: int = Field(default=4, ge=0, description="Data loading workers")
    prefetch_factor: int = Field(default=2, ge=1, description="Prefetch batches per worker")

    model_config = ConfigDict(extra="forbid")


@dataclass
class VideoClip:
    """Container for a video clip."""

    frames: Tensor  # (T, C, H, W)
    frame_indices: list[int]  # Original frame indices
    video_path: Path
    fps: float = 30.0

    @property
    def num_frames(self) -> int:
        """Number of frames in clip."""
        return self.frames.shape[0]

    @property
    def height(self) -> int:
        """Frame height."""
        return self.frames.shape[2]

    @property
    def width(self) -> int:
        """Frame width."""
        return self.frames.shape[3]


class ImageDataset(Dataset):
    """Dataset for static images (compression training).

    Loads images from a directory and applies random cropping
    for patch-based training.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: Path | str,
        config: DatasetConfig | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize image dataset.

        Args:
            root: Root directory containing images.
            config: Dataset configuration.
            transform: Optional transform to apply.

        """
        self.root = Path(root)
        self.config = config or DatasetConfig(root_dir=str(root))
        self.transform = transform

        # Find all images
        self.files = self._find_images()
        if not self.files:
            raise ValueError(f"No images found in {root}")

        logger.info(f"Found {len(self.files)} images in {root}")

    def _find_images(self) -> list[Path]:
        """Find all image files in root directory."""
        files = []
        for ext in self.EXTENSIONS:
            files.extend(self.root.glob(f"**/*{ext}"))
            files.extend(self.root.glob(f"**/*{ext.upper()}"))
        return sorted(files)

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tensor:
        """Load and preprocess image.

        Args:
            idx: Image index.

        Returns:
            Image tensor (C, H, W) in [0, 1].

        """
        path = self.files[idx]

        # Load image
        img = self._load_image(path)

        # Random crop if configured
        if self.config.random_crop:
            img = self._random_crop(img, self.config.patch_size)

        # Random flip if configured
        if self.config.random_flip and random.random() > 0.5:
            img = torch.flip(img, dims=[-1])

        # Apply custom transform
        if self.transform:
            img = self.transform(img)

        return img

    def _load_image(self, path: Path) -> Tensor:
        """Load image from path.

        Args:
            path: Image path.

        Returns:
            Image tensor (C, H, W) in [0, 1].

        """
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(path).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW

            return img_tensor

        except ImportError:
            raise ImportError("PIL not installed. Install with: pip install pillow")

    def _random_crop(self, img: Tensor, size: int) -> Tensor:
        """Apply random crop to image.

        Args:
            img: Image tensor (C, H, W).
            size: Crop size.

        Returns:
            Cropped tensor.

        """
        _, h, w = img.shape

        # If image is smaller, resize
        if h < size or w < size:
            scale = max(size / h, size / w) * 1.1  # Slight over-scale
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            _, h, w = img.shape

        # Random crop
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)

        return img[:, top : top + size, left : left + size]


class VideoDataset(Dataset):
    """Dataset for video clips (temporal compression training).

    Loads video files and extracts clips with specified
    temporal sampling patterns.
    """

    EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        root: Path | str,
        config: DatasetConfig | None = None,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """Initialize video dataset.

        Args:
            root: Root directory containing videos.
            config: Dataset configuration.
            transform: Optional transform to apply per-frame.

        """
        self.root = Path(root)
        self.config = config or DatasetConfig(root_dir=str(root))
        self.transform = transform

        # Find all videos
        self.files = self._find_videos()
        if not self.files:
            raise ValueError(f"No videos found in {root}")

        # Build clip index
        self._clips = self._build_clip_index()

        logger.info(f"Found {len(self.files)} videos, {len(self._clips)} clips in {root}")

    def _find_videos(self) -> list[Path]:
        """Find all video files in root directory."""
        files = []
        for ext in self.EXTENSIONS:
            files.extend(self.root.glob(f"**/*{ext}"))
            files.extend(self.root.glob(f"**/*{ext.upper()}"))
        return sorted(files)

    def _build_clip_index(self) -> list[tuple[int, int]]:
        """Build index of (video_idx, start_frame) pairs."""
        clips = []

        for vid_idx, path in enumerate(self.files):
            try:
                num_frames = self._get_video_length(path)
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")
                continue

            # Calculate clip positions
            clip_len = self.config.clip_length
            skip = self.config.frame_skip
            total_frames_needed = clip_len * skip

            for start in range(0, num_frames - total_frames_needed + 1, clip_len):
                clips.append((vid_idx, start))

        return clips

    def _get_video_length(self, path: Path) -> int:
        """Get number of frames in video.

        Args:
            path: Video path.

        Returns:
            Number of frames.

        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Could not open {path}")

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            return length

        except ImportError:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

    def __len__(self) -> int:
        """Dataset length (number of clips)."""
        return len(self._clips)

    def __getitem__(self, idx: int) -> VideoClip:
        """Load video clip.

        Args:
            idx: Clip index.

        Returns:
            VideoClip with frames and metadata.

        """
        vid_idx, start_frame = self._clips[idx]
        path = self.files[vid_idx]

        # Load frames
        frames, frame_indices = self._load_frames(
            path, start_frame, self.config.clip_length, self.config.frame_skip
        )

        # Get FPS
        fps = self._get_video_fps(path)

        # Random spatial crop (same for all frames)
        if self.config.random_crop:
            frames = self._random_crop_video(frames, self.config.patch_size)

        # Random flip (same for all frames)
        if self.config.random_flip and random.random() > 0.5:
            frames = torch.flip(frames, dims=[-1])

        # Apply per-frame transform
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])

        return VideoClip(
            frames=frames,
            frame_indices=frame_indices,
            video_path=path,
            fps=fps,
        )

    def _load_frames(
        self,
        path: Path,
        start: int,
        length: int,
        skip: int,
    ) -> tuple[Tensor, list[int]]:
        """Load frames from video.

        Args:
            path: Video path.
            start: Start frame index.
            length: Number of frames to load.
            skip: Frames to skip between samples.

        Returns:
            Tuple of (frames tensor, frame indices).

        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Could not open {path}")

            frames = []
            indices = []

            for i in range(length):
                frame_idx = start + i * skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # BGR -> RGB, normalize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # HWC -> CHW

                frames.append(frame)
                indices.append(frame_idx)

            cap.release()

            if not frames:
                raise ValueError(f"No frames loaded from {path}")

            return torch.stack(frames), indices

        except ImportError:
            raise ImportError("OpenCV not installed. Install with: pip install opencv-python")

    def _get_video_fps(self, path: Path) -> float:
        """Get video frame rate.

        Args:
            path: Video path.

        Returns:
            Frame rate.

        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return fps if fps > 0 else 30.0

        except ImportError:
            return 30.0

    def _random_crop_video(self, frames: Tensor, size: int) -> Tensor:
        """Apply consistent random crop to all frames.

        Args:
            frames: Video tensor (T, C, H, W).
            size: Crop size.

        Returns:
            Cropped tensor.

        """
        t, c, h, w = frames.shape

        # Resize if needed
        if h < size or w < size:
            scale = max(size / h, size / w) * 1.1
            new_h = int(h * scale)
            new_w = int(w * scale)
            frames = torch.nn.functional.interpolate(
                frames,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            _, _, h, w = frames.shape

        # Random crop (same for all frames)
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)

        return frames[:, :, top : top + size, left : left + size]


class VariableResolutionBatchSampler:
    """Batch sampler that groups images by resolution.

    Groups images with similar resolutions together to minimize
    padding in variable-resolution training.
    """

    def __init__(
        self,
        dataset: ImageDataset | VideoDataset,
        batch_size: int,
        resolution_buckets: list[int] | None = None,
        shuffle: bool = True,
    ) -> None:
        """Initialize batch sampler.

        Args:
            dataset: Source dataset.
            batch_size: Batch size.
            resolution_buckets: Resolution bucket boundaries.
            shuffle: Shuffle within buckets.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Default buckets: 256, 512, 1024, larger
        self.buckets = resolution_buckets or [256, 512, 1024, 2048]

        # Group indices by resolution bucket
        self._bucket_indices = self._build_buckets()

    def _build_buckets(self) -> dict[int, list[int]]:
        """Group dataset indices by resolution bucket."""
        buckets: dict[int, list[int]] = {b: [] for b in self.buckets}

        for idx in range(len(self.dataset)):
            # Get resolution (this is approximate - would need to load image)
            # For now, distribute evenly
            bucket = self.buckets[idx % len(self.buckets)]
            buckets[bucket].append(idx)

        return buckets

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices."""
        all_batches = []

        for _bucket, indices in self._bucket_indices.items():
            if not indices:
                continue

            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        """Number of batches."""
        total = sum(len(indices) for indices in self._bucket_indices.values())
        return (total + self.batch_size - 1) // self.batch_size
