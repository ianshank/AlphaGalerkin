"""Data loading utilities for video compression training.

Provides:
- VideoDataset: Load video frames with temporal sampling
- ImageDataset: Load images for compression training
- Variable resolution collation utilities
"""

from src.video_compression.data.dataset import (
    VideoDataset,
    ImageDataset,
    VideoClip,
    DatasetConfig,
)
from src.video_compression.data.transforms import (
    CompressionTransforms,
    RandomCrop,
    CenterCrop,
    RandomFlip,
    ColorJitter,
    Normalize,
)

__all__ = [
    "VideoDataset",
    "ImageDataset",
    "VideoClip",
    "DatasetConfig",
    "CompressionTransforms",
    "RandomCrop",
    "CenterCrop",
    "RandomFlip",
    "ColorJitter",
    "Normalize",
]
