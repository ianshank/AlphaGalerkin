"""Data loading utilities for video compression training.

Provides:
- VideoDataset: Load video frames with temporal sampling
- ImageDataset: Load images for compression training
- Variable resolution collation utilities
- SyntheticVideoGenerator: Pure-torch synthetic video for demos/testing
"""

from src.video_compression.data.dataset import (
    DatasetConfig,
    ImageDataset,
    VideoClip,
    VideoDataset,
)
from src.video_compression.data.synthetic import (
    SyntheticPattern,
    SyntheticVideoConfig,
    SyntheticVideoGenerator,
    create_test_sequence,
)
from src.video_compression.data.transforms import (
    CenterCrop,
    ColorJitter,
    CompressionTransforms,
    Normalize,
    RandomCrop,
    RandomFlip,
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
    "SyntheticPattern",
    "SyntheticVideoConfig",
    "SyntheticVideoGenerator",
    "create_test_sequence",
]
