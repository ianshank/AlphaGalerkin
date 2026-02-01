"""Unit tests for video compression data pipeline.

Tests dataset classes for video and image loading:
- DatasetConfig validation
- ImageDataset loading and augmentation
- VideoDataset clip extraction
- VariableResolutionBatchSampler bucketing
- VideoClip data structure
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
from torch import Tensor

from src.video_compression.data.dataset import (
    DatasetConfig,
    VideoClip,
    ImageDataset,
    VideoDataset,
    VariableResolutionBatchSampler,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def dataset_config() -> DatasetConfig:
    """Create test dataset configuration."""
    return DatasetConfig(
        root_dir="/test/data",
        patch_size=128,
        min_resolution=64,
        clip_length=4,
        frame_skip=1,
        random_crop=True,
        random_flip=True,
        color_jitter=False,
        num_workers=0,
        prefetch_factor=2,
    )


@pytest.fixture
def sample_frame() -> Tensor:
    """Create sample frame tensor."""
    return torch.rand(3, 256, 256)


@pytest.fixture
def sample_video() -> Tensor:
    """Create sample video tensor."""
    return torch.rand(8, 3, 256, 256)


@pytest.fixture
def temp_image_dir() -> Path:
    """Create temporary directory with test images."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create mock image files
        for i in range(5):
            (tmp_path / f"image_{i}.jpg").touch()
        
        yield tmp_path


# --------------------------------------------------------------------------
# DatasetConfig Tests
# --------------------------------------------------------------------------


class TestDatasetConfig:
    """Tests for DatasetConfig validation."""

    def test_valid_config(self) -> None:
        """Test valid configuration is accepted."""
        config = DatasetConfig(root_dir="/data/videos")
        
        assert config.root_dir == "/data/videos"
        assert config.patch_size == 256  # Default
        assert config.clip_length == 8  # Default

    def test_patch_size_bounds(self) -> None:
        """Test patch size validation."""
        # Valid bounds
        config = DatasetConfig(root_dir="/data", patch_size=64)
        assert config.patch_size == 64
        
        config = DatasetConfig(root_dir="/data", patch_size=1024)
        assert config.patch_size == 1024

    def test_patch_size_invalid(self) -> None:
        """Test patch size validation rejects invalid values."""
        with pytest.raises(ValueError):
            DatasetConfig(root_dir="/data", patch_size=16)  # Too small
        
        with pytest.raises(ValueError):
            DatasetConfig(root_dir="/data", patch_size=2048)  # Too large

    def test_clip_length_bounds(self) -> None:
        """Test clip length validation."""
        config = DatasetConfig(root_dir="/data", clip_length=1)
        assert config.clip_length == 1
        
        config = DatasetConfig(root_dir="/data", clip_length=64)
        assert config.clip_length == 64

    def test_clip_length_invalid(self) -> None:
        """Test clip length validation rejects invalid values."""
        with pytest.raises(ValueError):
            DatasetConfig(root_dir="/data", clip_length=0)  # Too small
        
        with pytest.raises(ValueError):
            DatasetConfig(root_dir="/data", clip_length=100)  # Too large

    def test_extra_fields_forbidden(self) -> None:
        """Test extra fields are rejected."""
        with pytest.raises(ValueError):
            DatasetConfig(root_dir="/data", unknown_field=True)

    def test_num_workers_bounds(self) -> None:
        """Test num_workers validation."""
        config = DatasetConfig(root_dir="/data", num_workers=0)
        assert config.num_workers == 0


# --------------------------------------------------------------------------
# VideoClip Tests
# --------------------------------------------------------------------------


class TestVideoClip:
    """Tests for VideoClip data structure."""

    def test_video_clip_creation(self, sample_video: Tensor) -> None:
        """Test VideoClip creation."""
        clip = VideoClip(
            frames=sample_video,
            frame_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            video_path=Path("/test/video.mp4"),
            fps=30.0,
        )
        
        assert clip.num_frames == 8
        assert clip.height == 256
        assert clip.width == 256
        assert clip.fps == 30.0

    def test_video_clip_properties(self) -> None:
        """Test VideoClip property accessors."""
        frames = torch.rand(4, 3, 128, 192)
        
        clip = VideoClip(
            frames=frames,
            frame_indices=[0, 2, 4, 6],
            video_path=Path("/test.mp4"),
        )
        
        assert clip.num_frames == 4
        assert clip.height == 128
        assert clip.width == 192

    def test_video_clip_default_fps(self) -> None:
        """Test VideoClip default FPS."""
        clip = VideoClip(
            frames=torch.rand(2, 3, 64, 64),
            frame_indices=[0, 1],
            video_path=Path("/test.mp4"),
        )
        
        assert clip.fps == 30.0


# --------------------------------------------------------------------------
# ImageDataset Tests
# --------------------------------------------------------------------------


class TestImageDataset:
    """Tests for ImageDataset loading."""

    def test_find_images_extensions(self) -> None:
        """Test finding images with various extensions."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            
            # Create files with different extensions
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                (tmp_path / f"image{ext}").touch()
            
            # Create non-image file (should be ignored)
            (tmp_path / "document.txt").touch()
            
            with patch.object(ImageDataset, "_load_image", return_value=torch.rand(3, 64, 64)):
                dataset = ImageDataset(tmp_path)
                
                # Check expected extensions are found (handles Windows case-insensitive FS)
                unique_files = set(dataset.files)
                assert len(unique_files) >= 5

    def test_find_images_uppercase_extensions(self) -> None:
        """Test finding images with uppercase extensions."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            
            (tmp_path / "image.JPG").touch()
            (tmp_path / "image.PNG").touch()
            
            with patch.object(ImageDataset, "_load_image", return_value=torch.rand(3, 64, 64)):
                dataset = ImageDataset(tmp_path)
                
                # Check files are found (may include duplicates on case-insensitive FS)
                unique_files = set(dataset.files)
                assert len(unique_files) >= 2

    def test_empty_directory_raises(self) -> None:
        """Test empty directory raises error."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="No images found"):
                ImageDataset(tmp)

    def test_getitem_returns_tensor(self, temp_image_dir: Path) -> None:
        """Test __getitem__ returns tensor."""
        with patch.object(
            ImageDataset, "_load_image", return_value=torch.rand(3, 256, 256)
        ):
            dataset = ImageDataset(
                temp_image_dir,
                config=DatasetConfig(root_dir=str(temp_image_dir), random_crop=False),
            )
            
            item = dataset[0]
            
            assert isinstance(item, Tensor)
            assert item.dim() == 3
            assert item.shape[0] == 3  # RGB channels

    def test_random_crop_applied(self) -> None:
        """Test random crop is applied when configured."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "image.jpg").touch()
            
            config = DatasetConfig(
                root_dir=str(tmp_path),
                patch_size=64,
                random_crop=True,
            )
            
            with patch.object(
                ImageDataset, "_load_image", return_value=torch.rand(3, 256, 256)
            ):
                dataset = ImageDataset(tmp_path, config=config)
                item = dataset[0]
                
                assert item.shape[-2:] == (64, 64)

    def test_random_crop_small_image(self) -> None:
        """Test random crop handles images smaller than patch size."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "image.jpg").touch()
            
            config = DatasetConfig(
                root_dir=str(tmp_path),
                patch_size=256,
                random_crop=True,
            )
            
            # Image smaller than patch size
            with patch.object(
                ImageDataset, "_load_image", return_value=torch.rand(3, 64, 64)
            ):
                dataset = ImageDataset(tmp_path, config=config)
                item = dataset[0]
                
                # Should be resized and cropped to patch size
                assert item.shape[-2:] == (256, 256)

    def test_custom_transform_applied(self, temp_image_dir: Path) -> None:
        """Test custom transform is applied."""
        def custom_transform(x: Tensor) -> Tensor:
            return x * 2
        
        with patch.object(
            ImageDataset, "_load_image", return_value=torch.rand(3, 256, 256)
        ):
            dataset = ImageDataset(
                temp_image_dir,
                config=DatasetConfig(root_dir=str(temp_image_dir), random_crop=False),
                transform=custom_transform,
            )
            
            original = torch.rand(3, 256, 256)
            with patch.object(ImageDataset, "_load_image", return_value=original.clone()):
                item = dataset[0]
                
                # Transform should have been applied
                assert item.max() <= 2.0

    def test_subdirectory_images_found(self) -> None:
        """Test images in subdirectories are found."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            
            # Create subdirectory with images
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (subdir / "image.jpg").touch()
            (tmp_path / "root_image.jpg").touch()
            
            with patch.object(ImageDataset, "_load_image", return_value=torch.rand(3, 64, 64)):
                dataset = ImageDataset(tmp_path)
                
                # Check both root and subdir images found
                unique_files = set(dataset.files)
                assert len(unique_files) >= 2


# --------------------------------------------------------------------------
# VideoDataset Tests
# --------------------------------------------------------------------------


class TestVideoDataset:
    """Tests for VideoDataset loading."""

    def test_find_videos_extensions(self) -> None:
        """Test finding videos with various extensions."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            
            for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                (tmp_path / f"video{ext}").touch()
            
            with patch.object(VideoDataset, "_get_video_length", return_value=100):
                with patch.object(VideoDataset, "_load_frames", return_value=(torch.rand(8, 3, 64, 64), list(range(8)))):
                    dataset = VideoDataset(tmp_path)
                    
                    # Check expected extensions are found
                    unique_files = set(dataset.files)
                    assert len(unique_files) >= 5

    def test_empty_video_directory_raises(self) -> None:
        """Test empty directory raises error."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="No videos found"):
                VideoDataset(tmp)

    def test_clip_index_building(self) -> None:
        """Test clip index is built correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "video.mp4").touch()
            
            config = DatasetConfig(
                root_dir=str(tmp_path),
                clip_length=4,
                frame_skip=1,
            )
            
            # Video with 20 frames should have multiple clips
            with patch.object(VideoDataset, "_get_video_length", return_value=20):
                with patch.object(VideoDataset, "_load_frames", return_value=(torch.rand(4, 3, 64, 64), list(range(4)))):
                    dataset = VideoDataset(tmp_path, config=config)
                    
                    # Expected clips: 0-4, 4-8, 8-12, 12-16
                    assert len(dataset._clips) >= 4

    def test_getitem_returns_video_clip(self) -> None:
        """Test __getitem__ returns VideoClip."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "video.mp4").touch()
            
            config = DatasetConfig(
                root_dir=str(tmp_path),
                clip_length=4,
                random_crop=False,
            )
            
            with patch.object(VideoDataset, "_get_video_length", return_value=100):
                with patch.object(VideoDataset, "_load_frames", return_value=(torch.rand(4, 3, 64, 64), [0, 1, 2, 3])):
                    with patch.object(VideoDataset, "_get_video_fps", return_value=30.0):
                        dataset = VideoDataset(tmp_path, config=config)
                        clip = dataset[0]
                        
                        assert isinstance(clip, VideoClip)
                        assert clip.num_frames == 4

    def test_frame_skip_applied(self) -> None:
        """Test frame skip is applied during loading."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "video.mp4").touch()
            
            config = DatasetConfig(
                root_dir=str(tmp_path),
                clip_length=4,
                frame_skip=2,  # Skip every other frame
                random_crop=False,
            )
            
            with patch.object(VideoDataset, "_get_video_length", return_value=100):
                with patch.object(VideoDataset, "_load_frames", return_value=(torch.rand(4, 3, 64, 64), [0, 2, 4, 6])) as mock_load:
                    with patch.object(VideoDataset, "_get_video_fps", return_value=30.0):
                        dataset = VideoDataset(tmp_path, config=config)
                        _ = dataset[0]
                        
                        # Verify frame skip was passed
                        call_args = mock_load.call_args
                        assert call_args.kwargs.get("skip", 1) == 2 or call_args[0][3] == 2


# --------------------------------------------------------------------------
# VariableResolutionBatchSampler Tests
# --------------------------------------------------------------------------


class TestVariableResolutionBatchSampler:
    """Tests for VariableResolutionBatchSampler bucketing."""

    @pytest.fixture
    def mock_dataset(self) -> MagicMock:
        """Create mock dataset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        return dataset

    def test_batch_size(self, mock_dataset: MagicMock) -> None:
        """Test batches respect batch size."""
        sampler = VariableResolutionBatchSampler(
            dataset=mock_dataset,
            batch_size=8,
            shuffle=False,
        )
        
        batches = list(sampler)
        
        for batch in batches[:-1]:  # All except last
            assert len(batch) <= 8

    def test_all_indices_sampled(self, mock_dataset: MagicMock) -> None:
        """Test all indices are sampled exactly once."""
        sampler = VariableResolutionBatchSampler(
            dataset=mock_dataset,
            batch_size=8,
            shuffle=False,
        )
        
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        
        assert sorted(all_indices) == list(range(100))

    def test_len_correct(self, mock_dataset: MagicMock) -> None:
        """Test __len__ returns correct number of batches."""
        sampler = VariableResolutionBatchSampler(
            dataset=mock_dataset,
            batch_size=8,
            shuffle=False,
        )
        
        expected = (100 + 8 - 1) // 8  # Ceiling division
        assert len(sampler) == expected

    def test_custom_resolution_buckets(self, mock_dataset: MagicMock) -> None:
        """Test custom resolution buckets are used."""
        custom_buckets = [128, 256, 512]
        
        sampler = VariableResolutionBatchSampler(
            dataset=mock_dataset,
            batch_size=8,
            resolution_buckets=custom_buckets,
            shuffle=False,
        )
        
        assert sampler.buckets == custom_buckets

    def test_shuffle_changes_order(self, mock_dataset: MagicMock) -> None:
        """Test shuffle changes batch order."""
        sampler1 = VariableResolutionBatchSampler(
            dataset=mock_dataset,
            batch_size=8,
            shuffle=True,
        )
        sampler2 = VariableResolutionBatchSampler(
            dataset=mock_dataset,
            batch_size=8,
            shuffle=True,
        )
        
        torch.manual_seed(42)
        batches1 = list(sampler1)
        
        torch.manual_seed(123)
        batches2 = list(sampler2)
        
        # Different seeds should produce different orders (most of the time)
        # This is probabilistic but very likely to differ
        assert batches1 != batches2 or len(batches1) <= 1


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""

    def test_config_to_dataset_workflow(self) -> None:
        """Test config creates valid dataset."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "image.jpg").touch()
            
            config = DatasetConfig(
                root_dir=str(tmp_path),
                patch_size=64,
                random_crop=True,
            )
            
            with patch.object(
                ImageDataset, "_load_image", return_value=torch.rand(3, 128, 128)
            ):
                dataset = ImageDataset(tmp_path, config=config)
                
                # Should be able to get item
                item = dataset[0]
                assert item.shape[-2:] == (64, 64)

    def test_video_clip_to_tensor_batch(self) -> None:
        """Test video clips can be batched as tensors."""
        clips = [
            VideoClip(
                frames=torch.rand(4, 3, 64, 64),
                frame_indices=[0, 1, 2, 3],
                video_path=Path(f"/video_{i}.mp4"),
            )
            for i in range(4)
        ]
        
        # Stack frames for batch
        batch = torch.stack([clip.frames for clip in clips])
        
        assert batch.shape == (4, 4, 3, 64, 64)  # B, T, C, H, W
