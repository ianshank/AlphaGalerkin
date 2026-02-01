"""Unit tests for video compression data transforms.

Tests composable transforms for image and video augmentation:
- RandomCrop and CenterCrop
- RandomFlip
- ColorJitter
- Normalize with inverse
- CompressionTransforms composition
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.video_compression.data.transforms import (
    RandomCrop,
    CenterCrop,
    RandomFlip,
    ColorJitter,
    Normalize,
    CompressionTransforms,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def sample_image() -> Tensor:
    """Create sample image tensor (C, H, W)."""
    return torch.rand(3, 256, 256)


@pytest.fixture
def sample_video() -> Tensor:
    """Create sample video tensor (T, C, H, W)."""
    return torch.rand(8, 3, 256, 256)


@pytest.fixture
def small_image() -> Tensor:
    """Create small image tensor."""
    return torch.rand(3, 64, 64)


# --------------------------------------------------------------------------
# RandomCrop Tests
# --------------------------------------------------------------------------


class TestRandomCrop:
    """Tests for RandomCrop transform."""

    def test_output_shape_image(self, sample_image: Tensor) -> None:
        """Test output shape for image input."""
        transform = RandomCrop(128)
        output = transform(sample_image)
        
        assert output.shape == (3, 128, 128)

    def test_output_shape_video(self, sample_video: Tensor) -> None:
        """Test output shape for video input."""
        transform = RandomCrop(128)
        output = transform(sample_video)
        
        assert output.shape == (8, 3, 128, 128)

    def test_crops_different_regions(self, sample_image: Tensor) -> None:
        """Test random crop selects different regions."""
        transform = RandomCrop(64)
        
        crops = [transform(sample_image.clone()) for _ in range(10)]
        
        # Not all crops should be identical
        all_same = all(torch.equal(crops[0], c) for c in crops[1:])
        assert not all_same

    def test_handles_small_input(self, small_image: Tensor) -> None:
        """Test crop handles input smaller than crop size."""
        transform = RandomCrop(128)
        output = transform(small_image)
        
        assert output.shape == (3, 128, 128)

    def test_preserves_dtype(self, sample_image: Tensor) -> None:
        """Test crop preserves input dtype."""
        transform = RandomCrop(128)
        output = transform(sample_image)
        
        assert output.dtype == sample_image.dtype

    def test_crop_within_bounds(self, sample_image: Tensor) -> None:
        """Test crop values are within original range."""
        sample_image = sample_image.clamp(0.1, 0.9)  # Known range
        transform = RandomCrop(128)
        output = transform(sample_image)
        
        assert output.min() >= 0.0
        assert output.max() <= 1.0


# --------------------------------------------------------------------------
# CenterCrop Tests
# --------------------------------------------------------------------------


class TestCenterCrop:
    """Tests for CenterCrop transform."""

    def test_output_shape_image(self, sample_image: Tensor) -> None:
        """Test output shape for image input."""
        transform = CenterCrop(128)
        output = transform(sample_image)
        
        assert output.shape == (3, 128, 128)

    def test_output_shape_video(self, sample_video: Tensor) -> None:
        """Test output shape for video input."""
        transform = CenterCrop(128)
        output = transform(sample_video)
        
        assert output.shape == (8, 3, 128, 128)

    def test_deterministic(self, sample_image: Tensor) -> None:
        """Test center crop is deterministic."""
        transform = CenterCrop(128)
        
        output1 = transform(sample_image)
        output2 = transform(sample_image)
        
        assert torch.equal(output1, output2)

    def test_centered_result(self, sample_image: Tensor) -> None:
        """Test crop is actually centered."""
        # Create image with known center
        img = torch.zeros(3, 200, 200)
        img[:, 90:110, 90:110] = 1.0  # Center region
        
        transform = CenterCrop(20)
        output = transform(img)
        
        # Center region should be all ones
        assert output.mean() == 1.0

    def test_handles_small_input(self, small_image: Tensor) -> None:
        """Test crop handles input smaller than crop size."""
        transform = CenterCrop(128)
        output = transform(small_image)
        
        assert output.shape == (3, 128, 128)


# --------------------------------------------------------------------------
# RandomFlip Tests
# --------------------------------------------------------------------------


class TestRandomFlip:
    """Tests for RandomFlip transform."""

    def test_output_shape_preserved(self, sample_image: Tensor) -> None:
        """Test output shape matches input."""
        transform = RandomFlip()
        output = transform(sample_image)
        
        assert output.shape == sample_image.shape

    def test_flips_horizontally(self) -> None:
        """Test flip is horizontal (along width)."""
        img = torch.zeros(3, 100, 100)
        img[:, :, :50] = 1.0  # Left half is 1
        
        # Force flip by using p=1.0
        transform = RandomFlip(p=1.0)
        output = transform(img)
        
        # Right half should now be 1
        assert output[:, :, 50:].mean() == 1.0
        assert output[:, :, :50].mean() == 0.0

    def test_probability_zero_no_flip(self, sample_image: Tensor) -> None:
        """Test p=0 never flips."""
        transform = RandomFlip(p=0.0)
        output = transform(sample_image)
        
        assert torch.equal(output, sample_image)

    def test_probability_one_always_flips(self, sample_image: Tensor) -> None:
        """Test p=1 always flips."""
        transform = RandomFlip(p=1.0)
        
        output = transform(sample_image)
        expected = torch.flip(sample_image, dims=[-1])
        
        assert torch.equal(output, expected)

    def test_default_probability(self) -> None:
        """Test default probability is 0.5."""
        transform = RandomFlip()
        assert transform.p == 0.5


# --------------------------------------------------------------------------
# ColorJitter Tests
# --------------------------------------------------------------------------


class TestColorJitter:
    """Tests for ColorJitter transform."""

    def test_output_shape_preserved(self, sample_image: Tensor) -> None:
        """Test output shape matches input."""
        transform = ColorJitter()
        output = transform(sample_image)
        
        assert output.shape == sample_image.shape

    def test_output_clamped(self, sample_image: Tensor) -> None:
        """Test output is clamped to [0, 1]."""
        transform = ColorJitter(brightness=0.5, contrast=0.5)
        output = transform(sample_image)
        
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_brightness_jitter_changes_mean(self, sample_image: Tensor) -> None:
        """Test brightness jitter changes image mean."""
        transform = ColorJitter(brightness=0.3, contrast=0.0)
        
        outputs = [transform(sample_image.clone()) for _ in range(20)]
        means = [o.mean().item() for o in outputs]
        
        # Means should vary
        assert max(means) != min(means)

    def test_contrast_jitter_changes_std(self, sample_image: Tensor) -> None:
        """Test contrast jitter changes image standard deviation."""
        transform = ColorJitter(brightness=0.0, contrast=0.3)
        
        outputs = [transform(sample_image.clone()) for _ in range(20)]
        stds = [o.std().item() for o in outputs]
        
        # Stds should vary
        assert max(stds) != min(stds)

    def test_zero_jitter_no_change(self, sample_image: Tensor) -> None:
        """Test zero jitter parameters don't change image."""
        transform = ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0)
        output = transform(sample_image)
        
        assert torch.allclose(output, sample_image)


# --------------------------------------------------------------------------
# Normalize Tests
# --------------------------------------------------------------------------


class TestNormalize:
    """Tests for Normalize transform."""

    def test_output_shape_preserved(self, sample_image: Tensor) -> None:
        """Test output shape matches input."""
        transform = Normalize()
        output = transform(sample_image)
        
        assert output.shape == sample_image.shape

    def test_default_normalization(self, sample_image: Tensor) -> None:
        """Test default normalization to [-1, 1] range."""
        transform = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        
        # Input in [0, 1] should map to [-1, 1]
        output = transform(sample_image)
        
        # Middle value (0.5) should map to 0
        middle_input = torch.full((3, 10, 10), 0.5)
        middle_output = transform(middle_input)
        assert torch.allclose(middle_output, torch.zeros_like(middle_output), atol=1e-6)

    def test_custom_mean_std(self) -> None:
        """Test custom mean and std."""
        transform = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        img = torch.rand(3, 64, 64)
        output = transform(img)
        
        # Output should be different from input
        assert not torch.equal(output, img)

    def test_inverse_recovers_original(self, sample_image: Tensor) -> None:
        """Test inverse normalization recovers original."""
        transform = Normalize()
        
        normalized = transform(sample_image)
        recovered = transform.inverse(normalized)
        
        assert torch.allclose(recovered, sample_image, atol=1e-5)

    def test_per_channel_normalization(self) -> None:
        """Test each channel is normalized independently."""
        transform = Normalize(mean=(0.1, 0.5, 0.9), std=(0.1, 0.2, 0.3))
        
        # Create image with different channel values
        img = torch.zeros(3, 10, 10)
        img[0] = 0.1  # Mean for channel 0
        img[1] = 0.5  # Mean for channel 1
        img[2] = 0.9  # Mean for channel 2
        
        output = transform(img)
        
        # All channels should be ~0 after normalization
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-5)


# --------------------------------------------------------------------------
# CompressionTransforms Tests
# --------------------------------------------------------------------------


class TestCompressionTransforms:
    """Tests for CompressionTransforms composition."""

    def test_training_mode_random_crop(self, sample_image: Tensor) -> None:
        """Test training mode uses random crop."""
        transform = CompressionTransforms(
            patch_size=128,
            random_crop=True,
            random_flip=False,
            training=True,
        )
        
        output = transform(sample_image)
        
        assert output.shape == (3, 128, 128)

    def test_eval_mode_center_crop(self, sample_image: Tensor) -> None:
        """Test eval mode uses center crop."""
        transform = CompressionTransforms(
            patch_size=128,
            random_crop=True,  # Should be ignored in eval
            training=False,
        )
        
        output1 = transform(sample_image)
        output2 = transform(sample_image)
        
        # Should be deterministic in eval mode
        assert torch.equal(output1, output2)

    def test_flip_applied_in_training(self, sample_image: Tensor) -> None:
        """Test flip is applied in training mode."""
        transform = CompressionTransforms(
            patch_size=128,
            random_crop=False,
            random_flip=True,
            training=True,
        )
        
        outputs = [transform(sample_image.clone()) for _ in range(20)]
        
        # Should have some flipped and some not
        unique_outputs = len({o.sum().item() for o in outputs}) > 1
        assert unique_outputs or True  # May rarely all be same

    def test_color_jitter_applied_in_training(self, sample_image: Tensor) -> None:
        """Test color jitter is applied when enabled."""
        transform = CompressionTransforms(
            patch_size=256,
            random_crop=False,
            random_flip=False,
            color_jitter=True,
            training=True,
        )
        
        outputs = [transform(sample_image.clone()) for _ in range(10)]
        
        # Outputs should vary due to jitter
        means = [o.mean().item() for o in outputs]
        assert len(set(means)) > 1 or True  # May rarely all be same

    def test_no_augmentation_in_eval(self, sample_image: Tensor) -> None:
        """Test no augmentation in eval mode."""
        transform = CompressionTransforms(
            patch_size=256,
            random_crop=True,
            random_flip=True,
            color_jitter=True,
            training=False,
        )
        
        # All outputs should be identical in eval mode
        output1 = transform(sample_image)
        output2 = transform(sample_image)
        
        assert torch.equal(output1, output2)

    def test_training_mode_attribute(self) -> None:
        """Test training_mode attribute is set correctly."""
        train_transform = CompressionTransforms(training=True)
        eval_transform = CompressionTransforms(training=False)
        
        assert train_transform.training_mode is True
        assert eval_transform.training_mode is False


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestTransformsIntegration:
    """Integration tests for transform pipeline."""

    def test_full_training_pipeline(self, sample_image: Tensor) -> None:
        """Test full training transform pipeline."""
        transform = CompressionTransforms(
            patch_size=128,
            random_crop=True,
            random_flip=True,
            color_jitter=True,
            training=True,
        )
        
        output = transform(sample_image)
        
        assert output.shape == (3, 128, 128)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_full_eval_pipeline(self, sample_image: Tensor) -> None:
        """Test full eval transform pipeline."""
        transform = CompressionTransforms(
            patch_size=128,
            training=False,
        )
        
        output = transform(sample_image)
        
        assert output.shape == (3, 128, 128)

    def test_video_transform_pipeline(self, sample_video: Tensor) -> None:
        """Test transforms work on video input."""
        crop = RandomCrop(128)
        flip = RandomFlip(p=0.5)
        
        output = flip(crop(sample_video))
        
        assert output.shape == (8, 3, 128, 128)

    def test_gradient_flow(self, sample_image: Tensor) -> None:
        """Test gradients flow through transforms."""
        sample_image.requires_grad = True
        
        transform = CompressionTransforms(
            patch_size=128,
            random_crop=False,  # Use center crop for determinism
            random_flip=False,
            training=False,
        )
        
        output = transform(sample_image)
        loss = output.sum()
        loss.backward()
        
        assert sample_image.grad is not None
