"""Unit tests for padding utilities."""

from __future__ import annotations

import pytest
import torch

from src.video_compression.utils.padding import (
    DynamicPadding,
    PaddingConfig,
    PaddingInfo,
    PaddingMode,
    PadToMultiple,
    compute_padding,
    crop_to_original,
    pad_to_multiple,
)


class TestPaddingInfo:
    """Tests for PaddingInfo dataclass."""

    def test_create_info(self) -> None:
        """Test creating padding info."""
        info = PaddingInfo(
            original_height=1080,
            original_width=1920,
            padded_height=1088,
            padded_width=1920,
            pad_top=4,
            pad_bottom=4,
            pad_left=0,
            pad_right=0,
        )

        assert info.original_height == 1080
        assert info.pad_top + info.pad_bottom == 8

    def test_to_from_dict(self) -> None:
        """Test dictionary serialization."""
        original = PaddingInfo(
            original_height=720,
            original_width=1280,
            padded_height=736,
            padded_width=1280,
            pad_top=8,
            pad_bottom=8,
            pad_left=0,
            pad_right=0,
        )

        d = original.to_dict()
        recovered = PaddingInfo.from_dict(d)

        assert recovered.original_height == original.original_height
        assert recovered.pad_top == original.pad_top


class TestComputePadding:
    """Tests for compute_padding function."""

    @pytest.mark.parametrize(
        "height,width,align_to,expected_h,expected_w",
        [
            (64, 64, 16, 64, 64),  # Already aligned
            (60, 70, 16, 64, 80),  # Needs padding
            (1, 1, 8, 8, 8),  # Small input
            (1080, 1920, 16, 1088, 1920),  # HD video
            (100, 100, 32, 128, 128),  # Larger alignment
        ],
    )
    def test_padding_dimensions(
        self,
        height: int,
        width: int,
        align_to: int,
        expected_h: int,
        expected_w: int,
    ) -> None:
        """Test computed padding dimensions."""
        pad_info = compute_padding(height, width, align_to)

        assert pad_info.padded_height == expected_h
        assert pad_info.padded_width == expected_w
        assert pad_info.padded_height % align_to == 0
        assert pad_info.padded_width % align_to == 0

    def test_symmetric_padding(self) -> None:
        """Test symmetric padding distribution."""
        pad_info = compute_padding(60, 70, 16, symmetric=True)

        # Height: 60 -> 64 needs 4 padding
        # Width: 70 -> 80 needs 10 padding
        total_pad_h = pad_info.pad_top + pad_info.pad_bottom
        total_pad_w = pad_info.pad_left + pad_info.pad_right
        assert total_pad_h == 4
        assert total_pad_w == 10


class TestPadToMultiple:
    """Tests for pad_to_multiple function."""

    def test_basic_padding(self) -> None:
        """Test basic padding operation."""
        x = torch.rand(1, 3, 60, 70)

        padded, info = pad_to_multiple(x, align_to=16)

        assert padded.shape[2] % 16 == 0
        assert padded.shape[3] % 16 == 0
        assert info.original_height == 60
        assert info.original_width == 70

    def test_already_aligned(self) -> None:
        """Test input that is already aligned."""
        x = torch.rand(1, 3, 64, 64)

        padded, info = pad_to_multiple(x, align_to=16)

        assert padded.shape == x.shape
        assert info.pad_top == 0
        assert info.pad_left == 0

    @pytest.mark.parametrize("mode", list(PaddingMode))
    def test_all_padding_modes(self, mode: PaddingMode) -> None:
        """Test all padding modes."""
        x = torch.rand(1, 3, 60, 70)

        padded, info = pad_to_multiple(x, align_to=16, mode=mode)

        assert padded.shape[2] % 16 == 0
        assert padded.shape[3] % 16 == 0

    def test_constant_padding_value(self) -> None:
        """Test constant padding with specific value."""
        x = torch.ones(1, 3, 60, 70)

        padded, info = pad_to_multiple(
            x,
            align_to=16,
            mode=PaddingMode.CONSTANT,
            constant_value=0.5,
        )

        # Check padded region has constant value
        if info.pad_top > 0:
            top_region = padded[:, :, : info.pad_top, :]
            assert torch.allclose(top_region, torch.full_like(top_region, 0.5))

    def test_symmetric_padding(self) -> None:
        """Test symmetric padding distribution."""
        x = torch.rand(1, 3, 60, 60)

        padded, info = pad_to_multiple(x, align_to=16, symmetric=True)

        # 60 -> 64 needs 4, symmetric: 2 top, 2 bottom
        assert info.pad_top == info.pad_bottom == 2


class TestCropToOriginal:
    """Tests for crop_to_original function."""

    def test_basic_crop(self) -> None:
        """Test basic cropping operation."""
        original = torch.rand(1, 3, 60, 70)
        padded, info = pad_to_multiple(original, align_to=16)

        cropped = crop_to_original(padded, info)

        assert cropped.shape == original.shape

    def test_content_preservation(self) -> None:
        """Test that original content is preserved."""
        original = torch.rand(1, 3, 60, 70)
        padded, info = pad_to_multiple(original, align_to=16)

        cropped = crop_to_original(padded, info)

        assert torch.allclose(cropped, original)

    def test_various_alignments(self) -> None:
        """Test with various alignment values."""
        for align_to in [4, 8, 16, 32, 64]:
            original = torch.rand(1, 3, 45, 73)
            padded, info = pad_to_multiple(original, align_to=align_to)
            cropped = crop_to_original(padded, info)

            assert torch.allclose(cropped, original)


class TestPadToMultipleModule:
    """Tests for PadToMultiple nn.Module."""

    def test_forward(self) -> None:
        """Test forward pass."""
        config = PaddingConfig(align_to=16)
        module = PadToMultiple(config=config)
        x = torch.rand(1, 3, 60, 70)

        padded = module(x)
        info = module.last_padding_info

        assert padded.shape[2] % 16 == 0
        assert padded.shape[3] % 16 == 0
        assert info is not None

    def test_module_config(self) -> None:
        """Test module with custom config."""
        config = PaddingConfig(
            mode=PaddingMode.REPLICATE,
            align_to=32,
            symmetric=True,
        )
        module = PadToMultiple(config=config)

        x = torch.rand(1, 3, 50, 50)
        padded = module(x)

        assert padded.shape[2] % 32 == 0


class TestDynamicPadding:
    """Tests for DynamicPadding nn.Module."""

    def test_inferred_alignment(self) -> None:
        """Test auto-inferred alignment from downsample factor."""
        module = DynamicPadding(downsample_factor=16)

        x = torch.rand(1, 3, 60, 70)
        padded, info = module.pad(x)

        assert padded.shape[2] % 16 == 0
        assert padded.shape[3] % 16 == 0

    def test_inverse(self) -> None:
        """Test inverse (crop) operation."""
        module = DynamicPadding(downsample_factor=8)

        original = torch.rand(1, 3, 45, 55)
        padded, info = module.pad(original)

        # Apply inverse
        cropped = module.unpad(padded, info)

        assert cropped.shape == original.shape
        assert torch.allclose(cropped, original)


class TestBatchPadding:
    """Tests for padding with batched inputs."""

    def test_batch_dimension(self) -> None:
        """Test padding preserves batch dimension."""
        x = torch.rand(4, 3, 60, 70)  # Batch of 4

        padded, info = pad_to_multiple(x, align_to=16)

        assert padded.shape[0] == 4
        assert padded.shape[2] % 16 == 0

    def test_batch_crop(self) -> None:
        """Test cropping preserves batch dimension."""
        original = torch.rand(8, 3, 50, 60)
        padded, info = pad_to_multiple(original, align_to=16)
        cropped = crop_to_original(padded, info)

        assert cropped.shape == original.shape


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_pixel(self) -> None:
        """Test with single pixel input."""
        x = torch.rand(1, 3, 1, 1)

        # Use replicate mode which works with any size (reflect fails for small inputs)
        padded, info = pad_to_multiple(x, align_to=8, mode=PaddingMode.REPLICATE)

        assert padded.shape[2] == 8
        assert padded.shape[3] == 8

    def test_large_alignment(self) -> None:
        """Test with large alignment requirement."""
        x = torch.rand(1, 3, 100, 100)

        padded, info = pad_to_multiple(x, align_to=128)

        assert padded.shape[2] == 128
        assert padded.shape[3] == 128

    def test_aspect_ratio_preservation(self) -> None:
        """Test that aspect ratio is maintained in padded dimensions."""
        x = torch.rand(1, 3, 120, 180)  # 2:3 aspect

        padded, info = pad_to_multiple(x, align_to=16)
        cropped = crop_to_original(padded, info)

        # Original dimensions should be restored
        assert cropped.shape[2] == 120
        assert cropped.shape[3] == 180
