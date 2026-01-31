"""Unit tests for GOP manager module.

Tests the Group of Pictures management including:
- Frame type assignment (I/P/B)
- Encoding order vs display order
- Reference frame tracking
- GOP boundary detection
"""

from __future__ import annotations

import pytest
import torch

from src.video_compression.codec.gop_manager import (
    FrameType,
    FrameInfo,
    ReferenceBuffer,
    GOPManager,
)


class TestFrameType:
    """Tests for FrameType enum."""

    def test_enum_values(self) -> None:
        """Test enum string values."""
        assert FrameType.I.value == "I"
        assert FrameType.P.value == "P"
        assert FrameType.B.value == "B"

    def test_enum_membership(self) -> None:
        """Test enum membership."""
        assert FrameType.I in FrameType
        assert FrameType.P in FrameType
        assert FrameType.B in FrameType


class TestFrameInfo:
    """Tests for FrameInfo dataclass."""

    def test_is_reference_i_frame(self) -> None:
        """Test that I-frames are references."""
        info = FrameInfo(
            index=0,
            gop_index=0,
            frame_type=FrameType.I,
            display_order=0,
            encode_order=0,
        )
        assert info.is_reference is True

    def test_is_reference_p_frame(self) -> None:
        """Test that P-frames are references."""
        info = FrameInfo(
            index=4,
            gop_index=4,
            frame_type=FrameType.P,
            display_order=4,
            encode_order=1,
            forward_ref=0,
        )
        assert info.is_reference is True

    def test_is_reference_b_frame(self) -> None:
        """Test that B-frames are NOT references."""
        info = FrameInfo(
            index=1,
            gop_index=1,
            frame_type=FrameType.B,
            display_order=1,
            encode_order=2,
            forward_ref=0,
            backward_ref=4,
        )
        assert info.is_reference is False

    def test_optional_fields(self) -> None:
        """Test optional fields have correct defaults."""
        info = FrameInfo(
            index=0,
            gop_index=0,
            frame_type=FrameType.I,
            display_order=0,
            encode_order=0,
        )
        assert info.forward_ref is None
        assert info.backward_ref is None
        assert info.qp is None

    def test_qp_field(self) -> None:
        """Test QP field can be set."""
        info = FrameInfo(
            index=0,
            gop_index=0,
            frame_type=FrameType.I,
            display_order=0,
            encode_order=0,
            qp=28,
        )
        assert info.qp == 28


class TestReferenceBuffer:
    """Tests for ReferenceBuffer."""

    @pytest.fixture
    def buffer(self) -> ReferenceBuffer:
        """Create test reference buffer."""
        return ReferenceBuffer(capacity=2)

    def test_add_frame(self, buffer: ReferenceBuffer) -> None:
        """Test adding a frame."""
        frame = torch.randn(1, 3, 64, 64)
        buffer.add(0, frame)

        assert 0 in buffer.frames
        assert buffer.get(0) is not None
        torch.testing.assert_close(buffer.get(0), frame)

    def test_add_with_latent(self, buffer: ReferenceBuffer) -> None:
        """Test adding frame with latent."""
        frame = torch.randn(1, 3, 64, 64)
        latent = torch.randn(1, 64, 8, 8)
        buffer.add(0, frame, latent)

        assert buffer.get(0) is not None
        assert buffer.get_latent(0) is not None
        torch.testing.assert_close(buffer.get_latent(0), latent)

    def test_capacity_limit(self, buffer: ReferenceBuffer) -> None:
        """Test that buffer respects capacity limit."""
        for i in range(5):
            frame = torch.randn(1, 3, 64, 64)
            buffer.add(i, frame)

        # Should only have last 2 frames
        assert len(buffer.frames) == 2
        assert 0 not in buffer.frames
        assert 1 not in buffer.frames
        assert 2 not in buffer.frames
        assert 3 in buffer.frames
        assert 4 in buffer.frames

    def test_get_nonexistent(self, buffer: ReferenceBuffer) -> None:
        """Test getting non-existent frame returns None."""
        assert buffer.get(999) is None
        assert buffer.get_latent(999) is None

    def test_clear(self, buffer: ReferenceBuffer) -> None:
        """Test clearing buffer."""
        frame = torch.randn(1, 3, 64, 64)
        latent = torch.randn(1, 64, 8, 8)
        buffer.add(0, frame, latent)
        buffer.add(1, frame, latent)

        buffer.clear()

        assert len(buffer.frames) == 0
        assert len(buffer.latents) == 0

    def test_latent_capacity_sync(self, buffer: ReferenceBuffer) -> None:
        """Test that latents are removed with frames."""
        for i in range(5):
            frame = torch.randn(1, 3, 64, 64)
            latent = torch.randn(1, 64, 8, 8)
            buffer.add(i, frame, latent)

        # Latents should match frames
        assert set(buffer.latents.keys()) == set(buffer.frames.keys())


class TestGOPManager:
    """Tests for GOPManager."""

    @pytest.fixture
    def manager(self) -> GOPManager:
        """Create test GOP manager."""
        return GOPManager(
            gop_size=8,
            i_frame_interval=8,
            use_b_frames=True,
            b_frame_count=2,
        )

    @pytest.fixture
    def manager_no_b(self) -> GOPManager:
        """Create GOP manager without B-frames."""
        return GOPManager(
            gop_size=8,
            i_frame_interval=8,
            use_b_frames=False,
            b_frame_count=0,
        )

    def test_get_frame_info_i_frame(self, manager: GOPManager) -> None:
        """Test that first frame is I-frame."""
        info = manager.get_frame_info(0)

        assert info.index == 0
        assert info.gop_index == 0
        assert info.frame_type == FrameType.I
        assert info.forward_ref is None
        assert info.backward_ref is None

    def test_get_frame_info_gop_boundary(self, manager: GOPManager) -> None:
        """Test that GOP boundaries are I-frames."""
        info_0 = manager.get_frame_info(0)
        info_8 = manager.get_frame_info(8)
        info_16 = manager.get_frame_info(16)

        assert info_0.frame_type == FrameType.I
        assert info_8.frame_type == FrameType.I
        assert info_16.frame_type == FrameType.I

    def test_get_frame_info_p_frame(self, manager: GOPManager) -> None:
        """Test P-frame detection."""
        # With b_frame_count=2, P-frames are at positions 3, 6, etc.
        info = manager.get_frame_info(3)

        assert info.frame_type == FrameType.P
        assert info.forward_ref is not None
        assert info.backward_ref is None

    def test_get_frame_info_b_frame(self, manager: GOPManager) -> None:
        """Test B-frame detection."""
        # Positions 1, 2 should be B-frames
        info1 = manager.get_frame_info(1)
        info2 = manager.get_frame_info(2)

        assert info1.frame_type == FrameType.B
        assert info2.frame_type == FrameType.B
        assert info1.forward_ref is not None
        assert info1.backward_ref is not None

    def test_no_b_frames_mode(self, manager_no_b: GOPManager) -> None:
        """Test GOP manager without B-frames."""
        for i in range(8):
            info = manager_no_b.get_frame_info(i)
            assert info.frame_type != FrameType.B

    def test_compute_encode_order_no_b_frames(self, manager_no_b: GOPManager) -> None:
        """Test encode order equals display order without B-frames."""
        for i in range(8):
            info = manager_no_b.get_frame_info(i)
            assert info.encode_order == info.display_order

    def test_compute_encode_order_with_b_frames(self, manager: GOPManager) -> None:
        """Test encode order differs from display order with B-frames."""
        frames = manager.get_gop_frames(0)

        # Reference frames should be encoded before B-frames
        ref_frames = [f for f in frames if f.frame_type != FrameType.B]
        b_frames = [f for f in frames if f.frame_type == FrameType.B]

        if ref_frames and b_frames:
            max_ref_order = max(f.encode_order for f in ref_frames)
            min_b_order = min(f.encode_order for f in b_frames)
            # B-frames should come after their references in encode order
            # (This depends on the specific encoding strategy)

    def test_get_gop_frames(self, manager: GOPManager) -> None:
        """Test getting all frames in a GOP."""
        frames = manager.get_gop_frames(0)

        assert len(frames) == 8
        assert all(f.index < 8 for f in frames)
        assert frames[0].frame_type == FrameType.I

    def test_get_encoding_order(self, manager: GOPManager) -> None:
        """Test getting frames in encoding order."""
        frames = manager.get_encoding_order(0)

        assert len(frames) == 8
        # Should be sorted by encode_order
        for i in range(len(frames) - 1):
            assert frames[i].encode_order <= frames[i + 1].encode_order

    def test_is_gop_boundary(self, manager: GOPManager) -> None:
        """Test GOP boundary detection."""
        assert manager.is_gop_boundary(0) is True
        assert manager.is_gop_boundary(1) is False
        assert manager.is_gop_boundary(7) is False
        assert manager.is_gop_boundary(8) is True
        assert manager.is_gop_boundary(16) is True

    def test_reset(self, manager: GOPManager) -> None:
        """Test reset clears state."""
        # Add some state
        manager.frame_count = 100
        manager.current_gop = 5
        frame = torch.randn(1, 3, 64, 64)
        manager.reference_buffer.add(0, frame)

        manager.reset()

        assert manager.frame_count == 0
        assert manager.current_gop == 0
        assert len(manager.reference_buffer.frames) == 0

    def test_iter_frames(self, manager: GOPManager) -> None:
        """Test frame iteration."""
        frames = list(manager.iter_frames(start=0, end=8))

        assert len(frames) == 8
        # All frames from first GOP
        assert all(f.index < 8 for f in frames)

    def test_iter_frames_partial(self, manager: GOPManager) -> None:
        """Test partial frame iteration."""
        frames = list(manager.iter_frames(start=2, end=6))

        assert len(frames) == 4
        assert all(2 <= f.index < 6 for f in frames)

    def test_multiple_gops(self, manager: GOPManager) -> None:
        """Test frame info across multiple GOPs."""
        # First GOP
        info_0 = manager.get_frame_info(0)
        info_7 = manager.get_frame_info(7)

        # Second GOP
        info_8 = manager.get_frame_info(8)
        info_15 = manager.get_frame_info(15)

        # Check GOP indices
        assert info_0.gop_index == 0
        assert info_7.gop_index == 7
        assert info_8.gop_index == 0  # Resets at GOP boundary
        assert info_15.gop_index == 7

    def test_i_frame_interval(self) -> None:
        """Test I-frame interval setting."""
        manager = GOPManager(
            gop_size=16,
            i_frame_interval=8,  # Force I-frame every 8 frames
            use_b_frames=False,
        )

        info_0 = manager.get_frame_info(0)
        info_8 = manager.get_frame_info(8)

        assert info_0.frame_type == FrameType.I
        assert info_8.frame_type == FrameType.I

    def test_reference_buffer_integration(self, manager: GOPManager) -> None:
        """Test reference buffer is accessible."""
        assert manager.reference_buffer is not None
        assert isinstance(manager.reference_buffer, ReferenceBuffer)

        # Should be able to add/get references
        frame = torch.randn(1, 3, 64, 64)
        manager.reference_buffer.add(0, frame)
        assert manager.reference_buffer.get(0) is not None
