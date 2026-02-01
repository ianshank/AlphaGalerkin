"""Integration tests for video compression encode/decode cycle.

Tests the complete pipeline from raw frames to reconstructed frames,
including bitstream serialization and reference frame handling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import Tensor

from src.video_compression.config import (
    CodecConfig,
    EncoderConfig,
    DecoderConfig,
    QuantizerConfig,
    EntropyConfig,
    MCTSRateControlConfig,
    QuantizationMode,
)
from src.video_compression.codec.codec import (
    VideoCodec,
    CodecOutput,
    create_codec,
    load_codec,
    ReferenceFrameError,
)
from src.video_compression.codec.gop_manager import (
    GOPManager,
    FrameInfo,
    FrameType,
    ReferenceBuffer,
)
from src.video_compression.utils.bitstream import (
    BitstreamHeader,
    BitstreamWriter,
    BitstreamReader,
    FrameHeader,
    EncodedFrame,
    save_bitstream,
    load_bitstream,
)
from src.video_compression.utils.padding import (
    pad_to_multiple,
    crop_to_original,
    PaddingMode,
)


class TestEndToEndEncodeDecode:
    """End-to-end tests for full encode/decode cycle."""

    @pytest.fixture
    def codec_config(self) -> CodecConfig:
        """Create test codec configuration."""
        return CodecConfig(
            name="test_codec",
            encoder=EncoderConfig(
                name="test_encoder",
                latent_channels=64,
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_ffn=256,
                downsample_factor=8,
            ),
            decoder=DecoderConfig(
                name="test_decoder",
                latent_channels=64,
                n_layers=2,
                d_model=128,
                n_heads=4,
                d_ffn=256,
                upsample_factor=8,
            ),
            quantizer=QuantizerConfig(
                name="test_quantizer",
                mode=QuantizationMode.STE,
            ),
            entropy=EntropyConfig(
                name="test_entropy",
                hyper_channels=64,
                num_filters=64,
            ),
            mcts=MCTSRateControlConfig(
                name="test_mcts",
                gop_size=4,
                i_frame_interval=4,
                use_b_frames=False,
            ),
        )

    @pytest.fixture
    def codec(self, codec_config: CodecConfig) -> VideoCodec:
        """Create test codec."""
        return create_codec(codec_config, use_mcts_rate_control=False)

    @pytest.fixture
    def sample_frame(self) -> Tensor:
        """Create sample test frame."""
        return torch.rand(1, 3, 64, 64)

    @pytest.fixture
    def sample_video(self) -> list[Tensor]:
        """Create sample video sequence (8 frames)."""
        base_frame = torch.rand(1, 3, 64, 64)
        frames = [base_frame]

        # Add temporal variation
        for i in range(7):
            noise = torch.randn_like(base_frame) * 0.1
            frames.append((base_frame + noise).clamp(0, 1))

        return frames

    def test_single_frame_encode_decode(
        self,
        codec: VideoCodec,
        sample_frame: Tensor,
    ) -> None:
        """Test encoding and decoding a single I-frame."""
        codec.eval()

        # Create frame info
        frame_info = FrameInfo(
            index=0,
            gop_index=0,
            frame_type=FrameType.I,
            display_order=0,
            encode_order=0,
        )

        # Encode
        output = codec.encode_frame(sample_frame, frame_info)

        assert isinstance(output, CodecOutput)
        assert output.rate > 0
        assert output.distortion >= 0
        assert output.reconstructed.shape == sample_frame.shape

    def test_encode_decode_psnr(
        self,
        codec: VideoCodec,
        sample_frame: Tensor,
    ) -> None:
        """Test that encode/decode achieves reasonable PSNR."""
        codec.eval()

        frame_info = FrameInfo(
            index=0,
            gop_index=0,
            frame_type=FrameType.I,
            display_order=0,
            encode_order=0,
        )

        output = codec.encode_frame(sample_frame, frame_info)

        # Compute PSNR
        mse = torch.mean((sample_frame - output.reconstructed) ** 2).item()
        psnr = -10 * torch.log10(torch.tensor(mse + 1e-10)).item()

        # Should achieve at least 20 dB PSNR
        assert psnr > 20.0, f"PSNR too low: {psnr:.2f} dB"

    def test_video_sequence_encode(
        self,
        codec: VideoCodec,
        sample_video: list[Tensor],
    ) -> None:
        """Test encoding a video sequence."""
        codec.eval()
        codec.gop_manager.reset()

        outputs: list[CodecOutput] = []
        for frame_idx, frame in enumerate(sample_video):
            frame_info = codec.gop_manager.get_frame_info(frame_idx)
            output = codec.encode_frame(frame, frame_info, validate_refs=True)
            outputs.append(output)

        assert len(outputs) == len(sample_video)

        # First frame should be I-frame (higher bits)
        # Later frames should be P-frames (lower bits in general)
        assert outputs[0].rate > 0

    def test_reference_frame_storage(
        self,
        codec: VideoCodec,
        sample_video: list[Tensor],
    ) -> None:
        """Test that reference frames are properly stored."""
        codec.eval()
        codec.gop_manager.reset()

        for frame_idx, frame in enumerate(sample_video[:4]):
            frame_info = codec.gop_manager.get_frame_info(frame_idx)
            codec.encode_frame(frame, frame_info, validate_refs=True)

        # Check that reference buffer has frames
        ref_buffer = codec.gop_manager.reference_buffer
        assert len(ref_buffer.frames) > 0

    def test_gop_structure(
        self,
        codec: VideoCodec,
        sample_video: list[Tensor],
    ) -> None:
        """Test GOP structure is correctly maintained."""
        codec.eval()
        codec.gop_manager.reset()

        frame_types: list[FrameType] = []
        for frame_idx, frame in enumerate(sample_video):
            frame_info = codec.gop_manager.get_frame_info(frame_idx)
            frame_types.append(frame_info.frame_type)
            codec.encode_frame(frame, frame_info, validate_refs=True)

        # First frame should be I
        assert frame_types[0] == FrameType.I

        # With B-frames disabled and GOP=4, frame 4 should be I
        if len(frame_types) > 4:
            assert frame_types[4] == FrameType.I


class TestBitstreamRoundTrip:
    """Tests for bitstream serialization/deserialization."""

    @pytest.fixture
    def sample_header(self) -> BitstreamHeader:
        """Create sample bitstream header."""
        return BitstreamHeader(
            width=64,
            height=64,
            num_frames=4,
            frame_rate=30.0,
            gop_size=4,
            downsample_factor=8,
            latent_channels=64,
            padded_width=64,
            padded_height=64,
        )

    @pytest.fixture
    def sample_frames(self) -> list[EncodedFrame]:
        """Create sample encoded frames."""
        frames = []
        for i in range(4):
            header = FrameHeader(
                frame_idx=i,
                frame_type=FrameType.I if i == 0 else FrameType.P,
                data_length=1024,
                qp=28,
                forward_ref_idx=-1 if i == 0 else i - 1,
            )
            # Random data for testing
            data = bytes([i % 256] * 1024)
            frames.append(EncodedFrame(header=header, data=data))
        return frames

    def test_write_read_bitstream(
        self,
        sample_header: BitstreamHeader,
        sample_frames: list[EncodedFrame],
    ) -> None:
        """Test writing and reading bitstream file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.agk"

            # Write
            with BitstreamWriter(path, sample_header) as writer:
                for frame in sample_frames:
                    writer.write_frame(frame)

            # Read
            with BitstreamReader(path) as reader:
                assert reader.header.width == sample_header.width
                assert reader.header.height == sample_header.height
                assert reader.header.num_frames == sample_header.num_frames

                read_frames = list(reader)

            assert len(read_frames) == len(sample_frames)

            for orig, read in zip(sample_frames, read_frames):
                assert orig.header.frame_idx == read.header.frame_idx
                assert orig.header.frame_type == read.header.frame_type
                assert orig.data == read.data

    def test_convenience_functions(
        self,
        sample_header: BitstreamHeader,
        sample_frames: list[EncodedFrame],
    ) -> None:
        """Test save_bitstream and load_bitstream functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.agk"

            # Save
            bytes_written = save_bitstream(path, sample_header, sample_frames)
            assert bytes_written > 0

            # Load
            loaded_header, loaded_frames = load_bitstream(path)

            assert loaded_header.width == sample_header.width
            assert len(loaded_frames) == len(sample_frames)

    def test_invalid_magic_bytes(self) -> None:
        """Test that invalid magic bytes are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "invalid.agk"

            # Write invalid file
            with open(path, "wb") as f:
                f.write(b"INVALID")

            with pytest.raises(ValueError, match="Invalid file format"):
                with BitstreamReader(path) as reader:
                    pass


class TestPaddingIntegration:
    """Tests for padding integration with codec."""

    def test_non_aligned_resolution(self) -> None:
        """Test codec handles non-aligned resolutions."""
        # Create frame with non-aligned dimensions
        frame = torch.rand(1, 3, 60, 70)  # Not divisible by 16

        # Pad to multiple of 16
        padded, pad_info = pad_to_multiple(
            frame,
            align_to=16,
            mode=PaddingMode.REFLECT,
        )

        assert padded.shape[2] % 16 == 0
        assert padded.shape[3] % 16 == 0

        # Verify we can crop back
        cropped = crop_to_original(padded, pad_info)
        assert cropped.shape == frame.shape
        assert torch.allclose(cropped, frame)

    def test_padding_preserves_content(self) -> None:
        """Test that padding doesn't modify original content."""
        frame = torch.rand(1, 3, 48, 56)

        padded, pad_info = pad_to_multiple(
            frame,
            align_to=16,
            mode=PaddingMode.REFLECT,
        )

        # Original content should be preserved in center
        original_region = padded[
            :,
            :,
            pad_info.pad_top : pad_info.pad_top + 48,
            pad_info.pad_left : pad_info.pad_left + 56,
        ]
        assert torch.allclose(original_region, frame)


class TestReferenceFrameValidation:
    """Tests for reference frame validation."""

    @pytest.fixture
    def codec_config(self) -> CodecConfig:
        """Create test codec config with B-frames."""
        return CodecConfig(
            name="test_codec_bframes",
            encoder=EncoderConfig(
                name="encoder",
                latent_channels=64,
                n_layers=2,
                d_model=128,
                downsample_factor=8,
            ),
            decoder=DecoderConfig(
                name="decoder",
                latent_channels=64,
                n_layers=2,
                d_model=128,
                upsample_factor=8,
            ),
            entropy=EntropyConfig(
                name="entropy",
                hyper_channels=64,
                num_filters=64,
            ),
            mcts=MCTSRateControlConfig(
                name="mcts",
                gop_size=8,
                use_b_frames=True,
                b_frame_count=2,
            ),
        )

    def test_missing_reference_raises_error(
        self,
        codec_config: CodecConfig,
    ) -> None:
        """Test that missing reference frame raises error."""
        codec = create_codec(codec_config)
        codec.eval()
        codec.gop_manager.reset()

        frame = torch.rand(1, 3, 64, 64)

        # Try to encode P-frame without encoding I-frame first
        frame_info = FrameInfo(
            index=1,
            gop_index=1,
            frame_type=FrameType.P,
            display_order=1,
            encode_order=1,
            forward_ref=0,
        )

        with pytest.raises(ReferenceFrameError):
            codec.encode_frame(frame, frame_info, validate_refs=True)

    def test_reference_buffer_capacity(self) -> None:
        """Test reference buffer respects capacity."""
        buffer = ReferenceBuffer(capacity=2)

        # Add 3 frames
        for i in range(3):
            buffer.add(i, torch.rand(1, 3, 64, 64), torch.rand(1, 64, 8, 8))

        # Should only have 2 frames
        assert len(buffer.frames) == 2

        # Oldest should be removed
        assert buffer.get(0) is None
        assert buffer.get(1) is not None
        assert buffer.get(2) is not None


class TestCodecStatistics:
    """Tests for codec encoding statistics."""

    @pytest.fixture
    def codec(self) -> VideoCodec:
        """Create test codec."""
        config = CodecConfig(
            name="stats_test",
            encoder=EncoderConfig(name="e", latent_channels=64, downsample_factor=8),
            decoder=DecoderConfig(name="d", latent_channels=64, upsample_factor=8),
            entropy=EntropyConfig(name="e", hyper_channels=64, num_filters=64),
        )
        return create_codec(config)

    def test_stats_tracking(self, codec: VideoCodec) -> None:
        """Test that encoding statistics are tracked."""
        codec.eval()
        codec.gop_manager.reset()
        codec._reset_stats()

        frames = [torch.rand(1, 3, 64, 64) for _ in range(4)]

        for i, frame in enumerate(frames):
            frame_info = codec.gop_manager.get_frame_info(i)
            codec.encode_frame(frame, frame_info, validate_refs=True)

        stats = codec.get_encoding_stats()

        assert "total_bits" in stats
        assert "avg_bits_per_frame" in stats
        assert "avg_psnr" in stats
        assert "num_frames" in stats
        assert stats["num_frames"] == 4


class TestVariableResolution:
    """Tests for variable resolution support."""

    @pytest.fixture
    def codec(self) -> VideoCodec:
        """Create test codec."""
        config = CodecConfig(
            name="var_res_test",
            encoder=EncoderConfig(name="e", latent_channels=64, downsample_factor=8),
            decoder=DecoderConfig(name="d", latent_channels=64, upsample_factor=8),
            entropy=EntropyConfig(name="e", hyper_channels=64, num_filters=64),
        )
        return create_codec(config)

    @pytest.mark.parametrize(
        "height,width",
        [
            (64, 64),    # Aligned
            (64, 128),   # Different aspect ratio
            (128, 64),   # Tall
            (72, 96),    # Non-aligned
        ],
    )
    def test_various_resolutions(
        self,
        codec: VideoCodec,
        height: int,
        width: int,
    ) -> None:
        """Test codec handles various resolutions."""
        codec.eval()

        # Pad to multiple of downsample factor
        frame = torch.rand(1, 3, height, width)
        padded, pad_info = pad_to_multiple(
            frame,
            align_to=codec.config.encoder.downsample_factor,
        )

        frame_info = FrameInfo(
            index=0,
            gop_index=0,
            frame_type=FrameType.I,
            display_order=0,
            encode_order=0,
        )

        output = codec.encode_frame(padded, frame_info, validate_refs=False)

        # Crop back to original
        reconstructed = crop_to_original(output.reconstructed, pad_info)
        assert reconstructed.shape == frame.shape
