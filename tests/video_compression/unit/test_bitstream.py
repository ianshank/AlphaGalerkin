"""Unit tests for bitstream I/O utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.video_compression.utils.bitstream import (
    BitstreamHeader,
    BitstreamWriter,
    BitstreamReader,
    FrameHeader,
    EncodedFrame,
    save_bitstream,
    load_bitstream,
    MAGIC_BYTES,
    FORMAT_VERSION,
)
from src.video_compression.utils.padding import PaddingInfo
from src.video_compression.codec.gop_manager import FrameType


class TestBitstreamHeader:
    """Tests for BitstreamHeader validation."""

    def test_valid_header(self) -> None:
        """Test creating valid header."""
        header = BitstreamHeader(
            width=1920,
            height=1080,
            num_frames=100,
            padded_width=1920,
            padded_height=1088,
        )

        assert header.width == 1920
        assert header.height == 1080
        assert header.version == FORMAT_VERSION

    def test_fps_alias(self) -> None:
        """Test fps property alias."""
        header = BitstreamHeader(
            width=100,
            height=100,
            num_frames=1,
            frame_rate=24.0,
            padded_width=112,
            padded_height=112,
        )

        assert header.fps == 24.0
        assert header.fps == header.frame_rate

    def test_bpp_computation(self) -> None:
        """Test bits per pixel computation."""
        header = BitstreamHeader(
            width=100,
            height=100,
            num_frames=10,
            padded_width=112,
            padded_height=112,
            total_bits=1000000,
        )

        bpp = header.compute_bpp()
        expected = 1000000 / (100 * 100 * 10)
        assert bpp == expected

    def test_invalid_dimensions(self) -> None:
        """Test that invalid dimensions are rejected."""
        with pytest.raises(Exception):  # Pydantic validation error
            BitstreamHeader(
                width=0,  # Invalid
                height=100,
                num_frames=1,
                padded_width=112,
                padded_height=112,
            )


class TestFrameHeader:
    """Tests for FrameHeader serialization."""

    def test_serialize_deserialize(self) -> None:
        """Test round-trip serialization."""
        original = FrameHeader(
            frame_idx=42,
            frame_type=FrameType.P,
            data_length=12345,
            qp=28,
            forward_ref_idx=41,
            backward_ref_idx=-1,
        )

        data = original.to_bytes()
        recovered, consumed = FrameHeader.from_bytes(data)

        assert recovered.frame_idx == original.frame_idx
        assert recovered.frame_type == original.frame_type
        assert recovered.data_length == original.data_length
        assert recovered.qp == original.qp
        assert recovered.forward_ref_idx == original.forward_ref_idx
        assert recovered.backward_ref_idx == original.backward_ref_idx

    def test_serialize_with_padding_info(self) -> None:
        """Test serialization with padding info."""
        padding = PaddingInfo(
            original_height=1080,
            original_width=1920,
            padded_height=1088,
            padded_width=1920,
            pad_top=4,
            pad_bottom=4,
            pad_left=0,
            pad_right=0,
        )

        original = FrameHeader(
            frame_idx=0,
            frame_type=FrameType.I,
            data_length=100,
            padding_info=padding,
        )

        data = original.to_bytes()
        recovered, _ = FrameHeader.from_bytes(data)

        assert recovered.padding_info is not None
        assert recovered.padding_info.original_height == 1080
        assert recovered.padding_info.pad_top == 4

    def test_all_frame_types(self) -> None:
        """Test serialization for all frame types."""
        for frame_type in [FrameType.I, FrameType.P, FrameType.B]:
            header = FrameHeader(
                frame_idx=0,
                frame_type=frame_type,
                data_length=100,
            )

            data = header.to_bytes()
            recovered, _ = FrameHeader.from_bytes(data)

            assert recovered.frame_type == frame_type


class TestEncodedFrame:
    """Tests for EncodedFrame container."""

    def test_total_bits(self) -> None:
        """Test total bits calculation."""
        header = FrameHeader(
            frame_idx=0,
            frame_type=FrameType.I,
            data_length=1000,
        )

        frame = EncodedFrame(
            header=header,
            data=b"\x00" * 1000,
            z_data=b"\x00" * 250,
        )

        assert frame.total_bits == (1000 + 250) * 8


class TestBitstreamWriter:
    """Tests for BitstreamWriter."""

    @pytest.fixture
    def header(self) -> BitstreamHeader:
        """Create test header."""
        return BitstreamHeader(
            width=64,
            height=64,
            num_frames=2,
            padded_width=64,
            padded_height=64,
        )

    def test_write_creates_file(self, header: BitstreamHeader) -> None:
        """Test that writer creates valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.agk"

            with BitstreamWriter(path, header) as writer:
                frame = EncodedFrame(
                    header=FrameHeader(
                        frame_idx=0,
                        frame_type=FrameType.I,
                        data_length=100,
                    ),
                    data=b"\x00" * 100,
                )
                writer.write_frame(frame)

            assert path.exists()

            # Verify magic bytes
            with open(path, "rb") as f:
                magic = f.read(4)
            assert magic == MAGIC_BYTES

    def test_write_multiple_frames(self, header: BitstreamHeader) -> None:
        """Test writing multiple frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.agk"

            with BitstreamWriter(path, header) as writer:
                for i in range(5):
                    frame = EncodedFrame(
                        header=FrameHeader(
                            frame_idx=i,
                            frame_type=FrameType.I if i == 0 else FrameType.P,
                            data_length=100,
                        ),
                        data=b"\x00" * 100,
                    )
                    writer.write_frame(frame)

            # Verify by reading back
            _, frames = load_bitstream(path)
            assert len(frames) == 5


class TestBitstreamReader:
    """Tests for BitstreamReader."""

    @pytest.fixture
    def test_file(self) -> tuple[Path, BitstreamHeader]:
        """Create test bitstream file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.agk"
            header = BitstreamHeader(
                width=128,
                height=128,
                num_frames=3,
                padded_width=128,
                padded_height=128,
                frame_rate=25.0,
            )

            with BitstreamWriter(path, header) as writer:
                for i in range(3):
                    frame = EncodedFrame(
                        header=FrameHeader(
                            frame_idx=i,
                            frame_type=FrameType.I if i == 0 else FrameType.P,
                            data_length=50,
                        ),
                        data=bytes([i] * 50),
                    )
                    writer.write_frame(frame)

            yield path, header

    def test_read_header(self, test_file: tuple[Path, BitstreamHeader]) -> None:
        """Test reading header."""
        path, expected = test_file

        with BitstreamReader(path) as reader:
            assert reader.header.width == expected.width
            assert reader.header.height == expected.height
            assert reader.header.frame_rate == expected.frame_rate

    def test_iterate_frames(self, test_file: tuple[Path, BitstreamHeader]) -> None:
        """Test iterating over frames."""
        path, _ = test_file

        with BitstreamReader(path) as reader:
            frames = list(reader)

        assert len(frames) == 3

        for i, frame in enumerate(frames):
            assert frame.header.frame_idx == i
            assert frame.data == bytes([i] * 50)

    def test_version_check(self) -> None:
        """Test that future versions are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "future.agk"

            # Write file with future version
            with open(path, "wb") as f:
                f.write(MAGIC_BYTES)
                f.write((FORMAT_VERSION + 10).to_bytes(2, "little"))  # Future version
                f.write(b"\x00" * 100)

            with pytest.raises(ValueError, match="Unsupported format version"):
                with BitstreamReader(path) as reader:
                    pass


class TestConvenienceFunctions:
    """Tests for save_bitstream and load_bitstream."""

    def test_round_trip(self) -> None:
        """Test complete round trip."""
        header = BitstreamHeader(
            width=256,
            height=256,
            num_frames=4,
            padded_width=256,
            padded_height=256,
            total_bits=10000,
        )

        frames = [
            EncodedFrame(
                header=FrameHeader(
                    frame_idx=i,
                    frame_type=FrameType.I if i % 4 == 0 else FrameType.P,
                    data_length=100,
                    qp=28 + i,
                ),
                data=bytes(range(100)),
                z_data=bytes([i] * 25),
            )
            for i in range(4)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.agk"

            save_bitstream(path, header, frames)
            loaded_header, loaded_frames = load_bitstream(path)

            assert loaded_header.width == header.width
            assert loaded_header.total_bits == header.total_bits

            for orig, loaded in zip(frames, loaded_frames):
                assert orig.header.frame_idx == loaded.header.frame_idx
                assert orig.header.qp == loaded.header.qp
                assert orig.data == loaded.data
                assert orig.z_data == loaded.z_data
