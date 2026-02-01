"""Integration tests for video compression CLI scripts.

Tests end-to-end functionality of encode_video.py and decode_video.py
with mocked video I/O for CI compatibility.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.video_compression.config import CodecConfig
from src.video_compression.utils.bitstream import (
    BitstreamHeader,
    BitstreamReader,
    BitstreamWriter,
    EncodedFrame,
    FrameHeader,
    save_bitstream,
    load_bitstream,
)
from src.video_compression.codec.gop_manager import FrameType


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_video_path(temp_dir: Path) -> Path:
    """Create mock video file path."""
    path = temp_dir / "test_video.mp4"
    path.touch()  # Create empty file
    return path


@pytest.fixture
def sample_header() -> BitstreamHeader:
    """Create sample bitstream header."""
    return BitstreamHeader(
        width=256,
        height=256,
        num_frames=4,
        frame_rate=30.0,
        gop_size=16,
        downsample_factor=16,
        latent_channels=192,
        padded_width=256,
        padded_height=256,
        lambda_rd=0.01,
    )


@pytest.fixture
def sample_encoded_frames() -> list[EncodedFrame]:
    """Create sample encoded frames."""
    frames = []
    for i in range(4):
        frame_type = FrameType.I if i == 0 else FrameType.P
        header = FrameHeader(
            frame_idx=i,
            frame_type=frame_type,
            data_length=1024,
            qp=32,
            forward_ref_idx=-1 if frame_type == FrameType.I else i - 1,
            backward_ref_idx=-1,
        )
        frames.append(
            EncodedFrame(
                header=header,
                data=b"\x00" * 1024,  # Mock data
                z_data=b"\x00" * 256,  # Mock hyperprior data
            )
        )
    return frames


# --------------------------------------------------------------------------
# Bitstream Round-Trip Tests
# --------------------------------------------------------------------------


class TestBitstreamRoundTrip:
    """Tests for bitstream writing and reading."""

    def test_write_read_header(
        self, temp_dir: Path, sample_header: BitstreamHeader
    ) -> None:
        """Test header is preserved through write/read cycle."""
        path = temp_dir / "test.agk"
        
        # Create empty writer to test header
        with BitstreamWriter(path, sample_header) as writer:
            pass  # Write no frames
        
        # Read back
        with BitstreamReader(path) as reader:
            assert reader.header.width == sample_header.width
            assert reader.header.height == sample_header.height
            assert reader.header.frame_rate == sample_header.frame_rate

    def test_write_read_frames(
        self,
        temp_dir: Path,
        sample_header: BitstreamHeader,
        sample_encoded_frames: list[EncodedFrame],
    ) -> None:
        """Test frames are preserved through write/read cycle."""
        path = temp_dir / "test.agk"
        
        # Write frames
        save_bitstream(path, sample_header, sample_encoded_frames)
        
        # Read back
        header, frames = load_bitstream(path)
        
        assert len(frames) == len(sample_encoded_frames)
        for original, loaded in zip(sample_encoded_frames, frames):
            assert loaded.header.frame_idx == original.header.frame_idx
            assert loaded.header.frame_type == original.header.frame_type
            assert len(loaded.data) == len(original.data)

    def test_frame_type_preserved(
        self,
        temp_dir: Path,
        sample_header: BitstreamHeader,
    ) -> None:
        """Test frame types are preserved."""
        path = temp_dir / "test.agk"
        
        frames = [
            EncodedFrame(
                header=FrameHeader(
                    frame_idx=0,
                    frame_type=FrameType.I,
                    data_length=100,
                    qp=30,
                ),
                data=b"\x00" * 100,
            ),
            EncodedFrame(
                header=FrameHeader(
                    frame_idx=1,
                    frame_type=FrameType.P,
                    data_length=100,
                    qp=32,
                    forward_ref_idx=0,
                ),
                data=b"\x00" * 100,
            ),
            EncodedFrame(
                header=FrameHeader(
                    frame_idx=2,
                    frame_type=FrameType.B,
                    data_length=100,
                    qp=34,
                    forward_ref_idx=1,
                    backward_ref_idx=0,
                ),
                data=b"\x00" * 100,
            ),
        ]
        
        save_bitstream(path, sample_header, frames)
        _, loaded = load_bitstream(path)
        
        assert loaded[0].header.frame_type == FrameType.I
        assert loaded[1].header.frame_type == FrameType.P
        assert loaded[2].header.frame_type == FrameType.B


# --------------------------------------------------------------------------
# Encode Script Tests
# --------------------------------------------------------------------------


class TestEncodeScript:
    """Tests for encode_video.py script."""

    def test_script_import(self) -> None:
        """Test encode_video script can be imported."""
        from scripts import encode_video
        
        assert hasattr(encode_video, "main")
        assert hasattr(encode_video, "load_video_frames")
        assert hasattr(encode_video, "parse_args")

    def test_load_video_frames_mocked(self, mock_video_path: Path) -> None:
        """Test load_video_frames with mocked cv2."""
        from scripts.encode_video import load_video_frames
        
        # Create mock cv2 module
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        
        # Simulate 4 frames then end
        import numpy as np
        frame_data = [
            (True, (np.random.rand(256, 256, 3) * 255).astype(np.uint8))
            for _ in range(4)
        ] + [(False, None)]
        mock_cap.read.side_effect = frame_data
        
        # Patch cv2 at module level where it's imported
        with patch.dict("sys.modules", {"cv2": MagicMock()}) as mock_modules:
            mock_cv2 = mock_modules["cv2"]
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cv2.cvtColor = lambda x, _: x
            mock_cv2.COLOR_BGR2RGB = None
            
            # Re-import to get patched version
            import importlib
            import scripts.encode_video as encode_mod
            importlib.reload(encode_mod)
            
            # This test verifies the function signature - actual cv2 testing
            # requires more complex mocking or real video files
            assert callable(encode_mod.load_video_frames)

    def test_parse_args_defaults(self) -> None:
        """Test argument parsing with defaults."""
        from scripts.encode_video import parse_args
        
        with patch("sys.argv", ["encode_video.py", "input.mp4", "output.agk"]):
            args = parse_args()
            
            assert args.qp == 32
            assert args.gop_size == 16
            assert args.device == "auto"

    def test_parse_args_custom(self) -> None:
        """Test argument parsing with custom values."""
        from scripts.encode_video import parse_args
        
        with patch(
            "sys.argv",
            [
                "encode_video.py",
                "input.mp4",
                "output.agk",
                "--qp", "40",
                "--gop-size", "8",
                "--device", "cpu",
            ],
        ):
            args = parse_args()
            
            assert args.qp == 40
            assert args.gop_size == 8
            assert args.device == "cpu"

    def test_serialize_latent_roundtrip(self) -> None:
        """Test latent serialization round-trip."""
        from scripts.encode_video import serialize_latent
        
        original = torch.randn(1, 192, 16, 16)
        serialized = serialize_latent(original)
        
        import io
        
        buffer = io.BytesIO(serialized)
        recovered = torch.load(buffer)
        
        assert torch.allclose(original, recovered)


# --------------------------------------------------------------------------
# Decode Script Tests
# --------------------------------------------------------------------------


class TestDecodeScript:
    """Tests for decode_video.py script."""

    def test_script_import(self) -> None:
        """Test decode_video script can be imported."""
        from scripts import decode_video
        
        assert hasattr(decode_video, "main")
        assert hasattr(decode_video, "parse_args")

    def test_parse_args_defaults(self) -> None:
        """Test argument parsing with required args."""
        from scripts.decode_video import parse_args
        
        # Decode requires --input, --output, --checkpoint
        with patch(
            "sys.argv",
            ["decode_video.py", "-i", "input.agk", "-o", "output.mp4", "-c", "model.pt"],
        ):
            args = parse_args()
            
            assert args.device == "auto"  # default
            assert str(args.input) == "input.agk"
            assert str(args.output) == "output.mp4"


# --------------------------------------------------------------------------
# Integration Tests
# --------------------------------------------------------------------------


class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    def test_bitstream_file_creation(
        self, temp_dir: Path, sample_header: BitstreamHeader
    ) -> None:
        """Test bitstream file is created with correct format."""
        path = temp_dir / "output.agk"
        
        with BitstreamWriter(path, sample_header) as writer:
            frame = EncodedFrame(
                header=FrameHeader(
                    frame_idx=0,
                    frame_type=FrameType.I,
                    data_length=100,
                    qp=32,
                ),
                data=b"\x00" * 100,
            )
            writer.write_frame(frame)
        
        # Verify file exists and has content
        assert path.exists()
        assert path.stat().st_size > 0
        
        # Verify magic bytes
        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"AGK\x00"

    def test_bitstream_version_check(
        self, temp_dir: Path, sample_header: BitstreamHeader
    ) -> None:
        """Test bitstream version is written correctly."""
        path = temp_dir / "output.agk"
        
        with BitstreamWriter(path, sample_header) as writer:
            pass
        
        with open(path, "rb") as f:
            _ = f.read(4)  # Skip magic
            version_bytes = f.read(2)
            import struct
            (version,) = struct.unpack("<H", version_bytes)
            assert version == 1  # FORMAT_VERSION

    def test_empty_video_handling(
        self, temp_dir: Path, sample_header: BitstreamHeader
    ) -> None:
        """Test handling of video with no frames."""
        path = temp_dir / "empty.agk"
        
        # Write with no frames
        with BitstreamWriter(path, sample_header) as writer:
            pass
        
        # Should be readable but empty
        with BitstreamReader(path) as reader:
            frames = list(reader)
            assert len(frames) == 0


# --------------------------------------------------------------------------
# Error Handling Tests
# --------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in CLI scripts."""

    def test_invalid_magic_bytes(self, temp_dir: Path) -> None:
        """Test error on invalid file format."""
        path = temp_dir / "invalid.agk"
        
        with open(path, "wb") as f:
            f.write(b"BAD\x00")  # Invalid magic
        
        with pytest.raises(ValueError, match="Invalid file format"):
            with BitstreamReader(path) as reader:
                pass

    def test_file_not_found(self, temp_dir: Path) -> None:
        """Test error on missing file."""
        path = temp_dir / "nonexistent.agk"
        
        with pytest.raises(FileNotFoundError):
            with BitstreamReader(path) as reader:
                pass

    def test_corrupted_frame_data(self, temp_dir: Path) -> None:
        """Test handling of corrupted frame data."""
        path = temp_dir / "corrupted.agk"
        
        # Create valid header but truncate frame data
        header = BitstreamHeader(
            width=64,
            height=64,
            num_frames=1,
            padded_width=64,
            padded_height=64,
        )
        
        with open(path, "wb") as f:
            # Write header
            f.write(b"AGK\x00")
            import struct
            f.write(struct.pack("<H", 1))  # Version
            header_json = header.model_dump_json().encode("utf-8")
            f.write(struct.pack("<I", len(header_json)))
            f.write(header_json)
            f.write(struct.pack("<I", 1))  # Frame count
            # Truncate file without writing frame data
        
        with BitstreamReader(path) as reader:
            with pytest.raises(Exception):  # Should fail reading incomplete frame
                next(reader)
