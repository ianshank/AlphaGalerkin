"""Bitstream I/O for video compression file format.

Implements a versioned, extensible bitstream format:
- Magic bytes for format identification
- Version number for backwards compatibility
- JSON metadata header
- Binary frame data with length prefixes

File format (.agk):
    [MAGIC: 4 bytes] [VERSION: 2 bytes] [HEADER_LEN: 4 bytes]
    [HEADER: JSON] [FRAME_COUNT: 4 bytes]
    [FRAME_0_LEN: 4 bytes] [FRAME_0_DATA: bytes]
    [FRAME_1_LEN: 4 bytes] [FRAME_1_DATA: bytes]
    ...
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from pydantic import BaseModel, ConfigDict, Field

from src.video_compression.utils.padding import PaddingInfo
from src.video_compression.codec.gop_manager import FrameType

logger = logging.getLogger(__name__)

# Format constants
MAGIC_BYTES = b"AGK\x00"  # AlphaGalerkin Codec
FORMAT_VERSION = 1
HEADER_ENCODING = "utf-8"


class BitstreamHeader(BaseModel):
    """Header metadata for compressed video file."""

    # Format info
    version: int = Field(default=FORMAT_VERSION, description="Format version")

    # Video info
    width: int = Field(..., ge=1, description="Original video width")
    height: int = Field(..., ge=1, description="Original video height")
    num_frames: int = Field(..., ge=1, description="Number of frames")
    frame_rate: float = Field(default=30.0, gt=0, description="Frame rate (fps)")

    # Codec info
    gop_size: int = Field(default=16, ge=1, description="GOP size")
    downsample_factor: int = Field(default=16, ge=1, description="Encoder downsample")
    latent_channels: int = Field(default=192, ge=1, description="Latent channels")

    # Padding info (for reconstruction)
    padded_width: int = Field(..., ge=1, description="Padded width")
    padded_height: int = Field(..., ge=1, description="Padded height")

    # Model info
    model_hash: str = Field(default="", description="Model checkpoint hash")
    lambda_rd: float = Field(default=0.01, gt=0, description="R-D lambda used")

    # Computed fields
    total_bits: int = Field(default=0, ge=0, description="Total encoded bits")

    model_config = ConfigDict(extra="forbid")

    @property
    def fps(self) -> float:
        """Alias for frame_rate."""
        return self.frame_rate

    def compute_bpp(self) -> float:
        """Compute bits per pixel."""
        pixels = self.width * self.height * self.num_frames
        return self.total_bits / pixels if pixels > 0 else 0.0


@dataclass
class FrameHeader:
    """Per-frame metadata."""

    frame_idx: int
    frame_type: FrameType
    data_length: int  # Bytes
    qp: int = 32
    forward_ref_idx: int = -1  # -1 if none
    backward_ref_idx: int = -1  # -1 if none

    # Padding info for this frame
    padding_info: PaddingInfo | None = None

    def to_bytes(self) -> bytes:
        """Serialize to bytes."""
        # Pack fixed fields
        header = struct.pack(
            "<IcIHii",
            self.frame_idx,
            self.frame_type.value.encode("ascii"),
            self.data_length,
            self.qp,
            self.forward_ref_idx,
            self.backward_ref_idx,
        )

        # Add padding info if present
        if self.padding_info:
            pad_bytes = json.dumps(self.padding_info.to_dict()).encode(HEADER_ENCODING)
            header += struct.pack("<I", len(pad_bytes)) + pad_bytes
        else:
            header += struct.pack("<I", 0)

        return header

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple[FrameHeader, int]:
        """Deserialize from bytes.

        Returns:
            Tuple of (FrameHeader, bytes consumed).
        """
        # Unpack fixed fields
        fixed_size = struct.calcsize("<IcIHii")
        frame_idx, frame_type_byte, data_length, qp, fwd_ref, bwd_ref = struct.unpack(
            "<IcIHii", data[:fixed_size]
        )
        frame_type = FrameType(frame_type_byte.decode("ascii"))

        # Read padding info
        offset = fixed_size
        (pad_len,) = struct.unpack("<I", data[offset : offset + 4])
        offset += 4

        padding_info = None
        if pad_len > 0:
            pad_json = data[offset : offset + pad_len].decode(HEADER_ENCODING)
            padding_info = PaddingInfo.from_dict(json.loads(pad_json))
            offset += pad_len

        return (
            cls(
                frame_idx=frame_idx,
                frame_type=frame_type,
                data_length=data_length,
                qp=qp,
                forward_ref_idx=fwd_ref,
                backward_ref_idx=bwd_ref,
                padding_info=padding_info,
            ),
            offset,
        )


@dataclass
class EncodedFrame:
    """Container for encoded frame data."""

    header: FrameHeader
    data: bytes  # Entropy-coded latent symbols
    z_data: bytes = b""  # Hyperprior data (if any)

    @property
    def total_bits(self) -> int:
        """Total bits for this frame."""
        return (len(self.data) + len(self.z_data)) * 8


class BitstreamWriter:
    """Writer for compressed video bitstream.

    Usage:
        with BitstreamWriter(path, header) as writer:
            for frame in frames:
                writer.write_frame(encoded_frame)
    """

    def __init__(
        self,
        path: Path | str,
        header: BitstreamHeader,
    ) -> None:
        """Initialize bitstream writer.

        Args:
            path: Output file path.
            header: Video header metadata.
        """
        self.path = Path(path)
        self.header = header
        self._file: BinaryIO | None = None
        self._frames_written = 0
        self._total_bits = 0
        self._frame_offsets: list[int] = []

    def __enter__(self) -> BitstreamWriter:
        """Open file for writing."""
        self._file = open(self.path, "wb")
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file and finalize."""
        if self._file:
            self._finalize()
            self._file.close()
            self._file = None

    def _write_header(self) -> None:
        """Write file header."""
        if not self._file:
            raise RuntimeError("File not open")

        # Magic bytes
        self._file.write(MAGIC_BYTES)

        # Version
        self._file.write(struct.pack("<H", FORMAT_VERSION))

        # Header JSON (with placeholder for total_bits)
        header_json = self.header.model_dump_json().encode(HEADER_ENCODING)
        self._file.write(struct.pack("<I", len(header_json)))
        self._file.write(header_json)

        # Frame count placeholder (will be updated in finalize)
        self._frame_count_offset = self._file.tell()
        self._file.write(struct.pack("<I", 0))

        logger.debug(f"Wrote header ({len(header_json)} bytes)")

    def write_frame(self, frame: EncodedFrame) -> None:
        """Write encoded frame to bitstream.

        Args:
            frame: Encoded frame data.
        """
        if not self._file:
            raise RuntimeError("File not open")

        # Record offset for potential seeking
        self._frame_offsets.append(self._file.tell())

        # Write frame header
        frame_header_bytes = frame.header.to_bytes()
        self._file.write(struct.pack("<I", len(frame_header_bytes)))
        self._file.write(frame_header_bytes)

        # Write frame data
        self._file.write(struct.pack("<I", len(frame.data)))
        self._file.write(frame.data)

        # Write hyperprior data (if any)
        self._file.write(struct.pack("<I", len(frame.z_data)))
        if frame.z_data:
            self._file.write(frame.z_data)

        self._frames_written += 1
        self._total_bits += frame.total_bits

        logger.debug(
            f"Wrote frame {frame.header.frame_idx} "
            f"({len(frame.data)} bytes, type={frame.header.frame_type.value})"
        )

    def _finalize(self) -> None:
        """Finalize file and update header."""
        if not self._file:
            return

        # Update frame count
        self._file.seek(self._frame_count_offset)
        self._file.write(struct.pack("<I", self._frames_written))

        # Note: total_bits in header is not updated in-place
        # Would need to re-write entire header for that

        logger.info(
            f"Finalized bitstream: {self._frames_written} frames, "
            f"{self._total_bits} bits total"
        )


class BitstreamReader:
    """Reader for compressed video bitstream.

    Usage:
        with BitstreamReader(path) as reader:
            print(f"Video: {reader.header.width}x{reader.header.height}")
            for frame in reader:
                decoded = decode(frame)
    """

    def __init__(self, path: Path | str) -> None:
        """Initialize bitstream reader.

        Args:
            path: Input file path.
        """
        self.path = Path(path)
        self._file: BinaryIO | None = None
        self._header: BitstreamHeader | None = None
        self._frame_count: int = 0
        self._frames_read: int = 0

    @property
    def header(self) -> BitstreamHeader:
        """Get video header."""
        if self._header is None:
            raise RuntimeError("File not open")
        return self._header

    def __enter__(self) -> BitstreamReader:
        """Open file for reading."""
        self._file = open(self.path, "rb")
        self._read_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close file."""
        if self._file:
            self._file.close()
            self._file = None

    def _read_header(self) -> None:
        """Read and validate file header."""
        if not self._file:
            raise RuntimeError("File not open")

        # Check magic bytes
        magic = self._file.read(4)
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid file format (magic: {magic!r})")

        # Check version
        (version,) = struct.unpack("<H", self._file.read(2))
        if version > FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version {version} (max: {FORMAT_VERSION})"
            )

        # Read header JSON
        (header_len,) = struct.unpack("<I", self._file.read(4))
        header_json = self._file.read(header_len).decode(HEADER_ENCODING)
        self._header = BitstreamHeader.model_validate_json(header_json)

        # Read frame count
        (self._frame_count,) = struct.unpack("<I", self._file.read(4))

        logger.debug(
            f"Read header: {self._header.width}x{self._header.height}, "
            f"{self._frame_count} frames"
        )

    def __iter__(self) -> Iterator[EncodedFrame]:
        """Iterate over frames."""
        return self

    def __next__(self) -> EncodedFrame:
        """Read next frame."""
        if self._frames_read >= self._frame_count:
            raise StopIteration

        frame = self.read_frame()
        self._frames_read += 1
        return frame

    def read_frame(self) -> EncodedFrame:
        """Read a single frame.

        Returns:
            EncodedFrame with header and data.
        """
        if not self._file:
            raise RuntimeError("File not open")

        # Read frame header
        (header_len,) = struct.unpack("<I", self._file.read(4))
        header_bytes = self._file.read(header_len)
        frame_header, _ = FrameHeader.from_bytes(header_bytes)

        # Read frame data
        (data_len,) = struct.unpack("<I", self._file.read(4))
        data = self._file.read(data_len)

        # Read hyperprior data
        (z_len,) = struct.unpack("<I", self._file.read(4))
        z_data = self._file.read(z_len) if z_len > 0 else b""

        logger.debug(
            f"Read frame {frame_header.frame_idx} "
            f"({len(data)} bytes, type={frame_header.frame_type.value})"
        )

        return EncodedFrame(
            header=frame_header,
            data=data,
            z_data=z_data,
        )

    def seek_frame(self, frame_idx: int) -> EncodedFrame:
        """Seek to and read a specific frame.

        Note: This is slow as it reads through all frames.
        For random access, consider building an index.

        Args:
            frame_idx: Target frame index.

        Returns:
            EncodedFrame at index.
        """
        # Reset to start of frames
        # This is inefficient - a real implementation would use an index
        self._file.seek(0)
        self._read_header()
        self._frames_read = 0

        for frame in self:
            if frame.header.frame_idx == frame_idx:
                return frame

        raise ValueError(f"Frame {frame_idx} not found")


def save_bitstream(
    path: Path | str,
    header: BitstreamHeader,
    frames: list[EncodedFrame],
) -> int:
    """Convenience function to save frames to bitstream.

    Args:
        path: Output path.
        header: Video header.
        frames: List of encoded frames.

    Returns:
        Total bytes written.
    """
    with BitstreamWriter(path, header) as writer:
        for frame in frames:
            writer.write_frame(frame)

    return Path(path).stat().st_size


def load_bitstream(path: Path | str) -> tuple[BitstreamHeader, list[EncodedFrame]]:
    """Convenience function to load all frames from bitstream.

    Args:
        path: Input path.

    Returns:
        Tuple of (header, list of frames).
    """
    with BitstreamReader(path) as reader:
        header = reader.header
        frames = list(reader)

    return header, frames
