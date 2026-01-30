"""Codec implementation for video compression.

Components:
- EntropyCoder: Arithmetic/range coding for lossless bitstream
- GOPManager: Frame scheduling and reference management
- Codec: Complete encode/decode pipeline
"""

from src.video_compression.codec.entropy_coder import (
    EntropyCoder,
    RangeEncoder,
    RangeDecoder,
)
from src.video_compression.codec.gop_manager import GOPManager, FrameInfo
from src.video_compression.codec.codec import VideoCodec, CodecOutput

__all__ = [
    "EntropyCoder",
    "RangeEncoder",
    "RangeDecoder",
    "GOPManager",
    "FrameInfo",
    "VideoCodec",
    "CodecOutput",
]
