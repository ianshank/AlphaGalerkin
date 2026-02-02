"""Codec implementation for video compression.

Components:
- EntropyCoder: Arithmetic/range coding for lossless bitstream
- GOPManager: Frame scheduling and reference management
- Codec: Complete encode/decode pipeline
"""

from src.video_compression.codec.codec import (
    CodecOutput,
    ReferenceFrameError,
    VideoCodec,
    VideoHeader,
    create_codec,
    load_codec,
)
from src.video_compression.codec.entropy_coder import (
    EntropyCoder,
    RangeDecoder,
    RangeEncoder,
)
from src.video_compression.codec.gop_manager import (
    FrameInfo,
    FrameType,
    GOPManager,
    ReferenceBuffer,
)

__all__ = [
    "EntropyCoder",
    "RangeEncoder",
    "RangeDecoder",
    "GOPManager",
    "FrameInfo",
    "FrameType",
    "ReferenceBuffer",
    "VideoCodec",
    "CodecOutput",
    "VideoHeader",
    "ReferenceFrameError",
    "create_codec",
    "load_codec",
]
