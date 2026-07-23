"""Utility modules for video compression.

Provides:
- Frame padding/cropping for arbitrary resolution support
- Bitstream I/O for file serialization
- Video dataset utilities
- Logging utilities
"""

from src.video_compression.utils.bitstream import (
    BitstreamHeader,
    BitstreamReader,
    BitstreamWriter,
    EncodedFrame,
    FrameHeader,
    load_bitstream,
    save_bitstream,
)
from src.video_compression.utils.logging import (
    CodecLogContext,
    DecoderLogger,
    EncoderLogger,
    EncodingMetrics,
    LogLevel,
    configure_codec_logging,
    get_codec_logger,
    log_function_call,
)
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

__all__ = [
    # Padding
    "PadToMultiple",
    "DynamicPadding",
    "PaddingConfig",
    "PaddingInfo",
    "PaddingMode",
    "pad_to_multiple",
    "crop_to_original",
    "compute_padding",
    # Bitstream
    "BitstreamWriter",
    "BitstreamReader",
    "BitstreamHeader",
    "FrameHeader",
    "EncodedFrame",
    "save_bitstream",
    "load_bitstream",
    # Logging
    "get_codec_logger",
    "CodecLogContext",
    "EncoderLogger",
    "DecoderLogger",
    "EncodingMetrics",
    "configure_codec_logging",
    "log_function_call",
    "LogLevel",
]
