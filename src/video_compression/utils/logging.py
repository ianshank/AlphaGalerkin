"""Structured logging utilities for video compression codec.

Provides consistent logging across all codec components with:
- Context binding for debugging
- Timing utilities for performance analysis
- Metric logging for quality tracking
- Function decorators for automatic logging
"""

from __future__ import annotations

import functools
import logging
import time
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import structlog

F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(str, Enum):
    """Log levels for codec logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def get_codec_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger for codec component.

    Args:
        name: Component name (e.g., "encoder", "entropy").

    Returns:
        Configured structlog logger.

    """
    return structlog.get_logger(f"video_compression.{name}")


def log_function_call(
    logger_name: str = "codec",
    level: LogLevel = LogLevel.DEBUG,
    include_args: bool = True,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function calls with timing.

    Args:
        logger_name: Name for the logger.
        level: Log level for the message.
        include_args: Whether to log function arguments.
        include_result: Whether to log return value.

    Returns:
        Decorated function.

    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_codec_logger(logger_name)
            log_data: dict[str, Any] = {"function": func.__name__}

            if include_args and (args or kwargs):
                # Limit arg representation to avoid huge logs
                log_data["args_count"] = len(args)
                log_data["kwargs_keys"] = list(kwargs.keys())

            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start

                log_data["elapsed_ms"] = round(elapsed * 1000, 2)
                log_data["status"] = "success"

                if include_result and result is not None:
                    log_data["result_type"] = type(result).__name__

                getattr(logger, level.value.lower())("call", **log_data)
                return result

            except Exception as e:
                elapsed = time.perf_counter() - start
                log_data["elapsed_ms"] = round(elapsed * 1000, 2)
                log_data["status"] = "error"
                log_data["error"] = str(e)
                log_data["error_type"] = type(e).__name__

                logger.error("call_failed", **log_data)
                raise

        return wrapper  # type: ignore

    return decorator


@dataclass
class CodecLogContext:
    """Context manager for codec logging with metrics.

    Tracks timing, frame counts, and quality metrics across
    encoding/decoding operations.
    """

    name: str
    logger: structlog.BoundLogger = field(init=False)

    # Timing
    start_time: float = field(default=0.0, init=False)
    total_time: float = field(default=0.0, init=False)

    # Counters
    frames_processed: int = field(default=0, init=False)
    total_bits: int = field(default=0, init=False)

    # Quality metrics
    psnr_sum: float = field(default=0.0, init=False)
    ssim_sum: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize logger."""
        self.logger = get_codec_logger(self.name)

    def __enter__(self) -> CodecLogContext:
        """Start timing context."""
        self.start_time = time.perf_counter()
        self.logger.info("starting", context=self.name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """End timing context and log summary."""
        self.total_time = time.perf_counter() - self.start_time

        summary = {
            "total_time_s": round(self.total_time, 3),
            "frames": self.frames_processed,
        }

        if self.frames_processed > 0:
            summary["fps"] = round(self.frames_processed / self.total_time, 2)
            summary["avg_bits_per_frame"] = self.total_bits // self.frames_processed
            summary["avg_psnr"] = round(self.psnr_sum / self.frames_processed, 2)

        if exc_type is not None:
            self.logger.error(
                "failed",
                error=str(exc_val),
                **summary,
            )
        else:
            self.logger.info("completed", **summary)

    def log_frame(
        self,
        frame_idx: int,
        bits: int,
        psnr: float,
        ssim: float | None = None,
        **extra: Any,
    ) -> None:
        """Log per-frame metrics.

        Args:
            frame_idx: Frame index.
            bits: Encoded bits.
            psnr: PSNR in dB.
            ssim: Optional SSIM value.
            **extra: Additional metrics.

        """
        self.frames_processed += 1
        self.total_bits += bits
        self.psnr_sum += psnr
        if ssim is not None:
            self.ssim_sum += ssim

        self.logger.debug(
            "frame",
            idx=frame_idx,
            bits=bits,
            psnr=round(psnr, 2),
            ssim=round(ssim, 4) if ssim else None,
            **extra,
        )

    @contextmanager
    def timed(self, operation: str) -> Iterator[None]:
        """Time a sub-operation.

        Args:
            operation: Operation name.

        Yields:
            None.

        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.logger.debug(
                "timed",
                operation=operation,
                elapsed_ms=round(elapsed * 1000, 2),
            )


def configure_codec_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure logging for video compression module.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: Use JSON output format.

    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
    )


@dataclass
class EncodingMetrics:
    """Container for encoding metrics collection."""

    frame_count: int = 0
    total_bits: int = 0
    total_mse: float = 0.0
    total_psnr: float = 0.0
    total_ssim: float = 0.0
    i_frame_count: int = 0
    p_frame_count: int = 0
    b_frame_count: int = 0

    def update(
        self,
        bits: int,
        mse: float,
        frame_type: str = "P",
        ssim: float | None = None,
    ) -> None:
        """Update metrics with new frame.

        Args:
            bits: Bits for this frame.
            mse: MSE distortion.
            frame_type: Frame type (I, P, B).
            ssim: Optional SSIM value.

        """
        import math

        self.frame_count += 1
        self.total_bits += bits
        self.total_mse += mse

        # PSNR from MSE
        psnr = -10 * math.log10(mse) if mse > 0 else 100.0
        self.total_psnr += psnr

        if ssim is not None:
            self.total_ssim += ssim

        # Frame type counts
        if frame_type == "I":
            self.i_frame_count += 1
        elif frame_type == "P":
            self.p_frame_count += 1
        else:
            self.b_frame_count += 1

    def summary(self) -> dict[str, float]:
        """Get summary statistics.

        Returns:
            Dictionary of average metrics.

        """
        if self.frame_count == 0:
            return {}

        return {
            "total_frames": self.frame_count,
            "total_bits": self.total_bits,
            "avg_bits": self.total_bits / self.frame_count,
            "avg_psnr": self.total_psnr / self.frame_count,
            "avg_ssim": self.total_ssim / self.frame_count if self.total_ssim > 0 else 0.0,
            "i_frames": self.i_frame_count,
            "p_frames": self.p_frame_count,
            "b_frames": self.b_frame_count,
        }


class EncoderLogger:
    """Specialized logger for encoder operations."""

    def __init__(self, name: str = "encoder") -> None:
        """Initialize encoder logger.

        Args:
            name: Logger name.

        """
        self.logger = get_codec_logger(name)
        self.metrics = EncodingMetrics()

    def log_encode_start(
        self,
        num_frames: int,
        width: int,
        height: int,
        **extra: Any,
    ) -> None:
        """Log encoding start.

        Args:
            num_frames: Number of frames to encode.
            width: Frame width.
            height: Frame height.
            **extra: Additional parameters.

        """
        self.metrics = EncodingMetrics()  # Reset
        self.logger.info(
            "encode_start",
            num_frames=num_frames,
            resolution=f"{width}x{height}",
            **extra,
        )

    def log_frame(
        self,
        frame_idx: int,
        frame_type: str,
        bits: int,
        mse: float,
        qp: int,
        **extra: Any,
    ) -> None:
        """Log frame encoding.

        Args:
            frame_idx: Frame index.
            frame_type: Frame type (I, P, B).
            bits: Encoded bits.
            mse: MSE distortion.
            qp: QP value used.
            **extra: Additional metrics.

        """
        import math

        psnr = -10 * math.log10(mse + 1e-10)
        self.metrics.update(bits, mse, frame_type)

        self.logger.debug(
            "frame_encoded",
            idx=frame_idx,
            type=frame_type,
            bits=bits,
            psnr=round(psnr, 2),
            qp=qp,
            **extra,
        )

    def log_encode_complete(self, elapsed_s: float) -> dict[str, float]:
        """Log encoding completion.

        Args:
            elapsed_s: Total elapsed time in seconds.

        Returns:
            Summary statistics.

        """
        summary = self.metrics.summary()
        summary["elapsed_s"] = round(elapsed_s, 2)
        summary["fps"] = round(self.metrics.frame_count / elapsed_s, 2) if elapsed_s > 0 else 0.0

        self.logger.info("encode_complete", **summary)
        return summary


class DecoderLogger:
    """Specialized logger for decoder operations."""

    def __init__(self, name: str = "decoder") -> None:
        """Initialize decoder logger.

        Args:
            name: Logger name.

        """
        self.logger = get_codec_logger(name)
        self.frame_count = 0

    def log_decode_start(
        self,
        num_frames: int,
        width: int,
        height: int,
        **extra: Any,
    ) -> None:
        """Log decoding start.

        Args:
            num_frames: Number of frames to decode.
            width: Frame width.
            height: Frame height.
            **extra: Additional parameters.

        """
        self.frame_count = 0
        self.logger.info(
            "decode_start",
            num_frames=num_frames,
            resolution=f"{width}x{height}",
            **extra,
        )

    def log_frame(
        self,
        frame_idx: int,
        frame_type: str,
        **extra: Any,
    ) -> None:
        """Log frame decoding.

        Args:
            frame_idx: Frame index.
            frame_type: Frame type.
            **extra: Additional info.

        """
        self.frame_count += 1
        self.logger.debug(
            "frame_decoded",
            idx=frame_idx,
            type=frame_type,
            **extra,
        )

    def log_decode_complete(self, elapsed_s: float) -> None:
        """Log decoding completion.

        Args:
            elapsed_s: Total elapsed time.

        """
        fps = self.frame_count / elapsed_s if elapsed_s > 0 else 0.0
        self.logger.info(
            "decode_complete",
            frames=self.frame_count,
            elapsed_s=round(elapsed_s, 2),
            fps=round(fps, 2),
        )
