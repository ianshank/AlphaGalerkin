"""Coverage tests for video compression logging utilities.

Targets uncovered lines in src/video_compression/utils/logging.py:
    - LogLevel enum
    - get_codec_logger
    - log_function_call decorator (success/error paths)
    - CodecLogContext (enter/exit, log_frame, timed)
    - configure_codec_logging (text and JSON modes)
    - EncodingMetrics (update, summary with various frame types)
    - EncoderLogger (log_encode_start, log_frame, log_encode_complete)
    - DecoderLogger (log_decode_start, log_frame, log_decode_complete)
"""

from __future__ import annotations

import pytest

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

# ---------------------------------------------------------------------------
# LogLevel
# ---------------------------------------------------------------------------


class TestLogLevel:
    def test_values(self) -> None:
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"


# ---------------------------------------------------------------------------
# get_codec_logger
# ---------------------------------------------------------------------------


class TestGetCodecLogger:
    def test_returns_logger(self) -> None:
        logger = get_codec_logger("test_component")
        assert logger is not None


# ---------------------------------------------------------------------------
# log_function_call decorator
# ---------------------------------------------------------------------------


class TestLogFunctionCall:
    def test_success_path(self) -> None:
        @log_function_call("test", level=LogLevel.DEBUG, include_args=True, include_result=True)
        def add(a: int, b: int) -> int:
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_error_path(self) -> None:
        @log_function_call("test", level=LogLevel.DEBUG)
        def fail() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            fail()

    def test_no_args_logging(self) -> None:
        @log_function_call("test", include_args=False)
        def noop() -> str:
            return "ok"

        assert noop() == "ok"


# ---------------------------------------------------------------------------
# CodecLogContext
# ---------------------------------------------------------------------------


class TestCodecLogContext:
    def test_basic_context(self) -> None:
        ctx = CodecLogContext(name="test_ctx")
        with ctx:
            assert ctx.start_time > 0

        assert ctx.total_time > 0
        assert ctx.frames_processed == 0

    def test_log_frame(self) -> None:
        ctx = CodecLogContext(name="frame_test")
        with ctx:
            ctx.log_frame(frame_idx=0, bits=1000, psnr=35.0, ssim=0.95)
            ctx.log_frame(frame_idx=1, bits=800, psnr=33.0)

        assert ctx.frames_processed == 2
        assert ctx.total_bits == 1800
        assert ctx.psnr_sum == pytest.approx(68.0)
        assert ctx.ssim_sum == pytest.approx(0.95)

    def test_timed_sub_operation(self) -> None:
        ctx = CodecLogContext(name="timed_test")
        with ctx:
            with ctx.timed("sub_op"):
                x = sum(range(100))
        assert ctx.total_time > 0

    def test_exit_on_exception(self) -> None:
        ctx = CodecLogContext(name="error_test")
        with pytest.raises(RuntimeError):
            with ctx:
                ctx.log_frame(frame_idx=0, bits=500, psnr=30.0)
                raise RuntimeError("test error")

        # Should still record total_time
        assert ctx.total_time > 0
        assert ctx.frames_processed == 1


# ---------------------------------------------------------------------------
# configure_codec_logging
# ---------------------------------------------------------------------------


class TestConfigureCodecLogging:
    def test_configure_text(self) -> None:
        configure_codec_logging(level="DEBUG", json_output=False)

    def test_configure_json(self) -> None:
        configure_codec_logging(level="INFO", json_output=True)


# ---------------------------------------------------------------------------
# EncodingMetrics
# ---------------------------------------------------------------------------


class TestEncodingMetrics:
    def test_empty_summary(self) -> None:
        metrics = EncodingMetrics()
        assert metrics.summary() == {}

    def test_update_and_summary(self) -> None:
        metrics = EncodingMetrics()
        metrics.update(bits=1000, mse=0.01, frame_type="I")
        metrics.update(bits=500, mse=0.02, frame_type="P", ssim=0.95)
        metrics.update(bits=300, mse=0.03, frame_type="B")

        summary = metrics.summary()
        assert summary["total_frames"] == 3
        assert summary["total_bits"] == 1800
        assert summary["i_frames"] == 1
        assert summary["p_frames"] == 1
        assert summary["b_frames"] == 1
        assert summary["avg_bits"] == 600.0
        assert summary["avg_psnr"] > 0

    def test_zero_mse_psnr(self) -> None:
        metrics = EncodingMetrics()
        metrics.update(bits=100, mse=0.0, frame_type="I")
        summary = metrics.summary()
        assert summary["avg_psnr"] == 100.0  # Clamped for zero MSE


# ---------------------------------------------------------------------------
# EncoderLogger
# ---------------------------------------------------------------------------


class TestEncoderLogger:
    def test_full_workflow(self) -> None:
        enc_logger = EncoderLogger(name="test_encoder")
        enc_logger.log_encode_start(num_frames=10, width=1920, height=1080)

        enc_logger.log_frame(frame_idx=0, frame_type="I", bits=5000, mse=0.01, qp=22)
        enc_logger.log_frame(frame_idx=1, frame_type="P", bits=2000, mse=0.02, qp=26)

        summary = enc_logger.log_encode_complete(elapsed_s=1.5)
        assert summary["total_frames"] == 2
        assert summary["elapsed_s"] == 1.5
        assert summary["fps"] > 0

    def test_reset_on_new_start(self) -> None:
        enc_logger = EncoderLogger()
        enc_logger.log_frame(frame_idx=0, frame_type="I", bits=1000, mse=0.01, qp=22)
        assert enc_logger.metrics.frame_count == 1

        enc_logger.log_encode_start(num_frames=5, width=640, height=480)
        assert enc_logger.metrics.frame_count == 0


# ---------------------------------------------------------------------------
# DecoderLogger
# ---------------------------------------------------------------------------


class TestDecoderLogger:
    def test_full_workflow(self) -> None:
        dec_logger = DecoderLogger(name="test_decoder")
        dec_logger.log_decode_start(num_frames=5, width=1920, height=1080)

        dec_logger.log_frame(frame_idx=0, frame_type="I")
        dec_logger.log_frame(frame_idx=1, frame_type="P")

        assert dec_logger.frame_count == 2

        dec_logger.log_decode_complete(elapsed_s=0.5)

    def test_zero_elapsed(self) -> None:
        dec_logger = DecoderLogger()
        dec_logger.log_decode_start(num_frames=1, width=64, height=64)
        dec_logger.log_decode_complete(elapsed_s=0.0)
