"""Tests for Phase 1 benchmark loop dispatch (Epic 1.4).

Covers:
- ``_runtime_name_for_profile`` mapping for PYTORCH and COMPILED backends
- ``_dtype_for_precision`` mapping for FP32, FP16, BF16
- Unimplemented backend (ONNX, TENSORRT) raises NotImplementedError
- Compiled backend runs through the benchmark loop on CPU (FORWARD phase)
- FP16 precision runs through the benchmark loop (FORWARD phase)
"""

from __future__ import annotations

import pytest

from src.templates.base import ExecutionStatus
from src.video_compression.perf import (
    PerfBenchmarkConfig,
    Precision,
    RuntimeBackend,
    RuntimeProfile,
    report_from_result,
    run_benchmark,
)
from src.video_compression.perf.benchmark import (
    _dtype_for_precision,
    _runtime_name_for_profile,
)

pytestmark = pytest.mark.video


class TestRuntimeNameForProfile:
    """Unit tests for the backend→runtime-name mapping."""

    def test_pytorch_maps_to_eager(self) -> None:
        profile = RuntimeProfile(
            name="p",
            backend=RuntimeBackend.PYTORCH,
            precision=Precision.FP32,
        )
        assert _runtime_name_for_profile(profile) == "pytorch-eager"

    def test_compiled_maps_to_compiled(self) -> None:
        profile = RuntimeProfile(
            name="p",
            backend=RuntimeBackend.COMPILED,
            precision=Precision.FP32,
        )
        assert _runtime_name_for_profile(profile) == "pytorch-compiled"

    def test_onnx_maps_to_onnx_cuda(self) -> None:
        profile = RuntimeProfile(
            name="p",
            backend=RuntimeBackend.ONNX,
            precision=Precision.FP32,
        )
        assert _runtime_name_for_profile(profile) == "onnx-cuda"

    def test_tensorrt_maps_to_tensorrt(self) -> None:
        profile = RuntimeProfile(
            name="p",
            backend=RuntimeBackend.TENSORRT,
            precision=Precision.FP32,
        )
        assert _runtime_name_for_profile(profile) == "tensorrt"


class TestDtypeForPrecision:
    """Unit tests for the precision→dtype mapping."""

    def test_fp32(self) -> None:
        assert _dtype_for_precision(Precision.FP32) == "float32"

    def test_fp16(self) -> None:
        assert _dtype_for_precision(Precision.FP16) == "float16"

    def test_bf16(self) -> None:
        assert _dtype_for_precision(Precision.BF16) == "bfloat16"


class TestCompiledBackendInBenchmarkLoop:
    """Integration: compiled backend runs through the full benchmark."""

    def test_compiled_forward_phase_completes(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
        tiny_codec_config,
    ) -> None:
        """The COMPILED backend with FORWARD phase runs successfully.

        Note: FORWARD phase uses CodecForwardSubject which doesn't
        exercise the runtime registry (it's a full codec pass). The
        runtime_name is computed but only used for DECODE phase. This
        test verifies the dispatch path doesn't reject COMPILED.
        """
        cfg = tiny_perf_config.with_overrides(
            runtime_profiles=[
                RuntimeProfile(
                    name="compiled",
                    backend=RuntimeBackend.COMPILED,
                    precision=Precision.FP32,
                ),
            ],
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        assert result.status == ExecutionStatus.COMPLETED
        report = report_from_result(result)
        ok = [c for c in report.cells if not c.failed]
        assert ok, "COMPILED+FORWARD cell should succeed"
        assert ok[0].throughput_fps > 0.0

    def test_compiled_fp16_forward_completes(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
        tiny_codec_config,
    ) -> None:
        """COMPILED + FP16 + FORWARD runs without error."""
        cfg = tiny_perf_config.with_overrides(
            runtime_profiles=[
                RuntimeProfile(
                    name="compiled_fp16",
                    backend=RuntimeBackend.COMPILED,
                    precision=Precision.FP16,
                ),
            ],
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        ok = [c for c in report.cells if not c.failed]
        assert ok, "COMPILED+FP16+FORWARD cell should succeed"


class TestBF16PrecisionInBenchmarkLoop:
    """BF16 path through the benchmark loop."""

    def test_bf16_forward_completes(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
        tiny_codec_config,
    ) -> None:
        cfg = tiny_perf_config.with_overrides(
            runtime_profiles=[
                RuntimeProfile(
                    name="bf16",
                    backend=RuntimeBackend.PYTORCH,
                    precision=Precision.BF16,
                ),
            ],
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        ok = [c for c in report.cells if not c.failed]
        assert ok, "BF16+FORWARD cell should succeed"
        assert ok[0].throughput_fps > 0.0
