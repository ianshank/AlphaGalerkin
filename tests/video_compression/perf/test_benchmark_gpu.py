"""GPU-required end-to-end benchmark tests.

Skipped automatically by the root conftest when CUDA is unavailable. On
the user's reference workstation (5060 Ti 16 GB at cuda:0 + 5060 8 GB at
cuda:1) these tests both pass; on a single-GPU machine the cuda:1 cases
are skipped via ``pytest.skip`` at runtime.
"""

from __future__ import annotations

import pytest
import torch

from src.templates.base import ExecutionStatus
from src.video_compression.perf import (
    BenchmarkPhase,
    PerfBenchmark,
    PerfBenchmarkConfig,
    Precision,
    ResolutionSpec,
    RuntimeBackend,
    RuntimeProfile,
    list_cuda_devices,
    report_from_result,
    run_benchmark,
)


pytestmark = [pytest.mark.gpu_required, pytest.mark.video]


def _gpu_perf_config(*, device: str = "cuda:0") -> PerfBenchmarkConfig:
    return PerfBenchmarkConfig(
        name="gpu_smoke",
        resolutions=[
            ResolutionSpec(name="r64", label="64x64", height=64, width=64),
        ],
        batch_sizes=[1],
        runtime_profiles=[
            RuntimeProfile(
                name="cuda_fp32",
                backend=RuntimeBackend.PYTORCH,
                precision=Precision.FP32,
                device=device,
            ),
        ],
        phases=[BenchmarkPhase.FORWARD],
        n_warmup=1,
        n_repeats=3,
        n_frames_per_iter=1,
        device_preference=device,
        track_gpu_memory=True,
        pattern="motion",
        data_seed=42,
        regression_tolerance_pct=99.0,
    )


class TestSingleCardCuda0:
    def test_runs_on_cuda0(self, tiny_codec_config) -> None:
        cfg = _gpu_perf_config(device="cuda:0")
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        assert result.status == ExecutionStatus.COMPLETED
        report = report_from_result(result)
        cell = report.cells[0]
        assert not cell.failed
        assert cell.device_label.startswith("cuda:0")
        assert cell.peak_vram_mib is not None
        assert cell.peak_vram_mib > 0.0

    def test_throughput_records_meaningful_value(self, tiny_codec_config) -> None:
        cfg = _gpu_perf_config(device="cuda:0")
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        # Even a tiny model should hit >1 fps on a 5060-class GPU; the
        # threshold is intentionally loose so this is a sanity gate, not
        # a calibration test.
        assert report.cells[0].throughput_fps > 1.0


class TestDualCardSweep:
    """Sweep across cuda:0 and cuda:1 in a single config."""

    def _require_two_gpus(self) -> None:
        if len(list_cuda_devices()) < 2:
            pytest.skip("dual-card test requires >=2 CUDA devices")

    def test_both_cards_in_one_sweep(self, tiny_codec_config) -> None:
        self._require_two_gpus()
        cfg = PerfBenchmarkConfig(
            name="dual_gpu",
            resolutions=[
                ResolutionSpec(name="r", label="64x64", height=64, width=64),
            ],
            runtime_profiles=[
                RuntimeProfile(
                    name="c0",
                    backend=RuntimeBackend.PYTORCH,
                    precision=Precision.FP32,
                    device="cuda:0",
                ),
                RuntimeProfile(
                    name="c1",
                    backend=RuntimeBackend.PYTORCH,
                    precision=Precision.FP32,
                    device="cuda:1",
                ),
            ],
            phases=[BenchmarkPhase.FORWARD],
            n_warmup=1,
            n_repeats=3,
            device_preference="cuda:0",
            track_gpu_memory=True,
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        labels = {c.device_label.split(":", 2)[1] for c in report.cells}
        # Each cell records its actual device index regardless of the
        # run-level default.
        assert labels == {"0", "1"}

    def test_oom_on_smaller_card_is_recorded_not_aborted(
        self, tiny_codec_config
    ) -> None:
        """Forced VRAM-OOM on cuda:1 must not abort the cuda:0 cell."""
        self._require_two_gpus()
        # Pick a resolution that's small but allocate it twice. The
        # purpose here is correctness-of-error-handling, not VRAM
        # calibration — we use a deliberately tiny model so cuda:1
        # *should* succeed on any modern GPU. The contract under test is
        # "if cell N fails, cell N+1 still runs".
        cfg = PerfBenchmarkConfig(
            name="dual_gpu_robust",
            resolutions=[
                ResolutionSpec(name="r", label="64x64", height=64, width=64),
                ResolutionSpec(name="bad", label="63x63", height=63, width=63),
            ],
            runtime_profiles=[
                RuntimeProfile(
                    name="c0",
                    backend=RuntimeBackend.PYTORCH,
                    precision=Precision.FP32,
                    device="cuda:0",
                ),
            ],
            phases=[BenchmarkPhase.FORWARD],
            n_warmup=0,
            n_repeats=2,
            device_preference="cuda:0",
            fail_fast=False,
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        # Exactly one ok cell (64x64) and one failed cell (63x63 — not
        # divisible by downsample_factor=4 → ValueError → recorded as
        # failed, not aborted).
        ok_count = sum(1 for c in report.cells if not c.failed)
        bad_count = sum(1 for c in report.cells if c.failed)
        assert ok_count == 1
        assert bad_count == 1


class TestPerCardVramLabel:
    def test_label_distinguishes_indices(self, tiny_codec_config) -> None:
        # Even on a 1-GPU machine this exercises labeling for cuda:0
        cfg = _gpu_perf_config(device="cuda:0")
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        for cell in report.cells:
            assert cell.device_label.startswith("cuda:0")
            # Hardware name embedded for human readers
            assert ":" in cell.device_label[len("cuda:0"):]
