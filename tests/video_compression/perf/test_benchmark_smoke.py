"""End-to-end smoke tests for ``PerfBenchmark``.

These exercise the full benchmark pipeline against the real codec at
tiny resolutions. They run on CPU and complete in <30 seconds — they're
intended for CI gating, not headline measurements.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.templates.base import ExecutionStatus
from src.video_compression.perf import (
    BaselineRegistry,
    BenchmarkPhase,
    PerfBenchmark,
    PerfBenchmarkConfig,
    Precision,
    ResolutionSpec,
    RuntimeBackend,
    RuntimeProfile,
    baseline_from_report,
    report_from_result,
    run_benchmark,
)

pytestmark = pytest.mark.video


# -------------------------------------------------------------- happy path


class TestBenchmarkSmoke:
    def test_runs_to_completion(self, tiny_perf_config, tiny_codec_config) -> None:
        result = run_benchmark(tiny_perf_config, codec_config=tiny_codec_config)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.metrics["n_cells_failed"] == 0.0
        assert result.metrics["n_cells_ok"] >= 1

    def test_report_artifact_present(self, tiny_perf_config, tiny_codec_config) -> None:
        result = run_benchmark(tiny_perf_config, codec_config=tiny_codec_config)
        report = report_from_result(result)
        assert report.benchmark_id == result.run_id
        assert report.config_hash == tiny_perf_config.compute_hash()
        assert len(report.cells) == 1
        cell = report.cells[0]
        assert not cell.failed
        assert cell.throughput_fps > 0.0
        assert cell.latency_stats.count == tiny_perf_config.n_repeats

    def test_writes_json_when_output_path_set(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
        tiny_codec_config,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "report.json"
        cfg = tiny_perf_config.with_overrides(output_path=str(out))
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        assert out.exists()
        # Persisted JSON is well-formed and re-loadable
        loaded = json.loads(out.read_text())
        assert loaded["benchmark_id"] == result.run_id
        assert loaded["cells"][0]["throughput_fps"] > 0.0


# ----------------------------------------------------------- sweep coverage


class TestSweepCoverage:
    def test_multiple_resolutions(self, tiny_perf_config, tiny_codec_config) -> None:
        cfg = tiny_perf_config.with_overrides(
            resolutions=[
                ResolutionSpec(name="r16", label="16x16", height=16, width=16),
                ResolutionSpec(name="r32", label="32x32", height=32, width=32),
            ],
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        labels = {c.resolution_label for c in report.cells}
        assert labels == {"16x16", "32x32"}

    def test_multiple_batch_sizes(self, tiny_perf_config, tiny_codec_config) -> None:
        cfg = tiny_perf_config.with_overrides(batch_sizes=[1, 2])
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        batches = {c.batch_size for c in report.cells}
        assert batches == {1, 2}


# ----------------------------------------------------- error / skip handling


class TestErrorPaths:
    def test_resolution_not_divisible_by_downsample(
        self,
        tiny_perf_config,
        tiny_codec_config,
    ) -> None:
        # Codec downsample=4; pick height not divisible by 4
        cfg = tiny_perf_config.with_overrides(
            resolutions=[
                ResolutionSpec(name="bad", label="17x17", height=17, width=17),
            ],
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        assert any(c.failed for c in report.cells)
        assert any("downsample_factor" in (c.failure_reason or "") for c in report.cells)

    def test_unimplemented_phase_marked_skipped(
        self,
        tiny_perf_config,
        tiny_codec_config,
    ) -> None:
        cfg = tiny_perf_config.with_overrides(phases=[BenchmarkPhase.ENCODE])
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        assert all(c.failed for c in report.cells)
        assert all("not implemented" in (c.failure_reason or "") for c in report.cells)

    def test_unimplemented_backend_marked_skipped(
        self,
        tiny_perf_config,
        tiny_codec_config,
    ) -> None:
        cfg = tiny_perf_config.with_overrides(
            runtime_profiles=[
                RuntimeProfile(
                    name="onnx",
                    backend=RuntimeBackend.ONNX,
                    precision=Precision.FP32,
                ),
            ],
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        report = report_from_result(result)
        assert all(c.failed for c in report.cells)

    def test_fail_fast_propagates(self, tiny_perf_config, tiny_codec_config) -> None:
        cfg = tiny_perf_config.with_overrides(
            phases=[BenchmarkPhase.ENCODE],  # not implemented
            fail_fast=True,
        )
        bench = PerfBenchmark(config=cfg, codec_config=tiny_codec_config)
        # ``run`` catches exceptions; we expect a FAILED result, not a raise.
        result = bench.run()
        assert result.status == ExecutionStatus.FAILED


# ------------------------------------------------------- regression gating


class TestRegressionGate:
    def test_gate_passes_on_self_baseline(
        self,
        tiny_perf_config,
        tiny_codec_config,
        tmp_path: Path,
    ) -> None:
        # First run records a baseline
        first = run_benchmark(tiny_perf_config, codec_config=tiny_codec_config)
        report = report_from_result(first)
        baseline_path = tmp_path / "baseline.json"
        BaselineRegistry(baseline_from_report(report, hardware_tag="ci-self")).save(baseline_path)

        # Second run with very loose tolerance (CI is noisy) must pass.
        cfg = tiny_perf_config.with_overrides(
            baseline_path=str(baseline_path),
            regression_tolerance_pct=99.0,
        )
        second = run_benchmark(cfg, codec_config=tiny_codec_config)
        # Status COMPLETED means no regression detected at this tolerance.
        assert second.status == ExecutionStatus.COMPLETED
        assert "regression" in second.artifacts
        assert second.artifacts["regression"]["n_regressions"] == 0

    def test_synthetic_baseline_with_zero_throughput_triggers_failure(
        self,
        tiny_perf_config,
        tiny_codec_config,
        tmp_path: Path,
    ) -> None:
        # Run once to get the report shape, then build a forged baseline
        # that claims 10x higher throughput. The new run must regress.
        first = run_benchmark(tiny_perf_config, codec_config=tiny_codec_config)
        first_report = report_from_result(first)
        # Inflate every entry's throughput so the live run can't keep up
        forged_doc = baseline_from_report(first_report)
        forged_doc = forged_doc.with_overrides(
            entries=[
                e.with_overrides(throughput_fps=e.throughput_fps * 10.0) for e in forged_doc.entries
            ]
        )
        baseline_path = tmp_path / "forged.json"
        BaselineRegistry(forged_doc).save(baseline_path)

        cfg = tiny_perf_config.with_overrides(
            baseline_path=str(baseline_path),
            regression_tolerance_pct=5.0,
        )
        second = run_benchmark(cfg, codec_config=tiny_codec_config)
        assert second.status == ExecutionStatus.FAILED
        assert second.artifacts["regression"]["n_regressions"] >= 1

    def test_missing_baseline_path_warns_not_fails(
        self,
        tiny_perf_config,
        tiny_codec_config,
        tmp_path: Path,
    ) -> None:
        cfg = tiny_perf_config.with_overrides(
            baseline_path=str(tmp_path / "nonexistent.json"),
        )
        result = run_benchmark(cfg, codec_config=tiny_codec_config)
        # No baseline available means no gate; benchmark itself succeeded.
        assert result.status == ExecutionStatus.COMPLETED


# ------------------------------------------------------ result serialization


class TestSerialization:
    def test_report_json_round_trip(
        self,
        tiny_perf_config,
        tiny_codec_config,
        tmp_path: Path,
    ) -> None:
        result = run_benchmark(tiny_perf_config, codec_config=tiny_codec_config)
        report = report_from_result(result)
        as_json = report.to_json()
        decoded = json.loads(as_json)
        # All cell keys round-trip through JSON
        keys_via_json = {c["cell_key"] for c in decoded["cells"]}
        keys_direct = {c.cell_key for c in report.cells}
        assert keys_via_json == keys_direct


# ---------------------------------------------------- defensive error paths
#
# Coverage fill-ins added in the follow-up PR off PR #75. These exercise
# the three reachable defensive-raise paths inside ``PerfBenchmark`` /
# ``report_from_result`` that the original test suite left at 97% on
# benchmark.py. The remaining uncovered lines (config.py:98 / 267,
# device.py:37 / 72, subjects.py:167) are dead code shielded by Pydantic
# Field bounds (``ge=16``, ``ge=1``, exhaustive enum check) — leaving
# alone rather than monkey-patching the validators to force coverage.


class TestDefensiveRaisePaths:
    """Cover the three reachable raise-paths inside the benchmark loop.

    These are essential safety nets, not edge cases — every one of them
    exists because the benchmark must be defensive about misconfigurations
    that would otherwise corrupt downstream artifacts. Lock them in.
    """

    def test_fail_fast_propagates_arbitrary_exception(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
        tiny_codec_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """benchmark.py:283 — non-NotImplementedError + fail_fast=True re-raises.

        The existing test_fail_fast_propagates uses ENCODE phase which
        triggers NotImplementedError, hitting a different except clause.
        This test forces a generic RuntimeError via a mock subject so
        the BLE001-marked except branch executes.
        """
        from unittest.mock import patch

        from src.video_compression.perf.benchmark import PerfBenchmark
        from src.video_compression.perf.subjects import BenchmarkSubject

        cfg = tiny_perf_config.with_overrides(fail_fast=True)
        bench = PerfBenchmark(config=cfg, codec_config=tiny_codec_config)

        # Force ``create_subject`` to return a subject whose ``prepare()``
        # raises a synthetic RuntimeError. This must NOT be a subclass of
        # NotImplementedError or the operator would route into the
        # already-tested skip path; the BLE001 ``except Exception`` branch
        # at benchmark.py:281-283 only fires for "real" runtime failures.
        class _RaisingSubject(BenchmarkSubject):
            def prepare(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("synthetic prepare-time failure for fail_fast test")

            def step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("step should not be reached")

            def teardown(self) -> None:
                return None

        with patch(
            "src.video_compression.perf.benchmark.create_subject",
            return_value=_RaisingSubject(),
        ):
            # The BaseExecutable.run() wrapper converts the raise into a
            # FAILED ExecutionResult — same outward contract as the
            # NotImplementedError path, but the inner ``raise`` at line
            # 283 IS hit, which is what we're locking in.
            result = bench.run()

        assert result.status == ExecutionStatus.FAILED

    def test_non_fp32_precision_runs_successfully(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
        tiny_codec_config,  # type: ignore[no-untyped-def]
    ) -> None:
        """Phase 1 activated FP16/BF16 in the benchmark loop.

        Mixed precision is now supported via the runtime's autocast.
        For FORWARD-phase subjects the benchmark runs the codec at FP32
        (the subject itself doesn't use the runtime registry), but the
        cell should complete without errors.
        """
        cfg = tiny_perf_config.with_overrides(
            runtime_profiles=[
                RuntimeProfile(
                    name="fp16_stub",
                    backend=RuntimeBackend.PYTORCH,
                    precision=Precision.FP16,
                )
            ],
        )
        bench = PerfBenchmark(config=cfg, codec_config=tiny_codec_config)
        result = bench.run()
        report = report_from_result(result)
        # FP16 cells should now complete successfully
        ok_cells = [c for c in report.cells if not c.failed]
        assert ok_cells, "FP16 cell should succeed now that Phase 1 is active"
        assert ok_cells[0].throughput_fps > 0.0

    def test_report_from_result_rejects_missing_artifact(
        self,
        tiny_perf_config: PerfBenchmarkConfig,
    ) -> None:
        """benchmark.py:579 -- ``report_from_result`` raises on bad input.

        Specifically: an ``ExecutionResult`` that wasn't produced by
        ``PerfBenchmark``. The function is a public utility; downstream
        tools must get a clear error rather than a silent ``KeyError``
        at the call site when they pass the wrong result type.
        """
        from src.templates.base import ExecutionResult

        # Construct an ExecutionResult that lacks the ``"report"`` key
        # in artifacts. This mirrors the case where another BaseExecutable
        # subclass result is mistakenly passed to ``report_from_result``.
        bogus = ExecutionResult(
            run_id="not-a-perf-run",
            config_hash=tiny_perf_config.compute_hash(),
            status=ExecutionStatus.COMPLETED,
            artifacts={},  # crucially: no "report" key
        )
        with pytest.raises(KeyError, match="report"):
            report_from_result(bogus)
