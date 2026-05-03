"""Tests for the GpuUtilizationProfiler context manager.

Validates the dmon-output parser and the no-nvidia-smi-binary fallback so
the profiler is safe to wire into every PINN solve regardless of host.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.research.gpu_profiler import (
    GpuUtilizationProfiler,
    GpuUtilizationReport,
    parse_dmon_output,
)

# Sample `nvidia-smi dmon -s pucvmt` output (no -o T timestamp prefix).
# First two lines are header (comment + column names); subsequent rows
# are data. Columns: gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk
_DMON_FIXTURE = """\
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa
# Idx      W      C      C      %      %      %      %      %      %
    0    125     62     58     87     74      0      0      0      0
    0    132     63     59     91     78      0      0      0      0
    0    128     62     58     85     72      0      0      0      0
"""


# Variant with FB-memory column at index 8 (driver-dependent).
_DMON_WITH_FB_MEM = """\
# gpu    pwr  gtemp  mtemp     sm    mem    enc    dec     fb
# Idx      W      C      C      %      %      %      %     MB
    0    125     62     58     87     74      0      0   6144
    0    132     63     59     91     78      0      0   7680
    0    128     62     58     85     72      0      0   6900
"""


# ---------------------------------------------------------------------------
# parse_dmon_output
# ---------------------------------------------------------------------------


class TestParseDmonOutput:
    def test_parses_three_samples(self) -> None:
        report = parse_dmon_output(_DMON_FIXTURE, gpu_indices=(0,), sample_interval_s=1.0)
        assert report.total_samples == 3
        assert report.mean_sm_util_pct == pytest.approx((87 + 91 + 85) / 3)
        assert report.mean_mem_util_pct == pytest.approx((74 + 78 + 72) / 3)

    def test_parses_fb_memory_when_present(self) -> None:
        report = parse_dmon_output(_DMON_WITH_FB_MEM, gpu_indices=(0,), sample_interval_s=1.0)
        assert report.peak_memory_mib == 7680

    def test_skips_header_lines(self) -> None:
        # Two header lines + three data rows; total_samples must reflect data rows only.
        report = parse_dmon_output(_DMON_FIXTURE, gpu_indices=(0,), sample_interval_s=1.0)
        assert report.total_samples == 3

    def test_empty_output(self) -> None:
        report = parse_dmon_output("", gpu_indices=(0,), sample_interval_s=1.0)
        assert report.total_samples == 0
        assert report.mean_sm_util_pct is None
        assert report.peak_memory_mib is None

    def test_only_headers(self) -> None:
        text = "#header line 1\n#header line 2\n"
        report = parse_dmon_output(text, gpu_indices=(0,), sample_interval_s=1.0)
        assert report.total_samples == 0

    def test_malformed_row_skipped(self) -> None:
        text = _DMON_FIXTURE + "garbage row that doesn't parse\n"
        report = parse_dmon_output(text, gpu_indices=(0,), sample_interval_s=1.0)
        assert report.total_samples == 3  # garbage row skipped, others counted

    def test_records_gpu_indices_and_interval(self) -> None:
        report = parse_dmon_output(_DMON_FIXTURE, gpu_indices=(0, 1), sample_interval_s=2.5)
        assert report.gpu_indices == (0, 1)
        assert report.sample_interval_s == 2.5

    def test_to_dict_serialises_all_fields(self) -> None:
        report = parse_dmon_output(_DMON_WITH_FB_MEM, gpu_indices=(0,), sample_interval_s=1.0)
        d = report.to_dict()
        assert d["gpu_indices"] == [0]
        assert d["total_samples"] == 3
        assert d["peak_memory_mib"] == 7680
        assert "mean_sm_util_pct" in d


# ---------------------------------------------------------------------------
# GpuUtilizationProfiler context manager
# ---------------------------------------------------------------------------


class TestGpuUtilizationProfiler:
    def test_no_op_when_gpu_indices_empty(self) -> None:
        """Empty gpu_indices means CPU run — no subprocess spawned."""
        with GpuUtilizationProfiler(gpu_indices=[]) as prof:
            pass  # noqa: PASS101
        assert prof.report is not None
        assert prof.report.total_samples == 0
        assert prof.report.gpu_indices == ()

    def test_no_op_when_nvidia_smi_missing(self) -> None:
        """FileNotFoundError on Popen must produce a clean empty report."""
        with patch("subprocess.Popen", side_effect=FileNotFoundError("no nvidia-smi")):
            with GpuUtilizationProfiler(gpu_indices=[0]) as prof:
                pass  # noqa: PASS101
        assert prof.report is not None
        assert prof.report.total_samples == 0
        assert prof.report.gpu_indices == (0,)

    def test_subprocess_terminate_on_exit(self, tmp_path: Path) -> None:
        """Profiler must terminate the dmon subprocess on context exit."""
        captured = tmp_path / "dmon.out"
        captured.write_text(_DMON_FIXTURE, encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc):
            with GpuUtilizationProfiler(
                gpu_indices=[0],
                sample_interval_s=1.0,
                output_path=captured,
            ) as prof:
                pass  # noqa: PASS101

        mock_proc.terminate.assert_called_once()
        assert prof.report is not None
        # Output file pre-populated by the test → parser sees 3 samples.
        assert prof.report.total_samples == 3

    def test_kill_on_terminate_timeout(self, tmp_path: Path) -> None:
        """If terminate hangs, profiler must fall back to kill()."""
        captured = tmp_path / "dmon.out"
        captured.write_text(_DMON_FIXTURE, encoding="utf-8")

        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        # First wait raises Timeout, second wait succeeds
        mock_proc.wait = MagicMock(
            side_effect=[subprocess.TimeoutExpired(cmd="dmon", timeout=5.0), None]
        )
        mock_proc.kill = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc):
            with GpuUtilizationProfiler(
                gpu_indices=[0],
                output_path=captured,
            ):
                pass  # noqa: PASS101

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    def test_temp_file_cleaned_on_exit(self) -> None:
        """When output_path is None, the auto-temp file must be removed."""
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        captured_paths: list[Path] = []

        # Capture the temp path the profiler picks so we can assert it's gone.
        original_init = GpuUtilizationProfiler.__enter__

        def spy_enter(self: GpuUtilizationProfiler) -> GpuUtilizationProfiler:
            ret = original_init(self)
            if self._captured_path is not None:
                captured_paths.append(self._captured_path)
                # Pre-populate the file so parser produces a report
                self._captured_path.write_text(_DMON_FIXTURE, encoding="utf-8")
            return ret

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch.object(GpuUtilizationProfiler, "__enter__", spy_enter):
                with GpuUtilizationProfiler(gpu_indices=[0]) as prof:
                    pass  # noqa: PASS101

        assert prof.report is not None
        assert len(captured_paths) == 1
        assert not captured_paths[
            0
        ].exists(), f"Temp dmon file {captured_paths[0]} was not cleaned up"


# ---------------------------------------------------------------------------
# Profiler fallback
# ---------------------------------------------------------------------------


class TestGpuUtilizationProfilerFallback:
    """Guards the ``report is never None`` contract added 2026-05-03."""

    def test_report_set_when_dmon_file_missing(self) -> None:
        """If Popen succeeds but dmon never writes output, report falls back."""
        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc):
            with GpuUtilizationProfiler(gpu_indices=[0]) as prof:
                # Delete the temp file the profiler created but never had
                # the subprocess populate, so __exit__ skips parsing.
                if prof._captured_path is not None and prof._captured_path.exists():
                    prof._captured_path.unlink()
        assert prof.report is not None
        assert prof.report.total_samples == 0
        assert prof.report.gpu_indices == (0,)


# ---------------------------------------------------------------------------
# GpuUtilizationReport
# ---------------------------------------------------------------------------


class TestGpuUtilizationReport:
    def test_to_dict_keys(self) -> None:
        r = GpuUtilizationReport(
            gpu_indices=(0,),
            sample_interval_s=1.0,
            total_samples=3,
            mean_sm_util_pct=87.5,
            mean_mem_util_pct=74.0,
            peak_memory_mib=7680,
            captured_path="/tmp/dmon.out",
        )
        d = r.to_dict()
        assert set(d.keys()) >= {
            "gpu_indices",
            "sample_interval_s",
            "total_samples",
            "mean_sm_util_pct",
            "mean_mem_util_pct",
            "peak_memory_mib",
            "captured_path",
        }
