"""GPU utilisation profiler wrapping ``nvidia-smi dmon``.

Embeds mean SM-utilisation, memory-utilisation, and peak-memory samples in
SolverResult.metadata so SBIR proposal tables can show whether a workload
is compute-bound or memory-bandwidth-bound.

Design choices:

- The profiler is a context manager. Enter spawns ``nvidia-smi dmon`` as a
  background subprocess; exit terminates and parses the captured CSV-like
  output. This keeps GPU samples flowing for the entire duration of the
  wrapped training loop without per-iteration overhead.
- If ``nvidia-smi`` is missing (e.g. CI on a no-GPU host) the profiler
  no-ops cleanly: ``__enter__`` returns a profiler whose ``report`` is an
  empty :class:`GpuUtilizationReport` with ``total_samples=0`` and
  ``mean_sm_util_pct=None``.
- We do not depend on ``pynvml`` to avoid the wheel pinning that NVML's
  Python bindings require.
"""

from __future__ import annotations

import contextlib
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# dmon column names per `nvidia-smi dmon -h`. We pin the metric set we ask
# for so the parser knows which numeric columns to read.
_DMON_METRIC_FLAGS = "pucvmt"
# Column indices into ``nvidia-smi dmon -s pucvmt`` whitespace-split rows.
# The first six columns are stable across driver versions:
#   gpu pwr gtemp mtemp sm mem
# The framebuffer-memory column is **not** stable: some drivers emit it
# before ``enc``/``dec``/``jpg``/``ofa``, others after, others not at all.
# Hardcoding ``[8]`` was wrong — on a driver without an ``fb`` column,
# index 8 is typically ``jpg`` utilisation (often 0), which we'd then
# silently report as peak FB-MiB. The parser now scans the dmon header
# (the ``# gpu pwr ...`` line) for an explicit ``fb`` column name and
# only records FB memory when found. Other indices stay constants
# because the first six columns are stable.
_DMON_COL_GPU = 0
_DMON_COL_SM_PCT = 4
_DMON_COL_MEM_PCT = 5
_DMON_MIN_COLUMNS = 6  # below this the row is malformed and gets skipped
# Header column-name token that identifies the framebuffer-memory column.
# nvidia-smi prints it lower-case as ``fb`` (or ``FB`` on some drivers).
_DMON_FB_HEADER_TOKEN = "fb"
# Process-termination guardrails.
_DEFAULT_TERMINATE_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class GpuUtilizationReport:
    """Aggregated GPU-utilisation summary for a captured interval."""

    gpu_indices: tuple[int, ...]
    sample_interval_s: float
    total_samples: int
    mean_sm_util_pct: float | None = None
    mean_mem_util_pct: float | None = None
    peak_memory_mib: int | None = None
    captured_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict for SolverResult.metadata."""
        return {
            "gpu_indices": list(self.gpu_indices),
            "sample_interval_s": self.sample_interval_s,
            "total_samples": self.total_samples,
            "mean_sm_util_pct": self.mean_sm_util_pct,
            "mean_mem_util_pct": self.mean_mem_util_pct,
            "peak_memory_mib": self.peak_memory_mib,
            "captured_path": self.captured_path,
        }


@dataclass
class GpuUtilizationProfiler:
    """Context manager that captures dmon samples for the wrapped block.

    Args:
        gpu_indices: GPU indices to monitor (e.g. ``[0]`` or ``[0, 1]``).
        sample_interval_s: Polling cadence in seconds. dmon supports
            integer second cadences; values below 1 are rounded up.
        output_path: Optional path to persist raw dmon output. When
            ``None`` a temp file is used and removed after parsing.
        terminate_timeout_s: Seconds to wait for ``nvidia-smi dmon`` to
            terminate gracefully on context exit before falling back to
            ``kill()``. Surfaced as a field so flaky-host CI can extend
            it without forking the profiler.

    """

    gpu_indices: list[int]
    sample_interval_s: float = 1.0
    output_path: Path | None = None
    terminate_timeout_s: float = _DEFAULT_TERMINATE_TIMEOUT_S

    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _captured_path: Path | None = field(default=None, init=False, repr=False)
    _is_temp_path: bool = field(default=False, init=False, repr=False)
    _effective_interval_s: float | None = field(default=None, init=False, repr=False)
    report: GpuUtilizationReport | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> GpuUtilizationProfiler:
        if not self.gpu_indices:
            self.report = GpuUtilizationReport(
                gpu_indices=(),
                sample_interval_s=self.sample_interval_s,
                total_samples=0,
            )
            return self

        if self.output_path is None:
            tmp = NamedTemporaryFile(mode="w", suffix=".dmon", delete=False, encoding="utf-8")
            tmp.close()
            self._captured_path = Path(tmp.name)
            self._is_temp_path = True
        else:
            self._captured_path = Path(self.output_path)
            self._is_temp_path = False

        # ``nvidia-smi dmon -d`` only accepts integer-second cadences, so
        # round up via ``math.ceil`` (NOT ``round`` — banker's rounding
        # would map 1.4 -> 1, under-sampling the workload). Stash the
        # ceil-rounded value as ``_effective_interval_s`` so the report
        # records the same cadence dmon actually used (the un-rounded
        # ``self.sample_interval_s`` would otherwise lie).
        self._effective_interval_s = float(max(1, math.ceil(self.sample_interval_s)))
        cmd = [
            "nvidia-smi",
            "dmon",
            "-i",
            ",".join(str(i) for i in self.gpu_indices),
            "-d",
            str(int(self._effective_interval_s)),
            "-s",
            _DMON_METRIC_FLAGS,
            "-f",
            str(self._captured_path),
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning(
                "gpu_profiler_disabled",
                reason="nvidia-smi binary not found on PATH",
                gpu_indices=list(self.gpu_indices),
            )
            self._process = None
            self.report = GpuUtilizationReport(
                gpu_indices=tuple(self.gpu_indices),
                sample_interval_s=self.sample_interval_s,
                total_samples=0,
            )

        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=self.terminate_timeout_s)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

        # Use the dmon-effective cadence (post-rounding) for both the
        # parser call and the empty-fallback report so the surface
        # ``GpuUtilizationReport.sample_interval_s`` always matches what
        # dmon actually polled at, never the un-rounded user request.
        report_interval_s = (
            self._effective_interval_s
            if self._effective_interval_s is not None
            else self.sample_interval_s
        )

        if self._captured_path is not None and self._captured_path.exists():
            try:
                with self._captured_path.open("r", encoding="utf-8") as f:
                    text = f.read()
                self.report = parse_dmon_output(
                    text,
                    gpu_indices=tuple(self.gpu_indices),
                    sample_interval_s=report_interval_s,
                    captured_path=str(self._captured_path),
                )
            finally:
                if self._is_temp_path:
                    with contextlib.suppress(OSError):
                        self._captured_path.unlink()

        # Contract: callers may always read ``profiler.report`` after the
        # ``with`` block. If dmon failed to write any output (subprocess
        # killed before flush, permission error, etc.) fall back to an
        # empty zero-sample report instead of leaving ``report = None``.
        if self.report is None:
            self.report = GpuUtilizationReport(
                gpu_indices=tuple(self.gpu_indices),
                sample_interval_s=report_interval_s,
                total_samples=0,
            )


def _find_fb_column_index(text: str) -> int | None:
    """Discover the FB-memory column index from the dmon header line.

    nvidia-smi dmon emits a column-name header line of the form::

        # gpu    pwr  gtemp  mtemp     sm    mem    enc    dec     fb

    Column names are whitespace-separated. We look for the explicit
    ``fb`` token; if absent (drivers that don't emit FB MiB), return
    ``None`` so the parser skips FB extraction entirely instead of
    misreading another numeric column (e.g. ``jpg`` utilisation) as
    framebuffer megabytes.

    The dmon `-o T` timestamp prefix flag is not used by this profiler,
    so the header's first token is always ``gpu``. We strip the leading
    ``#`` before tokenising.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#"):
            continue
        # Strip the comment marker and whitespace, then look for the
        # "gpu" anchor token to confirm this is the column-name header
        # (the second header line is a units row that doesn't contain
        # column names like "gpu", just unit strings like "Idx W ...").
        body = line.lstrip("#").strip()
        tokens = body.split()
        if not tokens or tokens[0].lower() != "gpu":
            continue
        for idx, tok in enumerate(tokens):
            if tok.lower() == _DMON_FB_HEADER_TOKEN:
                return idx
        # Found the column-name header but no fb column.
        return None
    return None


def parse_dmon_output(
    text: str,
    gpu_indices: tuple[int, ...],
    sample_interval_s: float,
    captured_path: str | None = None,
) -> GpuUtilizationReport:
    """Parse ``nvidia-smi dmon`` text output into a summary report.

    dmon emits a header (lines starting with ``#``) followed by sample
    rows. The metric flags we use (``pucvmt``) produce these stable
    columns after the GPU index: ``pwr  gtemp  mtemp  sm  mem``. Other
    columns (``enc``/``dec``/``jpg``/``ofa``/``fb``) are
    driver-dependent in both presence and ordering, so the FB-memory
    column index is discovered from the header at parse time.
    """
    sm_values: list[float] = []
    mem_values: list[float] = []
    fb_mem_values: list[int] = []
    total = 0
    # When ``gpu_indices`` is non-empty, only count rows for those indices.
    # An empty tuple means "no filter" (pre-2026-05-03 behaviour, used by
    # the no-op-when-no-GPU path that synthesises a zero-sample report).
    # This guards against pre-captured files or driver versions that
    # ignore ``-i`` and emit data for every GPU in the system, which
    # would otherwise corrupt the per-GPU means/peaks.
    accepted_indices: set[int] | None = set(gpu_indices) if gpu_indices else None
    fb_col: int | None = _find_fb_column_index(text)

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < _DMON_MIN_COLUMNS:
            continue
        try:
            row_gpu_idx = int(parts[_DMON_COL_GPU])
        except ValueError:
            continue
        if accepted_indices is not None and row_gpu_idx not in accepted_indices:
            continue
        try:
            sm_pct = float(parts[_DMON_COL_SM_PCT])
            mem_pct = float(parts[_DMON_COL_MEM_PCT])
        except (ValueError, IndexError):
            continue
        sm_values.append(sm_pct)
        mem_values.append(mem_pct)
        # Only attempt FB-memory parsing when the header confirmed an
        # ``fb`` column at a known index. Drivers without one yield
        # ``fb_col is None`` and we record no peak_memory_mib.
        if fb_col is not None and len(parts) > fb_col:
            with contextlib.suppress(ValueError, TypeError):
                fb_mem_values.append(int(float(parts[fb_col])))
        total += 1

    return GpuUtilizationReport(
        gpu_indices=gpu_indices,
        sample_interval_s=sample_interval_s,
        total_samples=total,
        mean_sm_util_pct=(float(sum(sm_values) / len(sm_values)) if sm_values else None),
        mean_mem_util_pct=(float(sum(mem_values) / len(mem_values)) if mem_values else None),
        peak_memory_mib=max(fb_mem_values) if fb_mem_values else None,
        captured_path=captured_path,
    )
