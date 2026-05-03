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
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

logger = logging.getLogger(__name__)

# dmon column names per `nvidia-smi dmon -h`. We pin the metric set we ask
# for so the parser knows which numeric columns to read.
_DMON_METRIC_FLAGS = "pucvmt"
# Column indices into ``nvidia-smi dmon -s pucvmt`` whitespace-split rows.
# The first six columns are stable across driver versions:
#   gpu pwr gtemp mtemp sm mem
# The framebuffer-memory column is driver-version-dependent (some drivers
# emit it before ``enc``/``dec``/``jpg``/``ofa``, others after, others not
# at all). The parser treats the optional FB column as best-effort: a
# misread is silently dropped via ``contextlib.suppress`` and the only
# downstream effect is a missing ``peak_memory_mib`` entry — never wrong
# data. Surfaced as named constants so a layout change is a one-line edit.
_DMON_COL_GPU = 0
_DMON_COL_SM_PCT = 4
_DMON_COL_MEM_PCT = 5
_DMON_COL_FB_MEM_MIB = 8  # NOTE: driver-version dependent; see comment above
_DMON_MIN_COLUMNS = 6  # below this the row is malformed and gets skipped
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

        cmd = [
            "nvidia-smi",
            "dmon",
            "-i",
            ",".join(str(i) for i in self.gpu_indices),
            "-d",
            str(max(1, int(round(self.sample_interval_s)))),
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
            logger.warning("nvidia-smi not found; GPU profiling disabled for this run")
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

        if self._captured_path is not None and self._captured_path.exists():
            try:
                with self._captured_path.open("r", encoding="utf-8") as f:
                    text = f.read()
                self.report = parse_dmon_output(
                    text,
                    gpu_indices=tuple(self.gpu_indices),
                    sample_interval_s=self.sample_interval_s,
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
                sample_interval_s=self.sample_interval_s,
                total_samples=0,
            )


def parse_dmon_output(
    text: str,
    gpu_indices: tuple[int, ...],
    sample_interval_s: float,
    captured_path: str | None = None,
) -> GpuUtilizationReport:
    """Parse ``nvidia-smi dmon`` text output into a summary report.

    dmon emits a header (lines starting with ``#``) followed by sample
    rows. The metric flags we use (``pucvmt``) produce these numeric
    columns after the GPU index: ``pwr  gtemp  mtemp  sm  mem  enc  dec
    jpg  ofa  mclk  pclk``.
    """
    sm_values: list[float] = []
    mem_values: list[float] = []
    fb_mem_values: list[int] = []
    total = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < _DMON_MIN_COLUMNS:
            continue
        try:
            int(parts[_DMON_COL_GPU])
        except ValueError:
            continue
        try:
            sm_pct = float(parts[_DMON_COL_SM_PCT])
            mem_pct = float(parts[_DMON_COL_MEM_PCT])
        except (ValueError, IndexError):
            continue
        sm_values.append(sm_pct)
        mem_values.append(mem_pct)
        # FB-memory column is driver-dependent and may not be present.
        if len(parts) > _DMON_COL_FB_MEM_MIB:
            with contextlib.suppress(ValueError, TypeError):
                fb_mem_values.append(int(float(parts[_DMON_COL_FB_MEM_MIB])))
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
