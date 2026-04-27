"""Codec performance benchmark harness.

Sweeps the cartesian product of (resolution, batch_size, runtime_profile,
phase), produces per-cell ``LatencyStats``/throughput, optionally compares
against a recorded baseline, and emits a JSON report.

The benchmark loop is subject-agnostic: a ``BenchmarkSubject`` Protocol
abstracts the thing being timed so future runtime backends drop in
without touching this file.

Logging contract: every log event from this module is structured (via
``BaseModuleLogger``) and bound to ``benchmark_id`` plus, where relevant,
the cell key. A ``--debug`` caller can grep on ``benchmark_id`` to
isolate a run's events.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog
import torch

from src.templates.base import BaseExecutable, ExecutionResult, ExecutionStatus
from src.templates.logging import create_logger_class
from src.video_compression.config import CodecConfig
from src.video_compression.data.synthetic import SyntheticPattern
from src.video_compression.perf.config import (
    BaselineDocument,
    BenchmarkPhase,
    PerfBenchmarkConfig,
    Precision,
    ResolutionSpec,
    RuntimeBackend,
    RuntimeProfile,
)
from src.video_compression.perf.device import device_label, resolve_device
from src.video_compression.perf.metrics import (
    LatencyStats,
    summarize_latencies,
    throughput_fps,
)
from src.video_compression.perf.subjects import BenchmarkSubject, create_subject

_module_logger = structlog.get_logger(__name__)
_BenchmarkLogger = create_logger_class("PerfBenchmark")

# Suffix used to label cells in baselines and reports. Surfaced as a
# constant so baselines remain comparable if the format ever needs to
# evolve (older baselines could be migrated by rewriting these keys).
CELL_KEY_TEMPLATE = "{resolution}|b{batch}|{profile}|{phase}"


def cell_key(
    resolution: ResolutionSpec,
    batch_size: int,
    profile: RuntimeProfile,
    phase: BenchmarkPhase,
) -> str:
    """Stable, human-readable cell identifier."""
    return CELL_KEY_TEMPLATE.format(
        resolution=resolution.label,
        batch=batch_size,
        profile=profile.display_key,
        phase=phase.value,
    )


@dataclass
class CellResult:
    """Per-cell measurement bundle."""

    cell_key: str
    resolution_label: str
    height: int
    width: int
    batch_size: int
    backend: RuntimeBackend
    precision: Precision
    phase: BenchmarkPhase
    latency_stats: LatencyStats
    throughput_fps: float
    peak_vram_mib: float | None
    device_label: str = ""
    failed: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Latency stats already provide their own dict; replace the dataclass
        # representation to keep enum values JSON-friendly.
        d["latency_stats"] = self.latency_stats.to_dict()
        d["backend"] = self.backend.value
        d["precision"] = self.precision.value
        d["phase"] = self.phase.value
        return d


@dataclass
class BenchmarkReport:
    """Full benchmark report.

    ``cells`` is the source of truth; everything else is derived. The
    JSON encoding is forward-compatible: readers should ignore unknown
    fields, and the ``schema_version`` lets future migrations target a
    specific shape.
    """

    schema_version: int = 1
    benchmark_id: str = ""
    config_hash: str = ""
    device: str = ""
    started_at: float = 0.0
    duration_s: float = 0.0
    cells: list[CellResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "config_hash": self.config_hash,
            "device": self.device,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "cells": [c.to_dict() for c in self.cells],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


class PerfBenchmark(BaseExecutable[PerfBenchmarkConfig]):
    """Codec performance benchmark.

    Composes ``BaseExecutable`` so the benchmark slots into the project's
    standard run-and-report pattern (timing, error handling, structured
    logging, ``ExecutionResult`` shape).
    """

    _executable_name: str = "perf_benchmark"
    _logger_class = _BenchmarkLogger

    def __init__(
        self,
        config: PerfBenchmarkConfig,
        *,
        codec_config: CodecConfig | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(config=config, run_id=run_id)
        # Codec config is decoupled so a caller can pin a specific model
        # config (e.g. the one matching a checkpoint they trained) while
        # still using the standard sweep.
        self._codec_config = codec_config or CodecConfig(name="benchmark_default")
        self._device: torch.device | None = None

    # ---------------------------------------------------------------- API

    def execute(self) -> ExecutionResult:
        """Entry point invoked by ``BaseExecutable.run``."""
        # Resolve the run-level device once for logging/headline reporting.
        # Per-cell devices may override this when a profile pins one.
        self._device = resolve_device(
            self.config.device_preference,
            context=self._executable_name,
        )

        report = BenchmarkReport(
            benchmark_id=self.run_id,
            config_hash=self.config.compute_hash(),
            device=device_label(self._device),
            started_at=time.time(),
        )

        self.logger.info(
            "benchmark.started",
            device=str(self._device),
            n_resolutions=len(self.config.resolutions),
            n_batch_sizes=len(self.config.batch_sizes),
            n_profiles=len(self.config.runtime_profiles),
            n_phases=len(self.config.phases),
        )

        # Sweep the cartesian product of dimensions. Order is
        # (phase, profile, resolution, batch_size); putting profile in the
        # outer loop minimises subject re-builds when a future backend has
        # an expensive prepare (e.g. torch.compile).
        cells = list(self._iter_cells())
        report.cells = []
        n_failed = 0

        for cell_idx, (resolution, batch_size, profile, phase) in enumerate(cells):
            key = cell_key(resolution, batch_size, profile, phase)
            cell_logger = self.logger.bind(cell=key, cell_idx=cell_idx)
            cell_logger.info(
                "cell.started",
                resolution=resolution.label,
                batch=batch_size,
                backend=profile.backend.value,
                precision=profile.precision.value,
                phase=phase.value,
            )
            try:
                cell_result = self._run_cell(
                    resolution=resolution,
                    batch_size=batch_size,
                    profile=profile,
                    phase=phase,
                    key=key,
                )
            except NotImplementedError as exc:
                # Phase requested a subject that doesn't exist yet (e.g.
                # encode/decode before Phase 4). Record a failure rather
                # than aborting; the report still contains useful data.
                if self.config.fail_fast:
                    raise
                n_failed += 1
                cell_logger.warning(
                    "cell.skipped_unimplemented",
                    reason=str(exc),
                )
                cell_result = self._failed_cell(
                    resolution=resolution,
                    batch_size=batch_size,
                    profile=profile,
                    phase=phase,
                    key=key,
                    reason=f"not implemented: {exc}",
                )
            except Exception as exc:  # noqa: BLE001 - benchmark must not abort the suite
                if self.config.fail_fast:
                    raise
                n_failed += 1
                cell_logger.exception(
                    "cell.failed",
                    error=str(exc),
                )
                cell_result = self._failed_cell(
                    resolution=resolution,
                    batch_size=batch_size,
                    profile=profile,
                    phase=phase,
                    key=key,
                    reason=f"runtime error: {exc}",
                )
            else:
                cell_logger.info(
                    "cell.completed",
                    throughput_fps=cell_result.throughput_fps,
                    latency_ms_p50=cell_result.latency_stats.percentile(50),
                    latency_ms_p99=cell_result.latency_stats.percentile(99),
                    peak_vram_mib=cell_result.peak_vram_mib,
                )
            report.cells.append(cell_result)

        report.duration_s = time.time() - report.started_at

        # Optional: regression check vs baseline
        regression_summary = self._regression_summary(report)

        # Optional: write the report to disk
        artifacts: dict[str, Any] = {"report": report.to_dict()}
        if regression_summary is not None:
            artifacts["regression"] = regression_summary

        if self.config.output_path:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report.to_json())
            self.logger.info("benchmark.report_written", path=str(output_path))
            artifacts["report_path"] = str(output_path)

        # Aggregate metrics for the ExecutionResult; throughput averaged
        # over successful cells gives a single headline number that is
        # useful in CI summaries.
        metrics = self._aggregate_metrics(report, n_failed=n_failed)

        status = (
            ExecutionStatus.COMPLETED
            if n_failed == 0
            else ExecutionStatus.COMPLETED  # partial success is still completion
        )
        # However, if a regression check was requested *and* failed, we
        # surface that as a failed status so CI gates can rely on it.
        if regression_summary is not None and regression_summary["regressions"]:
            status = ExecutionStatus.FAILED

        return self._create_result(
            status=status,
            metrics=metrics,
            artifacts=artifacts,
            metadata={
                "n_cells": len(report.cells),
                "n_failed_cells": n_failed,
            },
        )

    # ------------------------------------------------------ internals

    def _iter_cells(
        self,
    ) -> Iterator[tuple[ResolutionSpec, int, RuntimeProfile, BenchmarkPhase]]:
        for phase in self.config.phases:
            for profile in self.config.runtime_profiles:
                for resolution in self.config.resolutions:
                    for batch_size in self.config.batch_sizes:
                        yield resolution, batch_size, profile, phase

    def _run_cell(
        self,
        *,
        resolution: ResolutionSpec,
        batch_size: int,
        profile: RuntimeProfile,
        phase: BenchmarkPhase,
        key: str,
    ) -> CellResult:
        # Phase 0 only exposes the eager pytorch fp32 path; later phases
        # extend ``create_subject`` to dispatch on profile.
        if profile.backend is not RuntimeBackend.PYTORCH:
            raise NotImplementedError(
                f"runtime backend {profile.backend.value!r} requires "
                f"phase 1 (decoder runtime registry)",
            )
        if profile.precision is not Precision.FP32:
            raise NotImplementedError(
                f"precision {profile.precision.value!r} requires phase 1 (mixed-precision support)",
            )

        # Per-profile device override lets a single sweep cover both GPUs
        # (cuda:0 and cuda:1) without re-running the whole benchmark.
        cell_device = (
            resolve_device(profile.device, context=f"{self._executable_name}.{key}")
            if profile.device is not None
            else self._device
        )
        assert cell_device is not None

        subject: BenchmarkSubject = create_subject(
            phase=phase,
            codec_config=self._codec_config,
            device=cell_device,
            pattern=SyntheticPattern(self.config.pattern),
            seed=self.config.data_seed,
        )

        peak_vram_mib: float | None = None

        try:
            subject.prepare(
                batch_size=batch_size,
                height=resolution.height,
                width=resolution.width,
            )

            # Warmup
            for _ in range(self.config.n_warmup):
                subject.step()
            self._sync_device(cell_device)

            # Reset peak memory accounting *after* warmup so we capture
            # steady-state usage, not allocation transients.
            if cell_device.type == "cuda" and self.config.track_gpu_memory:
                torch.cuda.reset_peak_memory_stats(cell_device)

            latencies_ms: list[float] = []
            for _ in range(self.config.n_repeats):
                self._sync_device(cell_device)
                t0 = time.perf_counter()
                for _ in range(self.config.n_frames_per_iter):
                    subject.step()
                self._sync_device(cell_device)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            if cell_device.type == "cuda" and self.config.track_gpu_memory:
                peak_bytes = torch.cuda.max_memory_allocated(cell_device)
                peak_vram_mib = peak_bytes / (1024.0 * 1024.0)

        finally:
            subject.teardown()

        stats = summarize_latencies(latencies_ms)
        # Throughput accounts for n_frames_per_iter — each iteration
        # processed that many frames in batch_size batches.
        fps = throughput_fps(
            latencies_ms,
            frames_per_iter=batch_size * self.config.n_frames_per_iter,
        )

        return CellResult(
            cell_key=key,
            resolution_label=resolution.label,
            height=resolution.height,
            width=resolution.width,
            batch_size=batch_size,
            backend=profile.backend,
            precision=profile.precision,
            phase=phase,
            latency_stats=stats,
            throughput_fps=fps,
            peak_vram_mib=peak_vram_mib,
            device_label=device_label(cell_device),
        )

    def _failed_cell(
        self,
        *,
        resolution: ResolutionSpec,
        batch_size: int,
        profile: RuntimeProfile,
        phase: BenchmarkPhase,
        key: str,
        reason: str,
    ) -> CellResult:
        empty_stats = LatencyStats(
            count=0,
            mean_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            std_ms=0.0,
            percentiles_ms={50: 0.0, 90: 0.0, 99: 0.0},
        )
        return CellResult(
            cell_key=key,
            resolution_label=resolution.label,
            height=resolution.height,
            width=resolution.width,
            batch_size=batch_size,
            backend=profile.backend,
            precision=profile.precision,
            phase=phase,
            latency_stats=empty_stats,
            throughput_fps=0.0,
            peak_vram_mib=None,
            failed=True,
            failure_reason=reason,
        )

    def _sync_device(self, device: torch.device | None = None) -> None:
        target = device if device is not None else self._device
        if target is not None and target.type == "cuda":
            torch.cuda.synchronize(target)

    def _aggregate_metrics(
        self,
        report: BenchmarkReport,
        *,
        n_failed: int,
    ) -> dict[str, float]:
        ok_cells = [c for c in report.cells if not c.failed]
        if not ok_cells:
            return {
                "n_cells_total": float(len(report.cells)),
                "n_cells_failed": float(n_failed),
                "n_cells_ok": 0.0,
            }
        avg_throughput = sum(c.throughput_fps for c in ok_cells) / len(ok_cells)
        max_p99 = max(c.latency_stats.percentile(99) for c in ok_cells)
        return {
            "n_cells_total": float(len(report.cells)),
            "n_cells_failed": float(n_failed),
            "n_cells_ok": float(len(ok_cells)),
            "avg_throughput_fps": avg_throughput,
            "max_latency_ms_p99": max_p99,
            "duration_s": report.duration_s,
        }

    def _regression_summary(
        self,
        report: BenchmarkReport,
    ) -> dict[str, Any] | None:
        """Compare ``report`` against the configured baseline if present."""
        if not self.config.baseline_path:
            return None

        # Local import keeps a module-level cycle out of the file: the
        # baseline module imports the same config types we do.
        from src.video_compression.perf.baseline import BaselineRegistry

        baseline_path = Path(self.config.baseline_path)
        if not baseline_path.exists():
            self.logger.warning(
                "baseline.missing",
                path=str(baseline_path),
                hint="record a baseline first with --record-baseline",
            )
            return None

        registry = BaselineRegistry.load(baseline_path)
        diff_report = registry.compare_report(
            report,
            tolerance_pct=self.config.regression_tolerance_pct,
        )

        n_reg = len(diff_report.regressions)
        if n_reg:
            self.logger.error(
                "regression.detected",
                n_regressions=n_reg,
                first=diff_report.regressions[0].cell_key,
            )
        else:
            self.logger.info("regression.none")
        return diff_report.to_dict()


def run_benchmark(
    config: PerfBenchmarkConfig,
    *,
    codec_config: CodecConfig | None = None,
    run_id: str | None = None,
) -> ExecutionResult:
    """Convenience entry point for callers that don't want to subclass."""
    return PerfBenchmark(
        config=config,
        codec_config=codec_config,
        run_id=run_id,
    ).run()


def report_from_result(result: ExecutionResult) -> BenchmarkReport:
    """Reconstruct a ``BenchmarkReport`` from an ``ExecutionResult``.

    The benchmark stores the full report dict under ``artifacts["report"]``
    so that downstream tools can pick it up without re-running the sweep.
    """
    if "report" not in result.artifacts:
        raise KeyError(
            "ExecutionResult has no 'report' artifact; was it produced by PerfBenchmark?",
        )
    return _report_from_dict(result.artifacts["report"])


def _report_from_dict(data: dict[str, Any]) -> BenchmarkReport:
    cells = [
        CellResult(
            cell_key=c["cell_key"],
            resolution_label=c["resolution_label"],
            height=c["height"],
            width=c["width"],
            batch_size=c["batch_size"],
            backend=RuntimeBackend(c["backend"]),
            precision=Precision(c["precision"]),
            phase=BenchmarkPhase(c["phase"]),
            latency_stats=LatencyStats(
                count=c["latency_stats"]["count"],
                mean_ms=c["latency_stats"]["mean_ms"],
                min_ms=c["latency_stats"]["min_ms"],
                max_ms=c["latency_stats"]["max_ms"],
                std_ms=c["latency_stats"]["std_ms"],
                percentiles_ms={
                    int(k): float(v) for k, v in c["latency_stats"]["percentiles_ms"].items()
                },
            ),
            throughput_fps=c["throughput_fps"],
            peak_vram_mib=c["peak_vram_mib"],
            device_label=c.get("device_label", ""),
            failed=c.get("failed", False),
            failure_reason=c.get("failure_reason"),
        )
        for c in data["cells"]
    ]
    return BenchmarkReport(
        schema_version=data.get("schema_version", 1),
        benchmark_id=data.get("benchmark_id", ""),
        config_hash=data.get("config_hash", ""),
        device=data.get("device", ""),
        started_at=data.get("started_at", 0.0),
        duration_s=data.get("duration_s", 0.0),
        cells=cells,
    )


def baseline_from_report(
    report: BenchmarkReport,
    *,
    description: str = "",
    hardware_tag: str = "unknown",
    git_sha: str = "",
) -> BaselineDocument:
    """Convert a fresh report into a baseline document.

    Used by the ``benchmark_codec.py --record-baseline`` flow. Tolerances
    default to null per entry, which means the regression gate falls
    back to the run-level tolerance — that's the most permissive
    backwards-compatible default.
    """
    from src.video_compression.perf.baseline import (
        baseline_entry_from_cell,
    )

    entries = [baseline_entry_from_cell(c) for c in report.cells if not c.failed]
    return BaselineDocument(
        name="baseline",
        description=description,
        hardware_tag=hardware_tag,
        git_sha=git_sha,
        config_hash=report.config_hash,
        entries=entries,
    )
