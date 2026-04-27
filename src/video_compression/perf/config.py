"""Configuration schemas for codec performance benchmarking.

All tunables surface as Pydantic fields with explicit validation. There are
no hardcoded literals in the benchmark or baseline code paths — every knob
that affects measurement (resolution, batch size, warmup count, repetition
count, regression tolerance, runtime backend) flows through these schemas.

Schema versioning is explicit so that older baseline JSON files remain
loadable as new fields are added. ``BaselineEntry.schema_version`` and
``BaselineDocument.schema_version`` are the migration anchors.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from src.templates.config import BaseModuleConfig

# Schema versions are surfaced as constants (not magic literals) so callers
# and migrators can compare without duplicating the integer.
PERF_BENCHMARK_CONFIG_SCHEMA_VERSION: int = 1
PERF_BASELINE_DOCUMENT_SCHEMA_VERSION: int = 1
PERF_BASELINE_ENTRY_SCHEMA_VERSION: int = 1


class RuntimeBackend(str, Enum):
    """Decoder/encoder runtime backends.

    Currently only the eager PyTorch path is implemented. Phase 1 will add
    ``compiled`` (``torch.compile``), ``onnx`` (ONNX Runtime), and
    ``tensorrt`` (NVIDIA TensorRT). The enum is defined here so baseline
    files written today remain readable when those backends arrive.
    """

    PYTORCH = "pytorch"
    COMPILED = "compiled"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class Precision(str, Enum):
    """Numerical precision for inference."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class BenchmarkPhase(str, Enum):
    """Which slice of the codec a benchmark cell measures.

    - ``forward``: neural-only round-trip (encoder → entropy → decoder),
      no entropy coding. Lowest variance, used for headline throughput.
    - ``encode``: end-to-end encode including entropy coding.
    - ``decode``: end-to-end decode from a pre-encoded bitstream.
    """

    FORWARD = "forward"
    ENCODE = "encode"
    DECODE = "decode"


class ResolutionSpec(BaseModuleConfig):
    """A single (height, width) measurement target.

    A label is required so reports stay human-readable across
    non-standard resolutions (e.g. anamorphic or vertical video).
    """

    label: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Human-readable label, e.g. '1080p' or '512x256'.",
    )
    height: int = Field(
        ...,
        ge=16,
        le=8192,
        description="Frame height in pixels.",
    )
    width: int = Field(
        ...,
        ge=16,
        le=8192,
        description="Frame width in pixels.",
    )

    @model_validator(mode="after")
    def _validate_divisibility_hint(self) -> ResolutionSpec:
        # The codec's downsample factor is configurable, so we cannot enforce
        # divisibility here — that check happens in the benchmark when the
        # actual codec config is known. We only sanity-check non-degeneracy.
        if self.height * self.width < 16 * 16:
            raise ValueError(
                f"resolution {self.height}x{self.width} too small for any "
                f"realistic codec config",
            )
        return self


class RuntimeProfile(BaseModuleConfig):
    """A runtime backend × precision × device triple to measure.

    The benchmark sweeps the cartesian product of resolutions × batch sizes
    × runtime profiles. New backends are added without touching the
    benchmark loop.

    The ``device`` field lets a profile pin a specific accelerator. On a
    multi-GPU workstation (e.g. cuda:0 + cuda:1) you can record one
    profile per card so the sweep covers them independently. ``None`` (the
    default) inherits the run-level ``device_preference``.
    """

    backend: RuntimeBackend = Field(
        default=RuntimeBackend.PYTORCH,
        description="Runtime backend.",
    )
    precision: Precision = Field(
        default=Precision.FP32,
        description="Inference precision.",
    )
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default="default",
        description="torch.compile mode (only applies when backend=compiled).",
    )
    device: str | None = Field(
        default=None,
        description=(
            "Per-profile device override. Accepts 'cuda', 'cuda:N', 'cpu', "
            "or 'auto'. None inherits the run-level device_preference."
        ),
    )

    @property
    def display_key(self) -> str:
        """Stable string used as a dict key in reports and baselines.

        Includes the device when set so cells on cuda:0 vs cuda:1 produce
        distinct, comparable keys.
        """
        suffix = f"@{self.device}" if self.device else ""
        return f"{self.backend.value}-{self.precision.value}{suffix}"


class PerfBenchmarkConfig(BaseModuleConfig):
    """Complete benchmark configuration.

    No measurement-affecting parameter has a hardcoded value inside the
    benchmark code — they all live here.
    """

    schema_version: int = Field(
        default=PERF_BENCHMARK_CONFIG_SCHEMA_VERSION,
        ge=1,
        description="Config schema version for migration.",
    )

    # Sweep dimensions
    resolutions: list[ResolutionSpec] = Field(
        ...,
        min_length=1,
        description="Resolutions to sweep. No defaults — user must pick.",
    )
    batch_sizes: list[int] = Field(
        default_factory=lambda: [1],
        min_length=1,
        description="Batch sizes to sweep.",
    )
    runtime_profiles: list[RuntimeProfile] = Field(
        default_factory=lambda: [RuntimeProfile(name="default")],
        min_length=1,
        description="Runtime backends × precisions to sweep.",
    )
    phases: list[BenchmarkPhase] = Field(
        default_factory=lambda: [BenchmarkPhase.FORWARD],
        min_length=1,
        description="Which codec phases to measure.",
    )

    # Repetition / warmup
    n_warmup: int = Field(
        default=3,
        ge=0,
        le=1000,
        description="Warmup iterations per cell (excluded from stats).",
    )
    n_repeats: int = Field(
        default=10,
        ge=1,
        le=10000,
        description="Measurement iterations per cell.",
    )
    n_frames_per_iter: int = Field(
        default=1,
        ge=1,
        le=4096,
        description=(
            "Frames per measurement iteration. >1 amortizes per-call overhead."
        ),
    )

    # Hardware. Default is GPU-primary: ``"cuda"`` will fail loud if no
    # CUDA device is present, mirroring the rest of the project. CI uses
    # ``"cpu"`` explicitly via config/perf/smoke.yaml.
    device_preference: str = Field(
        default="cuda",
        description=(
            "Device-resolution preference. Accepts 'cuda', 'cuda:N', "
            "'cpu', or 'auto'. Defaults to 'cuda' (GPU-primary)."
        ),
    )
    track_gpu_memory: bool = Field(
        default=True,
        description="Record peak VRAM per cell when on CUDA.",
    )

    # Synthetic data
    pattern: Literal["gradient", "motion", "checkerboard", "waves", "noise"] = Field(
        default="motion",
        description="Synthetic pattern; 'motion' has the most realistic compressibility.",
    )
    data_seed: int = Field(
        default=20260427,
        ge=0,
        description="Seed for synthetic data generation.",
    )

    # Output
    output_path: str | None = Field(
        default=None,
        description=(
            "If set, write JSON report to this path. Otherwise return only."
        ),
    )

    # Regression-gate plumbing
    baseline_path: str | None = Field(
        default=None,
        description=(
            "If set, the benchmark loads this baseline and reports a "
            "RegressionReport in addition to raw numbers."
        ),
    )
    regression_tolerance_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Allowed degradation per metric, in percent. A throughput drop "
            "or latency rise beyond this fraction is reported as a "
            "regression."
        ),
    )

    # Failure handling
    fail_fast: bool = Field(
        default=False,
        description=(
            "If True, raise on the first cell failure. If False (default), "
            "record the failure in the report and continue with remaining "
            "cells."
        ),
    )

    @model_validator(mode="after")
    def _validate_repeat_counts(self) -> PerfBenchmarkConfig:
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        # Hard ceiling check: a sweep with too many cells × repeats is
        # almost certainly a misconfiguration. Cell count limit is large
        # enough that legitimate sweeps pass.
        cells = (
            len(self.resolutions)
            * len(self.batch_sizes)
            * len(self.runtime_profiles)
            * len(self.phases)
        )
        total_iters = cells * (self.n_warmup + self.n_repeats)
        if total_iters > 10**7:
            raise ValueError(
                f"sweep would run {total_iters} iterations across {cells} "
                f"cells; reduce one of resolutions / batch_sizes / "
                f"runtime_profiles / phases / n_repeats",
            )
        return self


class BaselineEntry(BaseModuleConfig):
    """Recorded measurement for a single benchmark cell.

    Schema is intentionally additive: new metrics or hardware fields can
    be added without invalidating older entries. We override the
    ``extra="forbid"`` default of ``BaseModuleConfig`` because forward-
    compatibility is the explicit goal here — a baseline written by a
    future binary must remain loadable by today's code.
    """

    # Override BaseModuleConfig's strict default so unknown future fields
    # are silently dropped on load.
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(
        default=PERF_BASELINE_ENTRY_SCHEMA_VERSION,
        ge=1,
        description="Entry schema version.",
    )
    cell_key: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable composite key: '<resolution>|b<batch>|<profile>|<phase>'."
        ),
    )
    resolution_label: str = Field(..., min_length=1)
    height: int = Field(..., ge=1)
    width: int = Field(..., ge=1)
    batch_size: int = Field(..., ge=1)
    runtime_backend: RuntimeBackend = Field(...)
    precision: Precision = Field(...)
    phase: BenchmarkPhase = Field(...)
    device_label: str = Field(
        default="",
        description=(
            "Device label captured at record time, e.g. "
            "'cuda:0:NVIDIA-GeForce-RTX-5060-Ti'. Empty for legacy "
            "baselines (migrated on read)."
        ),
    )

    # Recorded statistics (all in canonical units)
    throughput_fps: float = Field(
        ...,
        ge=0.0,
        description="Frames per second (mean over repeats).",
    )
    latency_ms_mean: float = Field(..., ge=0.0)
    latency_ms_p50: float = Field(..., ge=0.0)
    latency_ms_p90: float = Field(..., ge=0.0)
    latency_ms_p99: float = Field(..., ge=0.0)
    peak_vram_mib: float | None = Field(
        default=None,
        ge=0.0,
        description="Peak GPU memory in MiB; null on CPU.",
    )

    # Optional per-entry tolerance overrides
    tolerance_throughput_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Override regression tolerance for this entry's throughput "
            "comparison. Falls back to the run-level tolerance if null."
        ),
    )
    tolerance_latency_pct: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )


class BaselineDocument(BaseModuleConfig):
    """A persisted baseline file readable across schema versions.

    Loader is written so unknown fields are ignored (forward-compat) and
    so that pre-versioned files (no ``schema_version`` field) are migrated
    on read.
    """

    # Same forward-compat override as BaselineEntry.
    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(
        default=PERF_BASELINE_DOCUMENT_SCHEMA_VERSION,
        ge=1,
    )
    description: str = Field(
        default="",
        description="Free-form description of where/how this baseline was recorded.",
    )
    hardware_tag: str = Field(
        default="unknown",
        description=(
            "Free-form hardware identifier, e.g. 'rtx-3060-12g'. Used by "
            "human reviewers, not by the regression gate."
        ),
    )
    git_sha: str = Field(
        default="",
        description="Git SHA at which the baseline was recorded.",
    )
    config_hash: str = Field(
        default="",
        description="Hash of the PerfBenchmarkConfig that produced this.",
    )
    entries: list[BaselineEntry] = Field(
        default_factory=list,
        description="One entry per benchmark cell.",
    )
