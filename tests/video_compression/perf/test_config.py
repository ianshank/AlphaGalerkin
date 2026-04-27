"""Pydantic schema tests for the perf-benchmark configs.

These guard the public configuration surface: every tunable must remain
a validated field with no hardcoded literal in the consumer code.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.video_compression.perf.config import (
    PERF_BASELINE_DOCUMENT_SCHEMA_VERSION,
    PERF_BASELINE_ENTRY_SCHEMA_VERSION,
    PERF_BENCHMARK_CONFIG_SCHEMA_VERSION,
    BaselineDocument,
    BaselineEntry,
    BenchmarkPhase,
    PerfBenchmarkConfig,
    Precision,
    ResolutionSpec,
    RuntimeBackend,
    RuntimeProfile,
)


# ------------------------------------------------------------ ResolutionSpec


class TestResolutionSpec:
    def test_round_trip(self) -> None:
        spec = ResolutionSpec(name="r", label="1080p", height=1080, width=1920)
        assert spec.label == "1080p"
        assert spec.height == 1080
        assert spec.width == 1920

    @pytest.mark.parametrize(
        "h, w",
        [
            (8, 1920),    # too short
            (1920, 8),    # too narrow
            (-1, 1920),   # negative
            (1080, 16385),  # too wide (out of range)
        ],
    )
    def test_invalid_dimensions_rejected(self, h: int, w: int) -> None:
        with pytest.raises(ValidationError):
            ResolutionSpec(name="r", label="bad", height=h, width=w)

    def test_label_required(self) -> None:
        with pytest.raises(ValidationError):
            ResolutionSpec(name="r", label="", height=64, width=64)


# ----------------------------------------------------------- RuntimeProfile


class TestRuntimeProfile:
    def test_display_key_is_stable(self) -> None:
        a = RuntimeProfile(name="a", backend=RuntimeBackend.PYTORCH, precision=Precision.FP32)
        b = RuntimeProfile(name="b", backend=RuntimeBackend.PYTORCH, precision=Precision.FP32)
        assert a.display_key == b.display_key == "pytorch-fp32"

    def test_display_key_distinguishes_backends(self) -> None:
        a = RuntimeProfile(name="a", backend=RuntimeBackend.PYTORCH, precision=Precision.FP16)
        b = RuntimeProfile(name="b", backend=RuntimeBackend.ONNX, precision=Precision.FP16)
        assert a.display_key != b.display_key


# -------------------------------------------------------- PerfBenchmarkConfig


class TestPerfBenchmarkConfig:
    def _minimal(self, **overrides) -> dict:
        base = {
            "name": "test",
            "resolutions": [
                ResolutionSpec(name="r", label="64x64", height=64, width=64)
            ],
        }
        base.update(overrides)
        return base

    def test_minimal_construction(self) -> None:
        cfg = PerfBenchmarkConfig(**self._minimal())
        assert cfg.schema_version == PERF_BENCHMARK_CONFIG_SCHEMA_VERSION
        assert cfg.batch_sizes == [1]
        assert cfg.phases == [BenchmarkPhase.FORWARD]
        assert cfg.n_warmup == 3
        assert cfg.n_repeats == 10
        assert cfg.regression_tolerance_pct == 5.0

    def test_resolutions_required(self) -> None:
        with pytest.raises(ValidationError):
            PerfBenchmarkConfig(name="t", resolutions=[])

    @pytest.mark.parametrize("field, value", [
        ("n_warmup", -1),
        ("n_repeats", 0),
        ("n_frames_per_iter", 0),
        ("regression_tolerance_pct", -1.0),
        ("regression_tolerance_pct", 200.0),
    ])
    def test_field_bounds_enforced(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            PerfBenchmarkConfig(**self._minimal(**{field: value}))

    def test_excessive_sweep_rejected(self) -> None:
        # 100 res * 10 batch * 5 profiles * 3 phases * 1000 repeats = 15M iters
        with pytest.raises(ValidationError):
            PerfBenchmarkConfig(
                name="huge",
                resolutions=[
                    ResolutionSpec(name=f"r{i}", label=f"r{i}", height=64, width=64)
                    for i in range(50)
                ],
                batch_sizes=list(range(1, 11)),
                runtime_profiles=[
                    RuntimeProfile(name=f"p{i}") for i in range(5)
                ],
                phases=[BenchmarkPhase.FORWARD],
                n_repeats=10000,
                n_warmup=0,
            )

    def test_with_overrides_returns_new_instance(self) -> None:
        cfg = PerfBenchmarkConfig(**self._minimal())
        new = cfg.with_overrides(n_repeats=42)
        assert cfg.n_repeats == 10
        assert new.n_repeats == 42

    def test_compute_hash_is_stable(self) -> None:
        # Hashing the same instance twice must be byte-identical so the
        # hash can be used as a cache key. (Two independently constructed
        # instances may differ because nested BaseModuleConfig instances
        # carry a ``created_at`` field; that's a project-wide property of
        # ``compute_hash`` and not specific to our config.)
        cfg = PerfBenchmarkConfig(**self._minimal())
        assert cfg.compute_hash() == cfg.compute_hash()
        assert len(cfg.compute_hash()) == 16

    def test_compute_hash_changes_on_meaningful_edit(self) -> None:
        cfg = PerfBenchmarkConfig(**self._minimal())
        edited = cfg.with_overrides(n_repeats=cfg.n_repeats + 1)
        assert edited.compute_hash() != cfg.compute_hash()


# ----------------------------------------------------------- BaselineEntry


class TestBaselineEntry:
    def _entry(self, **overrides) -> BaselineEntry:
        base = dict(
            name="entry",
            cell_key="64x64|b1|pytorch-fp32|forward",
            resolution_label="64x64",
            height=64,
            width=64,
            batch_size=1,
            runtime_backend=RuntimeBackend.PYTORCH,
            precision=Precision.FP32,
            phase=BenchmarkPhase.FORWARD,
            throughput_fps=10.0,
            latency_ms_mean=100.0,
            latency_ms_p50=95.0,
            latency_ms_p90=110.0,
            latency_ms_p99=120.0,
        )
        base.update(overrides)
        return BaselineEntry(**base)

    def test_minimal_entry(self) -> None:
        entry = self._entry()
        assert entry.schema_version == PERF_BASELINE_ENTRY_SCHEMA_VERSION
        assert entry.peak_vram_mib is None
        assert entry.tolerance_throughput_pct is None

    def test_negative_throughput_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._entry(throughput_fps=-1.0)


# ---------------------------------------------------------- BaselineDocument


class TestBaselineDocument:
    def test_round_trip(self) -> None:
        doc = BaselineDocument(
            name="bl",
            description="test",
            hardware_tag="cpu",
            entries=[],
        )
        assert doc.schema_version == PERF_BASELINE_DOCUMENT_SCHEMA_VERSION
        as_json = doc.model_dump(mode="json")
        rehydrated = BaselineDocument.model_validate(as_json)
        assert rehydrated.schema_version == doc.schema_version
        assert rehydrated.hardware_tag == doc.hardware_tag

    def test_unknown_fields_ignored(self) -> None:
        # Forward-compat: a baseline produced by a future version may carry
        # fields we do not yet recognise. They should not break loading.
        # Note: BaseModuleConfig defaults to "ignore" for extras.
        raw = {
            "name": "bl",
            "schema_version": 1,
            "entries": [],
            "future_unknown_field": {"nested": [1, 2, 3]},
        }
        doc = BaselineDocument.model_validate(raw)
        assert doc.schema_version == 1
