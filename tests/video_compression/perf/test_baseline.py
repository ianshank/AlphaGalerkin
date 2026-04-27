"""Tests for BaselineRegistry, persistence, and regression diffs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.video_compression.perf.baseline import (
    BaselineDocument,
    BaselineRegistry,
    _migrate_baseline_document,
    baseline_entry_from_cell,
)
from src.video_compression.perf.benchmark import (
    BenchmarkReport,
    CellResult,
)
from src.video_compression.perf.config import (
    PERF_BASELINE_DOCUMENT_SCHEMA_VERSION,
    BaselineEntry,
    BenchmarkPhase,
    Precision,
    RuntimeBackend,
)
from src.video_compression.perf.metrics import LatencyStats


# ------------------------------------------------------------------ helpers


def _make_cell(
    *,
    cell_key: str = "64x64|b1|pytorch-fp32|forward",
    throughput_fps: float = 30.0,
    latency_p50: float = 33.0,
    latency_p99: float = 40.0,
    failed: bool = False,
    peak_vram_mib: float | None = None,
) -> CellResult:
    return CellResult(
        cell_key=cell_key,
        resolution_label="64x64",
        height=64,
        width=64,
        batch_size=1,
        backend=RuntimeBackend.PYTORCH,
        precision=Precision.FP32,
        phase=BenchmarkPhase.FORWARD,
        latency_stats=LatencyStats(
            count=10,
            mean_ms=33.0,
            min_ms=30.0,
            max_ms=42.0,
            std_ms=2.0,
            percentiles_ms={50: latency_p50, 90: 38.0, 99: latency_p99},
        ),
        throughput_fps=throughput_fps,
        peak_vram_mib=peak_vram_mib,
        failed=failed,
    )


def _make_report(cells: list[CellResult]) -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_id="test-id",
        config_hash="abc123",
        device="cpu",
        cells=cells,
    )


# ----------------------------------------------------- baseline_entry_from_cell


class TestEntryFromCell:
    def test_round_trip(self) -> None:
        cell = _make_cell()
        entry = baseline_entry_from_cell(cell)
        assert entry.cell_key == cell.cell_key
        assert entry.throughput_fps == cell.throughput_fps
        assert entry.latency_ms_p50 == cell.latency_stats.percentile(50)
        assert entry.latency_ms_p99 == cell.latency_stats.percentile(99)

    def test_failed_cell_rejected(self) -> None:
        cell = _make_cell(failed=True)
        with pytest.raises(ValueError, match="failed cell"):
            baseline_entry_from_cell(cell)


# ------------------------------------------------------------------ migration


class TestMigration:
    def test_unversioned_input_migrated_to_v1(self) -> None:
        raw = {"name": "old", "entries": []}
        migrated = _migrate_baseline_document(raw)
        assert migrated["schema_version"] == 1

    def test_input_dict_not_mutated(self) -> None:
        raw = {"name": "old", "entries": []}
        _migrate_baseline_document(raw)
        assert "schema_version" not in raw

    def test_future_schema_rejected(self) -> None:
        raw = {"name": "future", "schema_version": 999, "entries": []}
        with pytest.raises(ValueError, match="newer"):
            _migrate_baseline_document(raw)

    def test_current_schema_passes_through(self) -> None:
        raw = {
            "name": "ok",
            "schema_version": PERF_BASELINE_DOCUMENT_SCHEMA_VERSION,
            "entries": [],
        }
        migrated = _migrate_baseline_document(raw)
        assert migrated["schema_version"] == PERF_BASELINE_DOCUMENT_SCHEMA_VERSION


# ------------------------------------------------------ Registry persistence


class TestRegistryIO:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cell = _make_cell()
        entry = baseline_entry_from_cell(cell)
        doc = BaselineDocument(
            name="test",
            description="round-trip",
            hardware_tag="cpu-test",
            entries=[entry],
        )
        path = tmp_path / "baseline.json"
        BaselineRegistry(doc).save(path)

        loaded = BaselineRegistry.load(path)
        assert len(list(loaded.cell_keys)) == 1
        assert loaded.get(cell.cell_key) is not None
        assert loaded.document.hardware_tag == "cpu-test"

    def test_load_unversioned_file(self, tmp_path: Path) -> None:
        legacy = {
            "name": "legacy",
            "description": "before schema_version was added",
            "hardware_tag": "old-hw",
            "entries": [],
        }
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(legacy))
        loaded = BaselineRegistry.load(path)
        assert loaded.document.schema_version == 1

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(ValueError, match="not valid JSON"):
            BaselineRegistry.load(path)

    def test_load_root_must_be_object(self, tmp_path: Path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="object"):
            BaselineRegistry.load(path)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BaselineRegistry.load(tmp_path / "nonexistent.json")

    def test_unknown_fields_ignored_on_load(self, tmp_path: Path) -> None:
        forward_compat = {
            "name": "future",
            "schema_version": 1,
            "entries": [],
            "future_only_field": "should be ignored",
        }
        path = tmp_path / "future.json"
        path.write_text(json.dumps(forward_compat))
        loaded = BaselineRegistry.load(path)
        assert len(loaded.document.entries) == 0


# ------------------------------------------------------- compare_report


class TestRegressionDiff:
    def _registry_with(self, **cell_overrides) -> BaselineRegistry:
        cell = _make_cell(**cell_overrides)
        entry = baseline_entry_from_cell(cell)
        return BaselineRegistry(BaselineDocument(name="t", entries=[entry]))

    def test_no_regression_when_identical(self) -> None:
        reg = self._registry_with()
        report = _make_report([_make_cell()])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        assert diff.regressions == []
        assert diff.improvements == []

    def test_throughput_drop_flagged(self) -> None:
        reg = self._registry_with(throughput_fps=100.0)
        report = _make_report([_make_cell(throughput_fps=80.0)])  # 20% drop
        diff = reg.compare_report(report, tolerance_pct=5.0)
        assert any(d.metric == "throughput_fps" for d in diff.regressions)

    def test_throughput_gain_classified_as_improvement(self) -> None:
        reg = self._registry_with(throughput_fps=100.0)
        report = _make_report([_make_cell(throughput_fps=120.0)])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        assert any(d.metric == "throughput_fps" for d in diff.improvements)
        assert not diff.regressions

    def test_latency_p99_regression_flagged(self) -> None:
        reg = self._registry_with(latency_p99=10.0)
        report = _make_report([_make_cell(latency_p99=15.0)])  # +50%
        diff = reg.compare_report(report, tolerance_pct=10.0)
        assert any(d.metric == "latency_ms_p99" for d in diff.regressions)

    def test_within_tolerance_is_ok(self) -> None:
        reg = self._registry_with(throughput_fps=100.0)
        report = _make_report([_make_cell(throughput_fps=98.0)])  # 2% drop
        diff = reg.compare_report(report, tolerance_pct=5.0)
        assert diff.regressions == []
        # Each metric still produces an "ok" CellDiff
        ok_metrics = {d.metric for d in diff.diffs if d.status == "ok"}
        assert "throughput_fps" in ok_metrics

    def test_missing_in_observed(self) -> None:
        reg = self._registry_with(cell_key="present-in-baseline")
        report = _make_report([_make_cell(cell_key="present-in-observed")])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        assert "present-in-baseline" in diff.missing_in_observed
        assert "present-in-observed" in diff.missing_in_baseline

    def test_failed_observed_cell_excluded(self) -> None:
        reg = self._registry_with()
        report = _make_report([_make_cell(failed=True)])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        # Failed cells are not counted as observed
        assert diff.n_cells_observed == 0

    def test_vram_only_in_observed_marked_skipped(self) -> None:
        # Baseline recorded on CPU (no VRAM); observed has VRAM. The VRAM
        # comparison should be reported as "skipped" rather than crashing.
        reg = self._registry_with(peak_vram_mib=None)
        report = _make_report([_make_cell(peak_vram_mib=512.0)])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        skipped_metrics = {d.metric for d in diff.diffs if d.status == "skipped"}
        assert "peak_vram_mib" in skipped_metrics

    def test_per_entry_tolerance_override(self) -> None:
        # Tighten throughput tolerance from 50% to 1% via per-entry override.
        cell = _make_cell(throughput_fps=100.0)
        entry = baseline_entry_from_cell(cell)
        entry = entry.with_overrides(tolerance_throughput_pct=1.0)
        reg = BaselineRegistry(BaselineDocument(name="t", entries=[entry]))
        # 3% drop: would be ok at 5% but regressed at 1%
        report = _make_report([_make_cell(throughput_fps=97.0)])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        assert any(d.metric == "throughput_fps" for d in diff.regressions)

    def test_regression_dict_serializable(self) -> None:
        reg = self._registry_with(throughput_fps=100.0)
        report = _make_report([_make_cell(throughput_fps=50.0)])
        diff = reg.compare_report(report, tolerance_pct=5.0)
        as_dict = diff.to_dict()
        # Must be JSON-encodable
        json.dumps(as_dict)
        assert as_dict["n_regressions"] >= 1


# ----------------------------------------------------------------- Document


class TestBaselineDocumentEntries:
    def test_entry_preserved_through_save_load(self, tmp_path: Path) -> None:
        cell = _make_cell(throughput_fps=42.5, latency_p50=23.5, latency_p99=99.9)
        entry = baseline_entry_from_cell(cell)
        doc = BaselineDocument(name="t", entries=[entry])
        path = tmp_path / "b.json"
        BaselineRegistry(doc).save(path)
        loaded = BaselineRegistry.load(path)
        loaded_entry = loaded.get(cell.cell_key)
        assert loaded_entry is not None
        assert loaded_entry.throughput_fps == 42.5
        assert loaded_entry.latency_ms_p50 == 23.5
        assert loaded_entry.latency_ms_p99 == 99.9
