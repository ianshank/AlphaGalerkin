"""Baseline persistence and regression-gating.

A baseline is a versioned JSON document containing one ``BaselineEntry``
per benchmark cell. The registry handles:

  * **Loading** old baselines under their original schema (forward-compat:
    unknown fields are ignored, missing ``schema_version`` is migrated).
  * **Saving** new baselines.
  * **Comparing** a fresh ``BenchmarkReport`` against a baseline and
    producing a ``RegressionReport`` that lists each cell as ``ok``,
    ``regressed``, ``improved``, or ``missing``.

The comparison logic is metric-agnostic: throughput is treated as
"higher is better", latency and VRAM as "lower is better". Adding a new
metric is a one-line edit to ``_METRIC_DEFS``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import structlog

from src.video_compression.perf.config import (
    PERF_BASELINE_DOCUMENT_SCHEMA_VERSION,
    BaselineDocument,
    BaselineEntry,
    BenchmarkPhase,
    Precision,
    RuntimeBackend,
)
from src.video_compression.perf.metrics import regression_pct

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------- types


@dataclass(frozen=True)
class MetricDefinition:
    """Single comparable metric in a regression check.

    ``higher_is_better`` flips the sign of ``regression_pct`` so the same
    tolerance check works for throughput and latency. ``getter`` extracts
    the metric from a baseline entry; ``observed_getter`` from a fresh
    cell. They're separate because the source dataclasses differ but the
    metric semantics are the same.
    """

    name: str
    higher_is_better: bool
    baseline_getter: Callable[[BaselineEntry], float | None]
    observed_getter: Callable[[Any], float | None]
    tolerance_field: str | None = None  # which BaselineEntry field overrides tolerance


# We define the metric set at module scope so the regression gate can be
# extended in one place. ``observed_getter`` accepts ``Any`` to avoid an
# import cycle with ``benchmark.py``; the runtime type is ``CellResult``.
_METRIC_DEFS: list[MetricDefinition] = [
    MetricDefinition(
        name="throughput_fps",
        higher_is_better=True,
        baseline_getter=lambda e: e.throughput_fps,
        observed_getter=lambda c: c.throughput_fps,
        tolerance_field="tolerance_throughput_pct",
    ),
    MetricDefinition(
        name="latency_ms_p50",
        higher_is_better=False,
        baseline_getter=lambda e: e.latency_ms_p50,
        observed_getter=lambda c: c.latency_stats.percentile(50),
        tolerance_field="tolerance_latency_pct",
    ),
    MetricDefinition(
        name="latency_ms_p99",
        higher_is_better=False,
        baseline_getter=lambda e: e.latency_ms_p99,
        observed_getter=lambda c: c.latency_stats.percentile(99),
        tolerance_field="tolerance_latency_pct",
    ),
    MetricDefinition(
        name="peak_vram_mib",
        higher_is_better=False,
        baseline_getter=lambda e: e.peak_vram_mib,
        observed_getter=lambda c: c.peak_vram_mib,
        tolerance_field=None,  # VRAM uses run-level tolerance
    ),
]


@dataclass
class CellDiff:
    """Per-cell, per-metric regression result."""

    cell_key: str
    metric: str
    baseline_value: float | None
    observed_value: float | None
    delta_pct: float
    tolerance_pct: float
    status: str  # "ok" | "regressed" | "improved" | "skipped"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionReport:
    """Aggregated comparison result.

    ``regressions`` is the gate-relevant list — non-empty means the run
    failed the regression check.
    """

    baseline_path: str
    n_cells_baseline: int
    n_cells_observed: int
    diffs: list[CellDiff] = field(default_factory=list)
    regressions: list[CellDiff] = field(default_factory=list)
    improvements: list[CellDiff] = field(default_factory=list)
    missing_in_observed: list[str] = field(default_factory=list)
    missing_in_baseline: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_path": self.baseline_path,
            "n_cells_baseline": self.n_cells_baseline,
            "n_cells_observed": self.n_cells_observed,
            "n_diffs": len(self.diffs),
            "n_regressions": len(self.regressions),
            "n_improvements": len(self.improvements),
            "missing_in_observed": list(self.missing_in_observed),
            "missing_in_baseline": list(self.missing_in_baseline),
            "diffs": [d.to_dict() for d in self.diffs],
            "regressions": [d.to_dict() for d in self.regressions],
            "improvements": [d.to_dict() for d in self.improvements],
        }


# ----------------------------------------------------------------- migrations


def _migrate_baseline_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Migrate a raw baseline JSON dict to the current schema.

    Migration table:

    +----------------+----------------+--------------------------------+
    | from           | to             | change                         |
    +================+================+================================+
    | (unversioned)  | 1              | add ``schema_version`` field   |
    +----------------+----------------+--------------------------------+

    New schemas are added by appending to this table. Old code remains
    able to load every baseline ever recorded.
    """
    # Defensive copy so callers' dicts aren't mutated
    raw = dict(raw)

    schema_version = raw.get("schema_version")
    if schema_version is None:
        logger.info(
            "baseline.migration.unversioned_to_v1",
            keys=sorted(raw.keys()),
        )
        raw["schema_version"] = 1
        schema_version = 1

    # Future: if schema_version == 1 and current is 2, run 1->2 migration here.
    if schema_version > PERF_BASELINE_DOCUMENT_SCHEMA_VERSION:
        raise ValueError(
            f"baseline schema_version={schema_version} is newer than this "
            f"binary ({PERF_BASELINE_DOCUMENT_SCHEMA_VERSION}); upgrade the "
            f"package or pin a compatible baseline.",
        )
    return raw


# -------------------------------------------------------------------- helpers


def baseline_entry_from_cell(cell: Any) -> BaselineEntry:
    """Build a ``BaselineEntry`` from a successful ``CellResult``.

    Implemented in this module (not benchmark.py) so the dependency edge
    points one way: benchmark imports baseline, never the reverse.
    """
    if cell.failed:
        raise ValueError(
            f"cannot record baseline entry from failed cell {cell.cell_key!r}",
        )
    return BaselineEntry(
        name="baseline_entry",
        cell_key=cell.cell_key,
        resolution_label=cell.resolution_label,
        height=cell.height,
        width=cell.width,
        batch_size=cell.batch_size,
        runtime_backend=cell.backend,
        precision=cell.precision,
        phase=cell.phase,
        device_label=cell.device_label,
        throughput_fps=cell.throughput_fps,
        latency_ms_mean=cell.latency_stats.mean_ms,
        latency_ms_p50=cell.latency_stats.percentile(50),
        latency_ms_p90=cell.latency_stats.percentile(90),
        latency_ms_p99=cell.latency_stats.percentile(99),
        peak_vram_mib=cell.peak_vram_mib,
    )


# --------------------------------------------------------------------- registry


class BaselineRegistry:
    """In-memory view of a baseline document plus diff utilities.

    Construct via ``BaselineRegistry(document)`` or
    ``BaselineRegistry.load(path)``. ``compare_report`` is the gate.
    """

    def __init__(self, document: BaselineDocument) -> None:
        self._document = document
        self._by_key: dict[str, BaselineEntry] = {
            entry.cell_key: entry for entry in document.entries
        }

    # --- io

    @classmethod
    def load(cls, path: str | Path) -> BaselineRegistry:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"baseline file not found: {path}")
        raw_text = path.read_text()
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"baseline {path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"baseline {path} root must be a JSON object")
        migrated = _migrate_baseline_document(raw)
        # Pydantic ignores unknown fields when ``model_config.extra``
        # isn't set to "forbid" — BaseModuleConfig leaves it permissive,
        # which is what we want for forward compatibility.
        document = BaselineDocument.model_validate(migrated)
        logger.info(
            "baseline.loaded",
            path=str(path),
            schema_version=document.schema_version,
            n_entries=len(document.entries),
        )
        return cls(document)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._document.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info(
            "baseline.saved",
            path=str(path),
            n_entries=len(self._document.entries),
        )

    # --- accessors

    @property
    def document(self) -> BaselineDocument:
        return self._document

    @property
    def cell_keys(self) -> Iterable[str]:
        return self._by_key.keys()

    def get(self, cell_key: str) -> BaselineEntry | None:
        return self._by_key.get(cell_key)

    # --- comparison

    def compare_report(
        self,
        report: Any,
        *,
        tolerance_pct: float,
    ) -> RegressionReport:
        """Diff a ``BenchmarkReport`` against this baseline."""
        observed_by_key = {c.cell_key: c for c in report.cells if not c.failed}

        out = RegressionReport(
            baseline_path="",  # filled in by caller (we don't track our origin)
            n_cells_baseline=len(self._by_key),
            n_cells_observed=len(observed_by_key),
        )

        # Cells present in baseline but absent from observed.
        out.missing_in_observed = sorted(
            set(self._by_key) - set(observed_by_key),
        )
        out.missing_in_baseline = sorted(
            set(observed_by_key) - set(self._by_key),
        )

        for key in sorted(self._by_key):
            entry = self._by_key[key]
            if key not in observed_by_key:
                continue
            observed = observed_by_key[key]
            for metric in _METRIC_DEFS:
                diff = self._diff_metric(
                    cell_key=key,
                    entry=entry,
                    observed=observed,
                    metric=metric,
                    run_tolerance_pct=tolerance_pct,
                )
                if diff is None:
                    continue
                out.diffs.append(diff)
                if diff.status == "regressed":
                    out.regressions.append(diff)
                elif diff.status == "improved":
                    out.improvements.append(diff)

        return out

    def _diff_metric(
        self,
        *,
        cell_key: str,
        entry: BaselineEntry,
        observed: Any,
        metric: MetricDefinition,
        run_tolerance_pct: float,
    ) -> CellDiff | None:
        baseline_value = metric.baseline_getter(entry)
        observed_value = metric.observed_getter(observed)

        # Both null → metric is N/A on this cell (e.g. VRAM on CPU).
        if baseline_value is None and observed_value is None:
            return None
        # Mixed null is a real signal: hardware mismatch. Surface it.
        if baseline_value is None or observed_value is None:
            return CellDiff(
                cell_key=cell_key,
                metric=metric.name,
                baseline_value=baseline_value,
                observed_value=observed_value,
                delta_pct=0.0,
                tolerance_pct=run_tolerance_pct,
                status="skipped",
            )

        delta = regression_pct(
            baseline=baseline_value,
            observed=observed_value,
            higher_is_better=metric.higher_is_better,
        )

        # Per-entry tolerance overrides for the metric, if defined.
        per_entry_tolerance: float | None = None
        if metric.tolerance_field is not None:
            per_entry_tolerance = getattr(entry, metric.tolerance_field, None)
        tolerance = (
            per_entry_tolerance if per_entry_tolerance is not None else run_tolerance_pct
        )

        if delta > tolerance:
            status = "regressed"
        elif delta < -tolerance:
            status = "improved"
        else:
            status = "ok"

        return CellDiff(
            cell_key=cell_key,
            metric=metric.name,
            baseline_value=baseline_value,
            observed_value=observed_value,
            delta_pct=delta,
            tolerance_pct=tolerance,
            status=status,
        )


__all__ = [
    "BaselineDocument",
    "BaselineEntry",
    "BaselineRegistry",
    "BenchmarkPhase",
    "CellDiff",
    "MetricDefinition",
    "Precision",
    "RegressionReport",
    "RuntimeBackend",
    "baseline_entry_from_cell",
]
