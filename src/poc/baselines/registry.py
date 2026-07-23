"""Baseline persistence + regression-gating for PoC-scenario headline metrics.

``ScenarioBaselineRegistry`` records a baseline document from a completed run's
metrics and diffs a later run against it. The gate-relevant output is
``ScenarioRegressionReport.regressions`` — non-empty means the run regressed.

The comparison is metric-agnostic: each entry stores its own direction
(higher/lower-better) and tolerance, so the same code handles residuals,
solved-fraction, reduction-pct, and latency.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

from src.poc.baselines.schema import (
    MetricDirection,
    ScenarioBaselineDocument,
    ScenarioBaselineEntry,
    metric_key,
    migrate_baseline_document,
)

logger = structlog.get_logger(__name__)

# Floor on the baseline magnitude used as the percentage denominator so a
# near-zero baseline (e.g. a residual that hit the target) cannot produce an
# infinite drift. Surfaced as a named constant rather than an inline literal.
BASELINE_DENOM_FLOOR: float = 1e-12

# Default regression tolerance (percent) when neither the entry nor the caller
# specifies one. A named default, not a magic number at the call site.
DEFAULT_TOLERANCE_PCT: float = 10.0

# Observed metrics keyed scenario -> metric -> value.
ObservedMetrics = Mapping[str, Mapping[str, float]]


def regression_pct(*, baseline: float, observed: float, higher_is_better: bool) -> float:
    """Signed drift in percent where **positive means worse** than baseline.

    For ``higher_is_better`` metrics a drop is a regression; for lower-better
    metrics a rise is a regression. The sign convention lets a single
    ``delta > tolerance`` test gate both.
    """
    denom = max(abs(baseline), BASELINE_DENOM_FLOOR)
    raw_pct = (observed - baseline) / denom * 100.0
    return -raw_pct if higher_is_better else raw_pct


@dataclass
class MetricDiff:
    """Per-(scenario, metric) regression result."""

    key: str
    scenario_name: str
    metric_name: str
    baseline_value: float | None
    observed_value: float | None
    delta_pct: float
    tolerance_pct: float
    status: str  # "ok" | "regressed" | "improved" | "skipped"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioRegressionReport:
    """Aggregated baseline-vs-observed comparison."""

    baseline_path: str
    n_entries_baseline: int
    n_metrics_observed: int
    diffs: list[MetricDiff] = field(default_factory=list)
    regressions: list[MetricDiff] = field(default_factory=list)
    improvements: list[MetricDiff] = field(default_factory=list)
    missing_in_observed: list[str] = field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """True iff at least one metric regressed beyond tolerance."""
        return bool(self.regressions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_path": self.baseline_path,
            "n_entries_baseline": self.n_entries_baseline,
            "n_metrics_observed": self.n_metrics_observed,
            "n_diffs": len(self.diffs),
            "n_regressions": len(self.regressions),
            "n_improvements": len(self.improvements),
            "has_regressions": self.has_regressions,
            "missing_in_observed": list(self.missing_in_observed),
            "diffs": [d.to_dict() for d in self.diffs],
            "regressions": [d.to_dict() for d in self.regressions],
            "improvements": [d.to_dict() for d in self.improvements],
        }


def observed_from_result_dicts(
    result_dicts: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Extract ``scenario -> metric -> value`` from ScenarioResult-shaped dicts.

    Accepts the JSON written by ``src/poc/results.py`` (objects with
    ``scenario_name`` and a ``metrics`` mapping) *and* the research-loop
    ``ExecutionResult`` JSON (which labels the run with ``name``). Non-numeric
    metric values are skipped defensively.
    """
    observed: dict[str, dict[str, float]] = {}
    for raw in result_dicts:
        scenario = str(raw.get("scenario_name") or raw.get("name") or "").strip()
        if not scenario:
            continue
        metrics = raw.get("metrics") or {}
        bucket = observed.setdefault(scenario, {})
        for name, value in metrics.items():
            try:
                bucket[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
    return observed


class ScenarioBaselineRegistry:
    """In-memory view of a baseline document plus record/diff utilities."""

    def __init__(self, document: ScenarioBaselineDocument) -> None:
        self._document = document
        self._by_key: dict[str, ScenarioBaselineEntry] = {e.key: e for e in document.entries}

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, path: str | Path) -> ScenarioBaselineRegistry:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"baseline file not found: {path}")
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"baseline {path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"baseline {path} root must be a JSON object")
        migrated = migrate_baseline_document(raw)
        document = ScenarioBaselineDocument.model_validate(migrated)
        logger.info(
            "scenario_baseline_loaded",
            path=str(path),
            schema_version=document.schema_version,
            n_entries=len(document.entries),
        )
        return cls(document)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._document.model_dump(mode="json"), indent=2, sort_keys=True)
        )
        logger.info(
            "scenario_baseline_saved", path=str(path), n_entries=len(self._document.entries)
        )

    # ------------------------------------------------------------ factory

    @classmethod
    def from_observed(
        cls,
        observed: ObservedMetrics,
        *,
        higher_better_metrics: Iterable[str] = (),
        tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
        description: str = "",
        hardware_tag: str = "",
        git_sha: str = "",
        llm_backend: str = "",
    ) -> ScenarioBaselineRegistry:
        """Build a registry (and its document) from observed run metrics.

        Args:
            observed: ``scenario -> metric -> value`` mapping.
            higher_better_metrics: Metric names whose larger value is better
                (everything else is recorded as lower-better). Explicit so no
                direction heuristic is hardcoded.
            tolerance_pct: Per-entry regression tolerance to record.
            description: Human-readable baseline note (provenance).
            hardware_tag: Free-form hardware identifier (provenance).
            git_sha: Commit the baseline was recorded at (provenance).
            llm_backend: LLM backend used for the LLM arm, if any (provenance).

        """
        higher = set(higher_better_metrics)
        entries: list[ScenarioBaselineEntry] = []
        for scenario in sorted(observed):
            for metric in sorted(observed[scenario]):
                direction: MetricDirection = "higher_better" if metric in higher else "lower_better"
                entries.append(
                    ScenarioBaselineEntry(
                        scenario_name=scenario,
                        metric_name=metric,
                        value=float(observed[scenario][metric]),
                        direction=direction,
                        tolerance_pct=tolerance_pct,
                    )
                )
        document = ScenarioBaselineDocument(
            description=description,
            hardware_tag=hardware_tag,
            git_sha=git_sha,
            llm_backend=llm_backend,
            entries=entries,
        )
        return cls(document)

    # ---------------------------------------------------------- accessors

    @property
    def document(self) -> ScenarioBaselineDocument:
        return self._document

    def get(self, scenario_name: str, metric_name: str) -> ScenarioBaselineEntry | None:
        return self._by_key.get(f"{scenario_name}.{metric_key(metric_name)}")

    # --------------------------------------------------------- comparison

    def compare(
        self,
        observed: ObservedMetrics,
        *,
        baseline_path: str = "",
    ) -> ScenarioRegressionReport:
        """Diff observed metrics against this baseline.

        Each baseline entry is looked up in ``observed``; missing observed
        metrics are reported (and do not count as regressions — absence is a
        coverage gap, not a regression). Per-entry direction + tolerance drive
        the classification.
        """
        n_observed = sum(len(v) for v in observed.values())
        report = ScenarioRegressionReport(
            baseline_path=baseline_path,
            n_entries_baseline=len(self._by_key),
            n_metrics_observed=n_observed,
        )
        for key in sorted(self._by_key):
            entry = self._by_key[key]
            observed_value = self._lookup(observed, entry.scenario_name, entry.metric_name)
            if observed_value is None:
                report.missing_in_observed.append(key)
                continue
            delta = regression_pct(
                baseline=entry.value,
                observed=observed_value,
                higher_is_better=entry.direction == "higher_better",
            )
            if delta > entry.tolerance_pct:
                status = "regressed"
            elif delta < -entry.tolerance_pct:
                status = "improved"
            else:
                status = "ok"
            diff = MetricDiff(
                key=key,
                scenario_name=entry.scenario_name,
                metric_name=entry.metric_name,
                baseline_value=entry.value,
                observed_value=observed_value,
                delta_pct=delta,
                tolerance_pct=entry.tolerance_pct,
                status=status,
            )
            report.diffs.append(diff)
            if status == "regressed":
                report.regressions.append(diff)
            elif status == "improved":
                report.improvements.append(diff)
        logger.info(
            "scenario_baseline_compared",
            baseline_path=baseline_path,
            n_regressions=len(report.regressions),
            n_improvements=len(report.improvements),
            n_missing=len(report.missing_in_observed),
        )
        return report

    @staticmethod
    def _lookup(observed: ObservedMetrics, scenario: str, metric: str) -> float | None:
        scenario_metrics = observed.get(scenario)
        if scenario_metrics is None:
            return None
        value = scenario_metrics.get(metric)
        return None if value is None else float(value)


__all__ = [
    "BASELINE_DENOM_FLOOR",
    "DEFAULT_TOLERANCE_PCT",
    "MetricDiff",
    "ObservedMetrics",
    "ScenarioBaselineRegistry",
    "ScenarioRegressionReport",
    "observed_from_result_dicts",
    "regression_pct",
]
