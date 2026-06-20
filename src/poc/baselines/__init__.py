"""PoC-scenario headline baseline recording + regression gating.

Public surface:
    - ``ScenarioBaselineEntry`` / ``ScenarioBaselineDocument``: the versioned
      on-disk schema.
    - ``ScenarioBaselineRegistry``: ``from_observed`` / ``load`` / ``save`` /
      ``compare``.
    - ``ScenarioRegressionReport`` / ``MetricDiff``: comparison output.
    - ``observed_from_result_dicts``: adapt ScenarioResult JSON to the
      ``scenario -> metric -> value`` form ``compare`` expects.
"""

from __future__ import annotations

from src.poc.baselines.registry import (
    BASELINE_DENOM_FLOOR,
    DEFAULT_TOLERANCE_PCT,
    MetricDiff,
    ObservedMetrics,
    ScenarioBaselineRegistry,
    ScenarioRegressionReport,
    observed_from_result_dicts,
    regression_pct,
)
from src.poc.baselines.schema import (
    POC_BASELINE_DOCUMENT_SCHEMA_VERSION,
    POC_BASELINE_ENTRY_SCHEMA_VERSION,
    MetricDirection,
    ScenarioBaselineDocument,
    ScenarioBaselineEntry,
    migrate_baseline_document,
)

__all__ = [
    "BASELINE_DENOM_FLOOR",
    "DEFAULT_TOLERANCE_PCT",
    "POC_BASELINE_DOCUMENT_SCHEMA_VERSION",
    "POC_BASELINE_ENTRY_SCHEMA_VERSION",
    "MetricDiff",
    "MetricDirection",
    "ObservedMetrics",
    "ScenarioBaselineDocument",
    "ScenarioBaselineEntry",
    "ScenarioBaselineRegistry",
    "ScenarioRegressionReport",
    "migrate_baseline_document",
    "observed_from_result_dicts",
    "regression_pct",
]
