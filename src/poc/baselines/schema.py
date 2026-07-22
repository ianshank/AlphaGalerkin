"""Versioned baseline document for PoC-scenario headline metrics.

A *baseline* is a JSON document recording one :class:`ScenarioBaselineEntry`
per ``(scenario_name, metric_name)`` pair, together with the metric's
direction and a per-metric tolerance. It lets a later run be regression-gated
against a recorded headline (e.g. ``residual`` must not rise more than X%,
``solved_fraction`` must not drop more than Y%).

It follows the same schema-versioned, forward-compatible pattern (``extra="ignore"``,
explicit migration) as the other baseline registries, but the metric set is *data*, not
code: the direction and tolerance live in the document, so the same registry
handles residuals (lower-better), reduction-pct / solved-fraction
(higher-better), and latency (lower-better) without metric-specific branches.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Schema versions. Bump when the on-disk shape changes; ``_migrate_*`` then
# carries old documents forward so previously-recorded baselines still load.
POC_BASELINE_DOCUMENT_SCHEMA_VERSION: int = 1
POC_BASELINE_ENTRY_SCHEMA_VERSION: int = 1

# Whether a larger metric value is better. Stored per-entry so a single
# document can mix residuals (lower-better) and solved-fraction (higher-better).
MetricDirection = Literal["higher_better", "lower_better"]


class ScenarioBaselineEntry(BaseModel):
    """One recorded headline metric for a scenario.

    ``extra="ignore"`` makes the entry forward-compatible: a document written
    by a newer binary with extra fields still loads on an older one.
    """

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    schema_version: int = Field(
        default=POC_BASELINE_ENTRY_SCHEMA_VERSION,
        description="Entry schema version (for migration).",
    )
    scenario_name: str = Field(..., description="Scenario that produced the metric.", min_length=1)
    metric_name: str = Field(..., description="Metric key as it appears in ScenarioResult.metrics.")
    value: float = Field(..., description="Recorded headline value.")
    direction: MetricDirection = Field(
        ...,
        description=(
            "'lower_better' (e.g. residual/latency) or 'higher_better' (e.g. solved_fraction)."
        ),
    )
    tolerance_pct: float = Field(
        ...,
        ge=0.0,
        description=(
            "Allowed drift before a diff is flagged as a regression, as a "
            "percentage of the baseline magnitude."
        ),
    )

    @property
    def key(self) -> str:
        """Stable ``scenario.metric`` identifier used to align baseline vs observed."""
        return f"{self.scenario_name}.{metric_key(self.metric_name)}"


def metric_key(metric_name: str) -> str:
    """Normalise a metric name (currently identity; a hook for future aliasing)."""
    return metric_name


class ScenarioBaselineDocument(BaseModel):
    """A full baseline: provenance + the list of recorded metric entries."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    schema_version: int = Field(
        default=POC_BASELINE_DOCUMENT_SCHEMA_VERSION,
        description="Document schema version (for migration).",
    )
    description: str = Field(default="", description="Human-readable baseline note.")
    hardware_tag: str = Field(
        default="",
        description="Free-form hardware identifier (e.g. 'RTX 5060 Ti 16GiB').",
    )
    git_sha: str = Field(default="", description="Commit the baseline was recorded at.")
    llm_backend: str = Field(
        default="",
        description="LLM backend used for the LLM arm, if any (lm_studio/vllm/llama_cpp).",
    )
    entries: list[ScenarioBaselineEntry] = Field(
        default_factory=list,
        description="One entry per (scenario, metric) headline.",
    )


def migrate_baseline_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Carry a raw baseline JSON dict forward to the current document schema.

    Migration table:

    +---------------+----+-------------------------------+
    | from          | to | change                        |
    +===============+====+===============================+
    | (unversioned) | 1  | add ``schema_version`` field  |
    +---------------+----+-------------------------------+

    Raises:
        ValueError: The document declares a schema newer than this binary
            understands.

    """
    raw = dict(raw)  # defensive copy; never mutate the caller's dict
    schema_version = raw.get("schema_version")
    if schema_version is None:
        raw["schema_version"] = POC_BASELINE_DOCUMENT_SCHEMA_VERSION
        schema_version = POC_BASELINE_DOCUMENT_SCHEMA_VERSION
    if schema_version > POC_BASELINE_DOCUMENT_SCHEMA_VERSION:
        raise ValueError(
            f"baseline schema_version={schema_version} is newer than this "
            f"binary ({POC_BASELINE_DOCUMENT_SCHEMA_VERSION}); upgrade the "
            f"package or pin a compatible baseline."
        )
    return raw
