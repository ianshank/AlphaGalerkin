"""Top-level configuration for the certificate subpackage.

Thresholds reuse the canonical :class:`src.poc.config.MetricThreshold` — no
parallel schema, per the spec-tree peer-review correction captured in
``specs/verified_error_certificate.spec.md`` §6.

Follow-on PRs (Track A / Track B estimators) will extend this config with
their own knobs; the discipline is that *every* numeric knob is a typed field
with a description — no hardcoded values inside the estimator code paths.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.poc.config import MetricThreshold

# Cost-budget defaults. These are *placeholders* per the calibration
# convention: the Track A / Track B PRs will replace them with values pinned
# to a first measured run, mirroring ``stochastic_galerkin_nke``. Exposed as
# module constants so tests can reference them symbolically.
DEFAULT_TRACK_A_OVERHEAD_FRACTION: float = 0.10  # spec §5, "≤ 10% of solve"
DEFAULT_TRACK_B_BUDGET_S_CPU: float = 3600.0  # spec AC4, "1 h per solution on CPU"
DEFAULT_HEURISTIC_GRID_RESOLUTION: int = 64  # dense-grid residual sample count


class CertificateConfig(BaseModel):
    """Top-level knobs for building certificates.

    Deliberately narrow in this foundation PR: the follow-on Track A / Track B
    PRs extend the field set (backend selection, grid resolution overrides,
    etc.) rather than reshuffling what is here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    # Enablement -----------------------------------------------------------
    enabled: bool = Field(
        default=True,
        description=(
            "Whether to emit certificates during a scenario run. Off in unit "
            "tests that build a certificate by hand; on in production scenarios."
        ),
    )

    # Track selection ------------------------------------------------------
    prefer_rigorous: bool = Field(
        default=True,
        description=(
            "If True, prefer 'rigorous' Track B over the 'heuristic' dense-grid "
            "tier when a rigorous verifier is available. CI defaults flip this "
            "to False to keep smoke jobs fast."
        ),
    )

    # Cost budgets (AC4) --------------------------------------------------
    track_a_overhead_fraction: float = Field(
        default=DEFAULT_TRACK_A_OVERHEAD_FRACTION,
        gt=0.0,
        description=(
            "Maximum allowed Track A overhead as a fraction of the solve "
            "wall-clock, per spec §5. Warnings during calibration, hard gate "
            "after — same pattern as ``llm_call_p95_latency_ms``."
        ),
    )
    track_b_budget_s: float = Field(
        default=DEFAULT_TRACK_B_BUDGET_S_CPU,
        gt=0.0,
        description=(
            "Maximum wall-clock seconds allowed for a single Track B "
            "certification. Overruns emit a heuristic-tier certificate with "
            "``notes='budget_exceeded'`` (fail-closed, per skill guardrail)."
        ),
    )
    heuristic_grid_resolution: int = Field(
        default=DEFAULT_HEURISTIC_GRID_RESOLUTION,
        gt=0,
        description=(
            "Dense-grid resolution for the heuristic-tier residual max. Not "
            "used in this foundation PR; declared here so the Track B PR does "
            "not need to reshuffle the config schema."
        ),
    )

    # Thresholds (reuse canonical MetricThreshold — no parallel schema) ---
    thresholds: list[MetricThreshold] = Field(
        default_factory=list,
        description=(
            "Certificate-related pass/fail thresholds (e.g. effectivity index, "
            "coverage). Reuses :class:`src.poc.config.MetricThreshold` per the "
            "spec-tree peer-review correction — no parallel schema."
        ),
    )
