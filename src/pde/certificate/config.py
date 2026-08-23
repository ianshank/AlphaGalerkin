"""Top-level configuration for the certificate subpackage.

Thresholds reuse the canonical :class:`src.poc.config.MetricThreshold` — no
parallel schema, per the spec-tree peer-review correction captured in
``specs/verified_error_certificate.spec.md`` §6.

Follow-on PRs (Track A / Track B estimators) will extend this config with
their own knobs; the discipline is that *every* numeric knob is a typed field
with a description — no hardcoded values inside the estimator code paths.

**WS1 additions** (all backwards-compatible; every new field has a default so
pre-WS1 ``CertificateConfig()`` calls produce a byte-identical dump modulo
new keys). See ``specs/jax_track_b_verifier.spec.md`` §4 AC6.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.pde.certificate.types import (
    CertificationBudget,
    Device,
    Dtype,
    VerifierBackend,
)
from src.poc.config import MetricThreshold

# Cost-budget defaults. These are *placeholders* per the calibration
# convention: the Track A / Track B PRs will replace them with values pinned
# to a first measured run, mirroring ``stochastic_galerkin_nke``. Exposed as
# module constants so tests can reference them symbolically.
DEFAULT_TRACK_A_OVERHEAD_FRACTION: float = 0.10  # spec §5, "≤ 10% of solve"
DEFAULT_TRACK_B_BUDGET_S_CPU: float = 3600.0  # spec AC4, "1 h per solution on CPU"
DEFAULT_HEURISTIC_GRID_RESOLUTION: int = 64  # dense-grid residual sample count

# WS1 additions — defaults chosen so ``CertificateConfig()`` in a pre-WS1
# scenario still validates and still produces the historical certificate
# tier. Changing any of these is a scenario-level opt-in.
DEFAULT_VERIFIER_BACKEND: VerifierBackend = "heuristic_grid"
DEFAULT_DEVICE: Device = "auto"
DEFAULT_DTYPE: Dtype = "float64"
DEFAULT_RECORD_COMPILE_TIME: bool = True


def _default_budget() -> CertificationBudget:
    """Backwards-compat default: the historical Track B CPU budget."""
    return CertificationBudget(
        max_wall_s=DEFAULT_TRACK_B_BUDGET_S_CPU,
        max_mem_mb=None,
        allow_heuristic_fallback=False,
    )


class CertificateConfig(BaseModel):
    """Top-level knobs for building certificates.

    Deliberately narrow in this foundation PR: the follow-on Track A / Track B
    PRs extend the field set (backend selection, grid resolution overrides,
    etc.) rather than reshuffling what is here.
    """

    # ``extra="forbid"`` catches typos at construction (typos in a runtime
    # config are silent-bad; this class is *not* deserialised from a
    # schema-versioned document, so no forward-compat trade-off). Every
    # WS1 addition below has a default, so pre-WS1 ``CertificateConfig()``
    # calls still validate (spec §4 AC6).
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
            "Default Track B wall-clock ceiling used to seed "
            "``CertificationBudget.max_wall_s`` when a scenario does not "
            "provide its own budget. Per-call limits (and the actual "
            "fail-closed / heuristic-fallback behaviour) live on "
            "``CertificateConfig.budget`` — this field is not consumed "
            "directly by the WS1 verifier, kept here so the WS2 Track B PR "
            "does not need to reshuffle the config schema."
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

    # WS1: verifier boundary contract -------------------------------------
    verifier_backend: VerifierBackend = Field(
        default=DEFAULT_VERIFIER_BACKEND,
        description=(
            "Which :class:`~src.pde.certificate.interface.ResidualVerifier` "
            "to dispatch to. Default ``'heuristic_grid'`` matches the pre-WS1 "
            "behaviour (always-available, heuristic tier). Rigorous choices "
            "require the ``[certificate-rigorous]`` (or ``[jax]``) extra; "
            "attempting to use them on a base install raises "
            ":class:`~src.pde.certificate.types.VerifierUnavailableError` "
            "(spec §4 AC1, ADR 0003)."
        ),
    )
    budget: CertificationBudget = Field(
        default_factory=_default_budget,
        description=(
            "Wall-clock / memory ceiling for a single certification call. "
            "Independent from :attr:`track_b_budget_s`, which is the "
            "*scenario-level* budget; ``budget.max_wall_s`` gates the "
            "individual verifier call."
        ),
    )
    device: Device = Field(
        default=DEFAULT_DEVICE,
        description=(
            "Compute device for the verifier. ``'auto'`` defers to "
            ":func:`src.poc.device.resolve_device`; explicit ``'cpu'`` / "
            "``'cuda'`` / ``'tpu'`` / ``'mps'`` override."
        ),
    )
    dtype: Dtype = Field(
        default=DEFAULT_DTYPE,
        description=(
            "Floating precision the verifier operates in. ``float64`` for "
            "reproducibility; ``float32`` when accuracy tolerance allows and "
            "GPU throughput matters."
        ),
    )
    record_compile_time: bool = Field(
        default=DEFAULT_RECORD_COMPILE_TIME,
        description=(
            "Whether to separate ``compile_wall_s`` from ``steady_state_wall_s`` "
            "in the returned :class:`~src.pde.certificate.types."
            "CertifiedResidualBound`. Off only in synthetic tests where "
            "compile timing is irrelevant."
        ),
    )


__all__ = [
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "DEFAULT_HEURISTIC_GRID_RESOLUTION",
    "DEFAULT_RECORD_COMPILE_TIME",
    "DEFAULT_TRACK_A_OVERHEAD_FRACTION",
    "DEFAULT_TRACK_B_BUDGET_S_CPU",
    "DEFAULT_VERIFIER_BACKEND",
    "CertificateConfig",
]
