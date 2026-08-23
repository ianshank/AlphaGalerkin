"""Versioned ``Certificate`` artifact — the machine-checkable error bound.

Every pinned-scenario solution emits exactly one :class:`Certificate`. The
artifact is intentionally a *data record*, not a computation: the estimator
tracks (``track_a`` / ``track_b`` — follow-on PRs) build one of these and hand
it back. That separation is what makes the artifact useful across
verification backends (dReal, autoLiRPA, ∂-CROWN, or the cheap heuristic tier)
without leaking backend-specific types into the schema.

The forward-compat migration pattern mirrors
:mod:`src.poc.baselines.schema` and :mod:`src.video_compression.zoo`:
old JSON documents load via :func:`migrate_certificate_document`, and unknown
fields are ignored so a newer binary can drop *additional* keys without
breaking downstream readers.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bump when the on-disk schema changes; ``migrate_certificate_document`` then
# carries older documents forward. Keep this in lockstep with the migration
# table docstring.
CERTIFICATE_DOCUMENT_SCHEMA_VERSION: int = 1

# ``track`` distinguishes exact-Galerkin (residual a posteriori, Track A) from
# neural-operator (certified uniform residual, Track B) — see spec §2.
TrackKind = Literal["A", "B"]

# ``rigor`` labels heuristic-tier certificates so they never appear as rigorous
# in ``results/`` or business docs (fabrication-precedent guard — F0 backup
# and transfer-MSE incidents, ``CLAUDE.md`` 2026-07).
RigorKind = Literal["rigorous", "heuristic"]

# Norm the ``bound_value`` is expressed in. Deliberately open (str) rather than
# enum: adding L^inf / H^1 later must not require a schema bump.
DEFAULT_NORM: str = "L2"


class Certificate(BaseModel):
    """One certified error bound for one solution.

    Fields cover the spec §3 requirements: ``track`` and ``rigor`` are
    enum-enforced, cost telemetry is always present, and the stability-constant
    provenance is carried inline so downstream readers do not have to consult
    the registry to know how a bound was derived.

    ``extra="ignore"`` makes the artifact forward-compatible: a document
    written by a newer binary with extra fields still loads on an older one,
    matching the ``ScenarioBaselineEntry`` precedent.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, protected_namespaces=())

    schema_version: int = Field(
        default=CERTIFICATE_DOCUMENT_SCHEMA_VERSION,
        description="Certificate schema version (for forward-compat migration).",
    )

    # Identity ----------------------------------------------------------------
    certificate_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Stable per-artifact identifier (16-char hex; use "
            ":func:`src.pde.certificate.logging.new_certificate_id`). Used to "
            "correlate the artifact with its structlog events."
        ),
    )
    pde_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Value of :class:`src.pde.config.PDEType` this certificate applies "
            "to. Stored as ``str`` (not the enum) so the artifact JSON round-"
            "trips even against a binary that has added new operators."
        ),
    )
    scenario_name: str = Field(
        ...,
        min_length=1,
        description="Name of the pinned scenario that produced the solution.",
    )

    # Bound -------------------------------------------------------------------
    track: TrackKind = Field(
        ...,
        description=(
            "'A' for exact-Galerkin residual estimator (spec §2 Track A); "
            "'B' for certified neural-operator residual (Track B, "
            "arXiv:2603.19165)."
        ),
    )
    rigor: RigorKind = Field(
        ...,
        description=(
            "'rigorous' for machine-checkable bounds; 'heuristic' for the "
            "cheap dense-grid tier (CI-only)."
        ),
    )
    norm: str = Field(
        default=DEFAULT_NORM,
        min_length=1,
        description=(
            "Norm the ``bound_value`` is expressed in (e.g. 'L2', 'Linf', "
            "'H1'). Open string — new norms do not require a schema bump."
        ),
    )
    bound_value: float = Field(
        ...,
        ge=0.0,
        description=(
            "Certified upper bound ``||u - u_tilde|| ≤ C_0 * ||r(u_tilde)||`` "
            "in the declared norm. NaN/Inf rejected by validator."
        ),
    )
    residual_norm: float = Field(
        ...,
        ge=0.0,
        description=(
            "The measured residual norm ``||r(u_tilde)||``. ``bound_value = "
            "stability_constant * residual_norm`` when the stability entry is "
            "not ``unbounded_with_warning``."
        ),
    )

    # Stability provenance ----------------------------------------------------
    stability_constant: float | None = Field(
        default=None,
        description=(
            "The declared ``C_0`` (or inf-sup ``β``) used to scale the "
            "residual. ``None`` iff the registry entry is "
            "``unbounded_with_warning`` — in that case ``bound_value`` degrades "
            "to the residual norm and downstream code must render the "
            "'residual bound only — no error guarantee' string (spec AC5)."
        ),
    )
    stability_source: str = Field(
        ...,
        min_length=1,
        description=(
            "Value of :class:`StabilitySource` — 'analytic', 'estimated', or "
            "'unbounded_with_warning'. Stored as ``str`` for forward-compat."
        ),
    )
    stability_notes: str = Field(
        default="",
        description=(
            "Free-form provenance (e.g. 'Lax-Milgram, coercive bilinear form'; "
            "'Helmholtz k=5, empirical fit'; 'UNBOUNDED — Biharmonic; residual "
            "bound only')."
        ),
    )

    # Verifier ---------------------------------------------------------------
    verifier_backend: str = Field(
        default="",
        description=(
            "Backend that produced the residual: 'dorfler_indicator_2d', "
            "'autoLiRPA', 'delta_crown', 'dReal', 'dense_grid_heuristic'. "
            "Empty for placeholder / hand-built certificates in tests."
        ),
    )
    verifier_version: str = Field(
        default="",
        description="Verifier version pin (e.g. 'auto_LiRPA==0.4.0').",
    )

    # Cost telemetry (AC4) ---------------------------------------------------
    cert_wall_s: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock seconds spent producing this certificate.",
    )
    cert_peak_mem_mb: float = Field(
        default=0.0,
        ge=0.0,
        description="Peak RSS in MiB during certification (0.0 if unmeasured).",
    )
    cert_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Cost in USD on metered hardware (None off metered runs). "
            "Optional so free-tier CI never emits a bogus zero."
        ),
    )

    # Free-form provenance ---------------------------------------------------
    notes: str = Field(
        default="",
        description=(
            "Free-form provenance. Reserved values with defined semantics: "
            "'budget_exceeded' (heuristic emitted after Track B budget "
            "overrun — fail-closed, per skill guardrail); 'unbounded_operator' "
            "(no error guarantee; residual bound only)."
        ),
    )

    # --- Validators ---------------------------------------------------------

    @model_validator(mode="after")
    def _check_finite_bound(self) -> Certificate:
        """Reject NaN/Inf sneaking through ``ge=0.0`` on ``float``."""
        import math

        for field_name in ("bound_value", "residual_norm", "cert_wall_s", "cert_peak_mem_mb"):
            v = getattr(self, field_name)
            if not math.isfinite(v):
                raise ValueError(f"{field_name} must be finite, got {v!r}")
        if self.stability_constant is not None and not math.isfinite(self.stability_constant):
            raise ValueError(
                f"stability_constant must be finite when set, got {self.stability_constant!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_stability_consistency(self) -> Certificate:
        """``stability_constant is None`` iff source is ``unbounded_with_warning``.

        This is the load-bearing honesty invariant of AC5: an ``UNBOUNDED``
        operator (Helmholtz at high wavenumber, Biharmonic without a declared
        constant, ...) cannot ship a numeric ``C_0``, and conversely a
        numeric ``C_0`` cannot be paired with an ``UNBOUNDED`` source label.
        """
        is_unbounded = self.stability_source == "unbounded_with_warning"
        has_constant = self.stability_constant is not None
        if is_unbounded and has_constant:
            raise ValueError(
                "stability_source='unbounded_with_warning' requires "
                "stability_constant=None (spec AC5)"
            )
        if (not is_unbounded) and (not has_constant):
            raise ValueError(
                f"stability_source={self.stability_source!r} requires a numeric "
                f"stability_constant (spec AC5)"
            )
        return self


def migrate_certificate_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Carry a raw certificate JSON dict forward to the current schema.

    Migration table:

    +---------------+----+-------------------------------+
    | from          | to | change                        |
    +===============+====+===============================+
    | (unversioned) | 1  | add ``schema_version`` field  |
    +---------------+----+-------------------------------+

    Args:
        raw: JSON dict as read from disk / network. Never mutated in place —
            the returned dict is always a defensive copy.

    Returns:
        A new dict at the current schema version, safe to pass to
        :class:`Certificate` (extras ignored).

    Raises:
        ValueError: The document declares a schema newer than this binary
            understands.

    """
    raw = dict(raw)  # defensive copy; never mutate the caller's dict
    schema_version = raw.get("schema_version")
    if schema_version is None:
        raw["schema_version"] = CERTIFICATE_DOCUMENT_SCHEMA_VERSION
        schema_version = CERTIFICATE_DOCUMENT_SCHEMA_VERSION
    if not isinstance(schema_version, int):
        raise ValueError(
            f"certificate schema_version must be int, got {type(schema_version).__name__}"
        )
    if schema_version > CERTIFICATE_DOCUMENT_SCHEMA_VERSION:
        raise ValueError(
            f"certificate schema_version={schema_version} is newer than this "
            f"binary ({CERTIFICATE_DOCUMENT_SCHEMA_VERSION}); upgrade the package "
            f"or pin a compatible certificate document."
        )
    return raw


def _fresh_certificate_id_hex() -> str:
    """Small indirection so :mod:`.logging` can share the generator.

    Kept private (leading underscore) — not part of the public surface, only
    used to avoid an import cycle between ``certificate.py`` and ``logging.py``.
    """
    return uuid.uuid4().hex[:16]
