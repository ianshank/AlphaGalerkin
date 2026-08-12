"""Typed artifacts for the Track B verifier boundary (WS1, spec §3).

These types are additive to PR #1's :mod:`~src.pde.certificate.certificate`
module. They live in a *separate* file to avoid conflating the certificate
artifact (the on-disk JSON schema) with the verifier IO contract (the
in-process Python protocol).

Design principles (spec §3):

* Every type is a frozen Pydantic model with ``extra="ignore"`` for
  forward-compatibility. New optional fields land without breaking on-disk
  or in-process consumers.
* Numeric fields declare units in the field description.
* Backend selection is a closed :data:`VerifierBackend` ``Literal`` — IDE
  autocomplete catches typos at edit time.
* Coverage is a closed :data:`DomainCoverage` ``Literal`` — the
  ``"rigorous"`` rigor tier structurally requires ``"full"`` coverage (§4 AC3).
* :class:`VerifierUnavailableError` is the *only* exception verifier
  factories raise for missing dependencies. Silent fallback is forbidden
  (§4 AC1).

None of the code paths in this file imports ``torch`` or ``jax``. The AST
hot-path guard (:mod:`tests.pde.certificate.test_no_top_level_ml_import`)
enforces that at pre-commit time.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Closed Literals — every backend / coverage / rigor value the type system
# admits. Adding a new value is a deliberate edit and shows up in code review.
# ---------------------------------------------------------------------------

#: Registry keys for :class:`ResidualVerifier` implementations.
#:
#: * ``heuristic_grid`` — dense-grid residual max, always available (WS1).
#: * ``autolirpa`` — LiRPA family via ``auto_LiRPA`` (Torch, WS2).
#: * ``delta_crown`` — ``∂-CROWN`` optimisation-based bound (Torch, WS2).
#: * ``jax_verify`` — CROWN/IBP via ``jax_verify`` (JAX, WS2).
#: * ``dreal`` — SMT-based bound via ``dReal`` (WS2, optional).
VerifierBackend = Literal[
    "heuristic_grid",
    "autolirpa",
    "delta_crown",
    "jax_verify",
    "dreal",
]

#: How much of the certification domain the bound covers.
#:
#: * ``full`` — closed-form bound over the whole domain (rigorous).
#: * ``grid_sampled`` — sampled at a finite grid (heuristic tier only).
#: * ``partial`` — rigorous but only over a proper subdomain (still not
#:   admissible as ``rigor="rigorous"`` for a whole-domain certificate; see
#:   AC3).
DomainCoverage = Literal["full", "grid_sampled", "partial"]

#: Runtime rigor label. Matches
#: :data:`src.pde.certificate.certificate.RigorKind` so downstream artifacts
#: consume the same vocabulary.
BoundRigor = Literal["rigorous", "heuristic", "failed"]

#: Domain kind. ``rectangular`` covers the unit-box family PR #1 already
#: exercises; ``sdf`` reserved for the Leap 71 helical operators.
DomainKind = Literal["rectangular", "sdf"]

#: Device selection — mirrors :func:`src.poc.device.resolve_device` values.
Device = Literal["auto", "cpu", "cuda", "tpu", "mps"]

#: Floating-point precision.
Dtype = Literal["float32", "float64"]


# ---------------------------------------------------------------------------
# Explicit unavailability signal
# ---------------------------------------------------------------------------


class VerifierUnavailableError(RuntimeError):
    """Raised when a requested :class:`ResidualVerifier` cannot be constructed.

    The message names the missing extra (``"[certificate-rigorous]"``,
    ``"[jax]"``, ...) so a caller can act. Never caught silently by the
    dispatcher — spec §4 AC1 requires fail-closed behaviour.
    """

    def __init__(self, backend: str, extra: str, detail: str = "") -> None:
        self.backend = backend
        self.extra = extra
        self.detail = detail
        base = (
            f"verifier backend {backend!r} is unavailable in this environment; "
            f"install the optional extra {extra!r} (e.g. "
            f"``pip install -e '.[{extra}]'``)"
        )
        super().__init__(f"{base}: {detail}" if detail else base)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class HardwareMeta(BaseModel):
    """Provenance record baked into every :class:`CertifiedResidualBound`.

    Recorded so a later reviewer can determine whether a persisted bound is
    still comparable to a fresh run — swapping torch or jax versions
    invalidates parity comparisons.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    device: Device = Field(..., description="Compute device the bound ran on.")
    dtype: Dtype = Field(..., description="Floating precision used during certification.")
    torch_version: str | None = Field(
        default=None,
        description="``torch.__version__`` at run time, or ``None`` if torch is uninstalled.",
    )
    jax_version: str | None = Field(
        default=None,
        description="``jax.__version__`` at run time, or ``None`` if JAX is uninstalled.",
    )
    jax_verify_version: str | None = Field(
        default=None,
        description="``jax_verify.__version__`` at run time, or ``None`` if missing.",
    )
    cuda_capability: str | None = Field(
        default=None,
        description=(
            "``torch.cuda.get_device_capability`` render (e.g. ``sm_75``); ``None`` on CPU."
        ),
    )


class CertificationBudget(BaseModel):
    """Wall-clock / memory guard for a single certification call (§3 scope)."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    max_wall_s: float = Field(
        ...,
        gt=0.0,
        description="Hard wall-clock ceiling in seconds. Overrun → ``rigor='failed'``.",
    )
    max_mem_mb: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional resident-memory ceiling in MiB. ``None`` disables the check.",
    )
    allow_heuristic_fallback: bool = Field(
        default=False,
        description=(
            "If True, an overrun *may* return a heuristic-tier bound with "
            "``notes='budget_exceeded'``; if False, an overrun always fails "
            "closed. Defaults to ``False`` (spec §3 out-of-scope: silent "
            "downgrade)."
        ),
    )


class DomainSpec(BaseModel):
    """Description of the certification domain — closed form or functional.

    A rectangular domain is fully described by ``bounds``; an SDF domain by
    an :attr:`sdf_reference` string that names an evaluator registered
    elsewhere (:mod:`src.pde.sdf`). Verifiers consume the fields they need
    and raise :class:`VerifierUnavailableError` for kinds they cannot handle.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    kind: DomainKind = Field(..., description="Domain family.")
    bounds: tuple[tuple[float, float], ...] = Field(
        default=(),
        description=(
            "Axis-aligned bounding-box coordinates ``((x_min, x_max), ...)``. "
            "Required for ``kind='rectangular'``; used as a bounding hull for "
            "``kind='sdf'``."
        ),
    )
    sdf_reference: str | None = Field(
        default=None,
        description=(
            "Registry key for an SDF evaluator (see ``src.pde.sdf``). Required "
            "for ``kind='sdf'``; ignored for ``kind='rectangular'``."
        ),
    )
    grid_resolution: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Optional per-axis grid resolution for grid-sampled coverage; "
            "``None`` defers to ``CertificateConfig.heuristic_grid_resolution``."
        ),
    )

    @model_validator(mode="after")
    def _kind_matches_fields(self) -> DomainSpec:
        if self.kind == "rectangular":
            if not self.bounds:
                raise ValueError("kind='rectangular' requires non-empty bounds")
            for lo, hi in self.bounds:
                if not (hi > lo):
                    raise ValueError(f"bounds entry ({lo}, {hi}) must satisfy hi > lo")
        elif self.kind == "sdf":
            if self.sdf_reference is None:
                raise ValueError("kind='sdf' requires a non-null sdf_reference")
        return self

    @property
    def dimension(self) -> int:
        """Convenience: number of spatial dimensions (from bounds)."""
        return len(self.bounds)


class CertifiedModel(BaseModel):
    """Handle to a model the verifier can consume.

    :attr:`model_fn` is opaque — verifiers cast it into their native
    representation (e.g. a Torch ``nn.Module`` or a JAX callable). Kept as
    ``Any`` because the two frameworks disagree on the type, and forcing a
    common base class would defeat the purpose of the backend-neutral
    contract.
    """

    model_config = ConfigDict(
        frozen=True, extra="ignore", arbitrary_types_allowed=True, protected_namespaces=()
    )

    backend: Literal["torch", "jax", "numpy"] = Field(
        ...,
        description="Which framework produced this model. Verifiers reject mismatches.",
    )
    model_fn: Any = Field(
        ...,
        description=(
            "Callable ``(x) -> u(x)`` or a framework model object. Verifiers "
            "cast to their native form."
        ),
    )
    params: Any = Field(
        default=None,
        description=(
            "Framework-native parameters (JAX pytree, Torch state_dict). "
            "Some backends (Torch ``nn.Module``) fold params into ``model_fn`` "
            "and leave this ``None``."
        ),
    )
    traceable: bool = Field(
        default=False,
        description=(
            "True iff ``model_fn`` is safe to call under ``jax.jit`` or "
            "``torch.compile``. WS3's JAX batch wrapper reads this flag."
        ),
    )


class CertifiedResidualBound(BaseModel):
    """Verifier output: a certified upper bound on the residual over a domain.

    Fields required by §4 AC5 (cost accounting) are non-negative floats. The
    ``rigor``/``domain_coverage`` compatibility rule (AC3) is enforced at
    validation time so a caller cannot construct a bound that structurally
    lies about its own rigor.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    upper_bound: float = Field(
        ...,
        ge=0.0,
        description=(
            "Certified upper bound on ``||r(u_tilde)||`` over the declared "
            "domain, in the norm the verifier used (documented in ``notes``)."
        ),
    )
    rigor: BoundRigor = Field(
        ...,
        description=(
            "``'rigorous'`` — proven over the whole domain. "
            "``'heuristic'`` — dense-grid sample only. "
            "``'failed'`` — bound could not be computed (see ``failure_reason``)."
        ),
    )
    backend: VerifierBackend = Field(..., description="Which verifier produced this bound.")
    domain_coverage: DomainCoverage = Field(
        ...,
        description="How much of the domain the bound is asserted over.",
    )
    compile_wall_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock seconds spent inside JIT / graph compilation for this "
            "call. Non-zero only on the *first* call for a given shape/dtype "
            "(§4 AC5)."
        ),
    )
    cert_wall_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Total wall-clock seconds for the certification call, including "
            "compile time. ``cert_wall_s >= compile_wall_s`` is validator-"
            "enforced."
        ),
    )
    steady_state_wall_s: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock seconds excluding compile — the amortised cost a batch "
            "will pay. Equal to ``cert_wall_s - compile_wall_s`` in the "
            "common case; a verifier may record it separately for a warm run."
        ),
    )
    hardware_meta: HardwareMeta = Field(..., description="Run-time provenance.")
    failure_reason: str | None = Field(
        default=None,
        description=(
            "Populated iff ``rigor='failed'``. Names the failure class (missing "
            "dependency, budget overrun, verifier internal error)."
        ),
    )
    notes: str = Field(
        default="",
        description=(
            "Free-form provenance (norm used, sample density, algorithm "
            "specifics). Not consumed by validators."
        ),
    )

    @model_validator(mode="after")
    def _rigor_coverage_consistency(self) -> CertifiedResidualBound:
        # §4 AC3: rigorous ⇒ full coverage. Structural guard so "heuristic
        # dressed as rigorous" is impossible at the type-system level.
        if self.rigor == "rigorous" and self.domain_coverage != "full":
            raise ValueError(
                "rigor='rigorous' requires domain_coverage='full' "
                "(spec §4 AC3 — grid_sampled/partial coverage is at most 'heuristic')"
            )
        # §4 AC5: cost accounting is coherent.
        if self.cert_wall_s + 1e-12 < self.compile_wall_s:
            raise ValueError(
                f"cert_wall_s ({self.cert_wall_s}) must be >= compile_wall_s "
                f"({self.compile_wall_s}) (spec §4 AC5)"
            )
        # Failure semantics.
        if self.rigor == "failed" and not self.failure_reason:
            raise ValueError("rigor='failed' requires a non-empty failure_reason")
        if self.rigor != "failed" and self.failure_reason:
            raise ValueError(
                f"rigor={self.rigor!r} cannot carry a failure_reason (got {self.failure_reason!r})"
            )
        return self


__all__ = [
    "BoundRigor",
    "CertificationBudget",
    "CertifiedModel",
    "CertifiedResidualBound",
    "DomainCoverage",
    "DomainKind",
    "DomainSpec",
    "Device",
    "Dtype",
    "HardwareMeta",
    "VerifierBackend",
    "VerifierUnavailableError",
]
